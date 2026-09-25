"""End-to-end smoke tests: FastAPI -> LangGraph agent -> (fake) OpenRouter -> tools -> SQLite."""
import json

import pytest
from fastapi.testclient import TestClient

from langchain_core.messages import AIMessage

from app import agent, db
from app.main import app
from tests.fake_openrouter import FakeOpenRouter

ADMIN = ("admin", "pw")


@pytest.fixture()
def env(tmp_db, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setenv("ADMIN_PASSWORD", "pw")
    monkeypatch.setenv("PRIMARY_MODEL", "primary/model")
    monkeypatch.setenv("FALLBACK_MODEL", "fallback/model")
    fake = FakeOpenRouter()
    monkeypatch.setattr(agent, "TEST_TRANSPORT", fake.transport)
    with TestClient(app) as client:
        yield client, fake


def chat(client, uid, text):
    r = client.post("/api/chat", json={"user_id": uid, "message": text})
    assert r.status_code == 200, r.text
    return r.json()


def test_retail_question_end_to_end(env):
    client, fake = env
    out = chat(client, "U001", "Show me my retail properties")
    assert "₹21.20 Cr" in out["reply"]  # P001 ₹12 Cr + P003 ₹9.2 Cr, straight from the tool
    t = out["timing"]
    assert t["llm_calls"] == 2 and t["tool_calls"] == 1 and t["cost_usd"] == pytest.approx(0.0008)
    assert t["model"] == "primary/model" and t["fallback_used"] is False and out["attention_flags"] == []

    first = fake.requests[0]["body"]
    assert first["messages"][0]["role"] == "system" and "WhatsApp" in first["messages"][0]["content"]
    assert first["usage"] == {"include": True} and first["max_tokens"] == 700
    assert first["model"] == "primary/model"
    for tool in first["tools"]:
        params = set(tool["function"]["parameters"]["properties"])
        assert not params & {"user_id", "conversation_id", "uid"}, tool["function"]["name"]
    assert "user_id" not in json.dumps(first["tools"])

    detail = client.get("/api/admin/conversations/U001", auth=ADMIN).json()
    tr = detail["messages"][-1]["trace"]
    assert tr["model_used"] == "primary/model" and len(tr["llm_calls"]) == 2
    assert tr["tool_calls"][0]["name"] == "get_portfolio_analysis"
    assert tr["tool_calls"][0]["args"] == {"filters": {"property_type": "Retail"}}
    assert tr["llm_calls"][0]["requested_tools"] == ["get_portfolio_analysis"] and tr["llm_calls"][0]["cost"] == 0.0004
    assert detail["messages"][-1]["latency_ms"] == tr["total_latency_ms"]


def test_history_is_sent_and_windowed(env):
    client, fake = env
    for i in range(12):
        chat(client, "U002", f"hello {i}" if i % 2 else "Hi")
    fake.requests.clear()
    chat(client, "U002", "What is my total portfolio value?")
    sent = fake.requests[0]["body"]["messages"]
    assert len([m for m in sent if m["role"] in ("user", "assistant")]) == 21  # 20 history + the new message


def test_fallback_model_used_when_primary_is_down(env, monkeypatch):
    client, _ = env
    fake = FakeOpenRouter(fail_models={"primary/model"})
    monkeypatch.setattr(agent, "TEST_TRANSPORT", fake.transport)
    out = chat(client, "U001", "Show me my retail properties")
    assert "₹21.20 Cr" in out["reply"] and out["timing"]["fallback_used"] is True
    assert out["timing"]["model"] == "fallback/model"
    assert "fallback model used" in out["attention_flags"]
    assert db.get_conversation("U001")["needs_attention"] is True
    errs = client.get("/api/admin/conversations/U001", auth=ADMIN).json()["messages"][-1]["trace"]["llm_calls"]
    assert any(c["error"] and c["model"] == "primary/model" for c in errs)


def test_both_models_down_gives_graceful_reply(env, monkeypatch):
    client, _ = env
    monkeypatch.setattr(agent, "TEST_TRANSPORT", FakeOpenRouter(fail_models={"primary/model", "fallback/model"}).transport)
    out = chat(client, "U001", "What is my total portfolio value?")
    assert "flagged this for the team" in out["reply"]
    assert any(f.startswith("agent error") for f in out["attention_flags"])


def test_update_executes_and_flags_large_change(env):
    client, _ = env
    out = chat(client, "U001", "Change my Bandra retail property value to ₹12.5 crore")
    assert "₹12.00 Cr → ₹12.50 Cr" in out["reply"] and out["attention_flags"] == []
    assert db.get_property("U001", "P001")["current_estimated_value_inr"] == 125_000_000
    # the write tool itself returns the new totals (₹30.20 Cr), so the model never has to add 0.5 Cr to 29.70 Cr
    upd = [c for c in client.get("/api/admin/conversations/U001", auth=ADMIN).json()["messages"][-1]["trace"]["tool_calls"]]
    assert upd[0]["name"] == "update_property"
    from app.tools import TurnContext, build_tools
    now = json.loads({t.name: t for t in build_tools(TurnContext("U001", "conv_U001"))}["update_property"].invoke(
        {"property_ref": "P001", "field": "value", "new_value": "13 crore"}))["portfolio_now"]
    assert now["total_value_fmt"] == "₹30.70 Cr" and now["count"] == 3
    big = chat(client, "U001", "Change my Bandra retail property value to 30 crore")
    assert any("large value change" in f for f in big["attention_flags"])
    assert [a["field"] for a in db.audit_for_user("U001")][:2] == ["current_estimated_value_inr"] * 2


def test_add_property_asks_once_then_writes(env):
    client, _ = env
    first = chat(client, "U004", "Add a 3000 sq ft retail property in Indiranagar worth ₹4.2 crore")
    assert "annual rent" in first["reply"] and len(db.list_properties("U004")) == 3  # nothing written yet
    second = chat(client, "U004", "Add a 3000 sq ft retail property in Indiranagar worth ₹4.2 crore")
    assert "City assumed to be Bengaluru" in second["reply"]
    rows = db.list_properties("U004")
    new = [p for p in rows if p["property_id"] == "P013"][0]
    assert (new["type_group"], new["city"], new["current_estimated_value_inr"], new["occupancy_status"]) == \
        ("Retail", "Bengaluru", 42_000_000, "Vacant")


def test_scenario_does_not_write(env):
    client, _ = env
    before = db.list_properties("U001")
    out = chat(client, "U001", "What if I exclude the Bandra one?")
    assert "Scenario: excluding P001" in out["reply"]
    assert db.list_properties("U001") == before


def test_handoff_and_frustration_flags(env):
    client, _ = env
    out = chat(client, "U003", "I want to speak to a human about selling")
    assert any(f.startswith("agent asked for human") for f in out["attention_flags"])
    assert any("frustration keyword: 'human'" == f for f in out["attention_flags"])
    conv = client.get("/api/admin/conversations?needs_attention=true", auth=ADMIN).json()
    assert [c["user_id"] for c in conv] == ["U003"]
    client.post("/api/admin/conversations/U003/resolve", auth=ADMIN)
    assert client.get("/api/admin/conversations?needs_attention=true", auth=ADMIN).json() == []


def test_takeover_pauses_agent_and_human_reply_shows(env):
    client, fake = env
    client.post("/api/admin/conversations/U001/takeover", auth=ADMIN)
    n = len(fake.requests)
    out = chat(client, "U001", "Hello?")
    assert out["reply"] is None and out["agent_paused"] is True and len(fake.requests) == n  # no LLM call
    client.post("/api/admin/conversations/U001/reply", json={"content": "Hi Rahul, this is the team"}, auth=ADMIN)
    msgs = client.get("/api/conversations/U001/messages").json()
    assert [m["role"] for m in msgs["messages"]] == ["user", "human_agent"] and msgs["agent_paused"] is True
    client.post("/api/admin/conversations/U001/handback", auth=ADMIN)
    assert "properties worth" in chat(client, "U001", "What is my portfolio?")["reply"]
    hist = fake.requests[-2]["body"]["messages"]
    assert any(m["role"] == "assistant" and m["content"].startswith("[Team member]") for m in hist)


def test_new_user_can_chat_with_empty_portfolio(env):
    client, _ = env
    u = client.post("/api/users", json={"name": "Asha Rao", "phone": "+91 98765 43210"}).json()
    assert u["user_id"] == "U005" and u["phone"] == "+91 98765 43210"
    out = chat(client, u["user_id"], "What is my total portfolio value?")
    assert out["reply"] and out["timing"]["tool_calls"] == 1
    assert client.post("/api/users", json={"name": "X", "phone": "abc"}).status_code == 422


def test_admin_auth(env, monkeypatch):
    client, _ = env
    assert client.get("/api/admin/stats").status_code == 401
    assert client.get("/api/admin/stats", auth=("a", "wrong")).status_code == 401
    assert client.get("/admin").status_code == 401
    assert client.get("/admin", auth=ADMIN).status_code == 200
    assert client.get("/api/admin/stats", auth=ADMIN).json()["conversations"] == 4
    monkeypatch.setenv("ADMIN_PASSWORD", "")
    assert client.get("/api/admin/stats", auth=ADMIN).status_code == 503


def test_validation(env):
    client, _ = env
    assert client.post("/api/chat", json={"user_id": "U999", "message": "hi"}).status_code == 404
    assert client.post("/api/chat", json={"user_id": "U001", "message": "   "}).status_code == 422
    assert client.get("/api/conversations/U999/messages").status_code == 404


def test_bench_users_only_deletable(env):
    client, _ = env
    assert client.delete("/api/admin/users/U001", auth=ADMIN).status_code == 403
    u = client.post("/api/admin/users/U001/clone", json={"name": "bench-U001-1"}, auth=ADMIN).json()
    assert len(db.list_properties(u["user_id"])) == 3 and len(db.list_properties("U001")) == 3
    assert client.delete(f"/api/admin/users/{u['user_id']}", auth=ADMIN).status_code == 200
    assert db.get_user(u["user_id"]) is None


def test_truth_guard_retries_when_model_answers_from_memory(env, monkeypatch):
    client, _ = env
    fake = FakeOpenRouter(answer_from_memory_first=True)
    monkeypatch.setattr(agent, "TEST_TRANSPORT", fake.transport)
    out = chat(client, "U001", "Which one is performing better?")
    assert out["timing"]["truth_guard_retry"] is True and out["timing"]["tool_calls"] == 1
    assert "6.52%" not in out["reply"] and "properties worth" in out["reply"]  # the tool-backed answer, not the memory one
    assert "[System note" in fake.requests[-2]["body"]["messages"][-1]["content"]
    assert "[System note" not in [m for m in db.recent_messages("conv_U001", 5) if m["role"] == "user"][-1]["content"]


def test_greeting_needs_no_tool_and_no_retry(env):
    client, _ = env
    out = chat(client, "U001", "hello")
    assert out["timing"]["tool_calls"] == 0 and out["timing"]["truth_guard_retry"] is False


def test_whatsapp_format_cleanup():
    assert agent.whatsapp_format("**Bold** and\n## Header\nplain") == "*Bold* and\n*Header*\nplain"


@pytest.mark.parametrize("kind", ["length", "leak"])
def test_reply_guard_rejects_truncated_or_leaked_reasoning(env, monkeypatch, kind):
    client, _ = env
    monkeypatch.setattr(agent, "TEST_TRANSPORT", FakeOpenRouter(bad_reply={"primary/model": kind}).transport)
    out = chat(client, "U001", "What does my portfolio look like?")
    assert "properties worth" in out["reply"] and "We need" not in out["reply"] and "Actual portfolio:" not in out["reply"]
    assert out["timing"]["fallback_used"] is True and out["timing"]["model"] == "fallback/model"
    calls = client.get("/api/admin/conversations/U001", auth=ADMIN).json()["messages"][-1]["trace"]["llm_calls"]
    assert any((c["error"] or "").startswith("rejected") and c["model"] == "primary/model" for c in calls)


def test_special_spaces_in_amounts_are_normalised():
    assert agent.whatsapp_format("Value: ₹12.00\u202fCr, rent ₹72.0\u00a0L") == "Value: ₹12.00 Cr, rent ₹72.0 L"


def test_think_blocks_are_stripped():
    assert agent.whatsapp_format("<think>plan the answer</think>You have *3* properties.") == "You have *3* properties."
    assert agent.make_reply_guard().invoke(AIMessage("<think>hmm</think>Hello")).content == "Hello"
    with pytest.raises(agent.ReplyRejected):
        agent.make_reply_guard().invoke(AIMessage("<think>never closed and the answer is missing"))


def test_daily_free_quota_gets_a_clear_message(env, monkeypatch):
    client, _ = env
    fake = FakeOpenRouter(fail_models={"primary/model", "fallback/model"}, fail_status=429,
                          fail_message="Rate limit exceeded: free-models-per-day. Add 10 credits to unlock 1000 free model requests per day")
    monkeypatch.setattr(agent, "TEST_TRANSPORT", fake.transport)
    out = chat(client, "U001", "What does my portfolio look like?")
    assert "free AI quota for today" in out["reply"] and "brain" not in out["reply"]
    assert any("free daily request limit" in f for f in out["attention_flags"])


GROQ_MODEL = "llama-3.3-70b-versatile"


@pytest.fixture()
def groq_env(env, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.setenv("PRIMARY_MODEL", f"groq:{GROQ_MODEL}")
    monkeypatch.setenv("FALLBACK_MODEL", "fallback/model")
    return env


def test_model_spec_parsing():
    from app import config
    assert config.parse_model_spec("groq:llama-3.3-70b-versatile") == ("groq", "llama-3.3-70b-versatile")
    assert config.parse_model_spec("groq:openai/gpt-oss-120b") == ("groq", "openai/gpt-oss-120b")
    assert config.parse_model_spec("poolside/laguna-xs-2.1:free") == ("openrouter", "poolside/laguna-xs-2.1:free")
    assert config.parse_model_spec("openrouter:openai/gpt-4.1-mini") == ("openrouter", "openai/gpt-4.1-mini")
    assert config.parse_model_spec("openai/gpt-4.1-mini") == ("openrouter", "openai/gpt-4.1-mini")


def test_groq_primary_is_called_directly_with_its_own_key(groq_env):
    client, fake = groq_env
    out = chat(client, "U001", "Show me my retail properties")
    assert "₹21.20 Cr" in out["reply"] and out["timing"]["fallback_used"] is False
    req = fake.requests[0]
    assert req["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert req["headers"]["authorization"] == "Bearer gsk-test"          # the Groq key, not the OpenRouter one
    assert req["body"]["model"] == GROQ_MODEL                            # provider prefix stripped
    assert "usage" not in req["body"] and req["body"]["max_tokens"] == 700  # OpenRouter-only param not sent to Groq
    assert "x-title" not in req["headers"]
    assert out["timing"]["model"] == f"groq:{GROQ_MODEL}"                # trace keeps the provider prefix
    call = client.get("/api/admin/conversations/U001", auth=ADMIN).json()["messages"][-1]["trace"]["llm_calls"][0]
    assert call["model"] == f"groq:{GROQ_MODEL}" and call["cost_estimated"] is True and call["cost"] > 0


def test_gpt_oss_on_groq_asks_for_low_reasoning(groq_env, monkeypatch):
    client, fake = groq_env
    monkeypatch.setenv("PRIMARY_MODEL", "groq:openai/gpt-oss-120b")
    chat(client, "U001", "hello")
    assert fake.requests[0]["body"]["model"] == "openai/gpt-oss-120b"
    assert fake.requests[0]["body"]["reasoning_effort"] == "low"


def test_groq_failure_falls_back_to_openrouter(groq_env, monkeypatch):
    client, _ = groq_env
    fake = FakeOpenRouter(fail_models={GROQ_MODEL})
    monkeypatch.setattr(agent, "TEST_TRANSPORT", fake.transport)
    out = chat(client, "U001", "Show me my retail properties")
    assert "₹21.20 Cr" in out["reply"] and out["timing"]["fallback_used"] is True
    assert out["timing"]["model"] == "fallback/model" and "fallback model used" in out["attention_flags"]
    hosts = [r["url"].split("/")[2] for r in fake.requests]
    assert "api.groq.com" in hosts and "openrouter.ai" in hosts
    assert [r["headers"]["authorization"] for r in fake.requests if "openrouter" in r["url"]][0] == "Bearer sk-test"


def test_groq_model_without_key_is_skipped_not_a_crash(groq_env, monkeypatch):
    client, fake = groq_env
    monkeypatch.delenv("GROQ_API_KEY")
    out = chat(client, "U001", "Show me my retail properties")
    assert "₹21.20 Cr" in out["reply"] and out["timing"]["model"] == "fallback/model"
    assert out["timing"]["fallback_used"] is False  # the skipped model never counted as the primary
    assert all("groq.com" not in r["url"] for r in fake.requests)


def test_no_keys_at_all_gives_a_clear_error(groq_env, monkeypatch):
    client, _ = groq_env
    monkeypatch.delenv("GROQ_API_KEY")
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    out = chat(client, "U001", "hello")
    assert "flagged this for the team" in out["reply"]
    assert any("No LLM API key configured" in f for f in out["attention_flags"])


def test_compact_tool_schemas_are_smaller_but_equivalent(tmp_db):
    from langchain_core.utils.function_calling import convert_to_openai_tool
    from app.tools import TurnContext, build_tools
    tools = build_tools(TurnContext("U001", "conv_U001"))
    full = json.dumps([convert_to_openai_tool(t) for t in tools], separators=(",", ":"))
    compact_specs = [agent.compact_tool_schema(t) for t in tools]
    compact = json.dumps(compact_specs, separators=(",", ":"))
    assert len(compact) < 0.8 * len(full)
    assert '"title"' not in compact and '"null"' not in compact and '"default"' not in compact
    by_name = {c["function"]["name"]: c["function"]["parameters"] for c in compact_specs}
    filters = by_name["get_portfolio_analysis"]["properties"]["filters"]["properties"]
    assert filters["min_value"]["type"] == "string" and "description" in filters["min_value"]   # Optional[str] collapsed
    assert by_name["add_property"]["required"] if "required" in by_name["add_property"] else True
    assert set(by_name["update_property"]["required"]) == {"property_ref", "field", "new_value"}
    upd = {t.name: t for t in tools}["update_property"]
    with pytest.raises(Exception):
        upd.invoke({"property_ref": "P001"})


def test_groq_models_do_not_sleep_on_retry_after(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    assert agent.make_chat_model("groq:openai/gpt-oss-120b").max_retries == 0   # fail over instead of a ~30 s sleep
    assert agent.make_chat_model("some/model:free").max_retries == 1


def test_write_tools_return_both_yields(tmp_db):
    from app.tools import TurnContext, build_tools
    upd = {t.name: t for t in build_tools(TurnContext("U001", "conv_U001"))}["update_property"]
    now = json.loads(upd.invoke({"property_ref": "P001", "field": "value", "new_value": "14 crore"}))["portfolio_now"]
    assert now["total_value_fmt"] == "₹31.70 Cr" and now["yield_on_income_producing_assets_pct"] == 5.69
