"""A scripted stand-in for OpenRouter's /chat/completions, as an httpx.MockTransport.

Speaks the real wire format (tool_calls, usage.cost), so the whole LangChain path is exercised
without a network or API key. Routing is keyword-based and only good enough for tests / offline bench.
"""
import json
import re
import time

import httpx


def _tool_call(i, name, args):
    return {"id": f"call_{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def route(messages: list[dict]) -> list[tuple[str, dict]]:
    """Pick tool calls from the last user message (and earlier assistant turns for follow-ups)."""
    user = next(m["content"] for m in reversed(messages) if m["role"] == "user")
    t = user.lower()
    asked_optional = any(m["role"] == "assistant" and "annual rent" in (m.get("content") or "").lower() for m in messages)
    if re.search(r"\b(human|lawyer|sell my|want to sell)\b", t):
        return [("flag_for_human", {"reason": user[:80]})]
    if t.startswith("add "):
        area = re.search(r"([\d,]+)\s*sq", t)
        val = re.search(r"([\d.]+)\s*(crore|cr|lakh|l)\b", t)
        loc = re.search(r"\bin ([a-z ]+?)(?: worth|$)", t)
        typ = next((x for x in ("retail", "office", "residential") if x in t), "retail")
        args = {"property_type": typ.title(), "locality": loc.group(1).title() if loc else None,
                "area_sqft": float(area.group(1).replace(",", "")) if area else None,
                "value": f"{val.group(1)} {val.group(2)}" if val else None}
        if asked_optional:
            args["optional_details_asked"] = True
        return [("add_property", args)]
    if re.match(r"(change|update|set) ", t) and "value" in t:
        val = re.search(r"([\d.]+)\s*(crore|cr|lakh|l)\b", t)
        ref = "Bandra retail" if "bandra" in t else "P001"
        return [("update_property", {"property_ref": ref, "field": "value", "new_value": f"{val.group(1)} {val.group(2)}"})]
    if "exclude" in t and "bandra" in t:
        return [("get_portfolio_analysis", {"exclude_properties": ["Bandra"]})]
    if re.search(r"above .*crore", t):
        return [("get_portfolio_analysis", {"filters": {"min_value": "10 crore"}})]
    if "retail" in t or "performing" in t or "better" in t:
        return [("get_portfolio_analysis", {"filters": {"property_type": "Retail"}})]
    if "mumbai" in t:
        return [("get_portfolio_analysis", {"filters": {"location": "Mumbai"}})]
    if re.fullmatch(r"(hi|hello|hey|thanks|thank you)[!. ]*", t.strip()):
        return []
    return [("get_portfolio_analysis", {})]


def final_text(tool_msgs: list[str]) -> str:
    lines = []
    for raw in tool_msgs:
        d = json.loads(raw)
        st = d.get("status")
        if st == "ok":
            pre = (d.get("scenario_description") or "") + " " if d.get("is_hypothetical") else ""
            lines.append(f"{pre}{d['totals']['count']} properties worth {d['totals']['value_fmt']}, rent {d['totals']['annual_rent_fmt']}.")
        elif st == "updated":
            lines.append(f"Updated {d['property_id']} {d['field']}: {d['old']} → {d['new']}.")
        elif st == "added":
            lines.append(f"Added {d['property']['property_id']}. " + " ".join(d["assumptions"]))
        elif st == "ask_optional":
            lines.append("What's the annual rent, and is it tenanted, vacant or self-occupied?")
        elif st == "missing_fields":
            lines.append("I still need: " + ", ".join(d["missing"]))
        elif st == "ambiguous":
            lines.append("Which one: " + ", ".join(c["property_id"] for c in d["candidates"]) + "?")
        elif st == "confirmation_required":
            lines.append("Please confirm removal (yes/no).")
        elif st == "flagged":
            lines.append("I've asked a teammate to follow up.")
        else:
            lines.append(f"Status: {st}")
    return " ".join(lines)


class FakeOpenRouter:
    def __init__(self, latency_s: float = 0.0, fail_models=(), fail_status: int = 500, cost: float = 0.0004,
                 answer_from_memory_first: bool = False, bad_reply: dict | None = None, fail_message: str = "provider down"):
        self.fail_message = fail_message
        self.bad_reply = bad_reply or {}  # model -> "length" | "leak": that model's final answers are unusable
        self.latency_s, self.fail_models, self.fail_status, self.cost = latency_s, set(fail_models), fail_status, cost
        self.answer_from_memory_first = answer_from_memory_first
        self.requests: list[dict] = []
        self.transport = httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append({"body": body, "headers": dict(request.headers), "url": str(request.url)})
        model = body["model"]
        if model in self.fail_models:
            return httpx.Response(self.fail_status, json={"error": {"message": self.fail_message, "code": self.fail_status}})
        if self.latency_s:
            time.sleep(self.latency_s)
        msgs = body["messages"]
        tool_results = []
        for m in reversed(msgs):
            if m["role"] != "tool":
                break
            tool_results.insert(0, m["content"])
        if tool_results:
            msg = {"role": "assistant", "content": final_text(tool_results)}
            finish = "stop"
        elif self.answer_from_memory_first and "[System note" not in msgs[-1]["content"]:
            msg = {"role": "assistant", "content": "**P003** has the best yield at 6.52%."}  # figures, no tool call
            finish = "stop"
        else:
            calls = route(msgs)
            if calls:
                msg = {"role": "assistant", "content": None,
                       "tool_calls": [_tool_call(i, n, a) for i, (n, a) in enumerate(calls)]}
                finish = "tool_calls"
            else:
                msg = {"role": "assistant", "content": "Hi! Ask me anything about your portfolio."}
                finish = "stop"
        if finish == "stop" and self.bad_reply.get(model) == "length":
            msg["content"], finish = "Your portfolio has 3 properties. Actual portfolio:", "length"
        elif finish == "stop" and self.bad_reply.get(model) == "leak":
            msg["content"] = "We need to answer the user's question: \"what does my portfolio look like\". We have the tool output."
        ptoks = sum(len(json.dumps(m)) for m in msgs) // 4
        return httpx.Response(200, json={
            "id": "gen-test", "object": "chat.completion", "created": 0, "model": model,
            "choices": [{"index": 0, "finish_reason": finish, "message": msg}],
            "usage": {"prompt_tokens": ptoks, "completion_tokens": 40, "total_tokens": ptoks + 40,
                      # only OpenRouter reports cost; Groq's usage block has tokens only
                      **({"cost": self.cost} if "openrouter.ai" in str(request.url) else {})}})
