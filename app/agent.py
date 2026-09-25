"""LangGraph ReAct agent on OpenRouter: primary model with a cross-provider fallback."""
from __future__ import annotations

import re
import time
import warnings
from dataclasses import dataclass, field

import httpx
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI

from . import config
from .prompts import SYSTEM_PROMPT
from .tools import TurnContext, build_tools
from .tracing import TraceHandler

warnings.filterwarnings("ignore", message=".*create_react_agent.*")
from langgraph.prebuilt import create_react_agent  # noqa: E402  (after the warning filter)

# Tests / offline benchmarking point this at an httpx.MockTransport instead of the network.
TEST_TRANSPORT: httpx.BaseTransport | None = None

FALLBACK_REPLY = ("Sorry, I'm having trouble reaching my brain right now. I've flagged this for the team and "
                  "someone will get back to you shortly.")
TRUTH_NOTE = ("\n\n[System note: your previous draft stated figures without calling a tool. Call the appropriate tool "
              "now to fetch fresh data, then answer using only its output.]")
_FACTUAL = re.compile(r"[₹%]|\d")

# Chain-of-thought text that ended up in the visible reply.
_LEAK = re.compile(r"^\s*(we need|we should|we must|we have to|i need to|i should|the user (is asking|asks|wants|says|question)|"
                   r"let'?s (craft|answer|think|see|produce|compose|write)|let me (think|craft|see|check)|okay,? (so|the|let)|first,? (we|i))",
                   re.I)
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S | re.I)


class ReplyRejected(Exception):
    """Raised inside the model chain so with_fallbacks moves on to the next model."""


def make_reply_guard(on_reject=None):
    def check(msg):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            if isinstance(msg.content, str) and "<think>" in msg.content.lower():
                msg = msg.model_copy(update={"content": _THINK_BLOCK.sub("", msg.content).strip()})
            text = _text(msg)
            reason = None
            if (msg.response_metadata or {}).get("finish_reason") == "length":
                reason = "reply truncated (max_tokens reached)"
            elif "<think>" in text.lower() or _LEAK.search(text):
                reason = "reasoning leaked into the reply"
            if reason:
                if on_reject:
                    on_reject((msg.response_metadata or {}).get("model_name"), reason)
                raise ReplyRejected(reason)
        return msg
    return RunnableLambda(check)


QUOTA_REPLY = ("I can't answer right now: this demo's free AI quota for today has been used up. It resets daily "
               "(or the owner can add OpenRouter credits). I've flagged this for the team.")

EMPTY_REPLY = "Sorry, I couldn't put an answer together for that. Could you rephrase, or shall I ask a teammate to help?"


@dataclass
class TurnResult:
    reply: str
    llm_calls: list = field(default_factory=list)
    tool_calls: list = field(default_factory=list)
    total_latency_ms: int = 0
    model_used: str | None = None
    fallback_used: bool = False
    error: str | None = None
    ctx: TurnContext | None = None
    truth_guard_retry: bool = False


def make_chat_model(spec: str) -> ChatOpenAI:
    """spec is an OpenRouter model ID, or 'groq:<groq model id>' to call Groq's OpenAI-compatible API directly."""
    provider, model = config.parse_model_spec(spec)
    # Sent via extra_body so the payload carries `max_tokens` (langchain-openai would send `max_completion_tokens`).
    extra: dict = {"max_tokens": config.LLM_MAX_TOKENS}
    if provider == "groq":
        base_url, headers = config.GROQ_BASE_URL, {}
        if "gpt-oss" in model:  # keep hidden reasoning short so it can't exhaust max_tokens
            extra["reasoning_effort"] = "low"
    else:
        base_url, headers = config.OPENROUTER_BASE_URL, {"X-Title": "AI Portfolio Analyst"}
        extra["usage"] = {"include": True}  # OpenRouter usage accounting -> usage.cost
    kwargs = dict(
        model=model,
        api_key=config.provider_api_key(provider) or "missing-key",
        base_url=base_url,
        timeout=config.LLM_TIMEOUT_S,
        # Groq answers 429 with a long Retry-After; fail over to the next model instead of sleeping.
        max_retries=0 if provider == "groq" else config.LLM_MAX_RETRIES,
        temperature=0.2,
        extra_body=extra,
        default_headers=headers,
        metadata={"model_spec": spec},  # lets the trace callback name the provider
    )
    if TEST_TRANSPORT is not None:
        kwargs["http_client"] = httpx.Client(transport=TEST_TRANSPORT)
    return ChatOpenAI(**kwargs)


def compact_tool_schema(tool) -> dict:
    """OpenAI-format tool spec minus pydantic boilerplate (titles, `default: null`, `anyOf: [x, null]`).
    Only the LLM-facing copy shrinks: the tool object (and its pydantic validation) is unchanged."""
    def walk(node):
        if isinstance(node, dict):
            node = {k: walk(v) for k, v in node.items() if k not in ("title", "default")}
            opts = [o for o in node.get("anyOf", []) if o.get("type") != "null"]
            if "anyOf" in node and len(opts) == 1:
                return {**{k: v for k, v in node.items() if k != "anyOf"}, **opts[0]}
            return node
        if isinstance(node, list):
            return [walk(x) for x in node]
        return node
    return walk(convert_to_openai_tool(tool))


def build_model(tools, on_reject=None):
    guard = make_reply_guard(on_reject)
    chain = config.model_chain()  # configured models whose provider has a key; chain[0] is the effective primary
    if not chain:
        raise RuntimeError("No LLM API key configured: set OPENROUTER_API_KEY and/or GROQ_API_KEY (for groq: models)")
    schemas = [compact_tool_schema(t) for t in tools]
    bound = [make_chat_model(m).bind_tools(schemas) | guard for m in chain]
    return bound[0].with_fallbacks(bound[1:])


def _to_messages(history: list[dict]):
    out = []
    for m in history:
        if m["role"] == "user":
            out.append(HumanMessage(m["content"]))
        elif m["role"] == "human_agent":
            out.append(AIMessage("[Team member] " + m["content"]))
        else:
            out.append(AIMessage(m["content"]))
    return out


def whatsapp_format(text: str) -> str:
    """Models drift into Markdown; WhatsApp bold is a single asterisk and it has no headers."""
    text = text.replace("\u202f", " ").replace("\u00a0", " ")  # gpt-oss emits narrow no-break spaces inside amounts
    text = _THINK_BLOCK.sub("", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"*\1*", text)
    text = re.sub(r"(?m)^#{1,6}\s+(.+)$", r"*\1*", text)
    return text.strip()


def _text(msg) -> str:
    c = msg.content
    if isinstance(c, list):
        c = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in c)
    return (c or "").strip()


def run_turn(user_id: str, conversation_id: str, user_text: str, history: list[dict]) -> TurnResult:
    t0 = time.perf_counter()
    ctx = TurnContext(user_id=user_id, conversation_id=conversation_id)
    handler = TraceHandler()
    reply, error, retried = "", None, False
    try:
        tools = build_tools(ctx)
        model = build_model(tools, handler.reject)  # a callable stops langgraph binding tools again
        graph = create_react_agent(lambda state, runtime: model, tools, prompt=SYSTEM_PROMPT)
        msgs = _to_messages(history)
        cfg = {"callbacks": [handler], "recursion_limit": 12}
        result = graph.invoke({"messages": msgs + [HumanMessage(user_text)]}, config=cfg)
        last = result["messages"][-1]
        reply = _text(last) if isinstance(last, AIMessage) else ""
        if reply and not handler.tool_calls and _FACTUAL.search(reply):
            # figures without a tool call: the model answered from memory, so retry once and force a fetch
            retried = True
            result = graph.invoke({"messages": msgs + [HumanMessage(user_text + TRUTH_NOTE)]}, config=cfg)
            last = result["messages"][-1]
            reply = _text(last) if isinstance(last, AIMessage) else ""
        reply = whatsapp_format(reply)
        if not reply:
            error = "empty model reply"
            reply = EMPTY_REPLY
    except Exception as e:  # all models failed, timeout, recursion limit
        error = f"{type(e).__name__}: {e}"[:300]
        reply = FALLBACK_REPLY
        if "free-models-per-day" in str(e):  # account-level cap: every model fails the same way
            error, reply = "OpenRouter free daily request limit reached (free-models-per-day)", QUOTA_REPLY
    total = int((time.perf_counter() - t0) * 1000)

    ok = [c for c in handler.llm_calls if not c.get("error")]
    chain = config.model_chain()
    fallback_used = bool(chain) and any(c["model"] and c["model"] != chain[0] for c in handler.llm_calls)
    return TurnResult(reply=reply, llm_calls=sorted(handler.llm_calls, key=lambda c: c["started_ms"]),
                      tool_calls=sorted(handler.tool_calls, key=lambda c: c["started_ms"]), total_latency_ms=total,
                      model_used=(ok[-1]["model"] if ok else None), fallback_used=fallback_used, error=error, ctx=ctx, truth_guard_retry=retried)


def warm_up() -> None:
    """Build one graph at startup so the first real request doesn't pay import / schema-conversion costs."""
    if not config.model_chain():
        return  # nothing configured; main.py already warned
    ctx = TurnContext(user_id="warmup", conversation_id="warmup")
    tools = build_tools(ctx)
    model = build_model(tools)
    create_react_agent(lambda state, runtime: model, tools, prompt=SYSTEM_PROMPT)
