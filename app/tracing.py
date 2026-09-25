"""LangChain callback handler: per-LLM-call and per-tool-call timings, tokens and cost -> turn_traces."""
from __future__ import annotations

import json
import threading
import time
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

# Fallback price table (USD per token) used only when OpenRouter's usage block has no `cost`.
PRICES = {"openai/gpt-4.1-mini": (0.4e-6, 1.6e-6), "google/gemini-2.5-flash": (0.3e-6, 2.5e-6),
          # Groq returns no cost; approximate list prices, shown as estimates ("*") in the admin.
          "groq:llama-3.3-70b-versatile": (0.59e-6, 0.79e-6), "groq:openai/gpt-oss-120b": (0.15e-6, 0.60e-6),
          "groq:openai/gpt-oss-20b": (0.075e-6, 0.30e-6)}


def summarise_result(content: Any, limit: int = 220) -> str:
    text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, default=str)
    try:
        d = json.loads(text)
    except (ValueError, TypeError):
        return text[:limit]
    if not isinstance(d, dict):
        return text[:limit]
    parts = [str(d.get("status", ""))]
    if d.get("error"):
        parts.append(str(d["error"]))
    if d.get("scenario_description"):
        parts.append(d["scenario_description"])
    if isinstance(d.get("totals"), dict):
        parts.append(f"{d['totals'].get('count')} properties, {d['totals'].get('value_fmt')}")
    for k in ("matches", "candidates"):
        if isinstance(d.get(k), list):
            parts.append(f"{len(d[k])} {k}")
    if d.get("property_id") and d.get("old") is not None:
        parts.append(f"{d['property_id']} {d.get('field')}: {d['old']} → {d['new']}")
    if isinstance(d.get("property"), dict):
        parts.append(f"{d['property'].get('property_id')} {d['property'].get('label')}")
    if isinstance(d.get("missing"), list):
        parts.append("missing " + ", ".join(d["missing"]))
    return " | ".join(x for x in parts if x)[:limit]


class TraceHandler(BaseCallbackHandler):
    raise_error = False

    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.llm_calls: list[dict] = []
        self.tool_calls: list[dict] = []
        self._llm: dict = {}
        self._tool: dict = {}
        self._lock = threading.Lock()

    def _ms(self) -> int:
        return int((time.perf_counter() - self.t0) * 1000)

    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, invocation_params=None, **kw):
        self._llm_start(run_id, metadata, invocation_params)

    def on_llm_start(self, serialized, prompts, *, run_id, metadata=None, invocation_params=None, **kw):
        self._llm_start(run_id, metadata, invocation_params)

    def _llm_start(self, run_id, metadata, invocation_params):
        # model_spec (set in agent.make_chat_model) keeps the provider prefix, e.g. "groq:llama-3.3-70b-versatile"
        model = (metadata or {}).get("model_spec") or (metadata or {}).get("ls_model_name") \
            or (invocation_params or {}).get("model_name") or (invocation_params or {}).get("model")
        self._llm[run_id] = {"model": model, "started_ms": self._ms(), "_t": time.perf_counter()}

    def on_llm_end(self, response, *, run_id, **kw):
        s = self._llm.pop(run_id, None)
        if s is None:
            return
        entry = {"model": s["model"], "started_ms": s["started_ms"],
                 "latency_ms": int((time.perf_counter() - s["_t"]) * 1000), "error": None}
        usage = (response.llm_output or {}).get("token_usage") or {}
        requested = []
        try:
            msg = response.generations[0][0].message
            um = getattr(msg, "usage_metadata", None) or {}
            meta = getattr(msg, "response_metadata", None) or {}
            usage = usage or meta.get("token_usage") or {}
            entry["input_tokens"] = um.get("input_tokens", usage.get("prompt_tokens"))
            entry["output_tokens"] = um.get("output_tokens", usage.get("completion_tokens"))
            details = um.get("input_token_details") or {}
            if details.get("cache_read"):
                entry["cached_input_tokens"] = details["cache_read"]
            requested = [tc["name"] for tc in (getattr(msg, "tool_calls", None) or [])]
            served = meta.get("model_name")
            if served and served != entry["model"]:
                entry["served_model"] = served
        except (AttributeError, IndexError):
            entry["input_tokens"], entry["output_tokens"] = usage.get("prompt_tokens"), usage.get("completion_tokens")
        entry["requested_tools"] = requested
        cost = usage.get("cost")
        if cost is None and entry["model"] in PRICES and entry.get("input_tokens") is not None:
            pi, po = PRICES[entry["model"]]
            cost, entry["cost_estimated"] = entry["input_tokens"] * pi + (entry.get("output_tokens") or 0) * po, True
        entry["cost"] = cost
        with self._lock:
            self.llm_calls.append(entry)

    def on_llm_error(self, error, *, run_id, **kw):
        s = self._llm.pop(run_id, None)
        if s is None:
            return
        with self._lock:
            self.llm_calls.append({"model": s["model"], "started_ms": s["started_ms"],
                                   "latency_ms": int((time.perf_counter() - s["_t"]) * 1000),
                                   "error": f"{type(error).__name__}: {error}"[:300], "cost": None})

    def reject(self, model, reason: str) -> None:
        """Reply guard rejected this model's answer (truncated / leaked reasoning): flag its call as failed."""
        with self._lock:
            for e in reversed(self.llm_calls):  # the guard runs right after the model call it is judging
                if not e.get("error"):
                    e["error"] = f"rejected: {reason}"
                    return

    def on_tool_start(self, serialized, input_str, *, run_id, inputs=None, **kw):
        args = inputs if isinstance(inputs, dict) else input_str
        self._tool[run_id] = {"name": (serialized or {}).get("name") or kw.get("name"), "args": args,
                              "started_ms": self._ms(), "_t": time.perf_counter()}

    def _tool_done(self, run_id, summary=None, error=None):
        s = self._tool.pop(run_id, None)
        if s is None:
            return
        entry = {"name": s["name"], "args": s["args"], "started_ms": s["started_ms"],
                 "latency_ms": round((time.perf_counter() - s["_t"]) * 1000, 1), "result_summary": summary, "error": error}
        with self._lock:
            self.tool_calls.append(entry)

    def on_tool_end(self, output, *, run_id, **kw):
        # Domain problems ({"status": "error"|"ambiguous"...}) are recoverable and only show in the summary;
        # exceptions go through on_tool_error and count as tool errors.
        self._tool_done(run_id, summarise_result(getattr(output, "content", output)), None)

    def on_tool_error(self, error, *, run_id, **kw):
        self._tool_done(run_id, None, f"{type(error).__name__}: {error}"[:300])
