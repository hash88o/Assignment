"""Rule-based attention flags, computed after every turn. No LLM involved."""
import re

from . import config

FRUSTRATION = ["wrong", "not what i asked", "useless", "human", "agent"]
_FRUSTRATION_RE = re.compile(r"(?<![a-z])(" + "|".join(re.escape(k) for k in FRUSTRATION) + r")(?![a-z])", re.I)


def compute_flags(user_text: str, *, handoff_reasons, large_changes, tool_calls, fallback_used, total_latency_ms,
                  error) -> list[str]:
    flags = []
    if handoff_reasons:
        flags.append("agent asked for human: " + "; ".join(handoff_reasons))
    for t in tool_calls:
        if t.get("error"):
            flags.append(f"tool error in {t['name']}")
            break
    if fallback_used:
        flags.append("fallback model used")
    if total_latency_ms and total_latency_ms > config.SLOW_TURN_MS:
        flags.append(f"slow reply ({total_latency_ms / 1000:.1f}s)")
    for c in large_changes:
        flags.append(f"large value change ({c})")
    m = _FRUSTRATION_RE.search(user_text or "")
    if m:
        flags.append(f"frustration keyword: '{m.group(1).lower()}'")
    if error:
        flags.append("agent error: " + error[:120])
    return flags
