"""Verify the configured model chain against the providers' real model lists (and optionally probe tool calling).

    python scripts/check_models.py            # list checks only: no tokens spent
    python scripts/check_models.py --probe    # + one tiny tool-calling request per model (spends a few requests)
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from app import config  # noqa: E402

PROBE_TOOL = {"type": "function", "function": {
    "name": "get_portfolio_analysis", "description": "Read the user's portfolio. Always call it before answering.",
    "parameters": {"type": "object", "properties": {"filters": {"type": "object", "properties": {"property_type": {"type": "string"}}}}}}}


def model_lists() -> dict[str, dict[str, dict]]:
    out = {}
    try:
        out["openrouter"] = {m["id"]: m for m in httpx.get(f"{config.OPENROUTER_BASE_URL}/models", timeout=30).json()["data"]}
    except Exception as e:  # noqa: BLE001
        print("  ! could not fetch OpenRouter model list:", e)
    if config.groq_api_key():
        try:
            r = httpx.get(f"{config.GROQ_BASE_URL}/models", headers={"Authorization": f"Bearer {config.groq_api_key()}"}, timeout=30)
            r.raise_for_status()
            out["groq"] = {m["id"]: m for m in r.json()["data"]}
        except Exception as e:  # noqa: BLE001
            print("  ! could not fetch Groq model list (bad key?):", e)
    return out


def probe(provider: str, model: str) -> str:
    base = config.GROQ_BASE_URL if provider == "groq" else config.OPENROUTER_BASE_URL
    body = {"model": model, "max_tokens": 200, "tools": [PROBE_TOOL],
            "messages": [{"role": "system", "content": "Always call a tool before stating portfolio facts."},
                         {"role": "user", "content": "Show me my retail properties"}]}
    t = time.time()
    r = httpx.post(f"{base}/chat/completions", headers={"Authorization": f"Bearer {config.provider_api_key(provider)}"},
                   json=body, timeout=60)
    ms = int((time.time() - t) * 1000)
    if r.status_code != 200:
        return f"HTTP {r.status_code} in {ms} ms: {r.text[:140]}"
    msg = r.json()["choices"][0]["message"]
    calls = [(c["function"]["name"], c["function"]["arguments"]) for c in msg.get("tool_calls") or []]
    return f"{ms} ms, tool call: {calls or 'NONE (answered in text)'}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()
    chain = config.configured_models()
    print("Configured chain:", " -> ".join(chain))
    lists = model_lists()
    ok = True
    for i, spec in enumerate(chain):
        provider, model = config.parse_model_spec(spec)
        role = "primary " if i == 0 else "fallback"
        if not config.provider_api_key(provider):
            print(f"  [{role}] {spec}: SKIPPED, no {provider.upper()}_API_KEY set")
            continue
        info = lists.get(provider, {}).get(model)
        if info is None:
            print(f"  [{role}] {spec}: NOT FOUND in the {provider} model list"); ok = False
            continue
        extra = ""
        if provider == "openrouter":
            tools = "tools" in info.get("supported_parameters", [])
            free = str(info["pricing"].get("prompt")) == "0"
            extra = f" tools={'yes' if tools else 'NO'} {'free' if free else 'paid'}"
            ok &= tools
        print(f"  [{role}] {spec}: found{extra}")
        if args.probe:
            print("            probe:", probe(provider, model))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
