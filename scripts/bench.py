"""Latency benchmark -> LATENCY.md (+ scripts/bench_results.json).

Runs every sample_requests.csv prompt plus a scripted multi-turn conversation, N runs each.
Each run uses a throw-away clone of the user (bench-*), so seed data is never mutated.

    # live: against a running app (uses its real models)
    ADMIN_PASSWORD=... python scripts/bench.py --base-url http://localhost:8000 --runs 3
    # offline: in-process app + scripted fake OpenRouter; LLM latency is SIMULATED (--sim-llm-ms), everything else is real
    python scripts/bench.py --offline --sim-llm-ms 0
"""
import argparse
import csv
import json
import os
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

MULTI_TURN = [  # U001: retail -> which performs better -> exclude Bandra -> how does that change things -> an update
    "Tell me about my retail properties",
    "Which one is performing better?",
    "What if I exclude the Bandra property?",
    "How would that change my portfolio?",
    "Update the Bandra property's value to ₹14 Cr",
]


def pctl(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def fmt_ms(x):
    return "–" if x != x else (f"{x:,.0f} ms" if x < 10_000 else f"{x / 1000:.1f} s")


def mean(xs):
    return statistics.mean(xs) if xs else float("nan")


def make_client(args):
    pw = os.getenv("ADMIN_PASSWORD", "admin")
    if not args.offline:
        return httpx.Client(base_url=args.base_url, auth=("admin", pw), timeout=90), None
    os.environ["DB_PATH"] = os.path.join(tempfile.mkdtemp(), "bench.db")
    os.environ.update(OPENROUTER_API_KEY="offline", ADMIN_PASSWORD=pw)
    from fastapi.testclient import TestClient

    from app import agent
    from app.main import app
    from tests.fake_openrouter import FakeOpenRouter
    agent.TEST_TRANSPORT = FakeOpenRouter(latency_s=args.sim_llm_ms / 1000).transport
    tc = TestClient(app, base_url="http://bench")
    tc.auth = ("admin", pw)
    tc.__enter__()  # run lifespan (init_db + warm_up)
    return tc, tc


def chat(client, uid, text):
    t0 = time.perf_counter()
    r = client.post("/api/chat", json={"user_id": uid, "message": text})
    wall = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    d = r.json()
    t = d["timing"]
    pc = t["per_call_ms"]
    return {"wall_ms": wall, "total_ms": t["total_ms"], "llm_ms": t["llm_ms"], "tool_ms": t["tool_ms"],
            "overhead_ms": t["overhead_ms"], "llm_calls": t["llm_calls"], "tool_names": t["tool_names"],
            "call1_ms": pc[0] if pc else None, "call2_ms": pc[1] if len(pc) > 1 else None,
            "input_tokens": t["input_tokens"], "output_tokens": t["output_tokens"], "cost_usd": t["cost_usd"],
            "model": t["model"], "fallback_used": t["fallback_used"], "reply_words": len((d["reply"] or "").split())}


def with_clone(client, base_uid, label, fn):
    u = client.post(f"/api/admin/users/{base_uid}/clone", json={"name": f"bench-{base_uid}-{label}"})
    u.raise_for_status()
    uid = u.json()["user_id"]
    try:
        return fn(uid)
    finally:
        client.delete(f"/api/admin/users/{uid}")


def run(args, client):
    with open(ROOT / "data" / "sample_requests.csv", newline="", encoding="utf-8") as f:
        samples = list(csv.DictReader(f))
    results = []
    for run_i in range(args.runs):
        for s in samples:
            def one(uid, s=s):
                return chat(client, uid, s["user_request"])
            r = with_clone(client, s["user_id"], f"{s['request_id']}-r{run_i}", one)
            results.append({**r, "kind": "single", "id": s["request_id"], "prompt": s["user_request"], "run": run_i})
            print(f"  run {run_i + 1} {s['request_id']}: {r['total_ms']} ms, {r['llm_calls']} LLM calls")

        def convo(uid):
            return [chat(client, uid, m) for m in MULTI_TURN]
        for turn_i, r in enumerate(with_clone(client, "U001", f"convo-r{run_i}", convo), 1):
            results.append({**r, "kind": "multi", "id": f"T{turn_i}", "prompt": MULTI_TURN[turn_i - 1], "run": run_i})
            print(f"  run {run_i + 1} multi-turn {turn_i}: {r['total_ms']} ms, {r['llm_calls']} LLM calls")
    return results


def table(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    return "\n".join(out)


def by_prompt(results, kind):
    groups = {}
    for r in results:
        if r["kind"] == kind:
            groups.setdefault(r["id"], []).append(r)
    return groups


def render(args, results):
    tot = [r["total_ms"] for r in results]
    comp = lambda key, sel=lambda r: True: [r[key] for r in results if r[key] is not None and sel(r)]  # noqa: E731
    models = sorted({r["model"] for r in results if r["model"]})
    fb = sum(1 for r in results if r["fallback_used"])
    mode = ("**LIVE run** against `%s`. Models seen: %s." % (args.base_url, ", ".join(f"`{m}`" for m in models))) if not args.offline else (
        "> **Offline run: the LLM is a scripted fake with %d ms of *simulated* latency per call.** LLM numbers below are not real; "
        "tool time, framework overhead, call counts and payload sizes are real. Re-run live (`python scripts/bench.py --base-url ...`) "
        "to replace this file with measured end-to-end numbers." % args.sim_llm_ms)
    L = [f"# LATENCY.md", "", f"_Generated by `scripts/bench.py` on {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC · {args.runs} runs × "
         f"({len(by_prompt(results, 'single'))} sample prompts + {len(MULTI_TURN)}-turn conversation) = {len(results)} turns · "
         f"fallback used in {fb} of them_", "", mode, "", "## 1. Typical end-to-end response time", ""]
    L.append(table([["End-to-end (server, request → reply stored)", fmt_ms(pctl(tot, 50)), fmt_ms(pctl(tot, 95)), fmt_ms(mean(tot))],
                    ["Wall clock seen by the client (incl. HTTP)", fmt_ms(pctl([r["wall_ms"] for r in results], 50)),
                     fmt_ms(pctl([r["wall_ms"] for r in results], 95)), fmt_ms(mean([r["wall_ms"] for r in results]))]],
                   ["", "p50", "p95", "mean"]))
    L += ["", "## 2. Where the time goes", ""]
    rows = []
    for name, vals in [("LLM call 1 (decide: which tool?)", comp("call1_ms")),
                       ("LLM call 2 (write the reply from tool output)", comp("call2_ms")),
                       ("Tool execution (all tools in the turn)", comp("tool_ms")),
                       ("Overhead (graph build, callbacks, DB writes, HTTP handling)", comp("overhead_ms")),
                       ("LLM total per turn", comp("llm_ms"))]:
        rows.append([name, fmt_ms(pctl(vals, 50)), fmt_ms(pctl(vals, 95)), len(vals)])
    L.append(table(rows, ["Phase", "p50", "p95", "n"]))
    calls = comp("llm_calls")
    L += ["", f"- Average **{mean(calls):.2f} LLM calls per turn** (`{sum(1 for c in calls if c == 2)}` of {len(calls)} turns used exactly 2: one tool round-trip).",
          f"- Average tokens per turn: **{mean(comp('input_tokens')):,.0f} in / {mean(comp('output_tokens')):,.0f} out**.",
          f"- Average cost per turn: **${mean(comp('cost_usd')):.5f}** (OpenRouter `usage.cost`; ${sum(comp('cost_usd')):.4f} for this whole run)."
          if not args.offline else f"- Cost per turn: fake constant, not meaningful offline.",
          f"- Replies average {mean(comp('reply_words')):.0f} words (capped by `max_tokens=700` and the ~120-word prompt guideline).", ""]

    L += ["## 3. Sample requests", ""]
    rows = []
    for rid, rs in by_prompt(results, "single").items():
        rows.append([rid, rs[0]["prompt"][:58], fmt_ms(pctl([r["total_ms"] for r in rs], 50)), fmt_ms(max(r["total_ms"] for r in rs)),
                     f"{mean([r['llm_calls'] for r in rs]):.1f}", ", ".join(sorted({t for r in rs for t in r["tool_names"]})) or "–",
                     fmt_ms(mean([r["tool_ms"] for r in rs]))])
    L.append(table(rows, ["ID", "Prompt", "p50", "max", "LLM calls", "Tools", "Tool time"]))
    L += ["", f"## 4. Multi-turn conversation (retail → performing better → exclude Bandra → how does that change → update)", ""]
    rows = []
    for tid, rs in by_prompt(results, "multi").items():
        rows.append([tid, rs[0]["prompt"][:50], fmt_ms(pctl([r["total_ms"] for r in rs], 50)), fmt_ms(max(r["total_ms"] for r in rs)),
                     f"{mean([r['llm_calls'] for r in rs]):.1f}", ", ".join(sorted({t for r in rs for t in r["tool_names"]})) or "–",
                     f"{mean([r['input_tokens'] for r in rs]):,.0f}"])
    L.append(table(rows, ["Turn", "Message", "p50", "max", "LLM calls", "Tools", "Input tokens"]))
    L += ["", "Input tokens grow slightly per turn because the last-20-message history window fills up; the big constant is the system prompt + tool schemas + tool output.", ""]

    L += ["## 5. Slowest queries, and why", ""]
    allp = [(rs[0]["prompt"], pctl([r["total_ms"] for r in rs], 50), mean([r["llm_calls"] for r in rs]),
             sorted({t for r in rs for t in r["tool_names"]}), mean([r["input_tokens"] for r in rs]))
            for grp in (by_prompt(results, "single"), by_prompt(results, "multi")) for rs in grp.values()]
    for prompt, p50, ncalls, tools, intok in sorted(allp, key=lambda x: -x[1])[:3]:
        why = (f"{ncalls:.1f} LLM calls" + (" — an extra tool round-trip beyond the usual 2" if ncalls > 2.2 else " — the normal tool → reply pattern")
               + f"; tools: {', '.join(tools) or 'none'}; ~{intok:,.0f} input tokens")
        L.append(f"- **{fmt_ms(p50)}** — “{prompt[:70]}” — {why}.")
    L += ["", "General rule: latency ≈ (number of LLM calls) × (per-call latency). Tool time is a rounding error (in-process SQLite + pure Python), so every "
          "avoidable LLM round-trip (a `find_properties` before an update, a follow-up clarification) costs a full model call. Add-property "
          "always spans two turns by design (it asks for rent/occupancy once via `ask_optional`, then writes on the next message).", ""]
    L.append(STATIC_SECTIONS)
    return "\n".join(L)


STATIC_SECTIONS = """## 6. What we did to cut latency

| Lever | Effect |
|---|---|
| **Fast, cheap model** (`PRIMARY_MODEL`, a non-reasoning one where possible; cross-provider fallback chain) | The model only routes and phrases; all maths is in tools. Reasoning models are slower and can exhaust `max_tokens` on hidden reasoning, so they are avoided in the default chain. |
| **One rich analytics tool** (`get_portfolio_analysis`: filters, group-by, scenarios, comparison in a single call) | Typical question = 1 tool round-trip = **2 LLM calls** instead of 3–4. |
| **Property references resolved inside tools** (`update_property("Bandra retail", …)`, `exclude_properties=["Bandra"]`) | No separate `find_properties` hop before an update or a what-if. |
| **Pre-computed, pre-formatted tool output** (₹ strings, shares, yields, deltas, highlights) | The model copies instead of calculating, so replies are short and correct on the first try. Payloads are trimmed (compact JSON, no redundant fields) to keep call-2 input small. |
| **Capped `max_tokens` (700) and a ~120-word style rule** | Output tokens dominate generation time; WhatsApp replies are short anyway. |
| **Trimmed history window** (last 20 messages, text only — no stale tool payloads) | Bounded input size; portfolio state is never kept in context. |
| **In-process SQLite, one short connection per call** | Tool execution is a few ms (see §2), no network hop. |
| **Static system prompt first** | Identical prefix across users/turns so providers can cache it (OpenAI-style automatic prefix caching). Per-user facts arrive via tool output, not the prompt. |
| **20 s timeout, 1 retry, cross-provider fallback** | A dead provider is abandoned after at most 2 × 20 s (timeout × attempts) and the fallback model answers, instead of the request hanging. Worst case is therefore long; a lower timeout (~8–10 s) is the first thing to tighten once real p95s are known. |
| **Graph built per request but warmed at startup** | ~5 ms per request (measured); avoids a global mutable agent and keeps `user_id` injection trivially safe. |
| **No streaming** | WhatsApp delivers whole messages; streaming would add complexity without a UX win here. |

Not done (would help, but adds risk without a way to verify offline): OpenRouter provider routing preferences (`provider.sort = latency`), speculative parallel tool prefetch, a router model that skips the agent for greetings.

## 7. What would change at significantly higher volume

- **Postgres** (with a pooler) instead of SQLite: concurrent writers, replicas, and real migrations.
- **Queue + workers**: the web tier only enqueues; agent turns run in a worker pool with per-user ordering so two messages from one user never race.
- **Asynchronous WhatsApp webhooks**: acknowledge Meta's webhook immediately (<1 s), process in the background, reply via the send API. Latency then stops being a request timeout problem.
- **Semantic / result caching** for repeated analytical questions ("what's my total value?"), keyed by (user, portfolio version, normalised question); invalidated by writes.
- **A tiny router model** (or rules) to answer greetings/thanks/FAQs without invoking the agent at all.
- **Summarised long-term memory** instead of a raw last-20 window, so context stays small for very long conversations.
- **Per-user rate limits and budgets**, plus provider-level concurrency limits and backoff.
- **OpenTelemetry / Langfuse** instead of the homemade `turn_traces` table: distributed tracing, sampling, dashboards, and prompt/version tracking.
- **Prompt-cache-aware ordering and larger cached prefixes**, and per-model routing (cheap model by default, escalate on tool errors).
- **Horizontal scaling of the web tier** (stateless already, except the in-process pending-removal confirmations, which move to the DB/Redis).

## 8. Reproduce

```bash
# live (real models, real network): the app must be running with its API key(s) set
ADMIN_PASSWORD=... python scripts/bench.py --base-url http://localhost:8000 --runs 3
# offline (fake LLM): measures framework + tool overhead only
python scripts/bench.py --offline --sim-llm-ms 0
```

## 9. Live spot check (manual, not produced by this script)

Real end-to-end turns against the running app, read from the per-turn traces. Small samples, taken by hand; use the benchmark above for proper percentiles.

| Model chain | Turns | End-to-end | Single LLM call | Notes |
|---|---|---|---|---|
| `groq:openai/gpt-oss-120b` → `groq:openai/gpt-oss-20b` → OpenRouter free | 8, spaced ~22 s apart | 1.4–2.5 s (median ≈ 2.1 s) | ≈ 0.6–1.4 s | Tool time 1–11 ms. About half the turns were answered by the 20b fallback once the 120b model's 8,000 tokens/minute budget was spent. |
| Free OpenRouter models only | ~20 | 3–21 s | 1–18 s | Frequent 429/503 from upstream providers; most turns needed at least one fallback attempt. |

Findings that shaped the design:
- **Two LLM calls per turn dominate the time.** Tool execution is a few milliseconds, so the useful levers are the number of calls and the per-call latency of the chosen provider.
- **Retry-After sleeps are a hidden latency cost.** Groq answers a rate-limited request with a ~30 s `Retry-After`; the OpenAI SDK sleeps through it, which turned two turns into 31 s and 39 s. Groq models now use no in-place retry and fail over to the next model immediately.
- **Input size matters on rate-limited tiers.** A turn is roughly 6–8k tokens across its two calls against Groq's 8k-per-minute free limit, so tool schemas are compacted (about 25% smaller) and default breakdowns are trimmed.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--sim-llm-ms", type=int, default=0, help="offline: simulated latency per LLM call")
    ap.add_argument("--out", default=str(ROOT / "LATENCY.md"))
    args = ap.parse_args()
    client, tc = make_client(args)
    try:
        print(f"Benchmarking ({'offline' if args.offline else args.base_url}), {args.runs} runs...")
        results = run(args, client)
    finally:
        if tc is not None:
            tc.__exit__(None, None, None)
    Path(args.out).write_text(render(args, results), encoding="utf-8")
    (ROOT / "scripts" / "bench_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
