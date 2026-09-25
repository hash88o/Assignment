# AI Real Estate Portfolio Analyst

A WhatsApp-style chat where a property owner talks to an AI analyst about **their own** portfolio — add/update/remove properties, ask analytical questions, run what-if scenarios — plus a business console (`/admin`) to watch conversations, inspect every LLM/tool call, and step in as a human.

| | |
|---|---|
| **Live app** | _TODO: add the Render/Railway URL after deploying (see [Deploy](#deploy))_ — chat at `/`, admin at `/admin` |
| **Stack** | Python 3.11+, FastAPI, LangGraph ReAct agent, OpenRouter (free-tier models by default; primary + fallback chain), optional direct Groq, SQLite, vanilla HTML/JS |
| **Docs** | [SOUL.md](SOUL.md) agent definition · [ARCHITECTURE.md](ARCHITECTURE.md) diagram · [DECISIONS.md](DECISIONS.md) trade-offs · [LATENCY.md](LATENCY.md) performance |

## What it does
- **Understands the portfolio.** "Add a 3000 sq ft retail property in Indiranagar worth ₹4.2 crore" → validated, normalised, persisted (asks once for missing rent/occupancy).
- **Analyses it with tools, not memory.** Totals, rent, gross yield (on all assets *and* on rented assets), vacancy, breakdowns by type/city/region, highest/lowest, concentration flags, residential-vs-commercial comparison. Every number is computed in Python and returned pre-formatted (₹12.00 Cr, ₹45.0 L); the LLM never does arithmetic.
- **Keeps a conversation.** Last 20 messages as history; portfolio state is never held in context and is re-fetched by tools every turn.
- **Separates actual from hypothetical.** "What if I exclude the Bandra one?" runs an in-memory scenario, labelled *Scenario*, always contrasted with the actuals. Scenarios never write to the DB.
- **Takes actions safely.** Updates run immediately and echo old → new (large >30% value changes are flagged); removals are soft deletes that need an explicit yes; every mutation is written to `audit_log`.
- **Hands off to humans.** Legal/tax/transaction advice, "I want to buy/sell", "let me talk to a person", repeated confusion, or an unrecoverable tool error → `flag_for_human`.
- **Business console.** Conversation list with attention badges, transcript with collapsible per-turn trace (model, fallback, each LLM call and tool call with args/result/ms, tokens, cost), Mark resolved, Take over / Hand back, reply as a human (shown in the user's chat in a different colour), and a p50/p95/cost strip.

Try: *"What does my portfolio look like?"* · *"Which of my properties are above ₹10 crore?"* · *"Compare my residential and commercial exposure"* · *"Which one is performing better?"* → *"What if I exclude the Bandra one?"* · *"Change my Bandra retail property value to ₹12.5 crore"* · *"How much has my Worli flat appreciated?"* (answered honestly: no purchase data) · *"I want to talk to a person"*.

## Setup

### Local
```bash
python3 -m venv .venv && source .venv/bin/activate     # or: uv venv && uv pip install -r requirements-dev.txt
pip install -r requirements-dev.txt
cp .env.example .env                                    # then set OPENROUTER_API_KEY and ADMIN_PASSWORD
uvicorn app.main:app --reload                           # http://localhost:8000  (admin: /admin, any username + ADMIN_PASSWORD)
pytest                                                  # 93 tests: analytics + one end-to-end agent smoke suite (no API key needed)
```
No API key handy? `python scripts/dev_mock_server.py` runs the real app against a scripted fake OpenRouter — good for looking at the UIs and traces (replies are keyword-routed, not intelligent).

### Docker
```bash
docker build -t portfolio-analyst .
docker run -p 8000:8000 --env-file .env portfolio-analyst
```

### Deploy
`render.yaml` is included (Render → New → Blueprint → set `OPENROUTER_API_KEY` and `ADMIN_PASSWORD`). Railway works from the same `Dockerfile`. The app listens on `$PORT`, exposes `/healthz`, creates and seeds `app.db` from `data/*.csv` on first boot.

### Environment variables
| Var | Purpose | Default |
|---|---|---|
| `OPENROUTER_API_KEY` | OpenRouter key (default gateway) | – (needed for any non-`groq:` model) |
| `GROQ_API_KEY` | Groq key, used only by models written `groq:<model id>` | – (optional) |
| `PRIMARY_MODEL` | first-choice model: an OpenRouter ID, or `groq:<id>` | `poolside/laguna-xs-2.1:free` |
| `FALLBACK_MODEL` | comma-separated chain, tried in order if the primary errors, times out, is truncated or leaks reasoning | `cohere/north-mini-code:free,inclusionai/ling-3.0-flash-fin:free,openrouter/free` |
| `ADMIN_PASSWORD` | HTTP basic auth password for `/admin` (any username). **Admin is disabled if unset.** | – |
| `DB_PATH` | SQLite file | `app.db` |

The defaults are free-tier OpenRouter models, each confirmed on `https://openrouter.ai/api/v1/models` with tool support. A free account is capped at 50 free-model requests per day in total (a chat turn uses 2 or more), and free models are often rate-limited or overloaded upstream, so replies can be slow and the fallback chain is used often. Adding $10 of credits raises the cap to 1000/day. When the cap is reached the app says so and flags the conversation. For steadier, faster replies use a paid pair, e.g. `PRIMARY_MODEL=openai/gpt-4.1-mini`, `FALLBACK_MODEL=google/gemini-2.5-flash`, or the Groq chain below. Reasoning models are avoided in the default chain because hidden reasoning can exhaust `max_tokens`.

### Using Groq
Groq is an optional second provider, called directly through its OpenAI-compatible API (`https://api.groq.com/openai/v1`); OpenRouter stays the default gateway. Put `GROQ_API_KEY` in `.env` and prefix any model in the chain with `groq:`, e.g.

```bash
PRIMARY_MODEL=groq:openai/gpt-oss-120b
FALLBACK_MODEL=groq:openai/gpt-oss-20b,poolside/laguna-xs-2.1:free,cohere/north-mini-code:free
```

A Groq model can be the primary or any fallback. Its quota is separate from OpenRouter's, so it also protects against OpenRouter's free-tier daily cap. Models whose provider has no key are skipped with a startup warning, so the same `.env` works with either key. Traces show the provider (`groq:openai/gpt-oss-120b`); Groq reports no cost, so cost is estimated from an approximate price table and marked `*` in the admin. `groq:openai/gpt-oss-*` models are sent `reasoning_effort=low`. **Groq free-tier limits (read from its response headers): 8,000 tokens/minute and 1,000 requests/day *per model*.** One chat turn is roughly 6–8k tokens across its two LLM calls, so back-to-back turns can hit the per-minute cap. Groq models therefore do **not** retry in place (the SDK would sleep through a ~30 s `Retry-After`); they fail over immediately to the next model, which is why the recommended chain puts a second Groq model (its own token budget) and then OpenRouter free models behind the first. Tool schemas sent to the model are compacted (~25% smaller) to save tokens. Verify IDs and (optionally) probe a real tool call with `python scripts/check_models.py [--probe]`.

### Benchmark
`python scripts/bench.py --base-url http://localhost:8000 --runs 3` (needs `ADMIN_PASSWORD`) rewrites `LATENCY.md` with live numbers. It runs on throw-away cloned users, so seed data is untouched.

## Assumptions
- **Type normalisation** (raw label kept in `property_type_raw`): `Retail → Retail`; `Commercial Office` and `Office → Office` (same asset, two spellings — `sub_type` confirms it); `Residential → Residential`. `asset_class` is Commercial for Retail/Office, Residential otherwise. Free-text aliases (shop, showroom, flat, villa, IT park…) are mapped on input.
- **City normalisation:** `location` is split into `locality` + `city` + `metro`. Aliases: Gurgaon→Gurugram, Bangalore→Bengaluru, Bombay→Mumbai. `metro` groups Gurugram/Noida/Delhi as **Delhi NCR** (used for geographic concentration) while `city` keeps Gurugram vs Noida distinct. A state as the last token (`Alibaug, Maharashtra`) makes the locality the city, and **Alibaug is not folded into Mumbai**.
- **No purchase price and no dates anywhere →** appreciation, returns and annualised return are reported as *not available* (with what data would fix it). If a user supplies a purchase price, absolute gain is reported, but never an annualised return.
- **Vacancy = `Vacant` only.** `Self-occupied` earns ₹0 rent but is not vacancy. "Income-producing" = Tenanted with rent > 0. Yield is shown both on all assets and on income-producing assets.
- **`ownership_percent`** scales value and rent (economic interest); area is not scaled. All seed rows are 100%.
- **Soft deletes:** removing sets `status = Inactive` (recoverable, audited); Inactive rows are excluded from all analysis.
- **`portfolio_value_preference_inr`** is ambiguous (scalar or range; every user's actual portfolio is far above it), so it is stored and shown but **not** used to drive proactive "mismatch" insights. Those use `preferences` and `preferred_locations` instead.
- `tenant_status` is redundant with `occupancy_status` in the data; it is stored but not exposed to the model.
- Missing city on add → inferred (user's own property in that locality → small locality gazetteer → user's profile city) and **stated back** to the user. Unknown rent/occupancy after being asked once → `Vacant`, ₹0, and said so.
- Monthly rent is converted to annual by the tool (`monthly_rent` field), not by the model.
- New users start with an empty portfolio and phone is optional (seeded users have none).

## Known limitations
- **Ephemeral storage:** SQLite lives on the container disk; a redeploy resets to the seed data (conversations, edits and traces are lost). Fine for a demo, not for production.
- **No end-user authentication.** The chat API trusts the `user_id` in the request body (the "user switcher" is the login). The *agent* can only ever see the bound user's rows, but anyone who can call the API can impersonate a user. Admin is a single shared basic-auth password. No multi-tenant hardening, rate limiting or CSRF protection.
- **Single process.** The "confirm before removal" state is held in-process memory (restart = it asks again); two simultaneous messages from one user are not serialised.
- **Model availability changes.** Provider model lists change (for example `llama-3.3-70b-versatile` was not available on the Groq account used for testing, and `qwen/qwen3.8-27b` hit the per-minute token cap on its second call). Run `python scripts/check_models.py [--probe]` after changing keys or models. Request routing, fallback and skip-when-no-key are covered by tests against a fake server.
- **Groq free-tier limits.** 8,000 tokens/minute and 1,000 requests/day per model; sending messages faster than about one a minute moves turns onto the fallback models. In a test of eight spaced turns on the Groq chain (portfolio, most rent, what-if, update, repeat, comparison, appreciation, sell hand-off) each turn took 1.4–2.5 s, figures matched the tools, and the update reported the correct new totals.
- **Free OpenRouter models are slow and unreliable.** Observed 3–17 s per turn with frequent 429/503 responses, so the admin often shows "fallback model used" and "slow reply" flags. The committed `LATENCY.md` is an offline run (fake LLM); run `scripts/bench.py` against a live app with a higher-limit key for real numbers.
- **Small-model guard rails.** Weaker models answer follow-ups from history without a tool call, do their own arithmetic, use Markdown formatting, mis-group digits, or return truncated or reasoning text. Mitigations in code: a truth guard (retry once if figures appear with no tool call), write tools that return `portfolio_now` totals, `*_fmt` strings for money and area, WhatsApp formatting normalisation, and a reply guard that rejects truncated or reasoning-leaking answers so the next model answers. Wording and routing still vary between runs.
- **"Never do arithmetic" is enforced by the prompt and tool design (plus the truth guard above), not by a number-by-number check.** There is no post-hoc verification that reply numbers match tool output.
- **Attention flags are keyword rules:** "agent" also matches "real estate agent"; misses frustration phrased differently. Latency/fallback/tool-error flags are reliable.
- **History is text only** (last 20 messages); tool payloads aren't replayed, so follow-ups re-query (by design, for freshness, at the cost of an extra tool call).
- A Tenanted property added with unknown rent is stored with ₹0 rent, which understates yield until updated.
- `audit_log` records every mutation (old → new) but there is no undo button; reversal is manual.
- Out of scope: real WhatsApp API, market data, non-INR currency, fractional-ownership maths beyond the multiplier, i18n, tests beyond analytics + one agent smoke suite (the smoke suite also covers a few tool/API guards).

## External dependencies
- **OpenRouter** (`https://openrouter.ai/api/v1`) as the model gateway — needs an API key. Default (free-tier) models: **`poolside/laguna-xs-2.1:free`** (primary), then **`cohere/north-mini-code:free`**, **`inclusionai/ling-3.0-flash-fin:free`**, then **`openrouter/free`** (auto-router) as fallbacks — four different providers. Usage accounting via `usage: {include: true}`.
- **Groq** (optional; `https://api.groq.com/openai/v1`, key in `GROQ_API_KEY`) for `groq:` models.
- **Render or Railway** for hosting (Docker).
- **Python packages** (`requirements.txt`): `fastapi`, `uvicorn`, `langgraph` (`create_react_agent`), `langchain-core`, `langchain-openai`, `httpx`, `python-dotenv`. Dev: `pytest`. No vector DB, Redis, queue, or frontend build tooling; the UIs use no CDN assets.
- No property portal, CRM, or market-data API is used; all data is the synthetic CSVs in `data/`.

## Project layout
```
app/        main.py (FastAPI) · agent.py (LangGraph + OpenRouter + fallback) · tools.py (6 tools, user_id injected)
            analytics.py (pure maths, INR parse/format) · normalize.py · resolve.py · db.py · tracing.py · flags.py · prompts.py
static/     index.html (chat) · admin.html (console)
data/       seed CSVs + dataset notes
scripts/    bench.py (latency → LATENCY.md) · check_models.py (verify model IDs / probe) · dev_mock_server.py
tests/      test_analytics.py · test_agent_smoke.py · fake_openrouter.py
```
