# Decision log

Each entry: **Decision · Alternatives · Why · Trade-off.** Times and counts referenced here come from [LATENCY.md](LATENCY.md) and the test-suite.

## 1. One tool-calling agent, not multi-agent
- **Decision:** a single LangGraph ReAct agent (`create_react_agent`) with six tools.
- **Alternatives:** router → specialist agents (analytics / CRUD / advisor); a fixed workflow (classify → tool → answer).
- **Why:** the problem is "pick a tool and phrase the result". Every extra agent is another LLM hop (~1 s each) and another place to lose context. A single agent needs 2 LLM calls for the common question.
- **Trade-off:** one prompt has to cover reads, writes and hand-off; routing quality depends on one model. If the tool set grew past ~10, I would split by domain.

## 2. SQL + Python, not RAG or a vector DB
- **Decision:** SQLite rows and plain Python functions. No embeddings, no retrieval.
- **Alternatives:** stuff the portfolio in the prompt; vector search over property descriptions; text-to-SQL.
- **Why:** 12 structured rows. Questions are filters and aggregates (exact answers), not semantic lookups. Prompt-stuffing goes stale and leaks; text-to-SQL adds a failure mode for no gain.
- **Trade-off:** fuzzy references ("the Bandra one") are handled by a small token matcher, not embeddings. That would not scale to thousands of properties per user.

## 3. Deterministic tools do all the maths; the LLM only routes and phrases
- **Decision:** `analytics.py` computes every figure and returns pre-formatted INR strings beside the raw values; the prompt forbids arithmetic and unit conversion. Amounts go *into* tools as the user's own strings (`"4.2 crore"`) and `parse_inr` converts them.
- **Alternatives:** let the model calculate; a code-interpreter tool.
- **Why:** LLMs get percentages and crore/lakh conversions wrong at exactly the moments users care. Pure functions are unit-tested (62 tests, hand-verified totals).
- **Trade-off:** mostly enforced by prompt + tool design. Live testing on free models showed the prompt alone is not enough, so three guards are in code: (a) a *truth guard* re-runs a turn once if the reply has figures but no tool was called; (b) write tools return `portfolio_now` (new totals), because a model once added 29.70 + 0.5 and wrote 29.20; (c) replies are normalised to WhatsApp formatting and `<think>` blocks stripped; (d) a reply guard rejects truncated / reasoning-leaking answers so the fallback chain takes over. Still no number-by-number check that reply figures ⊆ tool output; that is the production upgrade.

## 4. A few rich tools, with references resolved inside them
- **Decision:** one read tool covering filters / group-by / comparison / scenarios; add/update/remove/flag; `find_properties` only for explicit disambiguation. `update_property`, `remove_property` and `exclude_properties` accept "P001" **or** a phrase like "Bandra retail".
- **Alternatives:** many small tools (`get_total`, `get_by_type`, `get_top_rent`, …); a mandatory `find_property` → `update_property` chain.
- **Why:** small tools multiply round-trips (each one is a full LLM call) and confuse routing. Resolving references inside the tool makes "change my Bandra property to ₹12.5 Cr" a single tool call; ambiguity returns candidates without writing.
- **Trade-off:** the analysis tool has a large argument schema and a 4–7 KB response. It's dereferenced and trimmed, but it is the biggest input-token cost per turn.

## 5. `user_id` is injected, never an LLM argument
- **Decision:** tools are built per request by a closure over `TurnContext(user_id, conversation_id)`. No tool schema contains a user identifier; every DB read/write takes `user_id` in its WHERE clause.
- **Alternatives:** pass `user_id` as an argument and instruct the model; LangGraph `InjectedState`.
- **Why:** prompt-level isolation is bypassable by a user who types "show me U002's portfolio". Here it is structurally impossible — a test asserts no tool schema has the parameter, and another that U001 cannot address U002's P004.
- **Trade-off:** rebuilding tools and the graph each request (~5 ms measured, warmed at startup). Not a security boundary between *users of the API* — see the no-auth limitation in the README.

## 6. Scenarios never write to the database
- **Decision:** exclusions, overrides and hypothetical additions are applied to an in-memory copy inside `portfolio_analysis`; results carry `is_hypothetical`, a `scenario_description` and a `delta_vs_actual` block (plus actual concentration flags for contrast).
- **Alternatives:** temporary DB rows; a "scenario mode" flag in conversation state.
- **Why:** the brief says to distinguish actual from hypothetical. Structural separation makes it impossible for "what if" to corrupt real data, and the delta comes from the same code path as the actuals.
- **Trade-off:** scenarios are single-turn; a multi-step "now also change X" re-sends the whole scenario each time (the model re-passes arguments from history).

## 7. Model choice and fallback
- **Decision:** env-configured models. Defaults are free-tier models: `PRIMARY_MODEL=poolside/laguna-xs-2.1:free`, `FALLBACK_MODEL=cohere/north-mini-code:free,inclusionai/ling-3.0-flash-fin:free,openrouter/free` (a comma-separated chain across four providers). Each was verified against OpenRouter's `/models` list *and* with a real two-step tool-calling probe. `primary.with_fallbacks([...])`, every candidate wrapped in a reply guard, 20 s timeout, 1 retry, `max_tokens=700`, usage accounting on. A paid pair (`openai/gpt-4.1-mini` → `google/gemini-2.5-flash`) is the recommended production setting; only env vars change. **Groq is an optional direct provider:** a model written `groq:<id>` is sent to Groq's OpenAI-compatible endpoint with `GROQ_API_KEY`, anywhere in the chain. OpenRouter remains the default gateway the brief asks for; Groq adds speed and an independent quota (useful because OpenRouter's free tier caps at 50 requests/day). Models whose provider key is missing are skipped, not fatal. Groq specifics: Groq's free tier is 8k tokens/minute per model and answers 429 with a long `Retry-After`, so Groq models use `max_retries=0` and fail over at once (an in-place retry stalled turns for 30–40 s); the recommended chain is `groq:openai/gpt-oss-120b → groq:openai/gpt-oss-20b → OpenRouter free models`. To stay under the token cap I compact the tool schemas sent to the model (drop pydantic titles/defaults/`anyOf null`, ~25% smaller) and trim default breakdowns; the tool objects and their validation are unchanged.
- **Alternatives:** Groq only (fast, but not a gateway and one provider); one bigger model; reasoning models; a router that escalates to a bigger model on hard questions; OpenRouter's server-side fallback list.
- **Why:** the model routes and phrases; the maths is elsewhere, so latency and cost matter more than depth. A cross-provider fallback means one provider outage can't take both down. Doing the fallback client-side lets the trace record exactly what happened.
- **Trade-off:** worst-case wait before a fallback answers is 2 × 20 s (timeout × attempts) — too long; tighten after seeing real p95s. Free models 429/503 often, so many turns use a fallback and take 3–17 s. Small models skip tools on follow-ups, do their own arithmetic and use Markdown, and reasoning models can spend the whole token budget thinking and return truncated or chain-of-thought replies. Hence the code-level guards in 3 and a reply guard that treats a truncated or reasoning-leaking answer as a model failure, so the next model answers. Free accounts are also capped at 50 requests/day; the app detects that and says so.

## 8. Attention flags are rule-based, not LLM-judged
- **Decision:** after each turn, deterministic rules: `flag_for_human` called, tool error, fallback used, latency > 10 s, value change > 30%, frustration keywords. The agent may *also* self-flag.
- **Alternatives:** an LLM classifier over each transcript; sentiment model.
- **Why:** free, instant, explainable (the reason text is shown in the admin), and no extra hop on the user's critical path.
- **Trade-off:** keyword rules have false positives ("agent" in "real estate agent") and misses. An async LLM triage pass over flagged/unflagged conversations would be the production upgrade.

## 9. SQLite, in-process
- **Decision:** plain `sqlite3`, WAL, one short connection per call, seeded from CSVs when the DB is empty/missing.
- **Alternatives:** Postgres; an in-memory dict; SQLModel.
- **Why:** zero infrastructure, sub-5 ms tool time, trivially portable for reviewers; the data is tiny.
- **Trade-off:** ephemeral on Render/Railway free tiers (redeploy = reset), single-writer, no migrations. Documented as a limitation; Postgres is the first thing to change (§ Production).

## 10. No streaming
- **Decision:** the API returns the whole reply; the UI shows a typing indicator.
- **Alternatives:** SSE/WebSocket token streaming.
- **Why:** WhatsApp delivers whole messages, replies are ~50–120 words, and the two-call tool pattern means the first call produces no user-visible text anyway.
- **Trade-off:** perceived latency equals full latency. Worth revisiting for a web-native product.

## 11. Data normalisation: merge Office labels, keep city *and* metro, don't invent history
- **Decision:** `Commercial Office` and `Office` → Office (raw label kept); `location` → locality + city + **metro** (Delhi NCR groups Gurugram/Noida; Alibaug stays its own city, not Mumbai); vacancy = Vacant only; yield reported on all assets and on income-producing assets; appreciation is `{"available": false, "reason": …}` because there are no purchase prices or dates.
- **Alternatives:** collapse to one city column ("Delhi NCR") as the brief's sketch suggested; treat Self-occupied as vacant; estimate appreciation.
- **Why:** U003's own stated preference is "Gurugram / Noida", so collapsing loses a real distinction; Alibaug in Mumbai would misreport U002's exposure and break "properties in Mumbai"; calling someone's own home "vacant" is wrong; inventing returns is worse than saying no.
- **Trade-off:** an extra `metro` column and a small static locality gazetteer for city inference. `portfolio_value_preference_inr` is stored but deliberately not used to generate insights (all four users are 2–6× over it under any reading).

## 12. Safe actions: execute updates, confirm deletes, ask once for optional facts
- **Decision:** explicit updates run immediately and echo old → new (value changes >30% are flagged for review); removal is a soft delete that the tool refuses to perform until it has issued a confirmation request; `add_property` writes nothing until type, location, area and value are present, then asks once about rent/occupancy before defaulting to Vacant/₹0. Every mutation writes `audit_log`.
- **Alternatives:** confirm every change; hard delete; never default.
- **Why:** friction where the user was explicit is annoying on WhatsApp, but deletes are the one irreversible-feeling action. The confirmation guard lives in the tool so a model that sets `confirmed=true` prematurely is ignored.
- **Trade-off:** pending confirmations are in-process memory (lost on restart, single instance). No undo UI yet, only the audit trail.

## Departures from the initial plan
Made after reading the data.
- `city` + `metro` instead of a single normalised city (see 11).
- `update_property` / `remove_property` take `property_ref` (ID **or** phrase), and scenarios take `exclude_properties` / `value_overrides` by reference, instead of IDs only (see 4).
- The analysis tool has numeric/type/location/occupancy **filters** and a `compare` argument, so "above ₹10 crore" and "residential vs commercial" are one call (a fuzzy `find_properties` can't express thresholds).
- `add_property` **infers a missing city** (user's own property in that locality → gazetteer → profile city) and states the assumption, rather than blocking on it (sample request R005 gives only "Indiranagar").
- Extra endpoints for the benchmark and demo: `POST /api/users`, admin clone/delete of `bench-*` users; the admin view also shows the user's portfolio and recent audit rows; `/api/chat` returns a `timing` breakdown.

## What I'd change in production
1. **Postgres** + migrations; conversation/trace tables partitioned; durable pending-action state.
2. **Real auth** for end users (WhatsApp number → identity via the Business API) and per-role admin auth with an audit trail; rate limits and per-user budgets.
3. **Queue + workers** and **asynchronous WhatsApp webhooks** (ack fast, process in the background, reply via the send API); per-user ordering.
4. **Observability:** OpenTelemetry / Langfuse instead of the homemade trace table; alerting on p95, fallback rate and tool errors.
5. **Answer verification:** check that numbers in the reply appear in tool output; an LLM-as-judge eval set built from `sample_requests.csv` plus adversarial cases (ambiguity, injection, unit slips).
6. **Latency:** lower the timeout, provider routing by latency, a router/small model for greetings, semantic cache keyed by portfolio version, prompt-cache-aware prefix layout.
7. **Memory:** summarised long-term memory; replay tool results selectively.
8. **Data:** purchase price + dates captured (unlocks appreciation and XIRR), undo from `audit_log`, soft-delete restore, currency/market data behind explicit, cited tools.
9. **LLM triage** of conversations for the admin inbox on top of the rule-based flags.
