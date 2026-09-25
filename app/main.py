"""FastAPI app: chat API, admin API, and the two static pages."""
from __future__ import annotations

import logging
import re
import secrets
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field

from . import agent, config, db
from .flags import compute_flags

log = logging.getLogger("portfolio")


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()  # creates + seeds from data/*.csv when the DB is missing or empty
    if not config.model_chain():
        log.warning("No usable LLM configured (set OPENROUTER_API_KEY and/or GROQ_API_KEY): chat replies will fail")
    elif config.skipped_models():
        log.warning("Skipping configured models with no API key: %s", ", ".join(config.skipped_models()))
    log.warning("LLM chain: %s", " -> ".join(config.model_chain()))
    if not config.admin_password():
        log.warning("ADMIN_PASSWORD is not set: /admin is disabled")
    agent.warm_up()
    yield


app = FastAPI(title="AI Real Estate Portfolio Analyst", lifespan=lifespan)
security = HTTPBasic(auto_error=False)


def require_admin(creds: HTTPBasicCredentials | None = Depends(security)) -> None:
    pw = config.admin_password()
    if not pw:
        raise HTTPException(503, "Admin console disabled: set ADMIN_PASSWORD")
    if creds is None or not secrets.compare_digest(creds.password.encode(), pw.encode()):
        raise HTTPException(401, "Admin login required", headers={"WWW-Authenticate": 'Basic realm="admin"'})


@app.get("/", include_in_schema=False)
def chat_page():
    return FileResponse(config.STATIC_DIR / "index.html")


@app.get("/admin", include_in_schema=False, dependencies=[Depends(require_admin)])
def admin_page():
    return FileResponse(config.STATIC_DIR / "admin.html")


@app.get("/healthz", include_in_schema=False)
def healthz():
    return {"ok": True}


class ChatRequest(BaseModel):
    user_id: str
    message: str = Field(min_length=1, max_length=2000)


class NewUser(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    phone: str | None = Field(None, max_length=20)


@app.get("/api/users")
def users():
    return db.list_users()


@app.post("/api/users", status_code=201)
def new_user(body: NewUser):
    phone = (body.phone or "").strip() or None
    if phone and not re.fullmatch(r"\+?[\d\s\-]{6,18}", phone):
        raise HTTPException(422, "Phone number looks invalid")
    return db.create_user(body.name.strip(), phone)


@app.get("/api/conversations/{user_id}/messages")
def messages(user_id: str, after_id: int = 0):
    conv = db.get_conversation(user_id)
    if not conv:
        raise HTTPException(404, "Unknown user")
    return {"messages": db.messages_with_latency(conv["conversation_id"], after_id), "agent_paused": conv["agent_paused"]}


@app.post("/api/chat")
def chat(req: ChatRequest):
    t0 = time.perf_counter()
    text = req.message.strip()
    if not text:
        raise HTTPException(422, "Empty message")
    conv = db.get_conversation(req.user_id)
    if not conv:
        raise HTTPException(404, "Unknown user")
    cid = conv["conversation_id"]

    user_msg = db.add_message(cid, "user", text)
    if conv["resolved"]:
        db.update_conversation(cid, resolved=0)
    if conv["agent_paused"]:  # a human has taken over: store the message, no agent reply
        return {"reply": None, "agent_paused": True, "user_message": user_msg}

    history = [m for m in db.recent_messages(cid, config.HISTORY_WINDOW + 1) if m["id"] != user_msg["id"]]
    res = agent.run_turn(req.user_id, cid, text, history[-config.HISTORY_WINDOW:])
    assistant = db.add_message(cid, "assistant", res.reply)
    total_ms = int((time.perf_counter() - t0) * 1000)

    db.insert_trace(cid, assistant["id"], res.model_used, res.fallback_used, res.llm_calls, res.tool_calls, total_ms, res.error)
    flags = compute_flags(text, handoff_reasons=res.ctx.handoff_reasons, large_changes=res.ctx.large_changes,
                          tool_calls=res.tool_calls, fallback_used=res.fallback_used, total_latency_ms=total_ms,
                          error=res.error)
    db.add_attention(cid, flags)

    llm_ms = sum(c["latency_ms"] for c in res.llm_calls)
    tool_ms = sum(c["latency_ms"] for c in res.tool_calls)
    return {
        "reply": res.reply, "agent_paused": False, "user_message": user_msg,
        "assistant_message": {**assistant, "latency_ms": total_ms}, "attention_flags": flags,
        "timing": {"total_ms": total_ms, "llm_ms": llm_ms, "tool_ms": round(tool_ms, 1),
                   "overhead_ms": round(total_ms - llm_ms - tool_ms, 1), "llm_calls": len(res.llm_calls),
                   "tool_calls": len(res.tool_calls), "tool_names": [c["name"] for c in res.tool_calls],
                   "input_tokens": sum(c.get("input_tokens") or 0 for c in res.llm_calls),
                   "output_tokens": sum(c.get("output_tokens") or 0 for c in res.llm_calls),
                   "cost_usd": sum(c.get("cost") or 0 for c in res.llm_calls),
                   "model": res.model_used, "fallback_used": res.fallback_used, "truth_guard_retry": res.truth_guard_retry,
                   "per_call_ms": [c["latency_ms"] for c in res.llm_calls]},
    }


admin = [Depends(require_admin)]


class HumanReply(BaseModel):
    content: str = Field(min_length=1, max_length=2000)


class CloneRequest(BaseModel):
    name: str = Field(min_length=1, max_length=60)


def _conv_or_404(user_id: str) -> dict:
    conv = db.get_conversation(user_id)
    if not conv:
        raise HTTPException(404, "Unknown user")
    return conv


@app.get("/api/admin/stats", dependencies=admin)
def admin_stats():
    return db.admin_stats()


@app.get("/api/admin/conversations", dependencies=admin)
def admin_conversations(needs_attention: bool = Query(False)):
    return db.admin_conversations(needs_attention)


@app.get("/api/admin/conversations/{user_id}", dependencies=admin)
def admin_conversation(user_id: str):
    conv = _conv_or_404(user_id)
    traces = db.traces_for_conversation(conv["conversation_id"])
    msgs = db.messages_with_latency(conv["conversation_id"])
    for m in msgs:
        m["trace"] = traces.get(m["id"])
    return {"conversation": conv, "user": db.get_user(user_id), "messages": msgs,
            "properties": db.list_properties(user_id, include_inactive=True), "audit": db.audit_for_user(user_id, 20)}


@app.post("/api/admin/conversations/{user_id}/resolve", dependencies=admin)
def admin_resolve(user_id: str):
    cid = _conv_or_404(user_id)["conversation_id"]
    db.update_conversation(cid, needs_attention=0, attention_reasons=[], resolved=1)
    return db.get_conversation(user_id)


@app.post("/api/admin/conversations/{user_id}/takeover", dependencies=admin)
def admin_takeover(user_id: str):
    db.update_conversation(_conv_or_404(user_id)["conversation_id"], agent_paused=1)
    return db.get_conversation(user_id)


@app.post("/api/admin/conversations/{user_id}/handback", dependencies=admin)
def admin_handback(user_id: str):
    db.update_conversation(_conv_or_404(user_id)["conversation_id"], agent_paused=0)
    return db.get_conversation(user_id)


@app.post("/api/admin/conversations/{user_id}/reply", dependencies=admin)
def admin_reply(user_id: str, body: HumanReply):
    return db.add_message(_conv_or_404(user_id)["conversation_id"], "human_agent", body.content.strip())


@app.post("/api/admin/users/{user_id}/clone", status_code=201, dependencies=admin)
def admin_clone(user_id: str, body: CloneRequest):
    """Copy a user's active properties into a fresh user (used by scripts/bench.py so it never mutates seed data)."""
    if not db.get_user(user_id):
        raise HTTPException(404, "Unknown user")
    return db.create_user(body.name.strip(), clone_from=user_id)


@app.delete("/api/admin/users/{user_id}", dependencies=admin)
def admin_delete_user(user_id: str):
    u = db.get_user(user_id)
    if not u:
        raise HTTPException(404, "Unknown user")
    if not u["name"].startswith("bench-"):
        raise HTTPException(403, "Only throw-away bench-* users can be deleted")
    db.delete_user(user_id)
    return {"deleted": user_id}
