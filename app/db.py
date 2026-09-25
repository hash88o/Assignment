"""SQLite layer: schema, CSV seeding, and small query helpers.

One short-lived connection per call (sqlite3.connect is ~50µs) so it is safe from FastAPI's
threadpool without a pool. All property reads/writes take user_id, so a tool bound to one user
can never touch another user's rows.
"""
from __future__ import annotations

import csv
import json
import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from . import config
from .normalize import metro_for, normalize_type, split_location

SCHEMA = """
CREATE TABLE IF NOT EXISTS users(
  user_id TEXT PRIMARY KEY, name TEXT NOT NULL, phone TEXT, city TEXT, preferences TEXT,
  preferred_locations TEXT, portfolio_value_pref TEXT, portfolio_value_min INTEGER, portfolio_value_max INTEGER,
  created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS properties(
  property_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, property_type_raw TEXT, type_group TEXT NOT NULL,
  asset_class TEXT NOT NULL, sub_type TEXT, locality TEXT, city TEXT, metro TEXT, area_sqft REAL NOT NULL,
  current_estimated_value_inr INTEGER NOT NULL, purchase_price_inr INTEGER, annual_rent_inr INTEGER NOT NULL DEFAULT 0,
  occupancy_status TEXT NOT NULL, tenant_status TEXT, ownership_percent REAL NOT NULL DEFAULT 100,
  status TEXT NOT NULL DEFAULT 'Active', updated_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_prop_user ON properties(user_id);
CREATE TABLE IF NOT EXISTS conversations(
  conversation_id TEXT PRIMARY KEY, user_id TEXT NOT NULL UNIQUE, agent_paused INTEGER NOT NULL DEFAULT 0,
  needs_attention INTEGER NOT NULL DEFAULT 0, attention_reasons TEXT NOT NULL DEFAULT '[]',
  resolved INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL, last_message_at TEXT);
CREATE TABLE IF NOT EXISTS messages(
  id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT NOT NULL, role TEXT NOT NULL,
  content TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conversation_id, id);
CREATE TABLE IF NOT EXISTS turn_traces(
  id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT NOT NULL, message_id INTEGER, model_used TEXT,
  fallback_used INTEGER NOT NULL DEFAULT 0, llm_calls TEXT NOT NULL DEFAULT '[]', tool_calls TEXT NOT NULL DEFAULT '[]',
  total_latency_ms INTEGER, error TEXT, created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_trace_msg ON turn_traces(message_id);
CREATE TABLE IF NOT EXISTS audit_log(
  id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, property_id TEXT, action TEXT NOT NULL,
  field TEXT, old_value TEXT, new_value TEXT, created_at TEXT NOT NULL);
"""

PROPERTY_COLS = ["property_id", "user_id", "property_type_raw", "type_group", "asset_class", "sub_type", "locality",
                 "city", "metro", "area_sqft", "current_estimated_value_inr", "purchase_price_inr", "annual_rent_inr",
                 "occupancy_status", "tenant_status", "ownership_percent", "status", "updated_at"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def conn():
    c = sqlite3.connect(config.db_path(), timeout=10)
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    except Exception:
        c.rollback()
        raise
    finally:
        c.close()


def _rows(cur) -> list[dict]:
    return [dict(r) for r in cur.fetchall()]


def _one(cur) -> dict | None:
    r = cur.fetchone()
    return dict(r) if r else None


def init_db() -> None:
    os.makedirs(os.path.dirname(config.db_path()) or ".", exist_ok=True)
    with conn() as c:
        c.execute("PRAGMA journal_mode=WAL")
        c.executescript(SCHEMA)
        empty = c.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0
    if empty:
        seed()


def _parse_pref_range(text: str | None) -> tuple[int | None, int | None]:
    nums = [int(x) for x in re.findall(r"\d+", text or "")]
    return (nums[0], nums[-1]) if nums else (None, None)


def seed() -> None:
    ts = now()
    with conn() as c:
        with open(config.DATA_DIR / "users.csv", newline="", encoding="utf-8") as f:
            for u in csv.DictReader(f):
                lo, hi = _parse_pref_range(u["portfolio_value_preference_inr"])
                c.execute("INSERT INTO users VALUES(?,?,?,?,?,?,?,?,?,?)",
                          (u["user_id"], u["name"], None, u["city"], u["preferences"], u["preferred_locations"],
                           u["portfolio_value_preference_inr"], lo, hi, ts))
                c.execute("INSERT INTO conversations(conversation_id,user_id,created_at) VALUES(?,?,?)",
                          (conversation_id_for(u["user_id"]), u["user_id"], ts))
        with open(config.DATA_DIR / "properties.csv", newline="", encoding="utf-8") as f:
            for p in csv.DictReader(f):
                locality, city = split_location(p["location"])
                t = normalize_type(p["property_type"]) or normalize_type(p["sub_type"])
                c.execute(f"INSERT INTO properties({','.join(PROPERTY_COLS)}) VALUES({','.join('?' * len(PROPERTY_COLS))})",
                          (p["property_id"], p["user_id"], p["property_type"], t["type_group"], t["asset_class"],
                           p["sub_type"], locality, city, metro_for(city), float(p["area_sqft"]),
                           int(p["current_estimated_value_inr"]),
                           int(p["purchase_price_inr"]) if p["purchase_price_inr"].strip() else None,
                           int(p["annual_rent_inr"]), p["occupancy_status"], p["tenant_status"],
                           float(p["ownership_percent"]), p["status"], ts))


def conversation_id_for(user_id: str) -> str:
    return f"conv_{user_id}"


def list_users() -> list[dict]:
    with conn() as c:
        return _rows(c.execute("SELECT user_id,name,phone,city FROM users ORDER BY user_id"))


def get_user(user_id: str) -> dict | None:
    with conn() as c:
        return _one(c.execute("SELECT * FROM users WHERE user_id=?", (user_id,)))


def create_user(name: str, phone: str | None = None, city: str | None = None, clone_from: str | None = None) -> dict:
    ts = now()
    with conn() as c:
        c.execute("BEGIN IMMEDIATE")
        last = c.execute("SELECT MAX(CAST(SUBSTR(user_id,2) AS INTEGER)) FROM users").fetchone()[0] or 0
        uid = f"U{last + 1:03d}"
        src = _one(c.execute("SELECT * FROM users WHERE user_id=?", (clone_from,))) if clone_from else None
        c.execute("INSERT INTO users VALUES(?,?,?,?,?,?,?,?,?,?)",
                  (uid, name, phone, city or (src or {}).get("city"), (src or {}).get("preferences"),
                   (src or {}).get("preferred_locations"), (src or {}).get("portfolio_value_pref"),
                   (src or {}).get("portfolio_value_min"), (src or {}).get("portfolio_value_max"), ts))
        c.execute("INSERT INTO conversations(conversation_id,user_id,created_at) VALUES(?,?,?)",
                  (conversation_id_for(uid), uid, ts))
        if src:
            for p in _rows(c.execute("SELECT * FROM properties WHERE user_id=? AND status='Active'", (clone_from,))):
                pid = _next_property_id(c)
                p.update(property_id=pid, user_id=uid, updated_at=ts)
                c.execute(f"INSERT INTO properties({','.join(PROPERTY_COLS)}) VALUES({','.join('?' * len(PROPERTY_COLS))})",
                          [p[k] for k in PROPERTY_COLS])
    return get_user(uid)


def delete_user(user_id: str) -> None:
    """Used by the bench script to clean up its throw-away users."""
    with conn() as c:
        conv = conversation_id_for(user_id)
        c.execute("DELETE FROM turn_traces WHERE conversation_id=?", (conv,))
        c.execute("DELETE FROM messages WHERE conversation_id=?", (conv,))
        c.execute("DELETE FROM conversations WHERE user_id=?", (user_id,))
        c.execute("DELETE FROM properties WHERE user_id=?", (user_id,))
        c.execute("DELETE FROM audit_log WHERE user_id=?", (user_id,))
        c.execute("DELETE FROM users WHERE user_id=?", (user_id,))


def list_properties(user_id: str, include_inactive: bool = False) -> list[dict]:
    q = "SELECT * FROM properties WHERE user_id=?" + ("" if include_inactive else " AND status='Active'")
    with conn() as c:
        return _rows(c.execute(q + " ORDER BY property_id", (user_id,)))


def get_property(user_id: str, property_id: str) -> dict | None:
    with conn() as c:
        return _one(c.execute("SELECT * FROM properties WHERE user_id=? AND property_id=?", (user_id, property_id)))


def _next_property_id(c) -> str:
    last = c.execute("SELECT MAX(CAST(SUBSTR(property_id,2) AS INTEGER)) FROM properties").fetchone()[0] or 0
    return f"P{last + 1:03d}"


def insert_property(user_id: str, data: dict) -> dict:
    ts = now()
    with conn() as c:
        c.execute("BEGIN IMMEDIATE")
        pid = _next_property_id(c)
        row = {"property_id": pid, "user_id": user_id, "purchase_price_inr": None, "tenant_status": None,
               "ownership_percent": 100, "status": "Active", "updated_at": ts, **data}
        row["tenant_status"] = "Yes" if row["occupancy_status"] == "Tenanted" else "No"
        c.execute(f"INSERT INTO properties({','.join(PROPERTY_COLS)}) VALUES({','.join('?' * len(PROPERTY_COLS))})",
                  [row.get(k) for k in PROPERTY_COLS])
        c.execute("INSERT INTO audit_log(user_id,property_id,action,field,old_value,new_value,created_at) VALUES(?,?,?,?,?,?,?)",
                  (user_id, pid, "add", "*", None, json.dumps({k: row[k] for k in PROPERTY_COLS if k not in ("updated_at",)}), ts))
    return get_property(user_id, pid)


def update_property_fields(user_id: str, property_id: str, updates: dict, action: str = "update") -> dict:
    """Apply several column updates atomically, one audit_log row per changed column."""
    ts = now()
    with conn() as c:
        old = _one(c.execute("SELECT * FROM properties WHERE user_id=? AND property_id=?", (user_id, property_id)))
        if not old:
            raise KeyError(property_id)
        if "occupancy_status" in updates:
            updates = {**updates, "tenant_status": "Yes" if updates["occupancy_status"] == "Tenanted" else "No"}
        for col, new in updates.items():
            if old[col] == new:
                continue
            c.execute(f"UPDATE properties SET {col}=?, updated_at=? WHERE user_id=? AND property_id=?",
                      (new, ts, user_id, property_id))
            c.execute("INSERT INTO audit_log(user_id,property_id,action,field,old_value,new_value,created_at) VALUES(?,?,?,?,?,?,?)",
                      (user_id, property_id, action, col, None if old[col] is None else str(old[col]),
                       None if new is None else str(new), ts))
    return get_property(user_id, property_id)


def audit_for_user(user_id: str, limit: int = 50) -> list[dict]:
    with conn() as c:
        return _rows(c.execute("SELECT * FROM audit_log WHERE user_id=? ORDER BY id DESC LIMIT ?", (user_id, limit)))


def get_conversation(user_id: str) -> dict | None:
    with conn() as c:
        r = _one(c.execute("SELECT * FROM conversations WHERE user_id=?", (user_id,)))
    return _conv(r)


def _conv(r: dict | None) -> dict | None:
    if r:
        for k in ("agent_paused", "needs_attention", "resolved"):
            r[k] = bool(r[k])
        r["attention_reasons"] = json.loads(r["attention_reasons"] or "[]")
    return r


def add_message(conversation_id: str, role: str, content: str) -> dict:
    ts = now()
    with conn() as c:
        cur = c.execute("INSERT INTO messages(conversation_id,role,content,created_at) VALUES(?,?,?,?)",
                        (conversation_id, role, content, ts))
        c.execute("UPDATE conversations SET last_message_at=? WHERE conversation_id=?", (ts, conversation_id))
        return {"id": cur.lastrowid, "conversation_id": conversation_id, "role": role, "content": content, "created_at": ts}


def recent_messages(conversation_id: str, limit: int) -> list[dict]:
    with conn() as c:
        rows = _rows(c.execute("SELECT id,role,content,created_at FROM messages WHERE conversation_id=? "
                               "ORDER BY id DESC LIMIT ?", (conversation_id, limit)))
    return rows[::-1]


def messages_with_latency(conversation_id: str, after_id: int = 0) -> list[dict]:
    with conn() as c:
        return _rows(c.execute(
            "SELECT m.id,m.role,m.content,m.created_at,t.total_latency_ms AS latency_ms FROM messages m "
            "LEFT JOIN turn_traces t ON t.message_id=m.id WHERE m.conversation_id=? AND m.id>? ORDER BY m.id",
            (conversation_id, after_id)))


def update_conversation(conversation_id: str, **fields) -> None:
    if not fields:
        return
    if "attention_reasons" in fields:
        fields["attention_reasons"] = json.dumps(fields["attention_reasons"])
    sets = ",".join(f"{k}=?" for k in fields)
    with conn() as c:
        c.execute(f"UPDATE conversations SET {sets} WHERE conversation_id=?", (*fields.values(), conversation_id))


def add_attention(conversation_id: str, reasons: list[str]) -> None:
    """Merge reasons into the conversation and raise the needs_attention flag."""
    if not reasons:
        return
    with conn() as c:
        cur = json.loads(c.execute("SELECT attention_reasons FROM conversations WHERE conversation_id=?",
                                   (conversation_id,)).fetchone()[0] or "[]")
        merged = cur + [r for r in reasons if r not in cur]
        c.execute("UPDATE conversations SET needs_attention=1, resolved=0, attention_reasons=? WHERE conversation_id=?",
                  (json.dumps(merged), conversation_id))


def insert_trace(conversation_id: str, message_id: int | None, model_used: str | None, fallback_used: bool,
                 llm_calls: list, tool_calls: list, total_latency_ms: int, error: str | None) -> None:
    with conn() as c:
        c.execute("INSERT INTO turn_traces(conversation_id,message_id,model_used,fallback_used,llm_calls,tool_calls,"
                  "total_latency_ms,error,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                  (conversation_id, message_id, model_used, int(fallback_used), json.dumps(llm_calls),
                   json.dumps(tool_calls), total_latency_ms, error, now()))


def traces_for_conversation(conversation_id: str) -> dict[int, dict]:
    with conn() as c:
        rows = _rows(c.execute("SELECT * FROM turn_traces WHERE conversation_id=?", (conversation_id,)))
    out = {}
    for r in rows:
        r["llm_calls"], r["tool_calls"] = json.loads(r["llm_calls"]), json.loads(r["tool_calls"])
        r["fallback_used"] = bool(r["fallback_used"])
        out[r["message_id"]] = r
    return out


def admin_conversations(only_attention: bool = False) -> list[dict]:
    q = ("SELECT c.*, u.name, u.city, "
         "(SELECT content FROM messages m WHERE m.conversation_id=c.conversation_id ORDER BY id DESC LIMIT 1) AS last_message, "
         "(SELECT COUNT(*) FROM messages m WHERE m.conversation_id=c.conversation_id) AS message_count "
         "FROM conversations c JOIN users u ON u.user_id=c.user_id")
    if only_attention:
        q += " WHERE c.needs_attention=1"
    q += " ORDER BY c.needs_attention DESC, COALESCE(c.last_message_at, c.created_at) DESC"
    with conn() as c:
        return [_conv(r) for r in _rows(c.execute(q))]


def admin_stats() -> dict:
    with conn() as c:
        convs = c.execute("SELECT COUNT(*), COALESCE(SUM(needs_attention),0) FROM conversations").fetchone()
        lat = sorted(r[0] for r in c.execute("SELECT total_latency_ms FROM turn_traces WHERE total_latency_ms IS NOT NULL"))
        calls = [json.loads(r[0]) for r in c.execute("SELECT llm_calls FROM turn_traces")]
    cost = sum((x.get("cost") or 0) for lc in calls for x in lc)

    def pctl(p):
        if not lat:
            return None
        return lat[min(len(lat) - 1, int(round(p / 100 * (len(lat) - 1))))]

    return {"conversations": convs[0], "flagged": convs[1], "turns": len(lat), "p50_ms": pctl(50),
            "p95_ms": pctl(95), "total_cost_usd": round(cost, 6)}
