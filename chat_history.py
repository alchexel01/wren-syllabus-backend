"""
Wren Chat History — email-scoped chat sync
============================================
Chat conversations follow the same identity model as premium.py: keyed
by the user's sign-in email, not device_id (see premium.py's module
docstring for the full reasoning — device UUIDs regenerate on
reinstall and silently orphan local data; email is what the user can
always type back in).

One row per (email, chat_id). The client already keeps one local JSON
file per chat shaped exactly like a ChatRecord below (id/title/date/
ts/history) — this module mirrors that shape on purpose, so a chat
fetched from here can be written straight to the local file format the
app already knows how to read, with no translation step on the client.

The backend is the source of truth; the client's local .json files are
a cache so the rest of the app keeps working (and stays usable)
offline. On every chat save the client pushes here; on sign-in it
downloads every chat for that email back down before doing anything
else — see AI.py's _sync_chats_from_backend / _push_chat_to_backend.

Storage: Postgres (DATABASE_URL env var), same reasoning as
premium.py — this service's own disk is wiped on every redeploy, which
would otherwise silently delete every saved conversation the next time
the backend ships a code change.

Endpoints (wired into main.py):
    POST   /chats/save              — upsert one chat for an email
    GET    /chats/{email}           — fetch every chat saved for an email
    DELETE /chats/{email}/{chat_id} — delete one chat
    DELETE /chats/{email}           — delete every chat for an email

All endpoints require the existing X-App-Secret header, same as every
other route on this backend.
"""

import os
import json
import logging
import datetime as dt
from typing import Optional, List, Any

import asyncpg
from fastapi import HTTPException
from pydantic import BaseModel

log = logging.getLogger("wren-backend")

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()

_pool = None


async def init_db():
    global _pool
    if not DATABASE_URL:
        log.error("[chat_history] DATABASE_URL not set - chat history endpoints will fail")
        return
    _pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
    async with _pool.acquire() as conn:
        await conn.execute(
            "CREATE TABLE IF NOT EXISTS chat_histories ("
            "email TEXT NOT NULL, "
            "chat_id TEXT NOT NULL, "
            "title TEXT NOT NULL DEFAULT 'Untitled', "
            "chat_date TEXT NOT NULL DEFAULT '', "
            "ts TIMESTAMPTZ NOT NULL, "
            "history JSONB NOT NULL, "
            "updated_at TIMESTAMPTZ NOT NULL DEFAULT now(), "
            "PRIMARY KEY (email, chat_id)"
            ")"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_histories_email "
            "ON chat_histories (email)"
        )
    log.info("[chat_history] DB pool ready, table ensured")


async def close_db():
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


def _require_pool(rid):
    if _pool is None:
        log.error(f"[{rid}] /chats: DB pool not initialized (DATABASE_URL missing?)")
        raise HTTPException(status_code=500,
                             detail={"error": "server missing DATABASE_URL", "request_id": rid})


def _normalize_email(email):
    return (email or "").strip().lower()


class ChatSaveRequest(BaseModel):
    email: str
    chat_id: str
    title: str = "Untitled"
    date: str = ""
    ts: Optional[str] = None
    history: List[Any] = []


class ChatRecord(BaseModel):
    id: str
    title: str
    date: str
    ts: str
    history: List[Any]


class ChatListResponse(BaseModel):
    chats: List[ChatRecord]


class OkResponse(BaseModel):
    ok: bool


def _parse_ts(raw: Optional[str]) -> dt.datetime:
    """Best-effort parse of the client's ISO timestamp string; falls
    back to "now" for anything missing/unparseable rather than
    rejecting the whole save over a cosmetic field."""
    if raw:
        try:
            parsed = dt.datetime.fromisoformat(raw)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=dt.timezone.utc)
            return parsed
        except Exception:
            pass
    return dt.datetime.now(dt.timezone.utc)


async def chats_save(payload: ChatSaveRequest, rid):
    _require_pool(rid)
    email = _normalize_email(payload.email)
    chat_id = (payload.chat_id or "").strip()
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail={"error": "a valid email is required", "request_id": rid})
    if not chat_id:
        raise HTTPException(status_code=400, detail={"error": "chat_id is required", "request_id": rid})

    ts = _parse_ts(payload.ts)

    async with _pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO chat_histories "
            "(email, chat_id, title, chat_date, ts, history, updated_at) "
            "VALUES ($1, $2, $3, $4, $5, $6::jsonb, now()) "
            "ON CONFLICT (email, chat_id) DO UPDATE SET "
            "title = EXCLUDED.title, chat_date = EXCLUDED.chat_date, "
            "ts = EXCLUDED.ts, history = EXCLUDED.history, updated_at = now()",
            email, chat_id, payload.title or "Untitled", payload.date or "",
            ts, json.dumps(payload.history),
        )
    log.info(f"[{rid}] /chats/save: email={email} chat_id={chat_id} "
             f"messages={len(payload.history)}")
    return OkResponse(ok=True)


async def chats_list(email, rid):
    _require_pool(rid)
    email = _normalize_email(email)
    if not email:
        raise HTTPException(status_code=400, detail={"error": "email is required", "request_id": rid})

    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT chat_id, title, chat_date, ts, history FROM chat_histories "
            "WHERE email = $1 ORDER BY updated_at DESC",
            email,
        )

    chats = []
    for r in rows:
        hist = r["history"]
        # asyncpg returns jsonb columns as raw text unless a codec is
        # registered — decode defensively either way rather than
        # assuming one representation.
        if isinstance(hist, str):
            try:
                hist = json.loads(hist)
            except Exception:
                hist = []
        chats.append(ChatRecord(
            id=r["chat_id"],
            title=r["title"],
            date=r["chat_date"],
            ts=r["ts"].isoformat(),
            history=hist or [],
        ))
    log.info(f"[{rid}] /chats/{email}: -> {len(chats)} chat(s)")
    return ChatListResponse(chats=chats)


async def chat_delete(email, chat_id, rid):
    _require_pool(rid)
    email = _normalize_email(email)
    async with _pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM chat_histories WHERE email = $1 AND chat_id = $2",
            email, chat_id,
        )
    log.info(f"[{rid}] /chats/{email}/{chat_id}: {result}")
    return OkResponse(ok=True)


async def chats_delete_all(email, rid):
    _require_pool(rid)
    email = _normalize_email(email)
    async with _pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM chat_histories WHERE email = $1", email)
    log.warning(f"[{rid}] /chats/{email}: deleted all - {result}")
    return OkResponse(ok=True)
