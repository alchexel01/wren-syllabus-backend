"""
Wren Premium — Paystack integration
====================================
One-time (non-recurring) premium purchase: N3,000 for 365 days of access.
Not a subscription — Paystack never auto-charges again. When the 365
days are up, the app simply goes back to free until the user pays again.

Storage: Postgres (DATABASE_URL env var, set automatically by Render
when you attach a Postgres instance to this service in the same
project). Deliberately NOT SQLite/a local file — this service's own
disk is wiped on every redeploy, which would silently un-premium every
paying user the next time you ship a code change. Postgres survives
that because it's a separate managed service.

Endpoints (wired into main.py):
    POST /premium/initialize        — start a Paystack transaction, get checkout URL
    GET  /premium/verify/{reference}— confirm a transaction, activate premium on success
    GET  /premium/status/{device_id}— is this device currently premium, until when

All three endpoints still require the existing X-App-Secret header,
same as every other route on this backend — this file doesn't relax
that, it just adds new checks on top for the Paystack-specific bits.
"""

import os
import time
import logging
import datetime as dt
from typing import Optional

import httpx
import asyncpg
from fastapi import HTTPException
from pydantic import BaseModel

log = logging.getLogger("wren-backend")

PAYSTACK_SECRET_KEY = os.environ.get("PAYSTACK_SECRET_KEY", "").strip()
PAYSTACK_BASE_URL = "https://api.paystack.co"

# NGN 3,000 in kobo (Paystack's smallest-unit convention — 100 kobo = N1).
# Kept as a constant here rather than trusting the client to send an
# amount, so nothing on the phone side can ever request a discounted
# charge by editing a local value.
PREMIUM_PRICE_KOBO = 300_000
PREMIUM_DURATION_DAYS = 365

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()

_pool: Optional[asyncpg.Pool] = None


async def init_db():
    """Call once at app startup. Creates the pool and the table if it
    doesn't exist yet. Safe to call on every boot — CREATE TABLE IF NOT
    EXISTS is a no-op once the table already exists."""
    global _pool
    if not DATABASE_URL:
        log.error("[premium] DATABASE_URL not set — premium endpoints will fail")
        return
    _pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
    async with _pool.acquire() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS premium_subscriptions (
                device_id    TEXT PRIMARY KEY,
                reference    TEXT NOT NULL,
                purchased_at TIMESTAMPTZ NOT NULL,
                expires_at   TIMESTAMPTZ NOT NULL,
                status       TEXT NOT NULL DEFAULT 'active'
            )
            """
        )
    log.info("[premium] DB pool ready, table ensured")


async def close_db():
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


def _require_pool(rid: str):
    if _pool is None:
        log.error(f"[{rid}] /premium: DB pool not initialized (DATABASE_URL missing?)")
        raise HTTPException(
            status_code=500,
            detail={"error": "server missing DATABASE_URL", "request_id": rid},
        )


def _require_paystack_key(rid: str):
    if not PAYSTACK_SECRET_KEY:
        log.error(f"[{rid}] /premium: server missing PAYSTACK_SECRET_KEY")
        raise HTTPException(
            status_code=500,
            detail={"error": "server missing PAYSTACK_SECRET_KEY", "request_id": rid},
        )


class InitializeRequest(BaseModel):
    device_id: str
    email: Optional[str] = None  # Paystack requires an email; we synthesize one if absent


class InitializeResponse(BaseModel):
    authorization_url: str
    reference: str


class StatusResponse(BaseModel):
    is_premium: bool
    expires_at: Optional[str] = None


async def premium_initialize(payload: InitializeRequest, rid: str) -> InitializeResponse:
    _require_pool(rid)
    _require_paystack_key(rid)

    device_id = (payload.device_id or "").strip()
    if not device_id:
        raise HTTPException(
            status_code=400,
            detail={"error": "device_id is required", "request_id": rid},
        )

    # Paystack requires an email on the initialize call even though we
    # don't otherwise collect one from the app. Devices don't have a
    # real email tied to them here, so we synthesize a stable
    # per-device placeholder. Using a real, resolvable domain (gmail.com)
    # rather than a made-up one like "device.wrenai.local" — Paystack's
    # validator rejects addresses on domains with no MX record, which
    # silently failed every initialize call under a generic 502.
    email = payload.email or f"wren-device-{device_id}@gmail.com"

    async with httpx.AsyncClient(timeout=20) as client:
        try:
            resp = await client.post(
                f"{PAYSTACK_BASE_URL}/transaction/initialize",
                headers={"Authorization": f"Bearer {PAYSTACK_SECRET_KEY}"},
                json={
                    "email": email,
                    "amount": PREMIUM_PRICE_KOBO,
                    "currency": "NGN",
                    "metadata": {"device_id": device_id},
                },
            )
        except httpx.RequestError as e:
            log.error(f"[{rid}] /premium/initialize: Paystack request failed: {e!r}")
            raise HTTPException(
                status_code=502,
                detail={"error": f"upstream request failed: {e}",
                        "source": "paystack", "request_id": rid},
            )

    body = resp.json()
    if resp.status_code != 200 or not body.get("status"):
        log.error(f"[{rid}] /premium/initialize: Paystack error {resp.status_code}: {body}")
        raise HTTPException(
            status_code=502,
            detail={"error": body.get("message", "paystack initialize failed"),
                    "source": "paystack", "request_id": rid},
        )

    data = body["data"]
    log.info(f"[{rid}] /premium/initialize: device={device_id} reference={data['reference']}")
    return InitializeResponse(
        authorization_url=data["authorization_url"],
        reference=data["reference"],
    )


async def premium_verify(reference: str, rid: str) -> StatusResponse:
    _require_pool(rid)
    _require_paystack_key(rid)

    async with httpx.AsyncClient(timeout=20) as client:
        try:
            resp = await client.get(
                f"{PAYSTACK_BASE_URL}/transaction/verify/{reference}",
                headers={"Authorization": f"Bearer {PAYSTACK_SECRET_KEY}"},
            )
        except httpx.RequestError as e:
            log.error(f"[{rid}] /premium/verify: Paystack request failed: {e!r}")
            raise HTTPException(
                status_code=502,
                detail={"error": f"upstream request failed: {e}",
                        "source": "paystack", "request_id": rid},
            )

    body = resp.json()
    if resp.status_code != 200 or not body.get("status"):
        log.error(f"[{rid}] /premium/verify: Paystack error {resp.status_code}: {body}")
        raise HTTPException(
            status_code=502,
            detail={"error": body.get("message", "paystack verify failed"),
                    "source": "paystack", "request_id": rid},
        )

    data = body["data"]
    paystack_status = data.get("status")  # "success", "failed", "abandoned", ...
    device_id = (data.get("metadata") or {}).get("device_id")

    if paystack_status != "success":
        log.info(f"[{rid}] /premium/verify: reference={reference} status={paystack_status} (not activating)")
        return StatusResponse(is_premium=False)

    if not device_id:
        # Shouldn't happen since we always set it at initialize time,
        # but guard against a malformed/replayed reference rather than
        # activating premium for nobody.
        log.error(f"[{rid}] /premium/verify: reference={reference} succeeded but has no device_id in metadata")
        raise HTTPException(
            status_code=500,
            detail={"error": "payment verified but no device_id on record", "request_id": rid},
        )

    now = dt.datetime.now(dt.timezone.utc)
    expires = now + dt.timedelta(days=PREMIUM_DURATION_DAYS)

    async with _pool.acquire() as conn:
        # Upsert: if this device already had a premium row (e.g. renewing
        # after a previous year lapsed), replace it rather than erroring.
        # Idempotent on reference: if the app polls /premium/verify
        # multiple times for the same successful reference (expected,
        # since polling doesn't know when to stop until it sees success),
        # this must not push expires_at further out on every poll.
        existing = await conn.fetchrow(
            "SELECT reference, expires_at FROM premium_subscriptions WHERE device_id = $1",
            device_id,
        )
        if existing and existing["reference"] == reference:
            expires = existing["expires_at"]
        else:
            await conn.execute(
                """
                INSERT INTO premium_subscriptions (device_id, reference, purchased_at, expires_at, status)
                VALUES ($1, $2, $3, $4, 'active')
                ON CONFLICT (device_id) DO UPDATE
                    SET reference = EXCLUDED.reference,
                        purchased_at = EXCLUDED.purchased_at,
                        expires_at = EXCLUDED.expires_at,
                        status = 'active'
                """,
                device_id, reference, now, expires,
            )

    log.info(f"[{rid}] /premium/verify: device={device_id} reference={reference} ACTIVATED until {expires.isoformat()}")
    return StatusResponse(is_premium=True, expires_at=expires.isoformat())


async def premium_status(device_id: str, rid: str) -> StatusResponse:
    _require_pool(rid)
    device_id = (device_id or "").strip()
    if not device_id:
        raise HTTPException(
            status_code=400,
            detail={"error": "device_id is required", "request_id": rid},
        )

    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT expires_at FROM premium_subscriptions WHERE device_id = $1",
            device_id,
        )

    if not row:
        return StatusResponse(is_premium=False)

    now = dt.datetime.now(dt.timezone.utc)
    expires_at = row["expires_at"]
    if expires_at <= now:
        return StatusResponse(is_premium=False, expires_at=expires_at.isoformat())

    return StatusResponse(is_premium=True, expires_at=expires_at.isoformat())
