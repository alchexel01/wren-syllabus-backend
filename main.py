"""
Wren Syllabus Backend
=======================
Small FastAPI service that serves JAMB syllabus grounding over HTTP,
so subject content can be updated/added without shipping a new APK.

Run locally:
    pip install fastapi uvicorn
    uvicorn main:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health                     — liveness check
    GET  /subjects                   — list loaded subjects
    GET  /exam-bodies                — list exam bodies with their available subjects
    POST /rag/context                — get grounding text for a query
    POST /admin/reload               — re-scan syllabus_data/ (needs API key)
    POST /premium/initialize         — start a Paystack transaction, get checkout URL
    GET  /premium/verify/{reference} — confirm payment, activate premium for 365 days
    GET  /premium/status/{device_id} — is this device currently premium, until when
    POST /premium/restore            — one-time: re-link an email's premium to a new device_id
    DELETE /premium/reset/{email}    — TESTING ONLY: wipe an email's premium record
    GET  /auth/google/start          — get a Google sign-in URL + session_id
    GET  /auth/google/callback       — Google redirects here after sign-in
    GET  /auth/google/status/{id}    — poll: has this session's sign-in completed?
    POST /chat                       — proxies to Groq chat completions
    POST /transcribe                 — proxies to Groq Whisper transcription
    GET  /model/{key}                — streams an offline model file from HF

See README.md for deployment and how AI.py should call this.
"""

import os
import sys
import json
import time
import uuid
import logging
import traceback
from fastapi import FastAPI, Header, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import StreamingResponse, PlainTextResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
import httpx

import rag_engine
import premium
import auth

app = FastAPI(title="Wren Syllabus Backend", version="1.0")


@app.on_event("startup")
async def _startup():
    await premium.init_db()


@app.on_event("shutdown")
async def _shutdown():
    await premium.close_db()

# ── Error / crash logging ────────────────────────────────────────────────
# Every request gets a short request_id. It's included in:
#   - every log line for that request (so you can grep one request's
#     whole story out of Render's log stream),
#   - every HTTPException/error JSON body sent back to the app, and
#   - AI.py's record_error() calls already tag entries with a 'source'
#     string (e.g. 'Chat/http-error') — pair that with the request_id
#     printed here and you can match a client-side log line to the
#     exact backend request that produced it.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,  # Render captures stdout as the service log
)
log = logging.getLogger("wren-backend")


def _rid() -> str:
    return uuid.uuid4().hex[:8]


@app.middleware("http")
async def _log_requests(request: Request, call_next):
    """Wraps every request: logs entry/exit/timing, and — critically —
    catches any exception that escapes a route handler. Without this,
    an unhandled exception in a route (bug, upstream surprise, etc.)
    just becomes an opaque 500 with an empty body and nothing in the
    logs pointing at where it happened. This guarantees every failure
    is logged with a traceback, a request_id, and which route it came
    from, before FastAPI has any chance to swallow it."""
    rid = _rid()
    request.state.rid = rid
    start = time.time()
    log.info(f"[{rid}] --> {request.method} {request.url.path}")
    try:
        response = await call_next(request)
    except Exception:
        dur_ms = int((time.time() - start) * 1000)
        tb = traceback.format_exc()
        log.error(
            f"[{rid}] !! UNHANDLED EXCEPTION in {request.method} "
            f"{request.url.path} after {dur_ms}ms\n{tb}"
        )
        return JSONResponse(
            status_code=500,
            content={
                "detail": "internal server error",
                "request_id": rid,
                "route": request.url.path,
            },
        )
    dur_ms = int((time.time() - start) * 1000)
    response.headers["X-Request-ID"] = rid
    log.info(
        f"[{rid}] <-- {request.method} {request.url.path} "
        f"{response.status_code} ({dur_ms}ms)"
    )
    return response

# Simple shared-secret admin key for the reload endpoint, so random
# internet traffic can't trigger reloads. Set this as an environment
# variable on whatever host you deploy to — never hardcode a real
# secret in the file itself.
ADMIN_KEY = os.environ.get("WREN_ADMIN_KEY", "")

# Shared password between the phone app and this backend. NOT a real
# provider key — just stops random internet traffic from riding on
# your Groq/HF credentials for free. Rotate this any time by changing
# the env var on Render; the app needs the same value set on its side
# (see WREN_APP_SECRET in AI.py).
#
# .strip() guards against the single most common cause of "the values
# look identical but auth still fails": Render's dashboard textbox (or
# a copy-paste) silently including a trailing space or newline in the
# saved env var. Without stripping, "secret" and "secret\n" compare as
# different strings even though they render identically on screen.
APP_SECRET = os.environ.get("WREN_APP_SECRET", "").strip()

# Real provider credentials. These never leave the server.
# Add GROQ_API_KEY_2, _3, etc. on Render if you want multiple keys in
# rotation — only _1 is wired up below; extend _groq_key() if/when you
# actually add more and want round-robin/failover behavior.
GROQ_API_KEY_1 = os.environ.get("GROQ_API_KEY_1", "")
HF_ACCESS_TOKEN = os.environ.get("HF_ACCESS_TOKEN", "")

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_TRANSCRIBE_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

# Registry of offline model files the app can download. Mirrors
# OFFLINE_MODELS in AI.py — add entries here if you add them there.
OFFLINE_MODELS = {
    "gemma3_1b": {
        "url": "https://huggingface.co/litert-community/Gemma3-1B-IT/"
               "resolve/main/gemma3-1b-it-int4.litertlm",
    },
}


import hashlib


def _fingerprint(s: str) -> str:
    """Never logs the actual secret — just enough to compare two
    values without ever printing either one: length + a short hash
    prefix. Two equal secrets always produce an identical fingerprint;
    two secrets that merely *look* the same (e.g. one has a trailing
    space, or was truncated on paste) will show either a different
    length or a different hash, which is the tell."""
    if not s:
        return "EMPTY"
    return f"len={len(s)} sha256={hashlib.sha256(s.encode()).hexdigest()[:10]}"


log.info(f"[startup] WREN_APP_SECRET fingerprint: {_fingerprint(APP_SECRET)}")


def _check_app_secret(x_app_secret: str, rid: str = "-"):
    """Every proxy route requires the shared app secret. Without this,
    anyone who finds this URL could spend your Groq/HF quota for free.
    Logged explicitly so a rotated/mismatched WREN_APP_SECRET shows up
    as a clear 401 in the logs instead of looking like a generic
    network failure on the client."""
    incoming = (x_app_secret or "").strip()
    if not APP_SECRET or incoming != APP_SECRET:
        log.warning(
            f"[{rid}] auth failed: X-App-Secret mismatch. "
            f"server={_fingerprint(APP_SECRET)} "
            f"received={_fingerprint(incoming)}"
        )
        raise HTTPException(status_code=401,
                             detail={"error": "invalid or missing app secret", "request_id": rid})


class ContextRequest(BaseModel):
    query: str
    subject: Optional[str] = None   # e.g. "Biology" — omit to search all subjects


class ContextResponse(BaseModel):
    context: str
    matched: bool


@app.get("/health")
def health():
    return {"status": "ok", "chunks_loaded": len(rag_engine.syllabus_rag.chunks)}


@app.get("/subjects")
def subjects():
    return {"subjects": rag_engine.list_subjects()}


SYLLABUS_DIR = os.path.join(os.path.dirname(__file__), "syllabus_data")


@app.get("/exam-bodies")
def exam_bodies():
    """Groups syllabus_data/*.json by their "exam_body" field so the
    client can search "which exam bodies do you have data for, and what
    subjects". An exam body with no JSON files here simply never appears
    in the response — adding a new one (e.g. WAEC) just means dropping
    its JSON files into syllabus_data/ with "exam_body": "WAEC" set; no
    code change is needed here.

    Response shape:
    {
      "exam_bodies": [
        {"name": "JAMB", "subjects": ["Biology", "Chemistry", ...]}
      ]
    }
    """
    bodies = {}

    if os.path.isdir(SYLLABUS_DIR):
        for fname in os.listdir(SYLLABUS_DIR):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(SYLLABUS_DIR, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue
            if not data:
                continue

            first = data[0]
            subject = first.get("subject") or fname[:-5].replace("_", " ").title()
            exam_body = first.get("exam_body") or "JAMB"

            bodies.setdefault(exam_body, set()).add(subject)

    result = [
        {"name": name, "subjects": sorted(subjects)}
        for name, subjects in sorted(bodies.items())
    ]
    return {"exam_bodies": result}


@app.post("/rag/context", response_model=ContextResponse)
def rag_context(req: ContextRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")
    context = rag_engine.get_context_for(req.query, subject=req.subject)
    matched = "NO MATCH" not in context
    return ContextResponse(context=context, matched=matched)


@app.post("/admin/reload")
def admin_reload(x_admin_key: str = Header(default="")):
    if not ADMIN_KEY or x_admin_key != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="invalid or missing admin key")
    n = rag_engine.reload_all()
    return {"reloaded": True, "chunks_loaded": n}


# ── Premium (Paystack) ───────────────────────────────────────────────────
# One-time N3,000 / 365-day purchase. Not a subscription — see
# premium.py for the full design notes and why Postgres (not this
# service's own disk) is the store of record.

@app.post("/premium/initialize", response_model=premium.InitializeResponse)
async def premium_initialize_route(
    payload: premium.InitializeRequest,
    req: Request,
    x_app_secret: str = Header(default=""),
):
    rid = req.state.rid
    _check_app_secret(x_app_secret, rid)
    return await premium.premium_initialize(payload, rid)


@app.get("/premium/verify/{reference}", response_model=premium.StatusResponse)
async def premium_verify_route(
    reference: str,
    req: Request,
    x_app_secret: str = Header(default=""),
):
    rid = req.state.rid
    _check_app_secret(x_app_secret, rid)
    return await premium.premium_verify(reference, rid)


@app.get("/premium/status/{device_id}", response_model=premium.StatusResponse)
async def premium_status_route(
    device_id: str,
    req: Request,
    x_app_secret: str = Header(default=""),
):
    rid = req.state.rid
    _check_app_secret(x_app_secret, rid)
    return await premium.premium_status(device_id, rid)


@app.post("/premium/restore", response_model=premium.StatusResponse)
async def premium_restore_route(
    payload: premium.RestoreRequest,
    req: Request,
    x_app_secret: str = Header(default=""),
):
    """One-time-per-email restore: re-links an email's existing premium
    purchase to whatever device_id is asking. Meant for the case where
    a reinstall regenerated the device's fallback UUID and orphaned a
    paying user's premium — see premium.py for the full identity model."""
    rid = req.state.rid
    _check_app_secret(x_app_secret, rid)
    return await premium.premium_restore(payload, rid)


@app.delete("/premium/reset/{email}", response_model=premium.StatusResponse)
async def premium_reset_route(
    email: str,
    req: Request,
    x_app_secret: str = Header(default=""),
):
    """TESTING ONLY — wipes any premium record for an email so the
    purchase flow can be re-run from scratch. Same X-App-Secret gate as
    every other route here; nothing extra-locked-down about it, so
    don't rely on this being hidden from anyone who has the app
    secret — it's meant for you during development, not as a
    production admin feature."""
    rid = req.state.rid
    _check_app_secret(x_app_secret, rid)
    return await premium.premium_reset(email, rid)


# ── Google Sign-In ────────────────────────────────────────────────────
# App opens a browser to /auth/google/start's authorization_url, Google
# redirects back to /auth/google/callback on this backend, and the app
# polls /auth/google/status/{session_id} until the verified email shows
# up — same shape as the Paystack initialize/verify polling above.
# See auth.py for the full design notes.

@app.get("/auth/google/start", response_model=auth.AuthStartResponse)
def auth_google_start_route(req: Request, x_app_secret: str = Header(default="")):
    rid = req.state.rid
    _check_app_secret(x_app_secret, rid)
    return auth.auth_google_start(rid)


@app.get("/auth/google/callback")
async def auth_google_callback_route(code: str, state: str, req: Request):
    # No X-App-Secret here on purpose — Google itself calls this URL
    # via browser redirect, not the app, so it can't attach that
    # header. Security instead comes from verifying the ID token's
    # signature server-side in auth.py.
    rid = req.state.rid
    return await auth.auth_google_callback(code, state, rid)


@app.get("/auth/google/status/{session_id}", response_model=auth.AuthStatusResponse)
def auth_google_status_route(session_id: str, req: Request, x_app_secret: str = Header(default="")):
    rid = req.state.rid
    _check_app_secret(x_app_secret, rid)
    return auth.auth_google_status(session_id, rid)


@app.post("/chat")
async def chat(request: dict, req: Request, x_app_secret: str = Header(default="")):
    """Forwards the app's chat-completion body to Groq, attaching the
    real Groq key server-side.

    IMPORTANT: the app (AI.py) always sends {"stream": True} and reads
    the response with rsp.iter_lines(), parsing Server-Sent-Events
    ('data: {...}' lines ending in 'data: [DONE]') as they arrive.
    This route MUST therefore open a real streaming connection to Groq
    and forward each chunk to the client as it arrives — buffering the
    whole reply into resp.content and wrapping it as a single chunk
    (the previous bug here) produces a byte blob the client's SSE
    parser never recognizes as valid, the loop exits with no [DONE],
    and the app treats that as a dead backend and falls back to
    Offline Intelligence / a "No internet connection" message even
    though nothing was actually wrong with connectivity.
    """
    rid = req.state.rid
    _check_app_secret(x_app_secret, rid)
    if not GROQ_API_KEY_1:
        log.error(f"[{rid}] /chat: server missing GROQ_API_KEY_1")
        raise HTTPException(status_code=500,
                             detail={"error": "server missing GROQ_API_KEY_1", "request_id": rid})

    is_streaming_req = bool(request.get("stream"))
    model = request.get("model", "?")
    log.info(f"[{rid}] /chat: model={model} stream={is_streaming_req}")

    client = httpx.AsyncClient(timeout=httpx.Timeout(30, read=60))
    try:
        upstream_req = client.build_request(
            "POST", GROQ_CHAT_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY_1}",
                "Content-Type": "application/json",
            },
            json=request,
        )
        upstream = await client.send(upstream_req, stream=True)
    except httpx.RequestError as e:
        await client.aclose()
        log.error(f"[{rid}] /chat: upstream (Groq) request failed: {e!r}")
        raise HTTPException(
            status_code=502,
            detail={"error": f"upstream request failed: {e}",
                    "source": "groq", "request_id": rid},
        )

    if upstream.status_code != 200:
        # Read the body BEFORE closing, so the actual reason Groq gave
        # (bad model name, invalid param, rate limit, auth, etc.) is
        # logged and returned instead of a bare status code.
        body = await upstream.aread()
        await upstream.aclose()
        await client.aclose()
        body_text = body.decode("utf-8", errors="replace")[:1000]
        log.error(
            f"[{rid}] /chat: Groq returned HTTP {upstream.status_code}: {body_text}"
        )
        raise HTTPException(
            status_code=upstream.status_code,
            detail={"error": body_text, "source": "groq", "request_id": rid},
        )

    async def _stream():
        chunk_count = 0
        byte_count = 0
        try:
            async for chunk in upstream.aiter_bytes():
                chunk_count += 1
                byte_count += len(chunk)
                yield chunk
        except Exception as e:
            # A failure mid-stream (Groq connection dropped, read
            # timeout, etc.) after the 200 status and headers have
            # already been sent to the client — we can no longer
            # change the HTTP status at this point, so log it loudly
            # server-side; the client sees this as a stream that ended
            # without a [DONE] sentinel (AI.py already detects and
            # logs that case itself as 'Chat/stream').
            log.error(
                f"[{rid}] /chat: stream broke after {chunk_count} chunks "
                f"({byte_count} bytes): {type(e).__name__}: {e}"
            )
        finally:
            log.info(f"[{rid}] /chat: stream finished — {chunk_count} chunks, {byte_count} bytes")
            await upstream.aclose()
            await client.aclose()

    return StreamingResponse(
        _stream(),
        status_code=200,
        media_type=upstream.headers.get("content-type", "text/event-stream"),
        headers={"X-Request-ID": rid},
    )


@app.post("/transcribe")
async def transcribe(
    req: Request,
    file: UploadFile = File(...),
    model: str = Form("whisper-large-v3-turbo"),
    response_format: str = Form("text"),
    x_app_secret: str = Header(default=""),
):
    """Forwards the recorded audio clip to Groq's Whisper endpoint,
    attaching the real Groq key server-side. Returns plain text, since
    the app reads resp.text.strip() directly."""
    rid = req.state.rid
    _check_app_secret(x_app_secret, rid)
    if not GROQ_API_KEY_1:
        log.error(f"[{rid}] /transcribe: server missing GROQ_API_KEY_1")
        raise HTTPException(status_code=500,
                             detail={"error": "server missing GROQ_API_KEY_1", "request_id": rid})

    audio_bytes = await file.read()
    log.info(f"[{rid}] /transcribe: {len(audio_bytes)} bytes, model={model}")

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.post(
                GROQ_TRANSCRIBE_URL,
                headers={"Authorization": f"Bearer {GROQ_API_KEY_1}"},
                files={"file": (file.filename, audio_bytes, file.content_type)},
                data={"model": model, "response_format": response_format},
            )
        except httpx.RequestError as e:
            log.error(f"[{rid}] /transcribe: upstream (Groq) request failed: {e!r}")
            raise HTTPException(
                status_code=502,
                detail={"error": f"upstream request failed: {e}",
                        "source": "groq", "request_id": rid},
            )

    if resp.status_code != 200:
        body_text = resp.text[:500]
        log.error(f"[{rid}] /transcribe: Groq returned HTTP {resp.status_code}: {body_text}")
        raise HTTPException(
            status_code=resp.status_code,
            detail={"error": body_text, "source": "groq", "request_id": rid},
        )

    return PlainTextResponse(resp.text)


@app.get("/model/{key}")
async def download_model(key: str, req: Request, x_app_secret: str = Header(default="")):
    """Streams an offline model file from Hugging Face, attaching the
    real HF token server-side. Streamed rather than buffered in memory
    since these files run several hundred MB."""
    rid = req.state.rid
    _check_app_secret(x_app_secret, rid)
    meta = OFFLINE_MODELS.get(key)
    if not meta:
        log.warning(f"[{rid}] /model/{key}: unknown model key")
        raise HTTPException(status_code=404,
                             detail={"error": "unknown model key", "request_id": rid})
    if not HF_ACCESS_TOKEN:
        log.error(f"[{rid}] /model/{key}: server missing HF_ACCESS_TOKEN")
        raise HTTPException(status_code=500,
                             detail={"error": "server missing HF_ACCESS_TOKEN", "request_id": rid})

    client = httpx.AsyncClient(timeout=httpx.Timeout(30, read=120))
    try:
        upstream_req = client.build_request(
            "GET", meta["url"],
            headers={"Authorization": f"Bearer {HF_ACCESS_TOKEN}"},
        )
        upstream = await client.send(upstream_req, stream=True)
    except httpx.RequestError as e:
        await client.aclose()
        log.error(f"[{rid}] /model/{key}: upstream (HF) request failed: {e!r}")
        raise HTTPException(
            status_code=502,
            detail={"error": f"upstream request failed: {e}",
                    "source": "huggingface", "request_id": rid},
        )

    if upstream.status_code in (401, 403):
        await upstream.aclose()
        await client.aclose()
        log.error(f"[{rid}] /model/{key}: Hugging Face auth failed ({upstream.status_code})")
        raise HTTPException(
            status_code=upstream.status_code,
            detail={"error": "Hugging Face auth failed", "source": "huggingface", "request_id": rid},
        )
    if upstream.status_code != 200:
        await upstream.aclose()
        await client.aclose()
        log.error(f"[{rid}] /model/{key}: HF returned HTTP {upstream.status_code}")
        raise HTTPException(
            status_code=upstream.status_code,
            detail={"error": "upstream error", "source": "huggingface", "request_id": rid},
        )

    async def _stream():
        byte_count = 0
        try:
            async for chunk in upstream.aiter_bytes():
                byte_count += len(chunk)
                yield chunk
        except Exception as e:
            log.error(f"[{rid}] /model/{key}: stream broke after {byte_count} bytes: {e!r}")
        finally:
            log.info(f"[{rid}] /model/{key}: stream finished — {byte_count} bytes")
            await upstream.aclose()
            await client.aclose()

    headers = {"X-Request-ID": rid}
    if "content-length" in upstream.headers:
        headers["content-length"] = upstream.headers["content-length"]

    return StreamingResponse(_stream(), status_code=200, headers=headers,
                              media_type="application/octet-stream")
