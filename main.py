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
    POST /rag/context                — get grounding text for a query
    POST /admin/reload               — re-scan syllabus_data/ (needs API key)
    POST /chat                       — proxies to Groq chat completions
    POST /transcribe                 — proxies to Groq Whisper transcription
    GET  /model/{key}                — streams an offline model file from HF

See README.md for deployment and how AI.py should call this.
"""

import os
import sys
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

app = FastAPI(title="Wren Syllabus Backend", version="1.0")

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
