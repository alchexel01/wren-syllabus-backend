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
from fastapi import FastAPI, Header, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse, PlainTextResponse
from pydantic import BaseModel
from typing import Optional
import httpx

import rag_engine

app = FastAPI(title="Wren Syllabus Backend", version="1.0")

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
APP_SECRET = os.environ.get("WREN_APP_SECRET", "")

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


def _check_app_secret(x_app_secret: str):
    """Every proxy route requires the shared app secret. Without this,
    anyone who finds this URL could spend your Groq/HF quota for free."""
    if not APP_SECRET or x_app_secret != APP_SECRET:
        raise HTTPException(status_code=401, detail="invalid or missing app secret")


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
async def chat(request: dict, x_app_secret: str = Header(default="")):
    """Forwards the app's chat-completion body to Groq, attaching the
    real Groq key server-side. Body/response shape is passed through
    as-is (OpenAI-compatible), since the app already expects
    data['choices'][0]['message']['content'] back."""
    _check_app_secret(x_app_secret)
    if not GROQ_API_KEY_1:
        raise HTTPException(status_code=500, detail="server missing GROQ_API_KEY_1")

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.post(
                GROQ_CHAT_URL,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY_1}",
                    "Content-Type": "application/json",
                },
                json=request,
            )
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"upstream request failed: {e}")

    return StreamingResponse(
        iter([resp.content]),
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "application/json"),
    )


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form("whisper-large-v3-turbo"),
    response_format: str = Form("text"),
    x_app_secret: str = Header(default=""),
):
    """Forwards the recorded audio clip to Groq's Whisper endpoint,
    attaching the real Groq key server-side. Returns plain text, since
    the app reads resp.text.strip() directly."""
    _check_app_secret(x_app_secret)
    if not GROQ_API_KEY_1:
        raise HTTPException(status_code=500, detail="server missing GROQ_API_KEY_1")

    audio_bytes = await file.read()

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.post(
                GROQ_TRANSCRIBE_URL,
                headers={"Authorization": f"Bearer {GROQ_API_KEY_1}"},
                files={"file": (file.filename, audio_bytes, file.content_type)},
                data={"model": model, "response_format": response_format},
            )
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"upstream request failed: {e}")

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text[:500])

    return PlainTextResponse(resp.text)


@app.get("/model/{key}")
async def download_model(key: str, x_app_secret: str = Header(default="")):
    """Streams an offline model file from Hugging Face, attaching the
    real HF token server-side. Streamed rather than buffered in memory
    since these files run several hundred MB."""
    _check_app_secret(x_app_secret)
    meta = OFFLINE_MODELS.get(key)
    if not meta:
        raise HTTPException(status_code=404, detail="unknown model key")
    if not HF_ACCESS_TOKEN:
        raise HTTPException(status_code=500, detail="server missing HF_ACCESS_TOKEN")

    client = httpx.AsyncClient(timeout=httpx.Timeout(30, read=120))
    try:
        req = client.build_request(
            "GET", meta["url"],
            headers={"Authorization": f"Bearer {HF_ACCESS_TOKEN}"},
        )
        upstream = await client.send(req, stream=True)
    except httpx.RequestError as e:
        await client.aclose()
        raise HTTPException(status_code=502, detail=f"upstream request failed: {e}")

    if upstream.status_code in (401, 403):
        await upstream.aclose()
        await client.aclose()
        raise HTTPException(status_code=upstream.status_code, detail="Hugging Face auth failed")
    if upstream.status_code != 200:
        await upstream.aclose()
        await client.aclose()
        raise HTTPException(status_code=upstream.status_code, detail="upstream error")

    async def _stream():
        try:
            async for chunk in upstream.aiter_bytes():
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()

    headers = {}
    if "content-length" in upstream.headers:
        headers["content-length"] = upstream.headers["content-length"]

    return StreamingResponse(_stream(), status_code=200, headers=headers,
                              media_type="application/octet-stream")
