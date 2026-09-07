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

See README.md for deployment and how AI.py should call this.
"""

import os
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import Optional

import rag_engine

app = FastAPI(title="Wren Syllabus Backend", version="1.0")

# Simple shared-secret admin key for the reload endpoint, so random
# internet traffic can't trigger reloads. Set this as an environment
# variable on whatever host you deploy to — never hardcode a real
# secret in the file itself.
ADMIN_KEY = os.environ.get("WREN_ADMIN_KEY", "")


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
