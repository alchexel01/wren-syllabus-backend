"""
Wren Google Sign-In
====================
Verifies Google identity server-side and hands the app back a verified
email, using the same "app opens a browser, polls a status endpoint"
pattern already used for Paystack checkout in premium.py.

Flow:
    1. App generates a random session_id, opens the browser to
       GOOGLE_AUTH_URL with state=session_id and our redirect_uri.
    2. User signs in with Google. Google redirects the browser to
       GET /auth/google/callback?code=...&state=session_id on THIS
       backend.
    3. We exchange the code for tokens (server-side — the Client
       Secret never touches the app), verify the ID token's signature
       against Google's public keys, and pull out the verified email.
    4. We store {session_id: email} in memory with a timestamp and
       show the user a plain "you can return to the app" page.
    5. The app polls GET /auth/google/status/{session_id} until it
       sees done=true, then treats that email exactly like the email
       from the old manual-entry onboarding screen (premium.py already
       keys everything off email, so nothing downstream needs to
       change).

Storage: in-memory dict, NOT Postgres. Unlike premium status, a login
session is only needed for a few minutes while the user is mid-flow —
it doesn't need to survive a redeploy, and every session is deleted
right after the app collects it (or expires on its own). If you scale
to multiple backend instances this dict won't be shared between them;
fine for a single Render instance, worth revisiting if you add more.

Env vars required (Render dashboard, same place as the others):
    GOOGLE_CLIENT_ID       — from the Web application OAuth client
    GOOGLE_CLIENT_SECRET   — from the same client (Client secret)
    GOOGLE_REDIRECT_URI    — must exactly match an Authorized redirect
                             URI registered on that client, e.g.
                             https://wren-syllabus-backend.onrender.com/auth/google/callback
"""

import os
import time
import logging
import secrets
from urllib.parse import urlencode
from typing import Optional

import httpx
from fastapi import HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from google.oauth2 import id_token as google_id_token
from google.auth.transport import requests as google_requests

log = logging.getLogger("wren-backend")

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "").strip()
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "").strip()
GOOGLE_REDIRECT_URI = os.environ.get("GOOGLE_REDIRECT_URI", "").strip()

GOOGLE_AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"

# How long an unfinished login session is kept before we give up on it
# and let it be garbage-collected. Generous, since a user might sit on
# Google's account picker for a while.
SESSION_TTL_SECONDS = 10 * 60

# session_id -> {"email": str, "created_at": float} once the callback
# lands. Sessions that never complete are just never inserted here —
# _purge_expired() below only needs to clean up completed-but-uncollected
# ones plus anything that's aged out.
_sessions: dict[str, dict] = {}

_google_request = google_requests.Request()


def _purge_expired():
    now = time.time()
    expired = [sid for sid, v in _sessions.items()
               if now - v["created_at"] > SESSION_TTL_SECONDS]
    for sid in expired:
        del _sessions[sid]


def _require_config(rid):
    missing = [name for name, val in (
        ("GOOGLE_CLIENT_ID", GOOGLE_CLIENT_ID),
        ("GOOGLE_CLIENT_SECRET", GOOGLE_CLIENT_SECRET),
        ("GOOGLE_REDIRECT_URI", GOOGLE_REDIRECT_URI),
    ) if not val]
    if missing:
        log.error(f"[{rid}] /auth/google: missing env vars: {', '.join(missing)}")
        raise HTTPException(
            status_code=500,
            detail={"error": f"server missing {', '.join(missing)}", "request_id": rid},
        )


class AuthStartResponse(BaseModel):
    authorization_url: str
    session_id: str


class AuthStatusResponse(BaseModel):
    done: bool
    email: Optional[str] = None


def auth_google_start(rid: str) -> AuthStartResponse:
    """Called by the app before opening the browser. Mints a fresh
    session_id and builds the Google consent-screen URL around it, so
    the Client ID/redirect URI live here on the backend rather than
    being hardcoded into the app (easier to rotate later)."""
    _require_config(rid)
    _purge_expired()

    session_id = secrets.token_urlsafe(24)
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email",
        "state": session_id,
        # Forces the account chooser rather than silently reusing
        # whatever Google session is already active in the browser —
        # matters most on a shared/test device.
        "prompt": "select_account",
    }
    auth_url = f"{GOOGLE_AUTH_ENDPOINT}?{urlencode(params)}"

    log.info(f"[{rid}] /auth/google/start: issued session_id={session_id[:8]}...")
    return AuthStartResponse(authorization_url=auth_url, session_id=session_id)


async def auth_google_callback(code: str, state: str, rid: str) -> HTMLResponse:
    """Google redirects here after the user signs in. Exchanges the
    code for tokens, verifies the ID token signature, and stashes the
    verified email under the session_id (Google's 'state' param) for
    the app to pick up via polling."""
    _require_config(rid)
    _purge_expired()

    session_id = state
    if not session_id:
        log.warning(f"[{rid}] /auth/google/callback: missing state param")
        return _result_page("Something went wrong — missing session. "
                             "Please return to the app and try again.", ok=False)

    async with httpx.AsyncClient(timeout=20) as client:
        try:
            resp = await client.post(
                GOOGLE_TOKEN_ENDPOINT,
                data={
                    "code": code,
                    "client_id": GOOGLE_CLIENT_ID,
                    "client_secret": GOOGLE_CLIENT_SECRET,
                    "redirect_uri": GOOGLE_REDIRECT_URI,
                    "grant_type": "authorization_code",
                },
            )
        except httpx.RequestError as e:
            log.error(f"[{rid}] /auth/google/callback: token exchange request failed: {e!r}")
            return _result_page("Couldn't reach Google. Please return to the app and try again.", ok=False)

    body = resp.json()
    if resp.status_code != 200 or "id_token" not in body:
        log.error(f"[{rid}] /auth/google/callback: token exchange failed "
                  f"{resp.status_code}: {body}")
        return _result_page("Sign-in failed. Please return to the app and try again.", ok=False)

    try:
        claims = google_id_token.verify_oauth2_token(
            body["id_token"], _google_request, GOOGLE_CLIENT_ID,
        )
    except ValueError as e:
        # Signature invalid, expired, wrong audience, etc. — never
        # trust an ID token we haven't verified against Google's keys.
        log.error(f"[{rid}] /auth/google/callback: ID token verification failed: {e}")
        return _result_page("Sign-in couldn't be verified. Please try again.", ok=False)

    if not claims.get("email_verified", False):
        log.warning(f"[{rid}] /auth/google/callback: email not verified on Google's side "
                    f"(email={claims.get('email')})")
        return _result_page(
            "That Google account's email isn't verified. Please verify it with "
            "Google first, then try again.", ok=False,
        )

    email = (claims.get("email") or "").strip().lower()
    if not email:
        log.error(f"[{rid}] /auth/google/callback: verified token had no email claim")
        return _result_page("Sign-in failed. Please return to the app and try again.", ok=False)

    _sessions[session_id] = {"email": email, "created_at": time.time()}
    log.info(f"[{rid}] /auth/google/callback: session_id={session_id[:8]}... "
             f"verified email={email}")
    return _result_page("You're signed in. You can close this tab and return to Wren.", ok=True)


def auth_google_status(session_id: str, rid: str) -> AuthStatusResponse:
    """Polled by the app after it opens the browser. Returns the
    verified email once the callback above has landed, then the
    session is consumed (deleted) so it can't be polled/reused again."""
    _purge_expired()
    session = _sessions.get(session_id)
    if not session:
        return AuthStatusResponse(done=False)

    email = session["email"]
    del _sessions[session_id]  # one-shot: collected exactly once
    log.info(f"[{rid}] /auth/google/status: session_id={session_id[:8]}... "
             f"collected email={email}")
    return AuthStatusResponse(done=True, email=email)


def _result_page(message: str, ok: bool) -> HTMLResponse:
    color = "#16a34a" if ok else "#dc2626"
    html = f"""
    <html>
      <head>
        <title>Sign in to Wren AI</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
      </head>
      <body style="font-family: -apple-system, sans-serif; text-align: center; padding: 48px 24px;">
        <p style="font-size: 15px; color: #6b7280; margin-bottom: 4px;">Wren AI</p>
        <p style="color: {color}; font-size: 18px;">{message}</p>
      </body>
    </html>
    """
    return HTMLResponse(content=html, status_code=200 if ok else 400)
