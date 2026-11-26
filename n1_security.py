"""Local-API hardening for the NinjaOne agent UI.

Threat model: the server executes approved NinjaOne writes, so any page or
process that can reach /api/* can try to drive it. This module adds:

* a per-launch API token (header ``x-n1-token``) that only the pages served by
  this app receive, so cross-site requests and DNS-rebinding pages cannot call
  the API (browsers will not let them read the token or send the header);
* a Host-header allowlist (DNS-rebinding defence);
* a refusal to bind to a non-loopback address without an explicit token and
  host allowlist;
* clickjacking headers on served HTML (the approve button must not be framable).

It does NOT protect against malware already running as the same user: such a
process can fetch ``/`` and read the token. Keep that limit in mind.
"""
from __future__ import annotations

import hmac
import ipaddress
import json
import os
import re
import secrets
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware

TOKEN_HEADER = "x-n1-token"
PUBLIC_API_PATHS = frozenset({"/api/health"})
LOOPBACK_NAMES = frozenset({"localhost", "127.0.0.1", "::1", "[::1]"})
BASE_DIR = Path(__file__).resolve().parent
_HEAD_RE = re.compile(r"<head[^>]*>", re.IGNORECASE)


def is_loopback(host: str | None) -> bool:
    value = str(host or "").strip().lower()
    if value in LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(value.strip("[]")).is_loopback
    except ValueError:
        return False


def resolve_token() -> str:
    """Use N1_API_TOKEN when set (stable across restarts), else a fresh random token."""
    configured = os.getenv("N1_API_TOKEN", "").strip()
    return configured or secrets.token_urlsafe(32)


def allowed_hosts() -> list[str]:
    configured = [h.strip() for h in os.getenv("ALLOWED_HOSTS", "").split(",") if h.strip()]
    return configured or ["127.0.0.1", "localhost", "[::1]"]


def assert_safe_bind(host: str | None) -> None:
    """Refuse to expose the approve/execute API beyond loopback without explicit opt-in."""
    if is_loopback(host):
        return
    if not os.getenv("N1_API_TOKEN", "").strip() or not os.getenv("ALLOWED_HOSTS", "").strip():
        raise RuntimeError(
            f"Refusing to bind to non-loopback HOST={host!r}. Set both N1_API_TOKEN and "
            "ALLOWED_HOSTS (comma-separated hostnames) to expose this service, and put it "
            "behind TLS."
        )


class TokenAuthMiddleware:
    """Pure-ASGI so Server-Sent Events streaming is not buffered or cancelled."""

    def __init__(self, app, token: str, public_paths=PUBLIC_API_PATHS) -> None:
        self.app = app
        self._token = token.encode("utf-8")
        self._public = frozenset(public_paths)

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            path = scope.get("path", "")
            if path.startswith("/api/") and path not in self._public:
                sent = dict(scope.get("headers") or []).get(TOKEN_HEADER.encode(), b"")
                if not hmac.compare_digest(sent, self._token):
                    response = JSONResponse({"detail": "Unauthorized"}, status_code=401)
                    await response(scope, receive, send)
                    return
        await self.app(scope, receive, send)


_FETCH_SHIM = (
    "<script>(()=>{const t=%s;const f=window.fetch.bind(window);"
    "window.fetch=(u,o={})=>{const req=typeof u==='string'?null:u;"
    "const url=new URL(req?req.url:u,location.href);"
    "if(url.origin!==location.origin)return f(u,o);"
    "const h=new Headers(o.headers||(req?req.headers:undefined));"
    "h.set('%s',t);return f(u,{...o,headers:h})}})();</script>"
)


def inject_token(html: str, token: str) -> str:
    # json.dumps does not escape "</script>"; neutralise HTML-significant characters.
    literal = json.dumps(token)
    for ch, esc in (
        ("<", "\\u003c"), (">", "\\u003e"), ("&", "\\u0026"),
        ("\u2028", "\\u2028"), ("\u2029", "\\u2029"),
    ):
        literal = literal.replace(ch, esc)
    shim = _FETCH_SHIM % (literal, TOKEN_HEADER)
    match = _HEAD_RE.search(html)
    if not match:
        return shim + html
    return html[:match.end()] + shim + html[match.end():]


def html_response(path: Path, token: str) -> HTMLResponse:
    html = inject_token(path.read_text(encoding="utf-8"), token)
    return HTMLResponse(html, headers={
        "Cache-Control": "no-store",
        "X-Frame-Options": "DENY",
        "Content-Security-Policy": "frame-ancestors 'none'",
        "Referrer-Policy": "no-referrer",
        "X-Content-Type-Options": "nosniff",
    })


def install(app: FastAPI, host: str | None, token: str | None = None) -> str:
    """Install host allowlist + token auth and the two token-bearing HTML routes.

    Must be called right after ``app = FastAPI(...)`` and BEFORE the /static mount so
    the explicit studio route wins over the static file handler.
    """
    assert_safe_bind(host)
    token = token or resolve_token()
    app.add_middleware(TokenAuthMiddleware, token=token)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts())  # outermost

    @app.get("/", include_in_schema=False)
    async def index():
        return html_response(BASE_DIR / "static" / "index.html", token)

    @app.get("/static/agent-execution-studio.html", include_in_schema=False)
    async def studio():
        return html_response(BASE_DIR / "static" / "agent-execution-studio.html", token)

    return token
