"""Discover a running Microsoft Foundry Local endpoint and its loaded models.

The CLI is the source of truth for dynamic ports. Explicit FOUNDRY_BASE_URL is
tried first as an override, followed by the endpoint reported by the CLI and
legacy ports used by earlier versions of this project.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
from typing import Any
from urllib.parse import urlparse

import requests

_CACHE_TTL_SECONDS = 5
_LEGACY_PORTS = (52072, 49934, 64414)
_CACHE: dict[str, Any] = {"at": 0.0, "value": None}
_LOCK = threading.Lock()


def _normalize_base_url(value: str) -> str | None:
    value = (value or "").strip().rstrip(".,;)")
    if not value:
        return None
    if "://" not in value:
        value = "http://" + value
    parsed = urlparse(value)
    # This app intentionally connects only to a local Foundry service.
    if parsed.scheme not in ("http", "https") or parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
        return None
    path = parsed.path.rstrip("/")
    if path.endswith("/models"):
        path = path[:-7].rstrip("/")
    if not path:
        path = "/v1"
    elif not (path.endswith("/v1") or path.endswith("/openai")):
        path += "/v1"
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def _urls_from_status(output: str) -> list[str]:
    found: list[str] = []
    # The CLI status output includes the local service URL. Accept both the
    # older text output and JSON output without depending on its exact schema.
    for match in re.findall(r"https?://[^\s\\"'<>]+", output or "", flags=re.I):
        base = _normalize_base_url(match)
        if base and base not in found:
            found.append(base)
    return found


def _status_output() -> str:
    executable = shutil.which("foundry") or shutil.which("foundry.exe")
    if not executable:
        return ""
    for args in ([executable, "server", "status", "--output", "json"],
                 [executable, "server", "status"]):
        try:
            result = subprocess.run(
                args, capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=6, check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            output = (result.stdout or "") + "\n" + (result.stderr or "")
            if result.returncode == 0 and output.strip():
                return output
        except (OSError, subprocess.TimeoutExpired):
            continue
    return ""


def _models_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        items = payload.get("data")
        if not isinstance(items, list):
            items = payload.get("models")
        if not isinstance(items, list):
            items = payload.get("items")
    elif isinstance(payload, list):
        items = payload
    else:
        items = []

    models: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        if isinstance(item, str):
            model_id, metadata = item.strip(), {}
        elif isinstance(item, dict):
            model_id = str(item.get("id") or item.get("model") or item.get("model_id") or item.get("name") or "").strip()
            metadata = item
        else:
            continue
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        capability_text = " ".join(str(metadata.get(key, "")) for key in ("id", "name", "description", "capabilities", "task"))
        vision = bool(re.search(r"(^|[\\W_])(vl|vision|multimodal)([\\W_]|$)", capability_text, re.I))
        models.append({
            "id": model_id,
            "label": model_id,
            "model_id": model_id,
            "vision": vision,
        })
    return models


def _fetch_models(base_url: str) -> list[dict[str, Any]] | None:
    for suffix in ("/models",):
        try:
            response = requests.get(base_url.rstrip("/") + suffix, timeout=2.5)
            if response.ok:
                models = _models_from_payload(response.json())
                if models:
                    return models
        except (requests.RequestException, ValueError):
            continue
    return None


def discover_foundry(force: bool = False) -> dict[str, Any]:
    """Return endpoint, loaded models, connection state, and diagnostic source."""
    now = time.monotonic()
    with _LOCK:
        cached = _CACHE.get("value")
        if not force and cached and now - _CACHE["at"] < _CACHE_TTL_SECONDS:
            return cached

        candidates: list[tuple[str, str]] = []
        override = _normalize_base_url(os.getenv("FOUNDRY_BASE_URL", ""))
        if override:
            candidates.append((override, "FOUNDRY_BASE_URL override"))

        status = _status_output()
        for endpoint in _urls_from_status(status):
            if all(endpoint != item[0] for item in candidates):
                candidates.append((endpoint, "foundry server status"))

        for port in _LEGACY_PORTS:
            endpoint = f"http://127.0.0.1:{port}/v1"
            if all(endpoint != item[0] for item in candidates):
                candidates.append((endpoint, "legacy-port fallback"))

        last_error = "Foundry Local CLI endpoint was not found and no candidate exposed /v1/models."
        for endpoint, source in candidates:
            models = _fetch_models(endpoint)
            if models:
                result = {
                    "connected": True,
                    "base_url": endpoint,
                    "models": models,
                    "source": source,
                    "error": None,
                }
                _CACHE.update(at=time.monotonic(), value=result)
                return result

        if candidates:
            last_error = "Foundry endpoint candidates were found, but none returned loaded models from /v1/models."
        result = {
            "connected": False,
            "base_url": None,
            "models": [],
            "source": None,
            "error": last_error,
        }
        _CACHE.update(at=time.monotonic(), value=result)
        return result


def is_vision_model(model_id: str, discovery: dict[str, Any] | None = None) -> bool:
    if discovery:
        for model in discovery.get("models", []):
            if model.get("id") == model_id:
                return bool(model.get("vision"))
    return bool(re.search(r"(^|[\\W_])(vl|vision|multimodal)([\\W_]|$)", model_id or "", re.I))
