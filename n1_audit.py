"""Append-only audit trail for write approvals.

One JSON object per line, written to a rotating file (default ``logs/n1-audit.log``).
Set ``N1_AUDIT_LOG`` to another path, or to ``off`` to disable.

Only an allowlist of non-sensitive fields is ever written. Request bodies, query
strings, headers, credentials and tokens can not be logged by construction, and the
chat session ID is stored only as a short hash. Audit logging never raises: a broken
log file must not block (or silently unblock) an approval.
"""
from __future__ import annotations

import hashlib
import json
import logging
import logging.handlers
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parent
_DEFAULT_PATH = _BASE_DIR / "logs" / "n1-audit.log"
_DISABLED = {"off", "none", "disabled", "false", "0"}
ALLOWED_FIELDS = frozenset({
    "action_id", "operation_id", "method", "path", "backend", "ok", "status_code",
    "reason", "error_class", "item_count", "skipped_count", "succeeded", "failed", "device_id",
})
_MAX_TEXT = 300

_logger = logging.getLogger("n1.audit")
_logger.setLevel(logging.INFO)
_logger.propagate = False
_lock = threading.Lock()
_configured = False


def configure(path: str | os.PathLike | None = None) -> None:
    """(Re)configure the audit sink. ``None`` reads ``N1_AUDIT_LOG`` / the default path."""
    global _configured
    with _lock:
        for handler in list(_logger.handlers):
            _logger.removeHandler(handler)
            handler.close()
        target = str(path) if path is not None else os.getenv("N1_AUDIT_LOG", "").strip()
        if target.lower() in _DISABLED:
            _logger.addHandler(logging.NullHandler())
        else:
            try:
                file_path = Path(target) if target else _DEFAULT_PATH
                file_path.parent.mkdir(parents=True, exist_ok=True)
                handler = logging.handlers.RotatingFileHandler(
                    file_path, maxBytes=5_000_000, backupCount=5, encoding="utf-8"
                )
                handler.setFormatter(logging.Formatter("%(message)s"))
                _logger.addHandler(handler)
            except OSError as exc:
                logging.getLogger("n1").warning("Audit log disabled: cannot open %s (%s)", target or _DEFAULT_PATH, exc)
                _logger.addHandler(logging.NullHandler())
        _configured = True


def session_hash(session_id: object) -> str:
    return hashlib.sha256(str(session_id).encode("utf-8")).hexdigest()[:12]


def audit(event: str, session_id: object = None, **fields) -> None:
    """Record one audit event. Unknown fields are dropped; never raises."""
    try:
        if not _configured:
            configure()
        record = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), "event": event}
        if session_id is not None:
            record["session"] = session_hash(session_id)
        for key, value in fields.items():
            if key in ALLOWED_FIELDS and value is not None:
                record[key] = value[:_MAX_TEXT] if isinstance(value, str) else value
        _logger.info(json.dumps(record, default=str, separators=(",", ":")))
    except Exception:  # pragma: no cover - auditing must never break the request path
        logging.getLogger("n1").exception("Audit logging failed")
