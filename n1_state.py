"""Bounded in-memory state shared by the Native and MCP backends.

Both stores subclass ``dict`` so existing code and tests keep working, while the
expiry / cap / cleanup logic lives in one place instead of being copy-pasted at every
call site. State is per process: run a single worker.
"""
from __future__ import annotations

import threading
import time
from typing import Any


class ApprovalStore(dict):
    """Pending write approvals keyed by action_id, with expiry and a hard cap."""

    def __init__(self, max_pending: int = 50) -> None:
        super().__init__()
        self.max_pending = max(1, int(max_pending))
        self.lock = threading.RLock()

    def purge_expired(self, now: float | None = None) -> int:
        """Drop expired actions; returns how many were removed. Call under ``lock``."""
        current = time.time() if now is None else now
        with self.lock:
            stale = [key for key, value in self.items() if current > value.get("expires_at", 0)]
            for key in stale:
                self.pop(key, None)
            return len(stale)

    def is_full(self) -> bool:
        with self.lock:
            return len(self) >= self.max_pending

    def has_session(self, session_id: str) -> bool:
        with self.lock:
            return any(item.get("session_id") == session_id for item in self.values())

    def clear_session(self, session_id: str) -> int:
        with self.lock:
            stale = [key for key, value in self.items() if value.get("session_id") == session_id]
            for key in stale:
                self.pop(key, None)
            return len(stale)


class BoundedSessionStore(dict):
    """Chat histories keyed by session_id, evicted by age and by count.

    Previously the only eviction was an explicit DELETE, so abandoned sessions grew
    without bound. Sessions are refreshed whenever they are written.
    """

    def __init__(self, max_sessions: int = 200, ttl_seconds: float = 24 * 3600) -> None:
        super().__init__()
        self.max_sessions = max(1, int(max_sessions))
        self.ttl_seconds = float(ttl_seconds)
        self._touched: dict[Any, float] = {}
        self._lock = threading.RLock()

    def __setitem__(self, key: Any, value: Any) -> None:
        with self._lock:
            super().__setitem__(key, value)
            self._touched[key] = time.monotonic()
            self._evict(protect=key)

    def __delitem__(self, key: Any) -> None:
        with self._lock:
            super().__delitem__(key)
            self._touched.pop(key, None)

    def pop(self, key: Any, *default: Any) -> Any:
        with self._lock:
            self._touched.pop(key, None)
            return super().pop(key, *default)

    def clear(self) -> None:
        with self._lock:
            super().clear()
            self._touched.clear()

    def _drop(self, key: Any) -> None:
        dict.pop(self, key, None)
        self._touched.pop(key, None)

    def _evict(self, protect: Any = None) -> None:
        now = time.monotonic()
        for key in list(self):                      # entries added via update()/patch.dict
            self._touched.setdefault(key, now)
        for key in [k for k in self._touched if k not in self]:
            del self._touched[key]
        for key in [k for k, at in self._touched.items() if k != protect and now - at > self.ttl_seconds]:
            self._drop(key)
        while len(self) > self.max_sessions:
            candidates = [k for k in self._touched if k != protect]
            if not candidates:
                break
            self._drop(min(candidates, key=self._touched.get))
