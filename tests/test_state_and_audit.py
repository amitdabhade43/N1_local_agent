import asyncio
import json
import time
from unittest.mock import patch

import pytest
import requests

import n1_audit
import n1_state
import ninja_agent as core


@pytest.fixture
def audit_file(tmp_path):
    path = tmp_path / "audit.log"
    n1_audit.configure(path)
    yield path
    n1_audit.configure("off")


def _events(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _action(action_id="a1", session="s1", **extra):
    base = {"action_id": action_id, "session_id": session, "operation_id": "op", "method": "PATCH",
            "path": "/v2/device/{id}", "query": {}, "body": {"secret": "BODY"}, "headers": {},
            "media_type": "application/json", "summary": "", "expires_at": time.time() + 60}
    base.update(extra)
    return base


# ------------------------------------------------------------------ audit module
def test_audit_writes_only_allowlisted_fields_and_hashes_session(audit_file):
    n1_audit.audit("action_prepared", "my-session-id", action_id="a", method="PATCH", path="/v2/x",
                   body={"password": "hunter2"}, token="TOKEN", query={"q": 1}, headers={"x": "y"})
    (event,) = _events(audit_file)
    assert event["event"] == "action_prepared" and event["method"] == "PATCH"
    assert set(event) <= {"ts", "event", "session", *n1_audit.ALLOWED_FIELDS}
    assert "body" not in event and "token" not in event and "query" not in event
    assert event["session"] != "my-session-id" and len(event["session"]) == 12
    assert "hunter2" not in audit_file.read_text() and "TOKEN" not in audit_file.read_text()


def test_audit_truncates_long_text(audit_file):
    n1_audit.audit("x", reason="r" * 5000)
    assert len(_events(audit_file)[0]["reason"]) == 300


def test_audit_never_raises_on_unwritable_path(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    n1_audit.configure(blocker / "sub" / "audit.log")  # parent is a file -> cannot open
    try:
        n1_audit.audit("action_prepared", action_id="a")  # must not raise
    finally:
        n1_audit.configure("off")


def test_audit_can_be_disabled(tmp_path):
    n1_audit.configure("off")
    n1_audit.audit("action_prepared", action_id="a")
    assert not list(tmp_path.iterdir())


# ------------------------------------------------------------------ native approval path
def test_native_execute_and_reject_are_audited(audit_file, monkeypatch):
    monkeypatch.setattr(core, "SCOPE", "monitoring management")
    monkeypatch.setattr(core, "http_request", lambda *a, **k: {"done": True})
    core._pending_actions["a1"] = _action("a1")
    core._pending_actions["a2"] = _action("a2")
    assert core.execute_pending_action("a1", "s1")["ok"] is True
    assert core.reject_pending_action("a2", "s1") is True
    assert core.execute_pending_action("a1", "s1")["ok"] is False  # already used
    assert [e["event"] for e in _events(audit_file)] == [
        "action_approved", "action_executed", "action_rejected", "action_not_found"]
    assert "BODY" not in audit_file.read_text()


def test_scope_block_is_audited_and_keeps_action(audit_file, monkeypatch):
    monkeypatch.setattr(core, "SCOPE", "monitoring")
    core._pending_actions["a3"] = _action("a3")
    assert core.execute_pending_action("a3", "s1")["ok"] is False
    assert "a3" in core._pending_actions
    assert _events(audit_file)[0]["event"] == "action_blocked"
    core._pending_actions.pop("a3", None)


def test_wrong_session_is_audited(audit_file):
    core._pending_actions["a4"] = _action("a4", session="owner")
    assert core.execute_pending_action("a4", "intruder")["ok"] is False
    assert _events(audit_file)[0]["event"] == "action_session_mismatch"
    core._pending_actions.pop("a4", None)


def test_unexpected_error_is_not_leaked_to_the_client(audit_file, monkeypatch):
    monkeypatch.setattr(core, "SCOPE", "management")

    def boom(*a, **k):
        raise RuntimeError("secret internal path C:/Users/x/.env")

    monkeypatch.setattr(core, "http_request", boom)
    core._pending_actions["a5"] = _action("a5")
    result = core.execute_pending_action("a5", "s1")
    assert result["ok"] is False
    assert "secret internal path" not in json.dumps(result)
    failed = [e for e in _events(audit_file) if e["event"] == "action_failed"][0]
    assert failed["error_class"] == "RuntimeError"


def test_http_error_and_validation_messages_are_still_shown(monkeypatch):
    monkeypatch.setattr(core, "SCOPE", "management")
    response = requests.Response()
    response.status_code = 403
    response._content = b"Forbidden by policy"

    def forbidden(*a, **k):
        raise requests.HTTPError(response=response)

    monkeypatch.setattr(core, "http_request", forbidden)
    core._pending_actions["a6"] = _action("a6")
    assert core.execute_pending_action("a6", "s1")["error"] == "NinjaOne HTTP 403: Forbidden by policy"

    def blocked(*a, **k):
        raise ValueError("DELETE requests are blocked.")

    monkeypatch.setattr(core, "http_request", blocked)
    core._pending_actions["a7"] = _action("a7")
    assert core.execute_pending_action("a7", "s1")["error"] == "DELETE requests are blocked."


def test_batch_is_audited_per_device_with_summary(audit_file, monkeypatch):
    monkeypatch.setattr(core, "SCOPE", "management")
    response = requests.Response()
    response.status_code = 404
    response._content = b"no such device"

    def flaky(method, path, *a, **k):
        if path.endswith("/2"):
            raise requests.HTTPError(response=response)
        return {"ok": 1}

    monkeypatch.setattr(core, "http_request", flaky)
    items = [{"device_name": f"PC-{i}", "path_params": {"id": i}, "query": {}, "headers": {},
              "body": {"organizationId": 7}, "media_type": "application/json"} for i in (1, 2, 3)]
    core._pending_actions["b1"] = _action("b1", batch_requests=items)
    result = core.execute_pending_action("b1", "s1")
    assert (result["total"], result["succeeded"], result["failed"], result["partial"]) == (3, 2, 1, True)
    assert result["results"][1]["error"] == "NinjaOne HTTP 404: no such device"
    events = _events(audit_file)
    assert [e["event"] for e in events] == [
        "action_approved", "batch_item_executed", "batch_item_failed", "batch_item_executed", "batch_executed"]
    assert events[2]["device_id"] == 2 and events[2]["status_code"] == 404
    assert (events[-1]["item_count"], events[-1]["succeeded"], events[-1]["failed"]) == (3, 2, 1)
    assert events[0]["item_count"] == 3


# ------------------------------------------------------------------ MCP approval path
class _ExplodingTool:
    name = "call"

    async def ainvoke(self, arguments):
        raise RuntimeError("gateway password=hunter2")


def test_mcp_unexpected_error_is_generic_and_audited(audit_file):
    import app as app_module
    app_module._mcp_pending_actions["m1"] = {
        "action_id": "m1", "session_id": "s1", "tool": _ExplodingTool(), "arguments": {},
        "method": "PATCH", "path": "/v2/device/1", "body": {"x": 1},
        "execution_mode": "direct_mcp_call", "expires_at": time.time() + 60}
    result = asyncio.run(app_module.execute_mcp_pending_action("m1", "s1"))
    assert result["ok"] is False and "hunter2" not in json.dumps(result)
    events = _events(audit_file)
    assert [e["event"] for e in events] == ["action_approved", "action_failed"]
    assert events[1]["backend"] == "mcp" and events[1]["error_class"] == "RuntimeError"


def test_mcp_reject_is_audited(audit_file):
    import app as app_module
    app_module._mcp_pending_actions["m2"] = {
        "action_id": "m2", "session_id": "s1", "method": "POST", "path": "/v2/x",
        "expires_at": time.time() + 60}
    assert app_module.reject_mcp_pending_action("m2", "s1") is True
    assert _events(audit_file)[0]["event"] == "action_rejected"


# ------------------------------------------------------------------ ApprovalStore
def test_approval_store_expiry_cap_and_session_helpers():
    store = n1_state.ApprovalStore(max_pending=2)
    store["old"] = {"session_id": "a", "expires_at": time.time() - 1}
    store["live"] = {"session_id": "a", "expires_at": time.time() + 60}
    assert store.purge_expired() == 1 and "old" not in store and "live" in store
    assert not store.is_full()
    store["other"] = {"session_id": "b", "expires_at": time.time() + 60}
    assert store.is_full()
    assert store.has_session("a") and not store.has_session("zzz")
    assert store.clear_session("a") == 1 and list(store) == ["other"]


def test_approval_store_is_patch_dict_compatible():
    store = n1_state.ApprovalStore()
    store["keep"] = {"session_id": "s", "expires_at": time.time() + 60}
    with patch.dict(store, {}, clear=True):
        assert len(store) == 0
    assert "keep" in store
    assert store.lock is not None


# ------------------------------------------------------------------ BoundedSessionStore
def test_session_store_caps_by_count_keeping_newest():
    store = n1_state.BoundedSessionStore(max_sessions=3)
    for index in range(5):
        store[f"s{index}"] = [index]
        time.sleep(0.002)
    assert set(store) == {"s2", "s3", "s4"}


def test_session_store_rewrite_refreshes_age():
    store = n1_state.BoundedSessionStore(max_sessions=2)
    store["a"] = [1]
    time.sleep(0.002)
    store["b"] = [2]
    time.sleep(0.002)
    store["a"] = [1, 1]          # a is now newer than b
    time.sleep(0.002)
    store["c"] = [3]
    assert set(store) == {"a", "c"}


def test_session_store_ttl_evicts_stale_sessions():
    store = n1_state.BoundedSessionStore(max_sessions=10, ttl_seconds=0.05)
    store["stale"] = [1]
    time.sleep(0.12)
    store["fresh"] = [2]
    assert set(store) == {"fresh"}


def test_session_store_behaves_like_a_dict_for_existing_callers():
    store = n1_state.BoundedSessionStore()
    store["x"] = [1]
    assert store.get("x") == [1] and store.get("nope", []) == []
    assert store.pop("x") == [1] and store.pop("x", None) is None
    store["y"] = [2]
    del store["y"]
    assert len(store) == 0
    with patch.dict(store, {"p": [9]}):
        assert store["p"] == [9]
        store["q"] = [1]            # eviction bookkeeping must tolerate patch.dict's update()
    assert "p" not in store
