import time

import pytest
from starlette.testclient import TestClient

import n1_security
import ninja_agent


@pytest.fixture(scope="module")
def client():
    import app as app_module
    with TestClient(app_module.app, base_url="http://127.0.0.1") as c:
        c.token = app_module.API_TOKEN
        yield c


def test_api_requires_token(client):
    assert client.post("/api/actions/x/reject", json={"session_id": "s"}).status_code == 401
    assert client.get("/api/config").status_code == 401


def test_wrong_token_rejected(client):
    r = client.post("/api/actions/x/reject", json={"session_id": "s"},
                    headers={"x-n1-token": "nope"})
    assert r.status_code == 401


def test_non_ascii_token_does_not_crash(client):
    r = client.post("/api/actions/x/reject", json={"session_id": "s"},
                    headers={"x-n1-token": "é".encode("latin-1")})
    assert r.status_code == 401


def test_valid_token_reaches_handler(client):
    r = client.post("/api/actions/x/reject", json={"session_id": "s"},
                    headers={"x-n1-token": client.token})
    assert r.status_code == 200
    assert r.json()["ok"] is False  # unknown action, but auth passed


def test_health_is_public(client):
    assert client.get("/api/health").status_code != 401


def test_foreign_host_header_rejected(client):
    r = client.get("/", headers={"host": "evil.example.com"})
    assert r.status_code == 400


@pytest.mark.parametrize("path", ["/", "/static/agent-execution-studio.html"])
def test_html_carries_token_and_frame_protection(client, path):
    r = client.get(path)
    assert r.status_code == 200
    assert client.token in r.text
    assert r.headers["x-frame-options"] == "DENY"
    assert "frame-ancestors 'none'" in r.headers["content-security-policy"]
    assert r.headers["cache-control"] == "no-store"


def test_other_static_files_still_served(client):
    # The explicit studio route must not break the static mount.
    assert client.get("/static/index.html").status_code == 200


def test_token_is_not_exposed_by_api(client):
    r = client.get("/api/config", headers={"x-n1-token": client.token})
    assert client.token not in r.text


def test_inject_token_without_head_tag():
    out = n1_security.inject_token("<p>hi</p>", "tok")
    assert out.startswith("<script>") and "tok" in out and out.endswith("<p>hi</p>")


def test_inject_token_cannot_break_out_of_script_tag():
    out = n1_security.inject_token("<head></head>", 'a"</script><img src=x onerror=1>')
    assert out.count("</script>") == 1  # only the shim's own closing tag
    assert "<img" not in out


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "[::1]", "127.0.0.2"])
def test_loopback_bind_allowed(host, monkeypatch):
    monkeypatch.delenv("N1_API_TOKEN", raising=False)
    n1_security.assert_safe_bind(host)


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.5", "myhost.corp"])
def test_non_loopback_bind_refused_without_opt_in(host, monkeypatch):
    monkeypatch.delenv("N1_API_TOKEN", raising=False)
    monkeypatch.delenv("ALLOWED_HOSTS", raising=False)
    with pytest.raises(RuntimeError):
        n1_security.assert_safe_bind(host)


def test_non_loopback_bind_allowed_with_token_and_hosts(monkeypatch):
    monkeypatch.setenv("N1_API_TOKEN", "x" * 32)
    monkeypatch.setenv("ALLOWED_HOSTS", "n1.corp")
    n1_security.assert_safe_bind("0.0.0.0")


def _pending(action_id="a1", session_id="s1"):
    return {
        "action_id": action_id, "session_id": session_id, "operation_id": "op",
        "method": "PATCH", "path": "/v2/x", "query": {}, "body": {}, "headers": {},
        "media_type": "application/json", "summary": "",
        "expires_at": time.time() + 60,
    }


def test_missing_management_scope_keeps_action_pending(monkeypatch):
    monkeypatch.setattr(ninja_agent, "SCOPE", "monitoring")
    monkeypatch.setitem(ninja_agent._pending_actions, "a1", _pending())
    result = ninja_agent.execute_pending_action("a1", "s1")
    assert result["ok"] is False and "management" in result["error"]
    assert "a1" in ninja_agent._pending_actions  # not burned


def test_action_consumed_once_when_scope_present(monkeypatch):
    calls = []
    monkeypatch.setattr(ninja_agent, "SCOPE", "monitoring management")
    monkeypatch.setattr(ninja_agent, "http_request", lambda *a, **k: calls.append(1) or {"ok": 1})
    monkeypatch.setitem(ninja_agent._pending_actions, "a2", _pending("a2"))
    assert ninja_agent.execute_pending_action("a2", "s1")["ok"] is True
    assert "a2" not in ninja_agent._pending_actions
    assert ninja_agent.execute_pending_action("a2", "s1")["ok"] is False
    assert len(calls) == 1



def test_native_approval_is_routed_by_action_id_even_if_backend_switches(client, monkeypatch):
    import app as app_module

    action = _pending("native-switch", "session-native")
    monkeypatch.setitem(ninja_agent._pending_actions, "native-switch", action)
    monkeypatch.setenv("AGENT_BACKEND", "mcp")

    response = client.post(
        "/api/actions/native-switch/reject",
        json={"session_id": "session-native"},
        headers={"x-n1-token": client.token},
    )

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert "native-switch" not in ninja_agent._pending_actions


def test_mcp_approval_is_routed_by_action_id_even_if_backend_switches(client, monkeypatch):
    import app as app_module

    action = {
        "action_id": "mcp-switch",
        "session_id": "session-mcp",
        "expires_at": time.time() + 60,
    }
    monkeypatch.setitem(app_module._mcp_pending_actions, "mcp-switch", action)
    monkeypatch.setenv("AGENT_BACKEND", "native")

    response = client.post(
        "/api/actions/mcp-switch/reject",
        json={"session_id": "session-mcp"},
        headers={"x-n1-token": client.token},
    )

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert "mcp-switch" not in app_module._mcp_pending_actions


def test_unknown_approval_action_returns_structured_failure(client, monkeypatch):
    monkeypatch.setenv("AGENT_BACKEND", "native")
    response = client.post(
        "/api/actions/not-pending/approve",
        json={"session_id": "session"},
        headers={"x-n1-token": client.token},
    )
    assert response.status_code == 200
    assert response.json()["ok"] is False
    assert "not found" in response.json()["error"].lower()



def test_streamed_approval_event_includes_server_session_id(client, monkeypatch):
    import json
    import app as app_module

    async def fake_direct(session_id, message):
        action = {
            "action_id": "stream-action",
            "method": "PATCH",
            "path": "/v2/device/1",
            "body": {"organizationId": 7},
        }
        activity = [{"kind": "approval_required", "action": action}]
        return "Approval required", activity

    monkeypatch.setattr(app_module, "try_direct_backend_request", fake_direct)
    response = client.post(
        "/api/chat/stream",
        json={"message": "move device 1 to organization 7", "provider": "foundry", "model": "test"},
        headers={"x-n1-token": client.token},
    )

    assert response.status_code == 200
    events = [
        json.loads(frame.split("data: ", 1)[1])
        for frame in response.text.split("\\n\\n")
        if frame.startswith("data: ")
    ]
    approval = next(
        event for event in events
        if event.get("type") == "progress"
        and (event.get("activity_event") or {}).get("kind") == "approval_required"
    )
    assert approval["session_id"]
    assert approval["activity_event"]["action"]["action_id"] == "stream-action"
