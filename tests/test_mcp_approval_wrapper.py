import asyncio
from types import SimpleNamespace

import app


def test_mcp_call_request_is_parsed_for_approval():
    assert app._mcp_write_request("ninjaone__call", {
        "endpoint": "PATCH /v2/user/end-user/{id}/device-access",
        "path_params": {"id": 144},
        "body": {"addDeviceIds": [22]},
    }) == (
        "PATCH",
        "/v2/user/end-user/144/device-access",
        {"addDeviceIds": [22]},
    )


def test_mcp_raw_call_request_is_parsed_for_approval_and_query_display():
    args = {"request": "POST /v2/user/end-users?sendInvitation=true", "body": {"firstName": "A"}}
    assert app._mcp_write_request("ninjaone__raw_call", args) == (
        "POST", "/v2/user/end-users", {"firstName": "A"}
    )
    assert app._mcp_request_query("ninjaone__raw_call", args) == {"sendInvitation": "true"}


def test_mcp_call_rejects_unresolved_or_absolute_requests():
    assert app._mcp_write_request("call", {
        "endpoint": "PATCH /v2/user/end-user/{id}/device-access",
        "path_params": {},
        "body": {},
    }) is None
    assert app._mcp_write_request("raw_call", {
        "request": "PATCH https://example.com/v2/user/1",
        "body": {},
    }) is None


def test_mcp_call_delete_method_is_identifiable_for_policy_guard():
    parsed = app._mcp_write_request("call", {"endpoint": "DELETE /v2/user/end-users/{id}"})
    assert parsed == ("DELETE", "/v2/user/end-users/{id}", None)


def test_approved_direct_mcp_call_invokes_unwrapped_tool_without_token(monkeypatch):
    calls = []

    class FakeTool:
        async def ainvoke(self, arguments):
            calls.append(arguments)
            return {"endpoint": "PATCH /v2/user/end-user/144/device-access",
                    "data": {"accessibleDeviceIds": [22]}}

    action_id = "test-approval-call"
    app._mcp_pending_actions[action_id] = {
        "action_id": action_id,
        "session_id": "test-session",
        "tool": FakeTool(),
        "arguments": {"endpoint": "PATCH /v2/user/end-user/144/device-access",
                      "path_params": {"id": 144},
                      "body": {"addDeviceIds": [22]}},
        "method": "PATCH",
        "path": "/v2/user/end-user/144/device-access",
        "body": {"addDeviceIds": [22]},
        "execution_mode": "direct_mcp_call",
        "expires_at": 9999999999,
    }
    monkeypatch.setattr(app, "parse_mcp_result", lambda value: value)

    result = asyncio.run(app.execute_mcp_pending_action(action_id, "test-session"))

    assert result["ok"] is True
    assert result["verified"] is True
    assert len(calls) == 1
    assert "approval_token" not in calls[0]
    assert action_id not in app._mcp_pending_actions
