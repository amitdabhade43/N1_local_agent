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


def test_wrapped_mcp_delete_is_blocked_before_schema_validation_or_execution(monkeypatch):
    from pydantic import BaseModel

    class CallArgs(BaseModel):
        endpoint: str
        path_params: dict | None = None
        query: dict | None = None
        body: dict | None = None

    invoked = []

    async def original_call(endpoint, path_params=None, query=None, body=None):
        invoked.append(endpoint)
        return {"unexpected": "execution"}

    tool = app.StructuredTool.from_function(
        coroutine=original_call,
        name="ninjaone__call",
        description="NinjaOne API call",
        args_schema=CallArgs,
        infer_schema=False,
    )

    def should_not_validate(*args, **kwargs):
        raise AssertionError("DELETE must be blocked before schema validation")

    monkeypatch.setattr(app, "_mcp_validate_call", should_not_validate)
    wrapped = app.wrap_mcp_write_tools([tool])

    result = asyncio.run(wrapped[0].ainvoke({
        "endpoint": "DELETE /v2/user/end-users/{id}",
    }))
    payload = __import__("json").loads(result)

    assert payload["blocked_delete"] is True
    assert payload["executed"] is False
    assert invoked == []


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



def test_mcp_call_is_checked_against_local_json_operation_and_typed_query(monkeypatch):
    core = app.ninja_native.core
    spec = {
        "paths": {
            "/v2/things/{id}": {
                "patch": {
                    "operationId": "patchThing",
                    "summary": "Update thing",
                }
            }
        }
    }
    operation = {
        "id": "patchThing",
        "method": "PATCH",
        "path": "/v2/things/{id}",
        "read_only": False,
        "params": [
            {"name": "id", "in": "path", "required": True, "schema": {"type": "integer"}},
            {"name": "enabled", "in": "query", "required": True, "schema": {"type": "boolean"}},
        ],
        "request_body": {
            "required": True,
            "media_type": "application/json",
            "schema": {"type": "object", "required": ["name"],
                       "properties": {"name": {"type": "string"}}},
        },
    }
    monkeypatch.setattr(core, "SPEC_DATA", spec)
    monkeypatch.setattr(core, "OPS", {"patchThing": operation})
    monkeypatch.setattr(core, "_argument_errors", lambda op, args: (
        ["invalid"] if not isinstance(args["query"].get("enabled"), bool) else []
    ))

    valid = app._mcp_validate_call("PATCH", "/v2/things/42", "ninjaone__call", {
        "endpoint": "PATCH /v2/things/{id}",
        "path_params": {"id": 42},
        "query": {"enabled": "true"},
        "body": {"name": "Thing"},
    })
    assert valid["ok"] is True
    assert valid["query"] == {"enabled": True}


def test_mcp_local_spec_contract_blocks_post_delete_semantics(monkeypatch):
    core = app.ninja_native.core
    monkeypatch.setattr(core, "SPEC_DATA", {
        "paths": {"/v2/things/delete": {
            "post": {"operationId": "deleteThings", "summary": "Delete things"}
        }}
    })
    monkeypatch.setattr(core, "OPS", {})

    result = app._mcp_validate_call("POST", "/v2/things/delete", "ninjaone__call", {
        "endpoint": "POST /v2/things/delete", "body": {}
    })
    assert result["ok"] is False
    assert result["blocked_delete"] is True


def test_raw_mcp_query_preserves_repeated_values():
    result = app._mcp_request_query("ninjaone__raw_call", {
        "request": "GET /v2/devices?type=WINDOWS&type=MAC"
    })
    assert result == {"type": ["WINDOWS", "MAC"]}



def test_mcp_spec_validation_rejects_unknown_endpoint_without_execution(monkeypatch):
    core = app.ninja_native.core
    monkeypatch.setattr(core, "SPEC_DATA", {"paths": {}})
    monkeypatch.setattr(core, "OPS", {})

    result = app._mcp_validate_call("POST", "/v2/unknown", "ninjaone__call", {
        "endpoint": "POST /v2/unknown",
        "body": {"any": "value"},
    })

    assert result["ok"] is False
    assert "not present" in result["error"]



def test_wrapped_mcp_write_prepares_card_without_invoking_the_underlying_tool(monkeypatch):
    from pydantic import BaseModel

    class CallArgs(BaseModel):
        endpoint: str
        path_params: dict | None = None
        query: dict | None = None
        body: dict | None = None

    invoked = []

    async def original_call(endpoint, path_params=None, query=None, body=None):
        invoked.append({"endpoint": endpoint, "path_params": path_params, "query": query, "body": body})
        return {"endpoint": endpoint, "data": {"updated": True}}

    tool = app.StructuredTool.from_function(
        coroutine=original_call,
        name="ninjaone__call",
        description="NinjaOne API call",
        args_schema=CallArgs,
        infer_schema=False,
    )
    monkeypatch.setattr(app, "_mcp_validate_call", lambda *args: {
        "ok": True,
        "operation": {"read_only": False},
    })
    wrapped = app.wrap_mcp_write_tools([tool])
    token = app._mcp_active_session.set("wrapper-approval-test")
    try:
        result = asyncio.run(wrapped[0].ainvoke({
            "endpoint": "PATCH /v2/user/end-user/{id}/device-access",
            "path_params": {"id": 144},
            "body": {"addDeviceIds": [22]},
        }))
    finally:
        app._mcp_active_session.reset(token)

    card = __import__("json").loads(result)
    assert card["approval_required"] is True
    assert card["method"] == "PATCH"
    assert card["path"] == "/v2/user/end-user/144/device-access"
    assert invoked == []
    pending = app._mcp_pending_actions[card["action_id"]]
    assert pending["execution_mode"] == "direct_mcp_call"
    app._mcp_pending_actions.pop(card["action_id"], None)
