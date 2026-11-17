import os
from unittest.mock import AsyncMock, patch

import pytest

import app
import ninja_native


class FakeTool:
    def __init__(self, name, result):
        self.name = name
        self.ainvoke = AsyncMock(return_value=result)


def toolset(execute_result=None):
    endpoint = "GET /v2/devices"
    return [
        FakeTool("discover_endpoints", {
            "semantic": {"mode": "fuzzy_fast_path", "selected": [{"endpoint": endpoint}]},
            "candidates": [{"endpoint": endpoint}],
        }),
        FakeTool("inspect_endpoint", {
            "method": "GET", "endpoint": endpoint, "parameters": [], "responses": {},
        }),
        FakeTool("execute_endpoint", execute_result or {
            "endpoint": endpoint, "executed": True, "complete": True,
            "data": [{"id": 1, "systemName": "VM-01"}, {"id": 2, "systemName": "VM-02"}],
            "elapsed_ms": 12.5,
        }),
    ]


def test_generic_read_gate_allows_simple_count_and_blocks_scoped_requests():
    assert app.is_simple_generic_read("How many devices are there?")
    assert app.is_simple_generic_read("How many users in NinjaOne?")
    assert not app.is_simple_generic_read("List devices assigned to Rene")
    assert not app.is_simple_generic_read("List devices from organization 6")
    assert not app.is_simple_generic_read("How many users for organization 6?")
    assert not app.is_simple_generic_read("Count devices with names starting with VM")
    assert not app.is_simple_generic_read("Compare users and devices")


@pytest.mark.asyncio
async def test_generic_mcp_count_uses_discovery_schema_and_get_result():
    tools = toolset()
    answer, activity = await app.run_generic_mcp_read(tools, "How many devices?")

    assert answer == "NinjaOne returned 2 records for devices."
    assert [item["tool"] for item in activity if item["kind"] == "call"] == [
        "discover_endpoints", "inspect_endpoint", "execute_endpoint"
    ]
    tools[0].ainvoke.assert_awaited_once_with({
        "intent": "How many devices?", "candidate_limit": 8,
    })
    tools[1].ainvoke.assert_awaited_once_with({"endpoint": "GET /v2/devices"})
    tools[2].ainvoke.assert_awaited_once_with({"endpoint": "GET /v2/devices"})


@pytest.mark.asyncio
async def test_generic_mcp_supports_namespaced_tool_names():
    tools = toolset()
    for tool in tools:
        tool.name = "ninjaone__" + tool.name
    answer, _ = await app.run_generic_mcp_read(tools, "How many devices?")

    assert answer == "NinjaOne returned 2 records for devices."


@pytest.mark.asyncio
async def test_generic_mcp_preflight_runs_before_provider_resolution():
    tools = toolset()

    class FakeSessionContext:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, exc_type, exc, traceback):
            return False

    class FakeClient:
        def session(self, server_name):
            assert server_name == "ninjaone"
            return FakeSessionContext()

    with patch.dict(os.environ, {"AGENT_BACKEND": "mcp"}):
        with patch.object(app, "get_tools", new=AsyncMock(return_value=tools)):
            with patch.object(app, "_mcp_connection", return_value=("stdio:test", FakeClient())):
                with patch.object(app, "load_mcp_tools", new=AsyncMock(return_value=tools)) as load_tools:
                    with patch.object(app, "detect_direct_tool_request", return_value=None):
                        result = await app.try_direct_backend_request("session-1", "How many devices?")

    assert result[0] == "NinjaOne returned 2 records for devices."
    load_tools.assert_awaited_once()
    assert [item["tool"] for item in result[1] if item["kind"] == "call"] == [
        "discover_endpoints", "inspect_endpoint", "execute_endpoint"
    ]


@pytest.mark.asyncio
async def test_generic_mcp_count_discloses_incomplete_result():
    tools = toolset({
        "endpoint": "GET /v2/devices", "executed": True, "complete": False,
        "data": [{"id": 1, "systemName": "VM-01"}],
    })
    answer, _ = await app.run_generic_mcp_read(tools, "How many devices?")

    assert "at least 1 record for devices" in answer
    assert "complete total is unknown" in answer


@pytest.mark.asyncio
async def test_generic_mcp_does_not_call_nonempty_result_empty_when_preview_missing():
    tools = toolset({
        "endpoint": "GET /v2/devices", "executed": True, "complete": False,
        "data": [], "items_fetched": 17,
    })
    answer, _ = await app.run_generic_mcp_read(tools, "List devices")

    assert "returned 17 records for devices" in answer
    assert "preview was unavailable" in answer


@pytest.mark.asyncio
async def test_generic_mcp_skips_cursor_pagination_instead_of_claiming_partial_total():
    tools = toolset()
    tools[1] = FakeTool("inspect_endpoint", {
        "method": "GET",
        "parameters": [{"name": "after", "in": "query", "required": False}],
        "responses": {
            "200": {
                "content": {
                    "application/json": {
                        "schema": {
                            "properties": {
                                "pageInfo": {
                                    "properties": {
                                        "nextCursor": {"type": "string"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
    })
    result = await app.run_generic_mcp_read(tools, "How many devices?")

    assert result is None
    tools[2].ainvoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_generic_mcp_skips_endpoints_with_required_parameters():
    tools = toolset()
    tools[1] = FakeTool("inspect_endpoint", {
        "method": "GET", "parameters": [{"name": "id", "in": "path", "required": True}],
        "responses": {},
    })
    result = await app.run_generic_mcp_read(tools, "How many devices?")

    assert result is None
    tools[2].ainvoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_generic_mcp_uses_declared_page_parameters():
    tools = toolset()
    tools.append(FakeTool("execute_paginated_endpoint", {
        "endpoint": "GET /v2/devices", "complete": True,
        "data": [{"id": 1}, {"id": 2}, {"id": 3}],
        "pages_fetched": 1, "items_fetched": 3, "elapsed_ms": 20,
    }))
    tools[1] = FakeTool("inspect_endpoint", {
        "method": "GET", "parameters": [
            {"name": "page", "in": "query", "required": False},
            {"name": "pageSize", "in": "query", "required": False},
        ], "responses": {},
    })
    answer, activity = await app.run_generic_mcp_read(tools, "How many devices?")

    assert answer == "NinjaOne returned 3 records for devices."
    tools[3].ainvoke.assert_awaited_once_with({
        "endpoint": "GET /v2/devices", "page_parameter": "page",
        "page_size_parameter": "pageSize", "page_size": 200,
        "start_page": 1, "max_pages": 20,
    })



def test_affirmative_reply_recovers_previous_mcp_mutation_only_for_preparation():
    class PreviousUserMessage:
        type = "human"
        content = "Add device 2QN9BY3 to PC Admin Rene Piuk device access list"

    with patch.dict(app._sessions, {"approval-recovery-test": [PreviousUserMessage()]}):
        with patch.dict(app._mcp_pending_actions, {}, clear=True):
            recovered, pending = app._recover_unprepared_mcp_mutation("approval-recovery-test", "yes")
    assert recovered == "Add device 2QN9BY3 to PC Admin Rene Piuk device access list"
    assert pending is False


def test_affirmative_reply_never_replaces_existing_mcp_approval_button():
    class PreviousUserMessage:
        type = "human"
        content = "Add device 2QN9BY3 to PC Admin Rene Piuk device access list"

    with patch.dict(app._sessions, {"approval-pending-test": [PreviousUserMessage()]}):
        with patch.dict(app._mcp_pending_actions, {
            "pending-action": {"session_id": "approval-pending-test"}
        }, clear=True):
            recovered, pending = app._recover_unprepared_mcp_mutation("approval-pending-test", "approved")
    assert recovered is None
    assert pending is True


def test_affirmative_reply_without_previous_mutation_does_not_invent_write():
    with patch.dict(app._sessions, {"approval-empty-test": []}):
        with patch.dict(app._mcp_pending_actions, {}, clear=True):
            recovered, pending = app._recover_unprepared_mcp_mutation("approval-empty-test", "yes")
    assert recovered is None
    assert pending is False



def test_native_affirmative_reply_recovers_previous_unprepared_mutation():
    history = [
        {"role": "user", "content": "Add device 2QN9BY3 to PC Admin Rene Piuk device access list"},
        {"role": "assistant", "content": "Please confirm to proceed."},
    ]
    assert ninja_native._recover_unprepared_mutation("yes", history) == history[0]["content"]


def test_native_affirmative_reply_does_not_recover_non_mutation():
    history = [{"role": "user", "content": "How many devices are there?"}]
    assert ninja_native._recover_unprepared_mutation("yes", history) is None


def test_native_affirmative_reply_does_not_match_arbitrary_text():
    history = [{"role": "user", "content": "Add device 2QN9BY3 to PC Admin Rene Piuk device access list"}]
    assert ninja_native._recover_unprepared_mutation("yes, please do this", history) is None
