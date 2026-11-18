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
            "selected_details": {
                "method": "GET", "endpoint": endpoint, "parameters": [], "responses": {},
            },
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
        "discover_endpoints", "execute_endpoint"
    ]
    tools[0].ainvoke.assert_awaited_once_with({
        "intent": "How many devices?", "candidate_limit": 8,
    })
    tools[1].ainvoke.assert_not_awaited()
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
        with patch.object(app, "get_tools", new=AsyncMock(return_value=tools)) as get_tools:
            with patch.object(app, "_mcp_connection", return_value=("stdio:test", FakeClient())):
                with patch.object(app, "load_mcp_tools", new=AsyncMock(return_value=tools)) as load_tools:
                    with patch.object(app, "detect_direct_tool_request", return_value=None):
                        result = await app.try_direct_backend_request("session-1", "How many devices?")

    assert result[0] == "NinjaOne returned 2 records for devices."
    get_tools.assert_not_awaited()
    load_tools.assert_awaited_once()
    assert [item["tool"] for item in result[1] if item["kind"] == "call"] == [
        "discover_endpoints", "execute_endpoint"
    ]


@pytest.mark.asyncio
async def test_complex_mcp_request_skips_throwaway_preflight_session():
    with patch.dict(os.environ, {"AGENT_BACKEND": "mcp"}):
        with patch.object(app, "_mcp_connection", side_effect=AssertionError("preflight must be skipped")):
            result = await app.try_direct_backend_request(
                "session-complex", "List devices assigned to Rene"
            )

    assert result is None


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
    tools[0].ainvoke.return_value["selected_details"] = {
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
    }
    result = await app.run_generic_mcp_read(tools, "How many devices?")

    assert result is None
    tools[2].ainvoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_generic_mcp_uses_openapi_after_id_cursor_pagination():
    tools = toolset()
    tools[0].ainvoke.return_value["selected_details"] = {
        "method": "GET",
        "parameters": [
            {"name": "pageSize", "in": "query", "required": False, "schema": {"type": "integer"}},
            {"name": "after", "in": "query", "required": False,
             "description": "Last Node ID from previous page", "schema": {"type": "integer"}},
        ],
        "responses": {
            "default": {"content": {"application/json": {"schema": {
                "type": "array", "items": {"type": "object", "properties": {
                    "id": {"type": "integer"}, "systemName": {"type": "string"}
                }}
            }}}}
        },
    }
    tools.append(FakeTool("execute_paginated_endpoint", {
        "endpoint": "GET /v2/devices", "complete": True,
        "data": [{"id": 1}, {"id": 2}], "pages_fetched": 1,
        "items_fetched": 2, "elapsed_ms": 18,
    }))

    answer, _ = await app.run_generic_mcp_read(tools, "How many devices?")

    assert answer == "NinjaOne returned 2 records for devices."
    tools[3].ainvoke.assert_awaited_once_with({
        "endpoint": "GET /v2/devices", "cursor_parameter": "after",
        "cursor_field": "id", "page_size_parameter": "pageSize",
        "page_size": 1000, "max_pages": 200,
    })
    tools[2].ainvoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_generic_mcp_skips_endpoints_with_required_parameters():
    tools = toolset()
    tools[0].ainvoke.return_value["selected_details"] = {
        "method": "GET", "parameters": [{"name": "id", "in": "path", "required": True}],
        "responses": {},
    }
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
    tools[0].ainvoke.return_value["selected_details"] = {
        "method": "GET", "parameters": [
            {"name": "page", "in": "query", "required": False},
            {"name": "pageSize", "in": "query", "required": False},
        ], "responses": {},
    }
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


@pytest.mark.parametrize("question", [
    'As per record, how many technicians have the "IT Support Technician" role?',
    'As per record its 78 technicians have the "IT Support Technician" role',
])
def test_technician_role_count_is_a_deterministic_generic_read(question):
    assert app._extract_technician_role_count_request(question) == "IT Support Technician"
    assert app.is_simple_generic_read(question)


@pytest.mark.asyncio
async def test_generic_mcp_counts_exact_technician_role_only_from_complete_result():
    endpoint = "GET /v2/user/technicians"
    records = [
        {"id": 1, "roles": ["IT Support Technician"]},
        {"id": 2, "roles": ["IT Support Technician", "Administrator"]},
        {"id": 3, "roles": ["Administrator"]},
        {"id": 4, "roles": ["it support technician"]},
    ]
    tools = [
        FakeTool("discover_endpoints", {
            "semantic": {"mode": "fuzzy_fallback", "selected": [{"endpoint": endpoint}]},
            "selected_details": {
                "method": "GET", "endpoint": endpoint, "parameters": [], "responses": {},
            },
        }),
        FakeTool("execute_endpoint", {
            "endpoint": endpoint, "executed": True, "complete": True,
            "truncated": False, "data": records, "items_fetched": 4, "elapsed_ms": 20,
        }),
    ]

    answer, activity = await app.run_generic_mcp_read(
        tools, 'As per record its 78 technicians have the "IT Support Technician" role'
    )

    assert answer == (
        '3 technicians have the role "IT Support Technician" according to the '
        'complete NinjaOne result (4 technicians checked).'
    )
    tools[0].ainvoke.assert_awaited_once_with({
        "intent": "List all technicians", "candidate_limit": 8,
    })
    tools[1].ainvoke.assert_awaited_once_with({"endpoint": endpoint})


@pytest.mark.asyncio
async def test_generic_mcp_role_count_does_not_claim_exact_total_for_incomplete_result():
    endpoint = "GET /v2/user/technicians"
    tools = [
        FakeTool("discover_endpoints", {
            "semantic": {"mode": "fuzzy_fast_path", "selected": [{"endpoint": endpoint}]},
            "selected_details": {
                "method": "GET", "endpoint": endpoint, "parameters": [], "responses": {},
            },
        }),
        FakeTool("execute_endpoint", {
            "endpoint": endpoint, "executed": True, "complete": False,
            "truncated": True, "data": [{"id": 1, "roles": ["IT Support Technician"]}],
            "items_fetched": 1, "elapsed_ms": 20,
        }),
    ]

    answer, _ = await app.run_generic_mcp_read(
        tools, 'How many technicians have the "IT Support Technician" role?'
    )

    assert answer.startswith("At least 1 technicians")
    assert "cannot confirm the complete total" in answer


def test_people_name_search_prompt_requires_all_categories_and_pagination():
    prompt = app.SYSTEM_PROMPT
    assert "getAllUserAndContacts" in prompt
    assert "TECHNICIAN, END_USER, and CONTACT" in prompt
    assert "searchCriteria" in prompt
    assert "Paginate the unified people endpoint completely" in prompt
    assert "stable record identity" in prompt


def test_activity_metric_is_not_mistaken_for_mcp_tool_execution():
    assert not app._has_mcp_tool_execution([
        {"kind": "metric", "stage": "llm_agent", "elapsed_ms": 1200},
    ])
    assert app._has_mcp_tool_execution([
        {"kind": "result", "tool": "execute_endpoint", "result": "[]"},
    ])
    assert app._has_mcp_tool_execution([
        {"kind": "call", "tool": "discover_endpoints", "arguments": {}},
    ])


def test_mcp_discovery_is_not_mistaken_for_a_successful_data_read():
    activity = [
        {"kind": "call", "tool": "discover_endpoints", "arguments": {"intent": "users"}},
        {"kind": "result", "tool": "discover_endpoints", "result": "{'selected': []}"},
        {"kind": "result", "tool": "inspect_endpoint", "result": "{'method': 'GET'}"},
    ]
    assert app._has_mcp_tool_execution(activity)
    assert not app._has_mcp_data_read_result(activity)


@pytest.mark.parametrize("tool_name", [
    "execute_endpoint", "execute_paginated_endpoint", "call", "raw_call",
    "ninjaone__execute_endpoint",
])
def test_successful_mcp_data_read_result_is_accepted(tool_name):
    assert app._has_mcp_data_read_result([
        {"kind": "result", "tool": tool_name,
         "result": '{"executed": true, "complete": true, "data": [{"id": 1}]}'},
    ])


@pytest.mark.parametrize("payload", [
    '{"error": "upstream unavailable"}',
    '{"executed": false, "error": "request failed"}',
    'Error: request timed out',
    '',
    None,
])
def test_failed_or_empty_mcp_read_result_is_not_accepted(payload):
    assert not app._has_mcp_data_read_result([
        {"kind": "result", "tool": "execute_endpoint", "result": payload},
    ])


def test_mcp_read_result_is_not_confused_with_discovery_or_metrics():
    assert not app._has_mcp_data_read_result([
        {"kind": "result", "tool": "discover_endpoints", "result": '{"selected": []}'},
        {"kind": "metric", "tool": "execute_endpoint", "result": '{"data": [{"id": 1}]}'},
    ])


def test_exact_count_requires_complete_or_authoritative_total_evidence():
    assert app._has_complete_mcp_count_evidence([
        {"kind": "result", "tool": "execute_paginated_endpoint",
         "result": '{"complete": true, "items_fetched": 250}'},
    ])
    assert app._has_complete_mcp_count_evidence([
        {"kind": "result", "tool": "call",
         "result": '{"data": [], "total": 42}'},
    ])
    assert not app._has_complete_mcp_count_evidence([
        {"kind": "result", "tool": "execute_endpoint",
         "result": '{"complete": false, "items_fetched": 50, "data": []}'},
    ])
    assert not app._has_complete_mcp_count_evidence([
        {"kind": "result", "tool": "discover_endpoints",
         "result": '{"complete": true, "total": 42}'},
    ])


def test_exact_count_intent_matches_natural_language_count_questions():
    import re

    pattern = re.compile(r"\\b(?:how many|count|number of|total)\\b", re.I)
    assert pattern.search("How many users contain Rene?")
    assert pattern.search("Count all technicians")
    assert not pattern.search("List all technicians")
