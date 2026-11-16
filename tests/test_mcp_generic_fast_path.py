import os
from unittest.mock import AsyncMock, patch

import pytest

import app


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
    assert not app.is_simple_generic_read("Count devices with names starting with VM")
    assert not app.is_simple_generic_read("Compare users and devices")


@pytest.mark.asyncio
async def test_generic_mcp_count_uses_discovery_schema_and_get_result():
    tools = toolset()
    answer, activity = await app.run_generic_mcp_read(tools, "How many devices?")

    assert answer == "2 records returned by NinjaOne for that request."
    assert [item["tool"] for item in activity if item["kind"] == "call"] == [
        "discover_endpoints", "inspect_endpoint", "execute_endpoint"
    ]
    tools[0].ainvoke.assert_awaited_once_with({
        "intent": "How many devices?", "candidate_limit": 8,
    })
    tools[1].ainvoke.assert_awaited_once_with({"endpoint": "GET /v2/devices"})
    tools[2].ainvoke.assert_awaited_once_with({"endpoint": "GET /v2/devices"})


@pytest.mark.asyncio
async def test_generic_mcp_preflight_runs_before_provider_resolution():
    tools = toolset()
    with patch.dict(os.environ, {"AGENT_BACKEND": "mcp"}):
        with patch.object(app, "get_tools", new=AsyncMock(return_value=tools)):
            with patch.object(app, "detect_direct_tool_request", return_value=None):
                result = await app.try_direct_backend_request("session-1", "How many devices?")

    assert result[0] == "2 records returned by NinjaOne for that request."


@pytest.mark.asyncio
async def test_generic_mcp_count_discloses_incomplete_result():
    tools = toolset({
        "endpoint": "GET /v2/devices", "executed": True, "complete": False,
        "data": [{"id": 1, "systemName": "VM-01"}],
    })
    answer, _ = await app.run_generic_mcp_read(tools, "How many devices?")

    assert "At least 1 records" in answer
    assert "complete total is unknown" in answer


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

    assert answer == "3 records returned by NinjaOne for that request."
    tools[3].ainvoke.assert_awaited_once_with({
        "endpoint": "GET /v2/devices", "page_parameter": "page",
        "page_size_parameter": "pageSize", "page_size": 200,
        "start_page": 1, "max_pages": 20,
    })
