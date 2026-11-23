import asyncio
import json

import app


def _tool_map(tools):
    return {tool.name: tool for tool in tools}


def test_mcp_schema_tools_delegate_to_shared_json_yaml_catalog(monkeypatch):
    core = app.ninja_native.core
    monkeypatch.setattr(app, "_ensure_mcp_schema_catalog", lambda: core)

    search_calls = []
    inspect_calls = []

    def search_fields(arguments):
        search_calls.append(arguments)
        return json.dumps({"matches": [{"name": "accessibleDeviceIds"}]})

    def inspect_schema(arguments):
        inspect_calls.append(arguments)
        return json.dumps({"schema_name": "EndUser", "schema_source": "yaml"})

    monkeypatch.setattr(core, "t_search_fields", search_fields)
    monkeypatch.setattr(core, "t_inspect_schema", inspect_schema)
    monkeypatch.setattr(core, "SPEC_SOURCE_REPORT", {
        "primary": "ninja_spec.json",
        "supplemental": "ninjaOne-API-core-resources.yaml",
        "comparison": {"schema_definition_conflicts": ["EndUser"]},
    })
    monkeypatch.setattr(core, "SCHEMA_INDEX", {
        ("json", "EndUser"): {},
        ("yaml", "EndUser"): {},
    })

    tools = _tool_map(app.build_mcp_schema_tools())

    assert {"local_search_fields", "local_inspect_schema", "local_schema_catalog_status"} <= set(tools)

    search_result = json.loads(tools["local_search_fields"].invoke({
        "query": "accessible device ids",
        "schema_source": "yaml",
        "limit": 12,
    }))
    assert search_result["matches"][0]["name"] == "accessibleDeviceIds"
    assert search_calls == [{
        "query": "accessible device ids",
        "schema_name": "",
        "operation_id": "",
        "schema_source": "yaml",
        "limit": 12,
    }]

    inspect_result = json.loads(tools["local_inspect_schema"].invoke({
        "schema_name": "EndUser",
        "schema_source": "yaml",
        "offset": 20,
        "limit": 40,
    }))
    assert inspect_result["schema_source"] == "yaml"
    assert inspect_calls == [{
        "schema_name": "EndUser",
        "schema_source": "yaml",
        "offset": 20,
        "limit": 40,
    }]

    status = json.loads(tools["local_schema_catalog_status"].invoke({}))
    assert status["yaml_loaded"] is True
    assert status["schema_sources"] == ["json", "yaml"]
    assert status["execution_authority"] == "ninja_spec.json"


def test_mcp_schema_catalog_initializes_without_ninjaone_oauth_credentials(monkeypatch):
    core = app.ninja_native.core
    loaded = []

    def fake_load_ops():
        loaded.append(True)
        core.SPEC_DATA = {"openapi": "3.0.0"}
        core.FIELD_INDEX = [{"name": "accessibleDeviceIds", "schema_source": "yaml"}]
        core.FIELD_SEARCH_INDEX = [(core.FIELD_INDEX[0], "accessible device ids", {"accessible", "device", "id"}, "accessible device ids")]
        core.SCHEMA_INDEX = {
            ("json", "EndUser"): {},
            ("yaml", "EndUser"): {},
        }
        core.SPEC_SOURCE_REPORT = {
            "primary": "ninja_spec.json",
            "supplemental": "ninjaOne-API-core-resources.yaml",
            "comparison": {},
        }
        return {"exampleOperation": {"method": "GET"}}

    monkeypatch.setattr(core, "SPEC_DATA", {})
    monkeypatch.setattr(core, "FIELD_INDEX", [])
    monkeypatch.setattr(core, "FIELD_SEARCH_INDEX", [])
    monkeypatch.setattr(core, "SCHEMA_INDEX", {})
    monkeypatch.setattr(core, "SPEC_SOURCE_REPORT", {})
    monkeypatch.setattr(core, "load_ops", fake_load_ops)
    monkeypatch.setattr(core, "Index", lambda operations: ("index", operations))

    result = app._ensure_mcp_schema_catalog()

    assert result is core
    assert loaded == [True]
    assert core.IDX == ("index", {"exampleOperation": {"method": "GET"}})
    assert set(core.SCHEMA_INDEX) == {("json", "EndUser"), ("yaml", "EndUser")}


def test_get_agent_injects_local_schema_tools_alongside_remote_mcp_tools(monkeypatch):
    from langchain_core.tools import StructuredTool

    captured = {}
    monkeypatch.setattr(app, "_agents", {})
    monkeypatch.setattr(app, "get_model", lambda *args: object())

    def fake_create_agent(**kwargs):
        captured.update(kwargs)
        return "fake-agent"

    monkeypatch.setattr(app, "create_agent", fake_create_agent)

    def find_endpoint(query: str) -> str:
        return query

    remote_tool = StructuredTool.from_function(
        func=find_endpoint,
        name="ninjaone__find_endpoint",
        description="Search the remote NinjaOne MCP endpoint catalog.",
    )

    result = asyncio.run(app.get_agent(
        "test-model",
        "http://127.0.0.1:1234/v1",
        [remote_tool],
        api_key="test-key",
    ))

    assert result == "fake-agent"
    tool_names = {tool.name for tool in captured["tools"]}
    assert "ninjaone__find_endpoint" in tool_names
    assert "local_search_fields" in tool_names
    assert "local_inspect_schema" in tool_names
    assert "local_schema_catalog_status" in tool_names
