import json

import ninja_agent


def test_ninjaone_tools_are_registered_from_the_shared_schemas():
    assert set(ninja_agent.NINJA_TOOL_REGISTRY.names()) >= set(ninja_agent.DISPATCH)
    assert set(ninja_agent.DISPATCH).issubset(set(ninja_agent.NINJA_TOOL_REGISTRY.names()))


def test_specialist_router_selects_access_lane_for_access_requests():
    agent = ninja_agent.specialist_for_request("add device access for Rene")
    assert agent.name == "access"
    assert "approval" in agent.instructions.lower()


def test_specialist_router_keeps_unknown_work_in_general_lane():
    assert ninja_agent.specialist_for_request("hello there").name == "general"


def test_execution_plan_reports_specialist_without_exposing_reasoning():
    plan = ninja_agent.build_execution_plan("show accessible devices for Rene")
    assert plan["specialist"]["name"] == "access"
    assert "not private model reasoning" in plan["disclosure"]


def test_shared_tool_registry_rejects_missing_required_arguments():
    result = ninja_agent.exec_tool("resolve_entity", {"kind": "device"})
    parsed = json.loads(result)
    assert "Missing required" in parsed["error"]


def test_shared_tool_registry_enforces_specialist_allowlist():
    result = ninja_agent.exec_tool("search_api", {"query": "devices"}, allowed=set())
    parsed = json.loads(result)
    assert "not permitted" in parsed["error"]

def test_ninjaone_specialist_instructions_load_the_matching_skill():
    instructions = ninja_agent.specialist_instructions("grant device access to Rene")
    assert "Active specialist: access" in instructions
    assert "Device access and permissions" in instructions
    assert "explicit UI approval" in instructions



def test_cross_resource_request_builds_bounded_specialist_team():
    plan = ninja_agent.build_execution_plan(
        "show accessible devices for user Rene"
    )
    names = [item["name"] for item in plan["specialist_team"]]

    assert len(names) <= 3
    assert "access" in names
    assert "inventory" in names
    assert "people" in names


def test_specialist_instructions_coordinate_team_without_parallel_model_fanout():
    instructions = ninja_agent.specialist_instructions(
        "show accessible devices for user Rene"
    )

    assert "Active specialist team:" in instructions
    assert "inventory:" in instructions
    assert "people:" in instructions
    assert "access:" in instructions
    assert "one coordinated investigation" in instructions
