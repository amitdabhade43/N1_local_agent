import json

import ninja_agent
import ninja_native


class FakeIndex:
    def search_scored(self, question, limit):
        return [
            (8.125, {
                "id": "getDevices",
                "method": "GET",
                "path": "/v2/devices",
                "summary": "List devices",
                "read_only": True,
            }),
            (4.5, {
                "id": "patchDevice",
                "method": "PATCH",
                "path": "/v2/device/{id}",
                "summary": "Update device",
                "read_only": False,
            }),
        ][:limit]


def test_execution_plan_ranks_spec_operations_and_marks_write_risk(monkeypatch):
    monkeypatch.setattr(ninja_agent, "IDX", FakeIndex())

    plan = ninja_agent.build_execution_plan("update device status", history=[{"role": "user"}])

    assert plan["intent"] == "mutation"
    assert plan["route"] == "spec_ranked_tool_workflow"
    assert plan["candidate_operations"][0]["operation_id"] == "getDevices"
    assert plan["candidate_operations"][0]["risk"] == "read_only"
    assert plan["candidate_operations"][1]["risk"] == "approval_required"
    assert "approval" in " ".join(plan["guardrails"]).lower()


def test_execution_plan_blocks_destructive_intent():
    plan = ninja_agent.build_execution_plan("delete this NinjaOne device")

    assert plan["intent"] == "blocked_destructive"
    assert plan["route"] == "policy_block"


def test_execution_plan_discloses_no_private_chain_of_thought():
    plan = ninja_agent.build_execution_plan("show device status")

    assert "disclosure" in plan
    assert "not private model reasoning" in plan["disclosure"]


def test_verifier_accepts_complete_call_api_result():
    result = ninja_agent.verify_tool_result(
        "call_api", json.dumps({"count": 12, "rows": [], "total": 12})
    )

    assert result["status"] == "verified"
    assert "pagination_complete" in result["checks"]


def test_verifier_marks_partial_results_instead_of_complete():
    result = ninja_agent.verify_tool_result(
        "call_api",
        json.dumps({"count": 50, "rows": [], "partial": True,
                    "pagination_note": "Only first page returned."}),
    )

    assert result["status"] == "partial"
    assert "first page" in result["note"]


def test_verifier_rejects_invalid_and_error_results():
    invalid = ninja_agent.verify_tool_result("call_api", "not-json")
    failed = ninja_agent.verify_tool_result(
        "resolve_entity", json.dumps({"error": "API unavailable"})
    )

    assert invalid["status"] == "failed"
    assert failed["status"] == "failed"


def test_verifier_distinguishes_prepared_mutation_from_executed_result():
    result = ninja_agent.verify_tool_result(
        "call_api", json.dumps({"approval_required": True, "action_id": "pending"})
    )

    assert result["status"] == "prepared"
    assert "not executed" in result["note"].lower()


def test_native_result_success_uses_shared_verification():
    assert ninja_native._native_tool_result_succeeded(
        "call_api", json.dumps({"count": 3, "rows": []})
    ) is True
    assert ninja_native._native_tool_result_succeeded(
        "call_api", json.dumps({"partial": True, "rows": []})
    ) is True
    assert ninja_native._native_tool_result_succeeded("call_api", "null") is False



def test_native_fast_path_publishes_execution_events_live(monkeypatch):
    monkeypatch.setattr(ninja_native, "_ensure", lambda: None)
    monkeypatch.setattr(
        ninja_agent,
        "build_execution_plan",
        lambda question, history=None: {
            "intent": "read",
            "route": "spec_ranked_tool_workflow",
            "candidate_operations": [],
        },
    )
    monkeypatch.setattr(ninja_agent, "recipe_add_device_access", lambda question: None)
    monkeypatch.setattr(ninja_agent, "fast_path", lambda question, history=None: "Direct answer")
    events = []

    answer, activity = ninja_native._run_inner(
        "stream-test-session",
        "test-model",
        "hello",
        "http://127.0.0.1:1234/v1",
        progress=events.append,
    )

    assert answer == "Direct answer"
    assert any(item.get("kind") == "execution_plan" for item in activity)
    streamed = [item.get("activity_event") for item in events if item.get("activity_event")]
    assert any(item.get("kind") == "execution_plan" for item in streamed)
    assert any(item.get("kind") == "call" and item.get("tool") == "fast_path" for item in streamed)
