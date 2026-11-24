import threading

import pytest

from agent_runtime import AgentDefinition, SubagentManager


def _agent(name, read_only=True):
    return AgentDefinition(
        name=name, description=name, instructions="Read-only specialist",
        allowed_tools=frozenset({"call_api"}), read_only=read_only,
    )


def test_run_many_executes_independent_read_only_specialists_concurrently():
    manager = SubagentManager(max_depth=1, max_children=2)
    barrier = threading.Barrier(2)
    observed = []

    def runner(agent, prompt, context, approval_gate):
        observed.append((agent.name, prompt, context))
        barrier.wait(timeout=2)
        return {"evidence": agent.name}

    results = manager.run_many(
        [(_agent("inventory"), "inspect devices"), (_agent("people"), "inspect users")],
        runner,
        parent_context={"tenant": "eu", "history": ["private"], "api_key": "secret"},
    )

    assert [item["agent"] for item in results] == ["inventory", "people"]
    assert all(item["status"] == "completed" for item in results)
    assert {item["result"]["evidence"] for item in results} == {"inventory", "people"}
    assert len(observed) == 2
    assert all(item[2] == {"tenant": "eu"} for item in observed)
    assert all(task.status == "completed" for task in manager.tasks.list())


def test_run_many_isolates_one_specialist_failure_from_other_results():
    manager = SubagentManager(max_depth=1, max_children=2)

    def runner(agent, prompt, context, approval_gate):
        if agent.name == "audit":
            raise RuntimeError("audit endpoint unavailable")
        return {"count": 3}

    results = manager.run_many(
        [(_agent("inventory"), "list devices"), (_agent("audit"), "inspect activities")],
        runner,
    )

    assert results[0]["status"] == "completed"
    assert results[0]["result"] == {"count": 3}
    assert results[1]["status"] == "failed"
    assert "audit endpoint unavailable" in results[1]["error"]


def test_run_many_rejects_mutating_specialists_even_with_approval_context():
    manager = SubagentManager(max_depth=1, max_children=2)
    with pytest.raises(PermissionError, match="read-only"):
        manager.run_many(
            [(_agent("writer", read_only=False), "change a device")],
            lambda *_: "must not run",
            approval_gate=object(),
        )


def test_run_many_enforces_fanout_and_unique_specialist_names():
    manager = SubagentManager(max_depth=1, max_children=2)
    runner = lambda *_: "ok"

    with pytest.raises(RuntimeError, match="fan-out"):
        manager.run_many(
            [(_agent("a"), "a"), (_agent("b"), "b"), (_agent("c"), "c")],
            runner,
        )
    with pytest.raises(ValueError, match="only once"):
        manager.run_many(
            [(_agent("inventory"), "first"), (_agent("inventory"), "second")],
            runner,
        )
