import json
import threading
from types import SimpleNamespace

import ninja_native


def _tool_call(call_id, operation_id):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(
            name="call_api",
            arguments=json.dumps({"operation_id": operation_id}),
        ),
    )


def test_independent_read_only_get_calls_execute_concurrently(monkeypatch):
    monkeypatch.setattr(
        ninja_native.core,
        "OPS",
        {
            "get_devices": {"method": "GET", "read_only": True},
            "get_users": {"method": "GET", "read_only": True},
        },
    )
    barrier = threading.Barrier(2)
    calls = []

    def fake_exec_tool(name, args, allowed=None):
        calls.append(args["operation_id"])
        barrier.wait(timeout=2)
        return json.dumps({"operation": args["operation_id"]})

    monkeypatch.setattr(ninja_native.core, "exec_tool", fake_exec_tool)

    results = ninja_native._execute_parallel_read_batch(
        [_tool_call("call-1", "get_devices"), _tool_call("call-2", "get_users")],
        allowed_tools={"call_api"},
    )

    assert set(results) == {"call-1", "call-2"}
    assert json.loads(results["call-1"][0])["operation"] == "get_devices"
    assert json.loads(results["call-2"][0])["operation"] == "get_users"
    assert set(calls) == {"get_devices", "get_users"}


def test_mutating_or_non_get_calls_fall_back_to_sequential_execution(monkeypatch):
    monkeypatch.setattr(
        ninja_native.core,
        "OPS",
        {
            "get_devices": {"method": "GET", "read_only": True},
            "patch_device": {"method": "PATCH", "read_only": False},
        },
    )
    monkeypatch.setattr(
        ninja_native.core,
        "exec_tool",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not execute in batch")),
    )

    assert ninja_native._execute_parallel_read_batch(
        [_tool_call("call-1", "get_devices"), _tool_call("call-2", "patch_device")],
        allowed_tools={"call_api"},
    ) == {}


def test_duplicate_get_calls_are_not_parallelized(monkeypatch):
    monkeypatch.setattr(
        ninja_native.core,
        "OPS",
        {"get_devices": {"method": "GET", "read_only": True}},
    )
    assert ninja_native._execute_parallel_read_batch(
        [_tool_call("call-1", "get_devices"), _tool_call("call-2", "get_devices")],
        allowed_tools={"call_api"},
    ) == {}
