import asyncio
import threading
import time

import pytest

import ninja_agent
from agent_runtime import (
    AgentDefinition, AgentRegistry, SubagentManager, TaskRegistry, ToolDefinition, ToolRegistry,
)


# ---------------------------------------------------------------- routing
def _registry():
    reg = AgentRegistry()
    reg.register(AgentDefinition("general", "g", "i"))
    reg.register(AgentDefinition("automation", "a", "i", keywords=("script", "scripts", "automation")))
    reg.register(AgentDefinition("access", "x", "i", keywords=("access", "accessible", "grant")))
    reg.register(AgentDefinition("tickets", "t", "i", keywords=("ticket", "tickets")))
    return reg


@pytest.mark.parametrize("text", [
    "show me the description field",       # 'script' inside 'description'
    "list accessories we own",             # 'access' inside 'accessories'
    "what is a ticketing philosophy",      # 'ticket' inside 'ticketing'
])
def test_substrings_no_longer_route(text):
    assert _registry().route(text).name == "general"


@pytest.mark.parametrize("text,expected", [
    ("run the scripts on this device", "automation"),
    ("how many tickets are open", "tickets"),
    ("Grant ACCESS to device 5", "access"),
    ("which Scripts failed", "automation"),
])
def test_whole_words_and_plurals_still_route(text, expected):
    assert _registry().route(text).name == expected


def test_route_is_route_many_first_and_default_weights_preserved():
    reg = _registry()
    text = "grant access to the tickets script"
    assert reg.route(text).name == reg.route_many(text, limit=1)[0].name == "access"  # 3+3 beats 2+2


def test_agent_weight_override_and_threshold():
    reg = AgentRegistry()
    reg.register(AgentDefinition("general", "g", "i"))
    reg.register(AgentDefinition("a", "a", "i", keywords=("alpha",), weights={"alpha": 5}))
    reg.register(AgentDefinition("b", "b", "i", keywords=("beta", "alpha")))
    assert [x.name for x in reg.route_many("alpha beta", limit=2)] == ["a", "b"]
    assert reg.route_many("alpha", min_score=2)[0].name == "a"
    assert reg.route_many("beta", min_score=2)[0].name == "general"  # weight 1 < threshold


def test_real_registry_does_not_misroute_description():
    assert ninja_agent.specialist_for_request("show the description of device 7").name != "automation"


# ---------------------------------------------------------------- tool registry
def _tool_registry():
    reg = ToolRegistry()
    reg.register(ToolDefinition(
        name="t", description="d", handler=lambda a: a,
        input_schema={"type": "object", "required": ["ids"], "additionalProperties": False,
                      "properties": {"ids": {"type": "array", "items": {"type": "integer"}},
                                     "cfg": {"type": "object",
                                             "properties": {"n": {"type": "integer"}}}}}))
    return reg


def test_nested_and_extra_arguments_are_rejected():
    reg = _tool_registry()
    assert reg.execute("t", {"ids": [1, 2]}) == {"ids": [1, 2]}
    with pytest.raises(ValueError):
        reg.execute("t", {"ids": [1, "x"]})                # array item type
    with pytest.raises(ValueError):
        reg.execute("t", {"ids": [1], "cfg": {"n": "no"}})  # nested type
    with pytest.raises(ValueError):
        reg.execute("t", {"ids": [1], "surprise": True})    # additionalProperties


def test_malformed_tool_schema_does_not_brick_tool():
    reg = ToolRegistry()
    reg.register(ToolDefinition("t", "d", handler=lambda a: "ok",
                                input_schema={"type": "object", "properties": {"x": {"type": 12}}}))
    assert reg.execute("t", {"x": 1}) == "ok"


def test_lazy_loader_runs_once_and_async_matches_sync():
    reg = ToolRegistry()
    loads = []
    reg.register_lazy("lazy", "d", lambda: loads.append(1) or (lambda a: "v"))
    assert reg.execute("lazy", {}) == "v"
    assert asyncio.run(reg.aexecute("lazy", {})) == "v"
    assert len(loads) == 1
    assert reg.get("lazy").lazy_loader is None


def test_permission_and_unknown_tool_errors_unchanged():
    reg = _tool_registry()
    with pytest.raises(PermissionError):
        reg.execute("t", {"ids": []}, allowed={"other"})
    with pytest.raises(KeyError):
        reg.execute("nope", {})


# ---------------------------------------------------------------- subagents
RO = AgentDefinition("ro", "d", "i", read_only=True)


def test_context_scrub_is_recursive_and_allowlist_applies():
    seen = {}
    mgr = SubagentManager(max_depth=2, context_allowlist=frozenset({"question", "auth"}))
    mgr.run(RO, "p", lambda a, p, ctx, g: seen.update(ctx),
            parent_context={"question": "q", "extra": "dropped-by-allowlist",
                            "auth": {"token": "SECRET", "org": 7, "items": [{"api_key": "K", "ok": 1}]}})
    assert seen == {"question": "q", "auth": {"org": 7, "items": [{"ok": 1}]}}


def test_context_scrub_without_allowlist_keeps_safe_keys():
    seen = {}
    SubagentManager(max_depth=2).run(
        RO, "p", lambda a, p, ctx, g: seen.update(ctx),
        parent_context={"org": 1, "nested": {"Bearer_Value": "x", "keep": 2}, "session_id": "s"})
    assert seen == {"org": 1, "nested": {"keep": 2}}


def test_child_limit_is_per_request_not_global():
    mgr = SubagentManager(max_depth=2, max_children=2)
    a, b = mgr.begin_request("a"), mgr.begin_request("b")
    release = threading.Event()
    started = threading.Semaphore(0)

    def blocker(agent, prompt, ctx, gate):
        started.release()
        release.wait(5)
        return "done"

    threads = [threading.Thread(target=mgr.run, args=(RO, "p", blocker, None, a)) for _ in range(2)]
    for t in threads:
        t.start()
    for _ in range(2):
        assert started.acquire(timeout=3)
    with pytest.raises(RuntimeError):                      # request A is full
        mgr.run(RO, "p", blocker, parent_task_id=a)
    assert mgr.run(RO, "p", lambda *x: "fine", parent_task_id=b) == "fine"  # B unaffected
    release.set()
    for t in threads:
        t.join(5)
    mgr.end_request(a)
    mgr.end_request(b, error="boom")
    assert mgr.tasks.get(a).status == "completed" and mgr.tasks.get(b).status == "failed"


def test_child_cap_is_atomic_under_contention():
    mgr = SubagentManager(max_depth=2, max_children=4)
    root = mgr.begin_request()
    release, entered = threading.Event(), threading.Semaphore(0)
    outcomes, lock = [], threading.Lock()

    def runner(agent, prompt, ctx, gate):
        entered.release()
        release.wait(5)
        return "ok"

    def attempt():
        try:
            mgr.run(RO, "p", runner, parent_task_id=root)
            outcome = "ok"
        except RuntimeError:
            outcome = "rejected"
        with lock:
            outcomes.append(outcome)

    threads = [threading.Thread(target=attempt) for _ in range(16)]
    for t in threads:
        t.start()
    for _ in range(4):
        assert entered.acquire(timeout=3)
    deadline = time.time() + 3
    while time.time() < deadline:
        with lock:
            if outcomes.count("rejected") == 12:
                break
        time.sleep(0.01)
    release.set()
    for t in threads:
        t.join(5)
    assert outcomes.count("ok") == 4 and outcomes.count("rejected") == 12


def test_run_many_times_out_slow_specialist_but_keeps_fast_results():
    mgr = SubagentManager(max_depth=1, max_children=4)
    slow_gate = threading.Event()
    fast = AgentDefinition("fast", "d", "i")
    slow = AgentDefinition("slow", "d", "i")

    def runner(agent, prompt, ctx, gate):
        if agent.name == "slow":
            slow_gate.wait(10)
        return agent.name

    started = time.time()
    try:
        results = mgr.run_many([(fast, "p"), (slow, "p")], runner, timeout=0.4)
        elapsed = time.time() - started
    finally:
        slow_gate.set()
    by_name = {r["agent"]: r for r in results}
    assert by_name["fast"]["status"] == "completed" and by_name["fast"]["result"] == "fast"
    assert by_name["slow"]["status"] == "failed" and "TimeoutError" in by_name["slow"]["error"]
    assert elapsed < 3, "run_many must not wait for the hung specialist"


def test_run_many_still_rejects_mutating_and_duplicate_specialists():
    mgr = SubagentManager(max_depth=1, max_children=4)
    with pytest.raises(PermissionError):
        mgr.run_many([(AgentDefinition("w", "d", "i", read_only=False), "p")], lambda *a: 1)
    with pytest.raises(ValueError):
        mgr.run_many([(RO, "p"), (RO, "q")], lambda *a: 1)


def test_children_of_none_is_root_only():
    reg = TaskRegistry()
    root = reg.create("root", "p")
    child = reg.create("child", "p", root.task_id)
    assert [t.task_id for t in reg.children_of(None)] == [root.task_id]
    assert [t.task_id for t in reg.children_of(root.task_id)] == [child.task_id]
    assert len(reg.list(None)) == 2  # legacy behaviour untouched
