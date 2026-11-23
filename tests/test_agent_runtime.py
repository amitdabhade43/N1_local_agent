import pytest

from agent_runtime import (
    AgentDefinition, AgentRegistry, EventBus, RuntimeEvent,
    SubagentManager, TaskRegistry, ToolDefinition, ToolRegistry,
)


def test_tool_registry_validates_schema_and_executes_shared_handler():
    registry = ToolRegistry()
    registry.register(ToolDefinition(
        name="lookup", description="lookup", handler=lambda args: args["name"].upper(),
        input_schema={"type": "object", "properties": {"name": {"type": "string"}},
                      "required": ["name"]},
    ))

    assert registry.execute("lookup", {"name": "rene"}) == "RENE"
    with pytest.raises(ValueError, match="Missing required"):
        registry.execute("lookup", {})
    with pytest.raises(TypeError, match="must be string"):
        registry.execute("lookup", {"name": 123})


def test_tool_registry_enforces_agent_tool_allowlist():
    registry = ToolRegistry()
    registry.register(ToolDefinition(name="read", description="read", handler=lambda _: "ok"))
    with pytest.raises(PermissionError, match="not permitted"):
        registry.execute("read", {}, allowed=set())


def test_lazy_tool_loads_only_on_first_execution():
    loaded = []
    registry = ToolRegistry()
    registry.register_lazy("lazy", "lazy tool",
                           lambda: (loaded.append(True) or (lambda args: "ready")))

    assert loaded == []
    assert registry.execute("lazy", {}) == "ready"
    assert loaded == [True]
    assert registry.execute("lazy", {}) == "ready"
    assert loaded == [True]


def test_event_bus_isolates_subscriber_failures():
    bus = EventBus()
    seen = []
    bus.subscribe(lambda _event: (_ for _ in ()).throw(RuntimeError("observer failed")))
    bus.subscribe(lambda event: seen.append(event.to_dict()))

    bus.publish(RuntimeEvent("tool_started", "task-1", stage="tool"))
    assert seen and seen[0]["task_id"] == "task-1"


def test_task_registry_tracks_lifecycle_and_rejects_invalid_transitions():
    tasks = TaskRegistry()
    task = tasks.create("inventory", "list devices")
    assert tasks.get(task.task_id).status == "queued"
    tasks.start(task.task_id)
    tasks.complete(task.task_id, {"count": 4})
    assert tasks.get(task.task_id).status == "completed"
    assert tasks.get(task.task_id).result == {"count": 4}
    with pytest.raises(ValueError, match="already completed"):
        tasks.cancel(task.task_id)


def test_agent_registry_routes_to_most_relevant_specialist():
    agents = AgentRegistry()
    general = AgentDefinition("general", "general", "general")
    access = AgentDefinition("access", "access", "access", keywords=("accessible", "device access"))
    agents.register(general)
    agents.register(access)

    assert agents.route("show accessible devices for Rene").name == "access"
    assert agents.route("hello").name == "general"


def test_subagent_context_isolated_and_approval_gate_is_inherited():
    tasks = TaskRegistry()
    manager = SubagentManager(tasks=tasks, max_depth=1, max_children=2)
    agent = AgentDefinition("researcher", "research", "read only", allowed_tools=frozenset({"search"}))
    received = {}
    approval_gate = object()

    def runner(definition, prompt, context, inherited_gate):
        received.update(agent=definition.name, prompt=prompt, context=context, gate=inherited_gate)
        return "research complete"

    result = manager.run(
        agent, "inspect device inventory", runner,
        parent_context={"tenant": "eu", "history": ["private chat"], "approval_token": "secret"},
        approval_gate=approval_gate,
    )

    assert result == "research complete"
    assert received["context"] == {"tenant": "eu"}
    assert received["gate"] is approval_gate
    assert tasks.list()[0].status == "completed"


def test_subagent_spawn_depth_is_bounded():
    manager = SubagentManager(max_depth=1)
    agent = AgentDefinition("worker", "worker", "worker")
    with pytest.raises(PermissionError, match="depth limit"):
        manager.run(agent, "task", lambda *_: "ok", depth=1)
