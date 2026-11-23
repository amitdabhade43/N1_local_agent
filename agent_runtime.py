"""Reusable runtime primitives inspired by ClawSpring.

The framework deliberately stays domain-neutral. NinjaOne-specific policy,
OpenAPI validation and approval remain owned by the NinjaOne adapter.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Callable
import json
import time
import inspect
import uuid


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class RuntimeEvent:
    kind: str
    task_id: str
    timestamp: str = field(default_factory=_utc_now)
    stage: str = ""
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "task_id": self.task_id,
            "timestamp": self.timestamp,
            "stage": self.stage,
            "message": self.message,
            **self.data,
        }


class EventBus:
    """Synchronous event fan-out; subscriber failures never break agent execution."""

    def __init__(self) -> None:
        self._subscribers: list[Callable[[RuntimeEvent], None]] = []
        self._lock = RLock()

    def subscribe(self, callback: Callable[[RuntimeEvent], None]) -> Callable[[], None]:
        with self._lock:
            self._subscribers.append(callback)

        def unsubscribe() -> None:
            with self._lock:
                if callback in self._subscribers:
                    self._subscribers.remove(callback)
        return unsubscribe

    def publish(self, event: RuntimeEvent) -> None:
        with self._lock:
            subscribers = tuple(self._subscribers)
        for callback in subscribers:
            try:
                callback(event)
            except Exception:
                # Observability must not turn a successful API call into a failure.
                continue


@dataclass
class AgentTask:
    task_id: str
    parent_task_id: str | None
    agent_name: str
    prompt: str
    status: str = "queued"
    created_at: str = field(default_factory=_utc_now)
    started_at: str | None = None
    completed_at: str | None = None
    result: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class TaskRegistry:
    """Tracks foreground/background work with bounded in-memory history."""

    TERMINAL = {"completed", "failed", "cancelled"}

    def __init__(self, history_limit: int = 200) -> None:
        self._tasks: dict[str, AgentTask] = {}
        self._lock = RLock()
        self._history_limit = max(1, int(history_limit))

    def create(self, agent_name: str, prompt: str, parent_task_id: str | None = None,
               metadata: dict[str, Any] | None = None) -> AgentTask:
        task = AgentTask(
            task_id=uuid.uuid4().hex,
            parent_task_id=parent_task_id,
            agent_name=agent_name,
            prompt=prompt,
            metadata=dict(metadata or {}),
        )
        with self._lock:
            self._tasks[task.task_id] = task
            self._trim_locked()
        return task

    def start(self, task_id: str) -> AgentTask:
        with self._lock:
            task = self._require(task_id)
            if task.status != "queued":
                raise ValueError(f"Task {task_id} cannot start from state {task.status}.")
            task.status = "running"
            task.started_at = _utc_now()
            return task

    def complete(self, task_id: str, result: Any) -> AgentTask:
        return self._finish(task_id, "completed", result=result)

    def fail(self, task_id: str, error: str) -> AgentTask:
        return self._finish(task_id, "failed", error=str(error))

    def cancel(self, task_id: str) -> AgentTask:
        return self._finish(task_id, "cancelled")

    def _finish(self, task_id: str, status: str, result: Any = None,
                error: str | None = None) -> AgentTask:
        with self._lock:
            task = self._require(task_id)
            if task.status in self.TERMINAL:
                raise ValueError(f"Task {task_id} is already {task.status}.")
            task.status = status
            task.result = result
            task.error = error
            task.completed_at = _utc_now()
            return task

    def get(self, task_id: str) -> AgentTask | None:
        with self._lock:
            return self._tasks.get(task_id)

    def list(self, parent_task_id: str | None = None) -> list[AgentTask]:
        with self._lock:
            values = list(self._tasks.values())
        if parent_task_id is not None:
            values = [task for task in values if task.parent_task_id == parent_task_id]
        return sorted(values, key=lambda task: task.created_at)

    def _require(self, task_id: str) -> AgentTask:
        task = self._tasks.get(task_id)
        if task is None:
            raise KeyError(f"Unknown task: {task_id}")
        return task

    def _trim_locked(self) -> None:
        excess = len(self._tasks) - self._history_limit
        if excess <= 0:
            return
        removable = [key for key, task in self._tasks.items()
                     if task.status in self.TERMINAL][:excess]
        for key in removable:
            self._tasks.pop(key, None)


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    handler: Callable[[dict[str, Any]], Any] | None = None
    input_schema: dict[str, Any] = field(default_factory=dict)
    read_only: bool = True
    concurrent_safe: bool = False
    category: str = "general"
    lazy_loader: Callable[[], Callable[[dict[str, Any]], Any]] | None = None


class ToolRegistry:
    """One registry for tool metadata, lazy loading, argument checks and execution."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self._lock = RLock()

    def register(self, tool: ToolDefinition) -> None:
        if not tool.name or not callable(tool.handler):
            raise ValueError("A registered tool needs a name and callable handler.")
        with self._lock:
            if tool.name in self._tools:
                raise ValueError(f"Tool already registered: {tool.name}")
            self._tools[tool.name] = tool

    def replace(self, tool: ToolDefinition) -> None:
        """Replace a dynamic adapter binding without changing its registry key."""
        if not tool.name or not callable(tool.handler):
            raise ValueError("A registered tool needs a name and callable handler.")
        with self._lock:
            self._tools[tool.name] = tool

    def register_lazy(self, name: str, description: str,
                      loader: Callable[[], Callable[[dict[str, Any]], Any]],
                      input_schema: dict[str, Any] | None = None,
                      read_only: bool = True, concurrent_safe: bool = False,
                      category: str = "general") -> None:
        if not callable(loader):
            raise ValueError("Lazy tool loader must be callable.")
        with self._lock:
            if name in self._tools:
                raise ValueError(f"Tool already registered: {name}")
            self._tools[name] = ToolDefinition(
                name=name, description=description, input_schema=input_schema or {},
                read_only=read_only, concurrent_safe=concurrent_safe,
                category=category, lazy_loader=loader,
            )

    def get(self, name: str) -> ToolDefinition | None:
        with self._lock:
            return self._tools.get(name)

    def names(self, allowed: set[str] | None = None) -> list[str]:
        with self._lock:
            names = list(self._tools)
        return sorted(name for name in names if allowed is None or name in allowed)

    def openai_schemas(self, allowed: set[str] | None = None) -> list[dict[str, Any]]:
        schemas = []
        with self._lock:
            tools = list(self._tools.values())
        for tool in tools:
            if allowed is not None and tool.name not in allowed:
                continue
            schemas.append({"type": "function", "function": {
                "name": tool.name, "description": tool.description,
                "parameters": tool.input_schema or {"type": "object", "properties": {}},
            }})
        return schemas

    def execute(self, name: str, arguments: dict[str, Any],
                allowed: set[str] | None = None,
                before_execute: Callable[[ToolDefinition, dict[str, Any]], None] | None = None) -> Any:
        with self._lock:
            tool = self._tools.get(name)
            if tool is None:
                raise KeyError(f"Unknown tool: {name}")
            if allowed is not None and name not in allowed:
                raise PermissionError(f"Agent is not permitted to use tool: {name}")
            if not isinstance(arguments, dict):
                raise TypeError("Tool arguments must be a JSON object.")
            self._validate_arguments(tool.input_schema, arguments)
            if before_execute:
                before_execute(tool, arguments)
            handler = tool.handler
            if handler is None:
                if tool.lazy_loader is None:
                    raise RuntimeError(f"Tool has no handler: {name}")
                handler = tool.lazy_loader()
                if not callable(handler):
                    raise TypeError(f"Lazy loader for {name} did not return a callable.")
                self._tools[name] = ToolDefinition(
                    name=tool.name, description=tool.description, handler=handler,
                    input_schema=tool.input_schema, read_only=tool.read_only,
                    concurrent_safe=tool.concurrent_safe, category=tool.category,
                )
        return handler(arguments)

    async def aexecute(self, name: str, arguments: dict[str, Any],
                       allowed: set[str] | None = None,
                       before_execute: Callable[[ToolDefinition, dict[str, Any]], None] | None = None) -> Any:
        """Async counterpart used by MCP adapters while preserving registry policy."""
        with self._lock:
            tool = self._tools.get(name)
            if tool is None:
                raise KeyError(f"Unknown tool: {name}")
            if allowed is not None and name not in allowed:
                raise PermissionError(f"Agent is not permitted to use tool: {name}")
            if not isinstance(arguments, dict):
                raise TypeError("Tool arguments must be a JSON object.")
            self._validate_arguments(tool.input_schema, arguments)
            if before_execute:
                before_execute(tool, arguments)
            handler = tool.handler
            if handler is None:
                if tool.lazy_loader is None:
                    raise RuntimeError(f"Tool has no handler: {name}")
                handler = tool.lazy_loader()
                if not callable(handler):
                    raise TypeError(f"Lazy loader for {name} did not return a callable.")
                self._tools[name] = ToolDefinition(
                    name=tool.name, description=tool.description, handler=handler,
                    input_schema=tool.input_schema, read_only=tool.read_only,
                    concurrent_safe=tool.concurrent_safe, category=tool.category,
                )
        result = handler(arguments)
        return await result if inspect.isawaitable(result) else result

    @staticmethod
    def _validate_arguments(schema: dict[str, Any], arguments: dict[str, Any]) -> None:
        required = schema.get("required") or []
        missing = [key for key in required if key not in arguments]
        if missing:
            raise ValueError("Missing required tool argument(s): " + ", ".join(missing))
        properties = schema.get("properties") or {}
        for key, value in arguments.items():
            definition = properties.get(key)
            if not isinstance(definition, dict):
                continue
            kind = definition.get("type")
            valid = {
                "string": lambda v: isinstance(v, str),
                "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
                "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
                "boolean": lambda v: isinstance(v, bool),
                "object": lambda v: isinstance(v, dict),
                "array": lambda v: isinstance(v, list),
            }.get(kind)
            if valid and not valid(value):
                raise TypeError(f"Tool argument '{key}' must be {kind}.")
            if "enum" in definition and value not in definition["enum"]:
                raise ValueError(f"Tool argument '{key}' must be one of: {definition['enum']}.")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if "minimum" in definition and value < definition["minimum"]:
                    raise ValueError(f"Tool argument '{key}' is below its minimum.")
                if "maximum" in definition and value > definition["maximum"]:
                    raise ValueError(f"Tool argument '{key}' exceeds its maximum.")


@dataclass(frozen=True)
class AgentDefinition:
    name: str
    description: str
    instructions: str
    keywords: tuple[str, ...] = ()
    allowed_tools: frozenset[str] = frozenset()
    max_iterations: int = 8
    read_only: bool = True


class AgentRegistry:
    """Routes work to named specialist lanes without duplicating API clients."""

    def __init__(self, default_agent: str = "general") -> None:
        self._agents: dict[str, AgentDefinition] = {}
        self.default_agent = default_agent

    def register(self, agent: AgentDefinition) -> None:
        if agent.name in self._agents:
            raise ValueError(f"Agent already registered: {agent.name}")
        self._agents[agent.name] = agent

    def get(self, name: str) -> AgentDefinition | None:
        return self._agents.get(name)

    def route(self, request: str) -> AgentDefinition:
        folded = str(request or "").casefold()
        ranked = []
        for agent in self._agents.values():
            if agent.name == self.default_agent:
                continue
            score = sum(3 if keyword in {"access", "accessible", "permission", "permissions", "grant", "revoke"} else 1 for keyword in agent.keywords if keyword in folded)
            if score:
                ranked.append((score, agent.name, agent))
        if not ranked:
            return self._agents[self.default_agent]
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return ranked[0][2]

    def all(self) -> list[AgentDefinition]:
        return list(self._agents.values())


@dataclass(frozen=True)
class SkillDefinition:
    name: str
    description: str
    instructions: str
    agents: frozenset[str] = frozenset({"general"})


class SkillRegistry:
    """Loads trusted, repository-local Markdown skills on demand."""

    def __init__(self) -> None:
        self._skills: dict[str, SkillDefinition] = {}

    def register(self, skill: SkillDefinition) -> None:
        if not skill.name or not skill.instructions.strip():
            raise ValueError("A skill needs a name and non-empty instructions.")
        self._skills[skill.name] = skill

    def load_directory(self, directory: str) -> int:
        from pathlib import Path

        root = Path(directory).resolve()
        if not root.is_dir():
            return 0
        loaded = 0
        for path in sorted(root.glob("*.md")):
            resolved = path.resolve()
            if root not in resolved.parents or not resolved.is_file():
                continue
            raw = resolved.read_text(encoding="utf-8")
            if not raw.startswith("---"):
                continue
            closing = raw.find("\n---", 3)
            if closing < 0:
                continue
            metadata_text = raw[3:closing].strip()
            body = raw[closing + 4:].strip()
            metadata: dict[str, str] = {}
            for line in metadata_text.splitlines():
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                metadata[key.strip().lower()] = value.strip().strip('"').strip("'")
            name = metadata.get("name") or resolved.stem
            description = metadata.get("description", "")
            agents = frozenset(
                item.strip() for item in metadata.get("agents", "general").split(",")
                if item.strip()
            )
            if not body:
                continue
            self.register(SkillDefinition(name, description, body, agents))
            loaded += 1
        return loaded

    def for_agent(self, agent_name: str, max_chars: int = 5000) -> str:
        matches = [
            skill for skill in self._skills.values()
            if agent_name in skill.agents or "*" in skill.agents
        ]
        matches.sort(key=lambda skill: skill.name)
        sections = [
            f"## Skill: {skill.name}\n{skill.description}\n{skill.instructions}"
            for skill in matches
        ]
        return "\n\n".join(sections)[:max(0, int(max_chars))]

    def names(self) -> list[str]:
        return sorted(self._skills)


class SubagentManager:
    """Bounded child execution with isolated context and inherited approval gate."""

    EXCLUDED_CONTEXT_KEYS = frozenset({
        "messages", "history", "trajectory", "session", "session_id",
        "pending_actions", "approval_token", "approval_secret",
    })

    def __init__(self, tasks: TaskRegistry | None = None, events: EventBus | None = None,
                 max_depth: int = 1, max_children: int = 4) -> None:
        self.tasks = tasks or TaskRegistry()
        self.events = events or EventBus()
        self.max_depth = max(0, int(max_depth))
        self.max_children = max(1, int(max_children))

    def run(self, agent: AgentDefinition, prompt: str,
            runner: Callable[[AgentDefinition, str, dict[str, Any], Any], Any],
            parent_context: dict[str, Any] | None = None,
            parent_task_id: str | None = None, depth: int = 0,
            approval_gate: Any = None) -> Any:
        if depth >= self.max_depth:
            raise PermissionError("Subagent spawn depth limit reached.")
        active = [task for task in self.tasks.list(parent_task_id)
                  if task.status in {"queued", "running"}]
        if len(active) >= self.max_children:
            raise RuntimeError("Subagent concurrency limit reached.")
        context = {key: value for key, value in (parent_context or {}).items()
                   if key not in self.EXCLUDED_CONTEXT_KEYS}
        task = self.tasks.create(agent.name, prompt, parent_task_id,
                                 {"depth": depth + 1, "allowed_tools": sorted(agent.allowed_tools)})
        self.tasks.start(task.task_id)
        self.events.publish(RuntimeEvent("task_started", task.task_id, stage="subagent",
                                         message=f"{agent.name} started"))
        try:
            result = runner(agent, prompt, context, approval_gate)
            self.tasks.complete(task.task_id, result)
            self.events.publish(RuntimeEvent("task_completed", task.task_id, stage="subagent",
                                             message=f"{agent.name} completed",
                                             data={"result_summary": str(result)[:500]}))
            return result
        except Exception as exc:
            self.tasks.fail(task.task_id, str(exc))
            self.events.publish(RuntimeEvent("task_failed", task.task_id, stage="subagent",
                                             message=f"{agent.name} failed",
                                             data={"error": str(exc)[:500]}))
            raise
