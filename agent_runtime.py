"""Reusable runtime primitives inspired by ClawSpring.

The framework deliberately stays domain-neutral. NinjaOne-specific policy,
OpenAPI validation and approval remain owned by the NinjaOne adapter.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace as dc_replace
from functools import lru_cache
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Callable
import json
import re
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

    def children_of(self, parent_task_id: str | None) -> list[AgentTask]:
        """Exact-parent filter. ``None`` means root-level tasks only (unlike ``list(None)``,
        which returns every task and would make child limits global across requests)."""
        with self._lock:
            values = [t for t in self._tasks.values() if t.parent_task_id == parent_task_id]
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

    def _checked(self, name: str, arguments: dict[str, Any], allowed: set[str] | None,
                 before_execute: Callable[[ToolDefinition, dict[str, Any]], None] | None) -> ToolDefinition:
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
        return tool

    def _resolve_handler(self, tool: ToolDefinition) -> Callable[[dict[str, Any]], Any]:
        if tool.handler is not None:
            return tool.handler
        if tool.lazy_loader is None:
            raise RuntimeError(f"Tool has no handler: {tool.name}")
        handler = tool.lazy_loader()  # may do I/O: deliberately outside the registry lock
        if not callable(handler):
            raise TypeError(f"Lazy loader for {tool.name} did not return a callable.")
        with self._lock:
            current = self._tools.get(tool.name)
            if current is not None and current.handler is not None:
                return current.handler  # another thread finished loading first
            self._tools[tool.name] = dc_replace(tool, handler=handler, lazy_loader=None)
        return handler

    def execute(self, name: str, arguments: dict[str, Any],
                allowed: set[str] | None = None,
                before_execute: Callable[[ToolDefinition, dict[str, Any]], None] | None = None) -> Any:
        tool = self._checked(name, arguments, allowed, before_execute)
        return self._resolve_handler(tool)(arguments)

    async def aexecute(self, name: str, arguments: dict[str, Any],
                       allowed: set[str] | None = None,
                       before_execute: Callable[[ToolDefinition, dict[str, Any]], None] | None = None) -> Any:
        """Async counterpart used by MCP adapters while preserving registry policy."""
        tool = self._checked(name, arguments, allowed, before_execute)
        result = self._resolve_handler(tool)(arguments)
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
        ToolRegistry._validate_deep(schema, arguments)

    @staticmethod
    def _validate_deep(schema: dict[str, Any], arguments: dict[str, Any]) -> None:
        """Second pass with jsonschema for what the shallow checks above cannot see."""
        if not schema:
            return
        try:
            from jsonschema import Draft7Validator
            from jsonschema.exceptions import SchemaError, best_match
        except ImportError:  # pragma: no cover - jsonschema is a declared dependency
            return
        try:
            Draft7Validator.check_schema(schema)
        except SchemaError:
            return  # a malformed tool schema must not make the tool unusable
        error = best_match(Draft7Validator(schema).iter_errors(arguments))
        if error is not None:
            where = ".".join(str(part) for part in error.absolute_path)
            raise ValueError(f"Invalid tool arguments{' at ' + where if where else ''}: {error.message}")


@dataclass(frozen=True)
class AgentDefinition:
    name: str
    description: str
    instructions: str
    keywords: tuple[str, ...] = ()
    allowed_tools: frozenset[str] = frozenset()
    max_iterations: int = 8
    read_only: bool = True
    # Per-keyword weight overrides (default comes from AgentRegistry.keyword_weights, else 1).
    weights: dict[str, int] = field(default_factory=dict, hash=False, compare=False)


DEFAULT_KEYWORD_WEIGHTS: dict[str, int] = {
    **{k: 3 for k in ("access", "accessible", "permission", "permissions", "grant", "revoke")},
    **{k: 2 for k in ("policy", "policies", "ticket", "tickets", "script", "scripts", "automation")},
}


@lru_cache(maxsize=512)
def _keyword_pattern(keyword: str) -> re.Pattern[str]:
    """Whole-word match with an optional plural suffix.

    Substring matching made "script" match "description" and "access" match
    "accessories"; word boundaries prevent that while "ticket" still matches "tickets".
    """
    return re.compile(rf"\b{re.escape(keyword.casefold())}(?:s|es)?\b")


class AgentRegistry:
    """Routes work to named specialist lanes without duplicating API clients."""

    def __init__(self, default_agent: str = "general",
                 keyword_weights: dict[str, int] | None = None, min_score: int = 1) -> None:
        self._agents: dict[str, AgentDefinition] = {}
        self.default_agent = default_agent
        self.keyword_weights = dict(DEFAULT_KEYWORD_WEIGHTS if keyword_weights is None else keyword_weights)
        self.min_score = max(1, int(min_score))

    def register(self, agent: AgentDefinition) -> None:
        if agent.name in self._agents:
            raise ValueError(f"Agent already registered: {agent.name}")
        self._agents[agent.name] = agent

    def get(self, name: str) -> AgentDefinition | None:
        return self._agents.get(name)

    def _score(self, agent: AgentDefinition, folded: str) -> int:
        weights = {**self.keyword_weights, **agent.weights}
        return sum(weights.get(keyword, 1) for keyword in agent.keywords
                   if _keyword_pattern(keyword).search(folded))

    def route_many(self, request: str, limit: int = 3, min_score: int | None = None) -> list[AgentDefinition]:
        """Select a small team of relevant specialists for cross-resource requests."""
        folded = str(request or "").casefold()
        threshold = self.min_score if min_score is None else max(1, int(min_score))
        ranked = []
        for agent in self._agents.values():
            if agent.name == self.default_agent:
                continue
            score = self._score(agent, folded)
            if score >= threshold:
                ranked.append((score, agent.name, agent))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        selected = [item[2] for item in ranked[:max(1, int(limit))]]
        return selected or [self._agents[self.default_agent]]

    def route(self, request: str) -> AgentDefinition:
        return self.route_many(request, limit=1)[0]

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
    """Bounded child execution with isolated context and inherited approval gate.

    Limits are enforced per ``parent_task_id``. Create one root per user request with
    ``begin_request`` and pass its id as ``parent_task_id`` so concurrent requests do not
    consume each other's child budget.

    Python threads cannot be killed: ``timeout`` stops *waiting* for a slow specialist and
    reports it as failed, but the runner itself must enforce its own HTTP/LLM deadlines.
    """

    EXCLUDED_CONTEXT_KEYS = frozenset({
        "messages", "history", "trajectory", "session", "session_id",
        "pending_actions", "approval_token", "approval_secret",
        "api_key", "client_secret", "credentials", "authorization",
        "access_token", "refresh_token", "password",
    })
    SECRET_TERMS = ("secret", "token", "credential", "password", "api_key", "apikey",
                    "authorization", "bearer", "cookie")

    def __init__(self, tasks: TaskRegistry | None = None, events: EventBus | None = None,
                 max_depth: int = 1, max_children: int = 4,
                 default_timeout: float | None = None,
                 context_allowlist: frozenset[str] | None = None) -> None:
        self.tasks = tasks or TaskRegistry()
        self.events = events or EventBus()
        self.max_depth = max(0, int(max_depth))
        self.max_children = max(1, int(max_children))
        self.default_timeout = default_timeout
        self.context_allowlist = frozenset(context_allowlist) if context_allowlist is not None else None
        self._spawn_lock = RLock()  # makes "count active children + create task" atomic

    # -- request roots -------------------------------------------------------------------
    def begin_request(self, label: str = "request", prompt: str = "") -> str:
        task = self.tasks.create(label, prompt)
        self.tasks.start(task.task_id)
        return task.task_id

    def end_request(self, task_id: str, error: str | None = None) -> None:
        task = self.tasks.get(task_id)
        if task is None or task.status in TaskRegistry.TERMINAL:
            return
        if error:
            self.tasks.fail(task_id, error)
        else:
            self.tasks.complete(task_id, None)

    # -- context isolation ---------------------------------------------------------------
    @classmethod
    def _is_secret_key(cls, key: Any) -> bool:
        folded = str(key).casefold()
        return folded in cls.EXCLUDED_CONTEXT_KEYS or any(term in folded for term in cls.SECRET_TERMS)

    @classmethod
    def sanitize_context(cls, value: Any, allowlist: frozenset[str] | None = None, _top: bool = True) -> Any:
        """Drop secret-looking keys at every depth; optionally allowlist top-level keys."""
        if isinstance(value, dict):
            clean = {}
            for key, item in value.items():
                if cls._is_secret_key(key):
                    continue
                if _top and allowlist is not None and key not in allowlist:
                    continue
                clean[key] = cls.sanitize_context(item, None, False)
            return clean
        if isinstance(value, (list, tuple, set, frozenset)):
            return [cls.sanitize_context(item, None, False) for item in value]
        return value

    # -- execution -----------------------------------------------------------------------
    def run(self, agent: AgentDefinition, prompt: str,
            runner: Callable[[AgentDefinition, str, dict[str, Any], Any], Any],
            parent_context: dict[str, Any] | None = None,
            parent_task_id: str | None = None, depth: int = 0,
            approval_gate: Any = None) -> Any:
        with self._spawn_lock:
            if depth >= self.max_depth:
                raise PermissionError("Subagent spawn depth limit reached.")
            active = [task for task in self.tasks.children_of(parent_task_id)
                      if task.status in {"queued", "running"}]
            if len(active) >= self.max_children:
                raise RuntimeError("Subagent concurrency limit reached.")
            task = self.tasks.create(agent.name, prompt, parent_task_id,
                                     {"depth": depth + 1, "allowed_tools": sorted(agent.allowed_tools)})
            self.tasks.start(task.task_id)
        context = self.sanitize_context(parent_context or {}, self.context_allowlist)
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

    def run_many(self, jobs: list[tuple[AgentDefinition, str]],
                 runner: Callable[[AgentDefinition, str, dict[str, Any], Any], Any],
                 parent_context: dict[str, Any] | None = None,
                 parent_task_id: str | None = None, depth: int = 0,
                 approval_gate: Any = None, timeout: float | None = None) -> list[dict[str, Any]]:
        """Execute independent read-only specialists concurrently with bounded fan-out.

        Each job gets a separate child task and sanitized context. Failures and timeouts
        are returned per job so one unavailable specialist does not discard other
        completed evidence. Mutating specialists are deliberately rejected here;
        writes must remain in the foreground approval workflow.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from concurrent.futures import TimeoutError as FuturesTimeout

        jobs = list(jobs or [])
        if not jobs:
            return []
        if len(jobs) > self.max_children:
            raise RuntimeError(
                f"Specialist fan-out exceeds the configured limit of {self.max_children}."
            )
        if depth >= self.max_depth:
            raise PermissionError("Subagent spawn depth limit reached.")
        names = [agent.name for agent, _prompt in jobs]
        if len(set(names)) != len(names):
            raise ValueError("A specialist may appear only once in a fan-out batch.")
        unsafe = [agent.name for agent, _prompt in jobs if not agent.read_only]
        if unsafe:
            raise PermissionError(
                "Parallel specialist execution is read-only; mutation specialists are blocked: "
                + ", ".join(unsafe)
            )

        active = [task for task in self.tasks.children_of(parent_task_id)
                  if task.status in {"queued", "running"}]
        if len(active) + len(jobs) > self.max_children:
            raise RuntimeError("Subagent concurrency limit reached.")

        wait_seconds = self.default_timeout if timeout is None else timeout
        results: dict[str, dict[str, Any]] = {}
        # No ``with`` block: its exit would join (and so wait for) hung workers.
        pool = ThreadPoolExecutor(max_workers=min(self.max_children, len(jobs)))
        try:
            futures = {
                pool.submit(
                    self.run, agent, prompt, runner, parent_context,
                    parent_task_id, depth, approval_gate,
                ): agent.name
                for agent, prompt in jobs
            }
            try:
                for future in as_completed(futures, timeout=wait_seconds):
                    name = futures[future]
                    try:
                        results[name] = {"agent": name, "status": "completed",
                                         "result": future.result(), "error": None}
                    except Exception as exc:
                        results[name] = {"agent": name, "status": "failed",
                                         "result": None, "error": f"{type(exc).__name__}: {exc}"}
            except FuturesTimeout:
                pass
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        for name in names:
            if name not in results:
                results[name] = {"agent": name, "status": "failed", "result": None,
                                 "error": f"TimeoutError: specialist did not finish within {wait_seconds}s"}
        return [results[agent.name] for agent, _prompt in jobs]
