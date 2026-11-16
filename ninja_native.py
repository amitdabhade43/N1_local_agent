"""Native NinjaOne backend for N1_local_agent's FastAPI UI (no MCP server needed).

The Foundry base URL is discovered by app.py and passed into run_native, so this
backend follows dynamic Foundry port changes just like the MCP backend.

Keeps the UI contract of app.py: returns (answer, activity) where activity is a list of
{"kind":"call","tool","arguments"} / {"kind":"result","tool","result"} items.
Reuses the spec index, tools and fast path from ninja_agent.py.
"""
import asyncio, hashlib, json, os, re, time

import requests
import httpx
from fastapi import HTTPException
from openai import OpenAI

import ninja_agent as core

MAX_STEPS = 10
_ready, _sessions, _clients = False, {}, {}


# Cache clients by endpoint because Foundry can receive a new port after restart.
def _client(base, api_key=None):
    # Cache separately by endpoint and credential identity; never log credentials.
    credential_fingerprint = hashlib.sha256((api_key or os.getenv("FOUNDRY_API_KEY", "local")).encode("utf-8")).hexdigest()[:16]
    key = (base, credential_fingerprint)
    if key not in _clients:
        client_options = {}
        gateway_base = os.getenv("LLM_GATEWAY_BASE_URL", "").strip().rstrip("/")
        gateway_base = gateway_base.removesuffix("/chat/completions").removesuffix("/models").rstrip("/")
        if gateway_base and base.rstrip("/").lower() == gateway_base.lower():
            ca_bundle = os.getenv("LLM_GATEWAY_CA_BUNDLE", "").strip()
            if ca_bundle:
                client_options["http_client"] = httpx.Client(verify=ca_bundle, timeout=120)
        _clients[key] = OpenAI(base_url=base, api_key=api_key or os.getenv("FOUNDRY_API_KEY", "local"),
                               timeout=120, max_retries=0, **client_options)
    return _clients[key]


def _is_mutation_request(content):
    """Conservatively identify user requests that ask the agent to change NinjaOne state."""
    text = re.sub(r"\s+", " ", str(content or "").strip().lower())
    if not text:
        return False
    return bool(re.search(
        r"\b(?:add|assign|grant|give|remove|revoke|update|change|modify|edit|"
        r"enable|disable|create|delete|reset|rename|move|invite|suspend|unsuspend|"
        r"activate|deactivate|unlink|link)\b",
        text,
    ))


def _text(content):
    if isinstance(content, str):
        return content
    return " ".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")


def _ensure():
    global _ready
    if _ready:
        return
    if not (core.CID and core.CSECRET):
        raise RuntimeError("Set NINJA_CLIENT_ID and NINJA_CLIENT_SECRET (and NINJA_BASE_URL) in .env")
    core.OPS = core.load_ops()
    core.IDX = core.Index(core.OPS)
    _ready = True



_AFFIRMATIVE_REPLY_RE = re.compile(r"^(?:yes|yeah|yep|sure|approved|approve|proceed|go ahead|do it|confirm|confirmed)\s*[.!]?$", re.I)


def _pending_for_session(session_id):
    with core._pending_lock:
        return any(item.get("session_id") == session_id for item in core._pending_actions.values())


def _recover_unprepared_mutation(question, history):
    """A conversational yes may resume preparation, but can never approve a write."""
    if not _AFFIRMATIVE_REPLY_RE.fullmatch(str(question or "").strip()):
        return None
    for item in reversed(history or []):
        if item.get("role") != "user":
            continue
        previous = _text(item.get("content", ""))
        if _is_mutation_request(previous):
            return previous
    return None


def _try_fast_path(session_id, content):
    """Execute only a deterministic read fast path without resolving an LLM provider."""
    core.set_active_session(session_id)
    try:
        _ensure()
        question = _text(content)
        # Supported write recipes must run before provider/model discovery. They
        # resolve targets and call the normal spec-validated write preparation
        # path, which creates a real pending action for the UI approval card.
        write_result = core.recipe_add_device_access(question)
        if write_result:
            try:
                prepared = json.loads(write_result)
            except (TypeError, ValueError):
                prepared = {}
            if prepared.get("approval_required"):
                return (
                    f'Approval required: {prepared.get("summary", "Add the requested device access")}. '
                    "Review the exact PATCH in the approval panel and choose Approve and execute. "
                    "No change has been made yet.",
                    [
                        {"kind": "call", "tool": "call_api", "arguments": {
                            "operation_id": prepared.get("operation_id"),
                            "method": prepared.get("method"), "path": prepared.get("path"),
                            "body": prepared.get("body"),
                        }},
                        {"kind": "result", "tool": "call_api", "result": write_result[:2500]},
                        {"kind": "approval_required", "action": prepared},
                    ],
                )
            return write_result, [
                {"kind": "call", "tool": "add_device_access", "arguments": {"question": question}},
                {"kind": "result", "tool": "add_device_access", "result": str(write_result)[:2500]},
            ]
        try:
            answer = core.fast_path(question)
        except requests.RequestException:
            # A failed optional read shortcut must not prevent the model/tool route.
            return None
        if not answer:
            return None
        return answer, [
            {"kind": "call", "tool": "fast_path", "arguments": {"question": question}},
            {"kind": "result", "tool": "fast_path", "result": answer[:4000]},
        ]
    finally:
        core.clear_active_session()


def _record_fast_path_turn(session_id, content, result):
    """Persist direct answers so a later turn can clarify or refer to them."""
    if result is None:
        return
    answer, _activity = result
    history = list(_sessions.get(session_id, []))
    history.extend([
        {"role": "user", "content": _text(content)},
        {"role": "assistant", "content": str(answer)},
    ])
    _sessions[session_id] = history[-12:]


_CLARIFICATION_RE = re.compile(
    r"\bplease\s+(?:confirm\s+which|clarify)\b"
    r"|\bwhich\s+(?:user|person|record|device|organization|technician|one)\b"
    r"|\bmultiple\b.{0,160}\b(?:matches|match|records|users|devices)\b",
    re.I | re.S,
)


def _has_pending_clarification(history):
    """Recognize when the last assistant turn explicitly asked the user to disambiguate."""
    for item in reversed(history or []):
        if item.get("role") == "assistant":
            return bool(_CLARIFICATION_RE.search(_text(item.get("content", ""))))
        if item.get("role") == "user":
            break
    return False


_CLARIFICATION_GUIDANCE = (
    "\n\nCONVERSATION CLARIFICATION: Your previous answer asked the user to choose "
    "between matching NinjaOne entities. The current user message is their clarification "
    "for the original request. Read the prior user request and your clarification question "
    "from conversation history, match the user's answer to one of the listed candidates, "
    "then resolve that exact entity through the appropriate API tool and finish the original "
    "request. Do not ask the user for operation IDs or endpoint names; those are internal. "
    "If the answer does not uniquely identify a candidate, ask one concise follow-up."
)


async def try_fast_path(session_id, content):
    """Async preflight used by app.py before potentially slow model discovery."""
    try:
        result = await asyncio.to_thread(_try_fast_path, session_id, content)
        if result is not None:
            await asyncio.to_thread(_record_fast_path_turn, session_id, content, result)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Native NinjaOne fast path failed: {exc}")

def _run_inner(session_id, model_id, content, base, progress=None, api_key=None, fast_path_checked=False):
    def emit(message, stage='working'):
        if progress:
            progress({'stage': stage, 'message': message})
    _ensure()
    q = _text(content)
    hist = _sessions.get(session_id, [])[-6:]
    emit("Checking whether this question can be answered directly.", "understanding")
    # A typed "yes"/"approved" is never an approval action. If the previous
    # mutation turn only produced prose and no pending action exists, resume
    # preparation from that original request so the UI can receive a real card.
    if _AFFIRMATIVE_REPLY_RE.fullmatch(q.strip()):
        if _pending_for_session(session_id):
            return (
                "There is already a pending NinjaOne approval request. Please use its "
                "Approve and execute or Reject button; a chat reply cannot approve it.",
                [],
            )
        recovered = _recover_unprepared_mutation(q, hist)
        if recovered:
            q = recovered
            emit("Resuming the previous change request to create its approval card.", "understanding")
    # Device-access grants are routed deterministically so the model cannot stall
    # while narrating a plan. The recipe only prepares a UI approval request.
    write_result = core.recipe_add_device_access(q)
    if write_result:
        try:
            prepared = json.loads(write_result)
        except (TypeError, ValueError):
            prepared = {}
        if prepared.get("approval_required"):
            emit("Resolved the end user and device; preparing the approval request.", "tool")
            activity = [
                {"kind": "call", "tool": "call_api", "arguments": {
                    "operation_id": prepared.get("operation_id"),
                    "method": prepared.get("method"), "path": prepared.get("path"),
                    "body": prepared.get("body"),
                }},
                {"kind": "result", "tool": "call_api", "result": write_result[:2500]},
                {"kind": "approval_required", "action": prepared},
            ]
            emit("The PATCH is prepared and waiting for your approval; it has not been executed.", "approval")
            return (
                f'Approval required: {prepared.get("summary", "Add the requested device access")}. '
                "Review the exact PATCH in the approval panel and choose Approve and execute. "
                "No change has been made yet.",
                activity,
            )
        return write_result, [{"kind": "call", "tool": "add_device_access", "arguments": {"question": q}},
                             {"kind": "result", "tool": "add_device_access", "result": write_result[:2500]}]

    fp = None if fast_path_checked else core.fast_path(q)  # preflight already checked when routed by app.py
    if fp:
        emit("Direct answer prepared; no model or API call was needed.", "complete")
        return fp, [{"kind": "call", "tool": "fast_path", "arguments": {"question": q}}]

    activity = []
    clarification_guidance = _CLARIFICATION_GUIDANCE if _has_pending_clarification(hist) else ""
    system = core.SYSTEM + core.candidates(q) + clarification_guidance + ("\n/no_think" if core.NO_THINK else "")
    msgs = [{"role": "system", "content": system}] + hist + [{"role": "user", "content": content}]
    answer = "I couldn't finish within the step limit. Try a more specific question."
    called = nudged = False
    steps_text = 0
    for step in range(MAX_STEPS):
        emit(f'Analyzing the request (step {step + 1} of {MAX_STEPS}).', 'model')
        inference_started = time.perf_counter()
        m = _client(base, api_key).chat.completions.create(
            model=model_id, messages=msgs, tools=core.TOOLS, temperature=0).choices[0].message
        activity.append({"kind": "metric", "stage": "llm_inference",
                         "elapsed_ms": round((time.perf_counter() - inference_started) * 1000, 1)})
        if not m.tool_calls:
            tc = core.text_call(m.content)
            if tc and steps_text < 3:
                steps_text += 1
                called = True
                emit(f'Calling NinjaOne operation: {tc[0]}', 'tool')
                tool_started = time.perf_counter()
                out = core.exec_tool(*tc)
                emit(f'Received a response from {tc[0]}.', 'result')
                activity.append({"kind": "call", "tool": tc[0], "arguments": tc[1]})
                activity.append({"kind": "result", "tool": tc[0], "result": out[:2500]})
                activity.append({"kind": "metric", "stage": "ninjaone_tool",
                                 "tool": tc[0], "elapsed_ms": round((time.perf_counter() - tool_started) * 1000, 1)})
                try:
                    prepared = json.loads(out)
                    if prepared.get("approval_required"):
                        activity.append({"kind": "approval_required", "action": prepared})
                except (ValueError, AttributeError):
                    pass
                msgs += [{"role": "assistant", "content": m.content},
                         {"role": "user", "content": core.text_call_followup(tc[0], out)}]
                continue
            txt = core.clean(m.content)
            tool_intent = bool(re.search(
                r"\b(?:i(?:\s+will|\x27ll)|going\s+to)\s+(?:call|use|invoke)\b|\b(?:retrieve|fetch|look\s+up)\s+(?:the\s+)?(?:list|records|data)\b",
                txt,
                re.I,
            ))
            question_words = set(re.findall(r"[a-z]+", q.lower()))
            lookup_intent = not called and any(
                word in core.NOUN_WORDS or (word.endswith("s") and word[:-1] in core.NOUN_WORDS)
                for word in question_words
            )
            # A state-changing request is not complete until the API layer has
            # returned an approval_required action. Never let a prose-only plan
            # masquerade as a prepared write, even after read/resolve tool calls.
            has_approval_action = any(item.get("kind") == "approval_required" for item in activity)
            mutation_without_approval = _is_mutation_request(q) and not has_approval_action
            bad = tool_intent or mutation_without_approval or (
                not called and bool(core.PLAN.search(txt) or core.BADCALL.search(txt) or lookup_intent)
            )
            if bad and not nudged:
                nudged = True
                msgs += [{"role": "assistant", "content": txt}, {"role": "user", "content": core.NUDGE}]
                continue
            answer = core.FALLBACK if bad else (txt or "The model returned no text response.")
            break
        called = True
        msgs.append(m.model_dump(exclude_none=True))
        for tc in m.tool_calls:
            emit(f"Calling NinjaOne operation: {tc.function.name}", "tool")
            tool_started = time.perf_counter()
            try:
                args = json.loads(tc.function.arguments or "{}")
                fn = core.DISPATCH.get(tc.function.name)
                out = fn(args) if fn else core.err("unknown tool")
            except requests.HTTPError as e:
                args, out = {}, core.err(f"HTTP {e.response.status_code}: {e.response.text[:200]}")
            except Exception as e:
                args, out = {}, core.err(f"{type(e).__name__}: {e}")
            activity.append({"kind": "call", "tool": tc.function.name, "arguments": args})
            emit(f"Received a response from {tc.function.name}.", "result")
            activity.append({"kind": "result", "tool": tc.function.name, "result": out[:2500]})
            activity.append({"kind": "metric", "stage": "ninjaone_tool",
                             "tool": tc.function.name, "elapsed_ms": round((time.perf_counter() - tool_started) * 1000, 1)})
            try:
                prepared = json.loads(out)
                if prepared.get("approval_required"):
                    activity.append({"kind": "approval_required", "action": prepared})
            except (ValueError, AttributeError):
                pass
            msgs.append({"role": "tool", "tool_call_id": tc.id, "content": out})
    approval_events = [item for item in activity if item.get("kind") == "approval_required"]
    if _is_mutation_request(q) and not approval_events:
        answer = (
            "I could not create a real NinjaOne approval request for this change. "
            "No write was executed and no change was made. Please retry with the target "
            "record and requested change clearly identified."
        )
    if approval_events:
        action = approval_events[-1].get("action", {})
        answer = (
            f"Prepared {action.get('method', 'write')} {action.get('path', '')}. "
            "It has NOT been executed. Review the exact request in the approval panel and "
            "choose Approve and execute or Reject."
        )
    _sessions[session_id] = (hist + [{"role": "user", "content": q},
                                     {"role": "assistant", "content": answer}])[-12:]
    return answer, activity

def _run(session_id, model_id, content, base, progress=None, api_key=None, fast_path_checked=False):
    core.set_active_session(session_id)
    try:
        return _run_inner(session_id, model_id, content, base, progress, api_key, fast_path_checked)
    finally:
        core.clear_active_session()


async def run_native(session_id, model_id, content, base_url, progress=None, api_key=None, fast_path_checked=False):
    try:
        return await asyncio.to_thread(_run, session_id, model_id, content, base_url, progress, api_key, fast_path_checked)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Native NinjaOne agent failed: {exc}")


def record_action_result(session_id, result):
    """Keep approved/rejected API outcomes available to the next chat turn."""
    history = _sessions.get(session_id, [])
    history.append({
        "role": "assistant",
        "content": "NinjaOne API approval outcome: " + json.dumps(result, default=str)[:3000],
    })
    _sessions[session_id] = history[-12:]


def clear_session(session_id):
    _sessions.pop(session_id, None)
    core.clear_pending_session(session_id)
