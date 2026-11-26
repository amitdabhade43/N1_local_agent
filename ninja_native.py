"""Native NinjaOne backend for N1_local_agent's FastAPI UI (no MCP server needed).

The Foundry base URL is discovered by app.py and passed into run_native, so this
backend follows dynamic Foundry port changes just like the MCP backend.

Keeps the UI contract of app.py: returns (answer, activity) where activity is a list of
{"kind":"call","tool","arguments"} / {"kind":"result","tool","result"} items.
Reuses the spec index, tools and fast path from ninja_agent.py.
"""
import asyncio, hashlib, json, os, re, time
from agent_runtime import RuntimeEvent

import requests
import httpx
from fastapi import HTTPException
from openai import OpenAI

import ninja_agent as core

try:
    MAX_STEPS = max(1, min(int(os.getenv("NINJA_AGENT_MAX_STEPS", "20")), 40))
except (TypeError, ValueError):
    MAX_STEPS = 20
_ready, _sessions, _clients = False, {}, {}


# Cache clients by endpoint because Foundry can receive a new port after restart.
def _requires_ninjaone_tool(content):
    """Identify requests requiring live NinjaOne evidence rather than model-only prose."""
    text = re.sub(r"\s+", " ", str(content or "").strip().casefold())
    if not text or core.GREET.match(text):
        return False
    if _is_mutation_request(text):
        return True
    words = set(re.findall(r"[a-z0-9]+", text))
    nouns = set(getattr(core, "NOUN_WORDS", set()))
    resource_mentioned = any(
        word in nouns or (word.endswith("s") and word[:-1] in nouns)
        for word in words
    )
    operation_language = bool(re.search(
        r"\b(?:how many|count|list|show|find|search|lookup|look up|get|retrieve|"
        r"details|status|created|creator|assigned|accessible|access|online|offline|"
        r"latest|newest|most recent|compare|investigate|check)\b",
        text,
    ))
    return resource_mentioned or operation_language


def _native_tool_result_succeeded(tool_name, output):
    """Use the shared verifier for live-data tool envelopes."""
    if tool_name not in {"call_api", "resolve_entity"}:
        return None
    verification = core.verify_tool_result(tool_name, output)
    return verification.get("status") in {"verified", "partial"}


def _model_completion(client, model_id, messages, require_tool=False):
    """Call the local OpenAI-compatible model, requiring a function call when supported."""
    kwargs = {
        "model": model_id,
        "messages": messages,
        "tools": core.TOOLS,
        "temperature": 0,
    }
    if require_tool:
        kwargs["tool_choice"] = "required"
    try:
        return client.chat.completions.create(**kwargs).choices[0].message
    except Exception as exc:
        message = str(exc).casefold()
        unsupported = (
            ("tool_choice" in message or "tool choice" in message)
            and any(term in message for term in (
                "unsupported", "unknown", "unrecognized", "not support",
                "invalid parameter", "unexpected", "not allowed",
            ))
        )
        if not (require_tool and unsupported):
            raise
        # Some OpenAI-compatible local runtimes do not implement tool_choice=required.
        # Retry with the explicit tool-call instruction and fail closed if still no tool.
        kwargs.pop("tool_choice", None)
        return client.chat.completions.create(**kwargs).choices[0].message


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
    # "give" is commonly conversational ("give me the list"), not a write
    # intent. Only treat it as a mutation when it explicitly grants access,
    # permissions, or a role. Keep the other action verbs as explicit intents.
    if re.search(r"\bgive\b.{0,60}\b(?:access|permission|permissions|role)\b", text):
        return True
    return bool(re.search(
        r"\b(?:add|assign|grant|remove|revoke|update|change|modify|edit|"
        r"enable|disable|create|delete|reset|rename|move|invite|suspend|unsuspend|"
        r"activate|deactivate|unlink|link)\b",
        text,
    ))

_DELETE_INTENT_RE = re.compile(
    r"\b(?:delete|erase|destroy|purge)\b|\bpermanently\s+remove\b|"
    r"\bremove\s+(?:the\s+)?(?:end[\s-]?user|user|technician|device|organization|contact|record)\b"
    r"(?!\s+(?:from|off|access|role|group|list)\b)",
    re.I,
)


def _is_delete_request(content):
    """Identify explicit destructive user intent; 'remove access' remains a valid non-delete change."""
    return bool(_DELETE_INTENT_RE.search(str(content or "")))


def _is_action_clarification(content):
    """Recognize a useful follow-up asking for required mutation details."""
    text = re.sub(r"\s+", " ", str(content or "").strip().lower())
    asks = re.search(
        r"\b(?:please\s+provide|provide\s+(?:the|your)|tell\s+me|"
        r"what\s+(?:is|are)|which|i\s+need|missing|required)\b",
        text,
    )
    names_field = re.search(
        r"\b(?:first[\s_-]?name|last[\s_-]?name|email(?:\s+address)?|"
        r"organization(?:\s+(?:name|id))?|phone|manager|invitation|portal\s+access)\b",
        text,
    )
    return bool(asks and names_field)


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


def _resolve_device_access_clarification(question, history):
    """Resolve a numbered/name/email reply to a prior device-access user clarification."""
    assistant_text = next(
        (_text(item.get("content", "")) for item in reversed(history or [])
         if item.get("role") == "assistant"),
        "",
    )
    # Some stored assistant turns may carry escaped newline sequences rather
    # than actual line breaks. Normalize both forms before parsing numbered choices.
    assistant_text = assistant_text.replace("\\n", "\n")
    request_pattern = (
        r"(?:"
        r"\b(?:add|grant|give|provide)\s+(?:the\s+)?(?:device\s+)?"
        r"[A-Za-z0-9][A-Za-z0-9._-]*\s+to\s+.+?\s+"
        r"(?:for\s+)?(?:the\s+)?(?:device\s+)?access(?:\s+list)?\s*$"
        r"|^\s*(?:please\s+)?(?:add|grant|give|provide)\s+"
        r"[A-Za-z0-9][A-Za-z0-9._-]*\s+"
        r"(?:system|device|computer|machine|endpoint)\s+access\s+(?:to|for)\s+.+?[.!?]*$"
        r"|^\s*(?:please\s+)?(?:add|grant|give|provide)\s+.+?\s+"
        r"(?:device\s+)?access\s+to\s+[A-Za-z0-9][A-Za-z0-9._-]*\s*[.!?]*$"
        r")"
    )
    original_request = next(
        (_text(item.get("content", "")) for item in reversed(history or [])
         if item.get("role") == "user"
         and re.search(request_pattern, _text(item.get("content", "")), re.I)),
        "",
    )

    if not assistant_text or not original_request:
        return None, None
    if "Which user should receive the device access" not in assistant_text:
        return None, None

    clarification_reply = str(question or "").strip()

    # Recover the requested device from either device-first or person-first wording.
    original_text = original_request.strip()
    request_match = re.search(
        r"\b(?:add|grant|give|provide)\s+(?:the\s+)?(?:device\s+)?"
        r"(?P<device>[A-Za-z0-9][A-Za-z0-9._-]*)\s+to\s+.+?\s+"
        r"(?:for\s+)?(?:the\s+)?(?:device\s+)?access(?:\s+list)?\s*$",
        original_text, re.I,
    )
    if not request_match:
        request_match = re.search(
            r"^\s*(?:please\s+)?(?:add|grant|give|provide)\s+"
            r"(?P<device>[A-Za-z0-9][A-Za-z0-9._-]*)\s+"
            r"(?:system|device|computer|machine|endpoint)\s+"
            r"access\s+(?:to|for)\s+.+?[.!?]*$",
            original_text, re.I,
        )
    if not request_match:
        request_match = re.search(
            r"^\s*(?:please\s+)?(?:add|grant|give|provide)\s+.+?\s+"
            r"(?:device\s+)?access\s+to\s+(?P<device>[A-Za-z0-9][A-Za-z0-9._-]*)\s*[.!?]*$",
            original_text, re.I,
        )
    if not request_match:
        return None, None
    device = request_match.group("device")

    candidates = []
    for line in assistant_text.splitlines():
        match = re.match(
            r"^\s*(\d+)\.\s+(.+?)(?:\s+<([^<>]+)>)?"
            r"(?:\s+\((?P<category>end-user|console user|technician|contact)(?: ID \d+)?\))?\s*$",
            line,
        )
        if match:
            candidates.append({
                "number": int(match.group(1)),
                "name": match.group(2).strip(),
                "email": (match.group(3) or "").strip(),
                "category": (match.group("category") or "").strip(),
            })
    if not candidates:
        return None, None

    # "PC Admin <name>" is a request to target a console-user identity, not
    # permission to substitute a same-name end-user record. The documented
    # device-access PATCH is end-user-only, so fail safely when that console
    # candidate is present.
    pc_admin = re.match(
        r"^(?:i\s+want\s+to\s+add\s+(?:it\s+)?(?:in|to)\s+)?"
        r"(?:the\s+)?pc\s+admin\s+(.+?)\s*[.!?]*$",
        clarification_reply,
        re.I,
    )
    if pc_admin:
        requested_name = pc_admin.group(1).strip().casefold()
        console_matches = [
            item for item in candidates
            if item.get("category") == "console user"
            and item["name"].casefold() == requested_name
        ]
        if len(console_matches) == 1:
            item = console_matches[0]
            identity = item["name"] + (f' <{item["email"]}>' if item["email"] else "")
            return None, (
                f'I found a matching console-user record for {identity}, but this candidate list '
                "does not establish that its role is PC Admin. The documented device-access PATCH "
                "supports end-user records only, so I will not substitute the same-name or "
                "same-email end-user record or prepare a change. I can inspect the console-user "
                "details first. No change was prepared."
            )

    # A clarification is not a lock on the conversation. Requests to inspect or
    # search for a person, change the target, or reject the candidate list should
    # return to normal Native routing instead of repeating the old choice prompt.
    if re.search(
        r"\b(?:check|show|find|search|lookup|look\s+up|details|who\s+is|"
        r"tell\s+me\s+about|i\s+want|i'd\s+like|not\s+in\s+the\s+list|"
        r"not\s+in\s+list|not\s+listed|none\s+of\s+these|another\s+user|"
        r"different\s+user|cancel|start\s+over)\b",
        clarification_reply,
        re.I,
    ):
        return None, None

    reply = str(question or "").strip().strip(" .")

    # Do not trap the whole conversation in a candidate-selection loop. Only
    # treat selection-shaped replies as attempts to choose a listed candidate.
    # Questions, greetings, cancellations, acknowledgements and new requests
    # must return to normal routing (including the original request context).
    if (
        not reply.isdigit()
        and re.search(
            r"\b(?:how|what|why|when|where|who|which|"
            r"count|number|many|total|list|search|find|check|show|details|"
            r"cancel|stop|nevermind|never\s+mind|"
            r"hi|hello|hey|thanks|thank\s+you|"
            r"no\s+change|not\s+required|not\s+needed|"
            r"start\s+over|new\s+request)\b",
            reply,
            re.I,
        )
    ):
        return None, None

    selected = []
    if reply.isdigit():
        selected = [item for item in candidates if item["number"] == int(reply)]
    else:
        folded = reply.casefold()
        selected = [
            item for item in candidates
            if folded == item["name"].casefold()
            or (item["email"] and folded == item["email"].casefold())
        ]

    if len(selected) != 1:
        choices = "\n".join(
            f'{item["number"]}. {item["name"]}'
            + (f' <{item["email"]}>' if item["email"] else "")
            + (f' ({item["category"]})' if item["category"] else "")
            for item in candidates
        )
        if selected:
            prompt = "That reply still matches more than one user. Please choose a unique option number or email address."
        else:
            prompt = "I couldn't match that reply to one listed user. Please choose a valid option number, full name, or email address."
        return None, (
            f"{prompt}\nWhich user should receive the device access for \"{device}\"?\n"
            f"{choices}\nNo change has been prepared."
        )

    user = selected[0]
    category = user.get("category")
    if category and category != "end-user":
        identity = user["name"] + (f' <{user["email"]}>' if user["email"] else "")
        return None, (
            f"The selected record {identity} is a {category}, not an end user. "
            "The documented NinjaOne device-access PATCH supports end-user records only. "
            "I will not substitute a same-name or same-email end-user record. "
            "No change was prepared."
        )
    # Older conversation histories may contain end-user candidate lists without
    # an explicit category label. Resume those selections, but let the current
    # recipe re-resolve the exact identity across every people inventory.
    target = user["name"]
    if user["email"]:
        target += f' <{user["email"]}>'
    return f"add {device} to {target} for device access", None


def _try_fast_path(session_id, content):
    """Execute only a deterministic read fast path without resolving an LLM provider."""
    core.set_active_session(session_id)
    try:
        question = _text(content)
        if _is_delete_request(question):
            refusal = (
                "I can't delete NinjaOne records or perform destructive delete actions. "
                "Delete requests are disabled by policy; no NinjaOne API operation was called."
            )
            return refusal, [{"kind": "policy", "stage": "blocked_delete",
                              "message": "Delete requests are blocked; no NinjaOne API operation was called."}]
        _ensure()
        # Chat text is never an approval action. Check this before clarification
        # handling and, critically, before any write recipe can prepare a new
        # pending action from the literal word "approve".
        if _AFFIRMATIVE_REPLY_RE.fullmatch(question.strip()):
            if _pending_for_session(session_id):
                message = (
                    "A real NinjaOne approval request is pending. Use its Approve and execute "
                    "or Reject button; a chat reply cannot approve it."
                )
            else:
                message = (
                    "There is no active NinjaOne approval request to approve. The previous "
                    "response did not create a real approval card, and no change was made. "
                    "Please resend the original change request; I must create the approval card "
                    "before you can approve it."
                )
            return message, [{
                "kind": "policy",
                "stage": "approval_not_pending",
                "message": "Chat confirmation was not treated as approval; no mutation was executed.",
            }]
        history = _sessions.get(session_id, [])
        resumed_question, clarification = _resolve_device_access_clarification(
            question, history
        )
        if clarification:
            return clarification, [
                {"kind": "call", "tool": "clarify_device_access_user", "arguments": {"reply": question}},
                {"kind": "result", "tool": "clarify_device_access_user", "result": clarification},
            ]
        if resumed_question:
            question = resumed_question
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
            answer = core.fast_path(question, history)
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
    "between matching NinjaOne entities. First determine whether the current message is "
    "actually selecting one of those candidates. A number, exact listed full name, or exact "
    "listed email may select a candidate. If the user names a different person, says the "
    "correct person is not listed, asks to inspect/search for a person, or changes the target, "
    "do NOT repeat the old choice prompt. Search the newly named identity in the appropriate "
    "NinjaOne people inventories and report the exact record type, ID, email and available "
    "details. Keep the original requested device/action in context when relevant, and use a valid candidate selection to finish the original request when the selected record type supports it. If it does not, explain the limitation without preparing a write. Never "
    "substitute an end-user record for a console user or technician merely because names or "
    "emails overlap. The documented device-access PATCH supports end-user records only; "
    "do not prepare a write for another record type. Do not ask the user for operation IDs "
    "or endpoint names; those are internal. If a genuinely ambiguous new lookup remains, ask one concise "
    "question that lets the user disambiguate it."
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


def _execute_parallel_read_batch(tool_calls, allowed_tools, seen_fingerprints=None):
    """Run a batch concurrently only when every call is a distinct documented read-only GET."""
    from concurrent.futures import ThreadPoolExecutor

    if len(tool_calls or []) < 2 or "call_api" not in set(allowed_tools or ()):
        return {}
    prepared = []
    fingerprints = set()
    for tc in tool_calls:
        function = getattr(tc, "function", None)
        name = getattr(function, "name", "")
        if name != "call_api":
            return {}
        try:
            args = json.loads(getattr(function, "arguments", "") or "{}")
        except (TypeError, ValueError):
            return {}
        operation = core.OPS.get(args.get("operation_id", ""))
        if not operation or operation.get("method") != "GET" or not operation.get("read_only"):
            return {}
        fingerprint = hashlib.sha256(json.dumps(
            [name, args], sort_keys=True, default=str
        ).encode("utf-8")).hexdigest()
        if fingerprint in fingerprints or fingerprint in (seen_fingerprints or {}):
            return {}
        fingerprints.add(fingerprint)
        prepared.append((tc, name, args))

    started = time.perf_counter()
    results = {}
    # Keep the initial limit deliberately small for the local Foundry/API host.
    with ThreadPoolExecutor(max_workers=min(2, len(prepared))) as pool:
        futures = {
            tc.id: pool.submit(core.exec_tool, name, args, allowed=allowed_tools)
            for tc, name, args in prepared
        }
        for tc, _name, _args in prepared:
            results[tc.id] = futures[tc.id].result()
    elapsed = time.perf_counter() - started
    per_call = elapsed * 1000 / max(1, len(prepared))
    return {tool_id: (result, per_call) for tool_id, result in results.items()}


def _run_inner(session_id, model_id, content, base, progress=None, api_key=None, fast_path_checked=False):
    def emit(message, stage='working'):
        if progress:
            progress({'stage': stage, 'message': message})

    activity = []

    def record_activity(event):
        activity.append(event)
        if progress:
            kind = event.get("kind", "activity")
            tool = event.get("tool") or ""
            labels = {
                "execution_plan": "Execution plan prepared",
                "call": f"Calling {tool}" if tool else "Calling NinjaOne operation",
                "result": f"Received result from {tool}" if tool else "Received operation result",
                "verification": f"Verified result from {tool}" if tool else "Verification completed",
                "metric": f"Timing recorded for {event.get('stage', 'operation')}",
                "approval_required": "Approval request prepared; no change executed",
                "policy": "Policy check",
                "clarification": "Clarification required",
            }
            progress({
                "stage": "activity",
                "message": labels.get(kind, "Execution activity"),
                "activity_event": event,
            })

    q = _text(content)
    hist = _sessions.get(session_id, [])[-6:]
    emit("Checking whether this question can be answered directly.", "understanding")
    if _is_delete_request(q):
        refusal = (
            "I can't delete NinjaOne records or perform destructive delete actions. "
            "Delete requests are disabled by policy. I can still help with supported "
            "non-delete changes, such as updating a record or revoking access, when the "
            "API provides a safe operation and you approve the exact request."
        )
        _sessions[session_id] = (hist + [
            {"role": "user", "content": q},
            {"role": "assistant", "content": refusal},
        ])[-12:]
        return refusal, [{"kind": "policy", "stage": "blocked_delete",
                          "message": "Delete requests are blocked; no NinjaOne API operation was called."}]
    if (
        re.search(
            r"\b(?:create|register|invite)\b.{0,50}\bend[\s-]?users?\b|"
            r"\badd\s+(?:a\s+|an\s+|new\s+)?end[\s-]?users?\b",
            q,
            re.I,
        )
        and not re.search(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", q, re.I)
    ):
        clarification = (
            "Absolutely—I can help create the end user in NinjaOne. "
            "Please provide the **first name, last name, and email address** (these are required). "
            "You can also include the organization name or ID, phone number, whether to enable "
            "full portal access, and whether to send an invitation. "
            "I’ll validate the details and show you the exact API request for approval; "
            "nothing will be created until you approve it."
        )
        _sessions[session_id] = (hist + [
            {"role": "user", "content": q},
            {"role": "assistant", "content": clarification},
        ])[-12:]
        return clarification, [{"kind": "clarification", "stage": "missing_required_fields",
                                "message": "Requested first name, last name and email before preparing the create-end-user request."}]
    _ensure()
    # A typed "approve"/"yes" is never an approval action and must not
    # re-run the previous mutation prompt. Re-running model inference here can
    # reproduce the same prose-only "preparing approval" loop indefinitely.
    if _AFFIRMATIVE_REPLY_RE.fullmatch(q.strip()):
        if _pending_for_session(session_id):
            message = (
                "A real NinjaOne approval request is pending. Use its Approve and execute "
                "or Reject button; a chat reply cannot approve it."
            )
        else:
            message = (
                "There is no active NinjaOne approval request to approve. The previous "
                "response did not create a real approval card, and no change was made. "
                "Please resend the original change request; I must create the approval card "
                "before you can approve it."
            )
        return message, [{
            "kind": "policy",
            "stage": "approval_not_pending",
            "message": "Chat confirmation was not treated as approval; no mutation was executed.",
        }]
    execution_plan = core.build_execution_plan(q, hist)
    plan_event = {"kind": "execution_plan", "plan": execution_plan}
    record_activity(plan_event)
    emit(
        f'Plan: {execution_plan["intent"]} via {execution_plan["route"]}; '
        f'{len(execution_plan["candidate_operations"])} specification candidate(s) ranked.',
        "planning",
    )
    # Device-access grants are routed deterministically so the model cannot stall
    # while narrating a plan. The recipe only prepares a UI approval request.
    if re.search(r"\b(?:device access|accessible devices?|grant .{1,60} access|add .{1,60} access)\b", q, re.I):
        emit("Resolving the requested person and device against NinjaOne inventory.", "resolution")
    else:
        emit("Checking supported direct workflows.", "routing")
    write_result = core.recipe_add_device_access(q)
    if write_result:
        try:
            prepared = json.loads(write_result)
        except (TypeError, ValueError):
            prepared = {}
        if prepared.get("approval_required"):
            emit("Resolved the end user and device; preparing the approval request.", "tool")
            record_activity({"kind": "call", "tool": "call_api", "arguments": {
                "operation_id": prepared.get("operation_id"),
                "method": prepared.get("method"), "path": prepared.get("path"),
                "body": prepared.get("body"),
            }})
            record_activity({"kind": "result", "tool": "call_api", "result": write_result[:2500]})
            record_activity({"kind": "approval_required", "action": prepared})
            emit("The PATCH is prepared and waiting for your approval; it has not been executed.", "approval")
            return (
                f'Approval required: {prepared.get("summary", "Add the requested device access")}. '
                "Review the exact PATCH in the approval panel and choose Approve and execute. "
                "No change has been made yet.",
                activity,
            )
        record_activity({"kind": "call", "tool": "add_device_access", "arguments": {"question": q}})
        record_activity({"kind": "result", "tool": "add_device_access", "result": write_result[:2500]})
        return write_result, activity

    fp = None if fast_path_checked else core.fast_path(q, hist)  # preflight already checked when routed by app.py
    if fp:
        emit("Direct answer prepared; no model or API call was needed.", "complete")
        record_activity({"kind": "call", "tool": "fast_path", "arguments": {"question": q}})
        return fp, activity

    clarification_guidance = _CLARIFICATION_GUIDANCE if _has_pending_clarification(hist) else ""
    specialist = core.specialist_for_request(q)
    system = (core.SYSTEM + core.specialist_instructions(q) + core.candidates(q)
              + clarification_guidance + ("\n/no_think" if core.NO_THINK else ""))
    msgs = [{"role": "system", "content": system}] + hist + [{"role": "user", "content": content}]
    answer = "I couldn't finish within the step limit. Try a more specific question."
    max_steps = min(MAX_STEPS, max(1, int(specialist.max_iterations)))
    try:
        max_seconds = max(15.0, min(float(os.getenv("NINJA_AGENT_MAX_SECONDS", "180")), 600.0))
    except (TypeError, ValueError):
        max_seconds = 180.0
    run_started = time.monotonic()
    tool_fingerprints = {}
    called = nudged = False
    steps_text = 0
    for step in range(max_steps):
        if time.monotonic() - run_started >= max_seconds:
            answer = "I reached the configured execution time budget before the request was fully verified. The results collected so far are shown in the execution activity; please narrow the request or retry."
            emit("Execution time budget reached; stopping safely.", "timeout")
            break
        emit(f'Analyzing the request (step {step + 1} of {max_steps}).', 'model')
        inference_started = time.perf_counter()
        require_tool = bool(nudged and not called and _requires_ninjaone_tool(q))
        m = _model_completion(
            _client(base, api_key), model_id, msgs, require_tool=require_tool
        )
        record_activity({"kind": "metric", "stage": "llm_inference",
                         "elapsed_ms": round((time.perf_counter() - inference_started) * 1000, 1)})
        if not m.tool_calls:
            tc = core.text_call(m.content)
            if tc and steps_text < 3:
                steps_text += 1
                called = True
                emit(f'Calling NinjaOne operation: {tc[0]}', 'tool')
                tool_started = time.perf_counter()
                record_activity({"kind": "call", "tool": tc[0], "arguments": tc[1]})
                fingerprint = hashlib.sha256(json.dumps(
                    [tc[0], tc[1]], sort_keys=True, default=str
                ).encode("utf-8")).hexdigest()
                tool_fingerprints[fingerprint] = tool_fingerprints.get(fingerprint, 0) + 1
                if tool_fingerprints[fingerprint] > 2:
                    out = core.err("Repeated identical tool call blocked by loop protection. Use the previous result or change the query.")
                    record_activity({"kind": "policy", "stage": "loop_guard",
                                     "message": f"Blocked repeated identical call to {tc[0]}."})
                else:
                    out = core.exec_tool(tc[0], tc[1], allowed=specialist.allowed_tools)
                emit(f'Received a response from {tc[0]}.', 'result')
                result_event = {"kind": "result", "tool": tc[0], "result": out[:2500]}
                result_ok = _native_tool_result_succeeded(tc[0], out)
                if result_ok is not None:
                    result_event["success"] = result_ok
                    result_event["verification"] = core.verify_tool_result(tc[0], out)
                    record_activity({"kind": "verification", "tool": tc[0], **result_event["verification"]})
                record_activity(result_event)
                record_activity({"kind": "metric", "stage": "ninjaone_tool",
                                 "tool": tc[0], "elapsed_ms": round((time.perf_counter() - tool_started) * 1000, 1)})
                try:
                    prepared = json.loads(out)
                    if prepared.get("approval_required"):
                        record_activity({"kind": "approval_required", "action": prepared})
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
            mutation_without_approval = (
                _is_mutation_request(q)
                and not has_approval_action
                and not _is_action_clarification(txt)
            )
            bad = tool_intent or mutation_without_approval or (
                not called
                and not _is_action_clarification(txt)
                and bool(core.PLAN.search(txt) or core.BADCALL.search(txt) or lookup_intent)
            )
            if bad and not nudged:
                nudged = True
                msgs += [{"role": "assistant", "content": txt}, {"role": "user", "content": core.NUDGE}]
                continue
            answer = core.FALLBACK if bad else (txt or "The model returned no text response.")
            break
        called = True
        msgs.append(m.model_dump(exclude_none=True))
        parallel_read_results = _execute_parallel_read_batch(
            m.tool_calls, specialist.allowed_tools, tool_fingerprints
        )
        if parallel_read_results:
            emit(
                f"Running {len(parallel_read_results)} independent read-only NinjaOne calls concurrently (maximum 2).",
                "tool",
            )
            record_activity({
                "kind": "parallel_batch",
                "mode": "read_only_get",
                "concurrency_limit": 2,
                "tools": [tc.function.name for tc in m.tool_calls],
            })
        for tc in m.tool_calls:
            emit(f"Calling NinjaOne operation: {tc.function.name}", "tool")
            tool_started = time.perf_counter()
            try:
                args = json.loads(tc.function.arguments or "{}")
            except (TypeError, ValueError):
                args = {}
            record_activity({"kind": "call", "tool": tc.function.name, "arguments": args})
            fingerprint = hashlib.sha256(json.dumps(
                [tc.function.name, args], sort_keys=True, default=str
            ).encode("utf-8")).hexdigest()
            tool_fingerprints[fingerprint] = tool_fingerprints.get(fingerprint, 0) + 1
            if tool_fingerprints[fingerprint] > 2:
                out = core.err("Repeated identical tool call blocked by loop protection. Use the previous result or change the query.")
                record_activity({"kind": "policy", "stage": "loop_guard",
                                 "message": f"Blocked repeated identical call to {tc.function.name}."})
            elif tc.id in parallel_read_results:
                out, parallel_elapsed_ms = parallel_read_results[tc.id]
                tool_started = time.perf_counter() - (parallel_elapsed_ms / 1000)
            else:
                out = core.exec_tool(
                    tc.function.name, args, allowed=specialist.allowed_tools
                )
            emit(f"Received a response from {tc.function.name}.", "result")
            result_event = {"kind": "result", "tool": tc.function.name, "result": out[:2500]}
            result_ok = _native_tool_result_succeeded(tc.function.name, out)
            if result_ok is not None:
                result_event["success"] = result_ok
                result_event["verification"] = core.verify_tool_result(tc.function.name, out)
                record_activity({"kind": "verification", "tool": tc.function.name, **result_event["verification"]})
            record_activity(result_event)
            record_activity({"kind": "metric", "stage": "ninjaone_tool",
                             "tool": tc.function.name, "elapsed_ms": round((time.perf_counter() - tool_started) * 1000, 1)})
            try:
                prepared = json.loads(out)
                if prepared.get("approval_required"):
                    record_activity({"kind": "approval_required", "action": prepared})
            except (ValueError, AttributeError):
                pass
            msgs.append({"role": "tool", "tool_call_id": tc.id, "content": out})
    approval_events = [item for item in activity if item.get("kind") == "approval_required"]
    live_api_results = [
        item for item in activity
        if item.get("kind") == "result" and item.get("tool") in {"call_api", "resolve_entity"}
    ]
    verified_read_result = any(item.get("success") is True for item in live_api_results)
    if (
        _requires_ninjaone_tool(q)
        and not _is_mutation_request(q)
        and not verified_read_result
        and not _is_action_clarification(answer)
    ):
        answer = (
            "I could not complete a verified live NinjaOne lookup for this request. "
            "The specification or endpoint discovery alone is not tenant data, so I won't guess. "
            "Please rephrase the target or request the relevant record details."
        )
    if (
        _is_mutation_request(q)
        and not approval_events
        and not _is_action_clarification(answer)
    ):
        answer = (
            "I could not prepare this NinjaOne change yet. No write was executed. "
            "I can help complete it: please provide any missing required details, or "
            "tell me the target record and the exact change you want."
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
    task = core.NINJA_TASKS.create(
        "orchestrator", _text(content), metadata={"backend": "native", "model": model_id}
    )
    core.NINJA_TASKS.start(task.task_id)

    def forward_event(event):
        if progress and event.task_id == task.task_id:
            progress(event.to_dict())

    unsubscribe = core.NINJA_EVENT_BUS.subscribe(forward_event)

    def publish(payload):
        payload = dict(payload or {})
        stage = str(payload.pop("stage", "working"))
        message = str(payload.pop("message", "Agent update"))
        core.NINJA_EVENT_BUS.publish(RuntimeEvent(
            "progress", task.task_id, stage=stage, message=message, data=payload
        ))

    try:
        publish({"stage": "task_start", "message": "Agent task started"})
        result = _run_inner(
            session_id, model_id, content, base, publish, api_key, fast_path_checked
        )
        core.NINJA_TASKS.complete(task.task_id, {
            "answer": result[0] if isinstance(result, tuple) else str(result)
        })
        publish({"stage": "task_complete", "message": "Agent task completed"})
        return result
    except Exception as exc:
        core.NINJA_TASKS.fail(task.task_id, str(exc))
        publish({"stage": "task_failed", "message": "Agent task failed safely"})
        raise
    finally:
        unsubscribe()
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
