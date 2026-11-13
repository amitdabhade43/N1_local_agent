#!/usr/bin/env python3
"""NinjaOne fuzzy-question agent.

Read-only, driven at runtime by the NinjaOne OpenAPI spec, tuned for local
~14B models (Qwen) served through an OpenAI-compatible API (Ollama, LM Studio, vLLM).

Design: 3 tools only (search_api, call_api, resolve_entity), a no-LLM fast path for
trivial questions, server-side pagination/filtering/counting, compact tool output.
"""
import ast, difflib, json, math, os, re, sys, time, threading, secrets
from collections import Counter
from urllib.parse import quote

import requests
from openai import OpenAI
from ninja_openapi import build_operations, describe_operation, operation_document, validate_payload, is_delete_operation

SPEC_URL = os.getenv("NINJA_SPEC_URL", "https://app.ninjarmm.com/apidocs/NinjaRMM-API-v2.json")
BASE = os.getenv("NINJA_BASE_URL", "https://eu.ninjarmm.com").rstrip("/")  # use your region host
CID, CSECRET = os.getenv("NINJA_CLIENT_ID", ""), os.getenv("NINJA_CLIENT_SECRET", "")
SCOPE = os.getenv("NINJA_SCOPE", "monitoring")
LLM_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:14b")
NO_THINK = os.getenv("NO_THINK", "0") == "1"  # set 1 for Qwen3 to skip thinking (faster)
MAX_STEPS, MAX_ROWS, MAX_CHARS, TTL = 6, 25, 6000, 60

OPS, IDX, SPEC_DATA = {}, None, {}
_tok, _cache = {"v": None, "exp": 0.0}, {}
_pending_actions, _pending_lock = {}, threading.Lock()
_active_session = threading.local()


# ---------------------------------------------------------------- spec index
def deref(spec, x):
    for _ in range(10):
        if not (isinstance(x, dict) and "$ref" in x):
            break
        node = spec
        for k in x["$ref"].lstrip("#/").split("/"):
            node = node.get(k, {}) if isinstance(node, dict) else {}
        x = node
    return x


def load_ops():
    """Load the bundled OpenAPI document and build the complete safe operation catalog."""
    global SPEC_DATA
    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ninja_spec.json")
    if os.path.exists(cache) and time.time() - os.path.getmtime(cache) < 86400:
        with open(cache, "r", encoding="utf-8") as handle:
            spec = json.load(handle)
    else:
        response = requests.get(SPEC_URL, timeout=60)
        response.raise_for_status()
        spec = response.json()
        with open(cache, "w", encoding="utf-8") as handle:
            json.dump(spec, handle)
    SPEC_DATA = spec
    return build_operations(spec)


SYN = {"count": "list", "many": "list", "total": "list", "number": "list", "machine": "device",
       "computer": "device", "endpoint": "device", "laptop": "device", "server": "device",
       "workstation": "device", "client": "organization", "customer": "organization",
       "org": "organization", "company": "organization", "tech": "technician",
       "admin": "technician", "enduser": "end user", "staff": "user", "person": "user"}


def toks(s):
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    return [t[:-1] if len(t) > 3 and t.endswith("s") else t for t in re.findall(r"[a-z0-9]+", s.lower())]


class Index:
    def __init__(self, ops):
        self.ops = ops
        self.docs = {i: set(toks(operation_document(o))) for i, o in ops.items()}
        df = Counter(t for d in self.docs.values() for t in d)
        n = max(len(self.docs), 1)
        self.idf = {t: math.log(1 + n / c) for t, c in df.items()}

    def search(self, q, k=5):
        qt = set()
        for t in toks(q):
            qt.add(t)
            qt.update(toks(SYN.get(t, "")))
        detail = bool(qt & {"detail", "id", "specific"})
        out = []
        for i, o in self.ops.items():
            if is_delete_operation(o["method"], o["path"], o["id"], o.get("summary", "")):
                continue
            segs = set(toks(o["path"])) - {"v2", "list"}
            s = sum(self.idf.get(t, 0) for t in qt if t in self.docs[i])
            s += sum(1.5 for t in qt if t in segs)
            s -= 0.15 * o["path"].count("/") + (0.5 if "{" in o["path"] and not detail else 0)
            if s > 0:
                out.append((s, o))
        out.sort(key=lambda x: -x[0])
        return [o for _, o in out[:k]]


def describe(o):
    ps = ", ".join(f'{p["name"]}({p["in"]}{",req" if p["required"] else ""})' for p in o["params"])
    body = o.get("request_body") or {}
    body_hint = ""
    if body:
        schema = body.get("schema") or {}
        required = schema.get("required", []) if isinstance(schema, dict) else []
        body_hint = f' | body: {body.get("media_type") or "unknown"}; required fields: {required}'
    return f'{o["id"]}: {o["method"]} {o["path"]} - {o["summary"]} | params: {ps or "none"}{body_hint}'


# ---------------------------------------------------------------- HTTP
def token():
    if _tok["v"] and time.time() < _tok["exp"] - 30:
        return _tok["v"]
    r = requests.post(f"{BASE}/ws/oauth/token", timeout=30, data={
        "grant_type": "client_credentials", "client_id": CID,
        "client_secret": CSECRET, "scope": SCOPE})
    r.raise_for_status()
    j = r.json()
    _tok.update(v=j["access_token"], exp=time.time() + j.get("expires_in", 3600))
    return _tok["v"]


def http_request(method, path, query=None, body=None, media_type="application/json", extra_headers=None):
    """Execute a validated NinjaOne request. GET responses are cached; writes are not."""
    method = method.upper()
    if method == "DELETE":
        raise ValueError("DELETE execution is disabled by policy.")
    if method not in {"GET", "POST", "PUT", "PATCH"}:
        raise ValueError(f"Unsupported HTTP method: {method}")

    query = query or {}
    key = (method, path, json.dumps(query, sort_keys=True), json.dumps(body, sort_keys=True, default=str),
           json.dumps(extra_headers or {}, sort_keys=True))
    hit = _cache.get(key) if method == "GET" else None
    if hit and time.time() - hit[0] < TTL:
        return hit[1]

    headers = {"Authorization": "Bearer " + token(), "Accept": "application/json"}
    for name, value in (extra_headers or {}).items():
        if name.lower() in {"authorization", "cookie", "host", "content-length"}:
            raise ValueError(f"Header '{name}' cannot be overridden.")
        headers[name] = str(value)
    kwargs = {"params": query, "timeout": 90, "headers": headers}
    if method != "GET" and body is not None:
        if media_type == "application/json":
            kwargs["json"] = body
            headers["Content-Type"] = "application/json"
        elif media_type == "application/x-www-form-urlencoded":
            kwargs["data"] = body
        else:
            raise ValueError(
                f"Request media type '{media_type}' is not supported by the JSON agent interface. "
                "This operation needs a file or multipart payload."
            )

    for attempt in (0, 1):
        response = requests.request(method, BASE + path, **kwargs)
        if response.status_code == 401 and attempt == 0:
            _tok["v"] = None
            kwargs["headers"]["Authorization"] = "Bearer " + token()
            continue
        break
    response.raise_for_status()
    if not response.content:
        data = {"ok": True, "status_code": response.status_code}
    else:
        try:
            data = response.json()
        except ValueError:
            data = {"status_code": response.status_code, "text": response.text[:4000]}

    if method == "GET":
        _cache[key] = (time.time(), data)
    else:
        _cache.clear()
    return data


def http_get(path, query=None, extra_headers=None):
    return http_request("GET", path, query, extra_headers=extra_headers)



def unwrap(data):
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in ("results", "data", "items", "content", "records"):
            if isinstance(data.get(k), list):
                return data[k]
        return [data]
    return []


def fetch(op, path, query, extra_headers=None):
    """Fetch rows; only auto-paginate when the OpenAPI declares an integer ID cursor."""
    params = {p["name"]: p for p in op["params"]}
    # Only integer ID cursors are auto-paginated. Activity timestamps and opaque
    # backup cursors also use the name "after"/"cursor" but have different semantics.
    after_schema = params.get("after", {}).get("schema", {})
    if not ("after" in params and "pageSize" in params and after_schema.get("type") == "integer"):
        return unwrap(http_get(path, query, extra_headers))
    q, rows, last = dict(query), [], None
    q.setdefault("pageSize", 1000)
    for _ in range(200):
        page = unwrap(http_get(path, q, extra_headers))
        if not page:
            break
        rows += page
        nxt = page[-1].get("id") if isinstance(page[-1], dict) else None
        if nxt is None or nxt == last:
            break
        last, q["after"] = nxt, nxt
    return rows


def op_by_path(path):
    return next((o for o in OPS.values() if o["method"] == "GET" and o["path"] == path), None)


# ---------------------------------------------------------------- shaping
def slim(r):
    if not isinstance(r, dict):
        return r
    o = {}
    for k, v in r.items():
        if isinstance(v, (str, int, float, bool)) and v != "":
            o[k] = v[:80] if isinstance(v, str) else v
        elif isinstance(v, dict) and v.get("name"):
            o[k + "_name"] = str(v["name"])[:60]
    return o


def shape(rows, where=None, contains=None, group_by=None, fields=None, count_only=False, limit=MAX_ROWS):
    if where:
        rows = [r for r in rows if isinstance(r, dict) and
                all(str(r.get(k)).lower() == str(v).lower() for k, v in where.items())]
    if contains:
        c = str(contains).lower()
        rows = [r for r in rows if c in json.dumps(r).lower()]
    out = {"total": len(rows)}
    if count_only:
        return out
    if group_by:
        def g(r):
            v = r.get(group_by) if isinstance(r, dict) else None
            return str(v.get("name") if isinstance(v, dict) else v)
        out["groups"] = dict(Counter(g(r) for r in rows).most_common(30))
        return out
    limit = min(int(limit or MAX_ROWS), 100)
    out["rows"] = [({k: r.get(k) for k in fields} if fields and isinstance(r, dict) else slim(r))
                   for r in rows[:limit]]
    if len(rows) > limit:
        out["truncated"] = f"showing {limit} of {len(rows)}; narrow with where/contains/fields"
    return out


# ---------------------------------------------------------------- tools
def err(msg):
    return json.dumps({"error": msg})


def t_search(a):
    hits = IDX.search(a.get("query", ""), int(a.get("limit", 8)))
    return "\n".join(describe(o) for o in hits) or err("no match; try different keywords")


def _compact_schema(schema, depth=0):
    if not isinstance(schema, dict):
        return schema
    allowed = ("type", "format", "description", "title", "required", "enum",
               "minimum", "maximum", "minLength", "maxLength", "minItems",
               "maxItems", "nullable", "additionalProperties")
    compact = {key: schema[key] for key in allowed if key in schema}
    if depth >= 4:
        if schema.get("properties") or schema.get("items"):
            compact["nested_schema_truncated"] = True
        return compact
    properties = schema.get("properties")
    if isinstance(properties, dict):
        compact["properties"] = {
            name: _compact_schema(value, depth + 1)
            for name, value in list(properties.items())[:35]
        }
        if len(properties) > 35:
            compact["properties_truncated"] = len(properties) - 35
    if isinstance(schema.get("items"), dict):
        compact["items"] = _compact_schema(schema["items"], depth + 1)
    return compact


def t_inspect(a):
    operation = OPS.get(a.get("operation_id", ""))
    if not operation:
        return err("Unknown operation_id. Use search_api first.")
    details = describe_operation(operation, include_schema=True)
    encoded = json.dumps(details, default=str)
    if len(encoded) > MAX_CHARS:
        details["request_body_schema"] = _compact_schema(details.get("request_body_schema") or {})
        details["schema_truncated"] = True
        encoded = json.dumps(details, default=str)
    return encoded


def _argument_errors(op, args):
    errors = []
    path_params = args.get("path_params") or {}
    query = args.get("query") or {}
    headers = args.get("headers") or {}
    declared = op.get("params", [])
    path_defs = {p["name"]: p for p in declared if p.get("in") == "path"}
    query_defs = {p["name"]: p for p in declared if p.get("in") == "query"}
    header_defs = {p["name"].lower(): p for p in declared if p.get("in") == "header"}

    missing_path = [name for name, p in path_defs.items() if p.get("required") and name not in path_params]
    if missing_path:
        errors.append("Missing path parameters: " + ", ".join(missing_path))
    unknown_path = sorted(set(path_params) - set(path_defs))
    if unknown_path:
        errors.append("Unknown path parameters: " + ", ".join(unknown_path))

    missing_query = [name for name, p in query_defs.items() if p.get("required") and name not in query]
    if missing_query:
        errors.append("Missing required query parameters: " + ", ".join(missing_query))
    missing_headers = [name for name, p in header_defs.items() if p.get("required") and name.lower() not in {h.lower() for h in headers}]
    if missing_headers:
        errors.append("Missing required headers: " + ", ".join(missing_headers))
    unknown_query = sorted(set(query) - set(query_defs))
    if unknown_query:
        errors.append("Unknown query parameters: " + ", ".join(unknown_query))

    for name in headers:
        if name.lower() not in header_defs:
            errors.append(f"Header '{name}' is not declared by the OpenAPI operation.")
        if name.lower() in {"authorization", "cookie", "host", "content-length"}:
            errors.append(f"Header '{name}' cannot be overridden by the agent.")

    for name, value in path_params.items():
        schema = path_defs.get(name, {}).get("schema") or {}
        errors.extend(f"path_params.{name}: {e}" for e in validate_payload(schema, value, SPEC_DATA))
    for name, value in query.items():
        schema = query_defs.get(name, {}).get("schema") or {}
        errors.extend(f"query.{name}: {e}" for e in validate_payload(schema, value, SPEC_DATA))

    request_body = op.get("request_body")
    body = args.get("body")
    if request_body and request_body.get("required") and body is None:
        errors.append("This operation requires a request body.")
    if body is not None:
        if not request_body:
            errors.append("This operation does not define a request body.")
        else:
            errors.extend(validate_payload(request_body.get("schema"), body, SPEC_DATA))
    return errors


def set_active_session(session_id):
    _active_session.value = session_id


def clear_active_session():
    _active_session.value = None


def _prepare_mutation(op, path, query, body, headers=None):
    session_id = getattr(_active_session, "value", None)
    if not session_id:
        return err("Mutation blocked: no active chat session is associated with this request.")
    request_body = op.get("request_body") or {}
    media_type = request_body.get("media_type") or "application/json"
    if media_type not in {"application/json", "application/x-www-form-urlencoded"}:
        return err(
            f"Mutation not prepared: {media_type} payloads require a file/multipart workflow "
            "that is not yet available in the chat UI."
        )
    action_id = secrets.token_urlsafe(24)
    action = {
        "action_id": action_id,
        "session_id": session_id,
        "operation_id": op["id"],
        "method": op["method"],
        "path": path,
        "query": query,
        "body": body,
        "headers": headers or {},
        "media_type": media_type,
        "summary": op.get("summary", ""),
        "expires_at": time.time() + 600,
    }
    with _pending_lock:
        expired = [key for key, value in _pending_actions.items() if time.time() > value.get("expires_at", 0)]
        for key in expired:
            _pending_actions.pop(key, None)
        if len(_pending_actions) >= 50:
            return err("Too many pending approval requests. Approve, reject or clear an existing request first.")
        _pending_actions[action_id] = action
    return json.dumps({
        "approval_required": True,
        "action_id": action_id,
        "method": action["method"],
        "path": action["path"],
        "summary": action["summary"],
        "query": query,
        "headers": headers or {},
        "body": body,
        "media_type": media_type,
        "expires_in_seconds": 600,
        "note": "The request has NOT been executed. The user must approve it in the chat UI."
    }, default=str)


def execute_pending_action(action_id, session_id):
    scopes = set(re.split(r"[\s,]+", SCOPE.strip().lower()))
    if "management" not in scopes:
        return {
            "ok": False,
            "error": (
                "The configured NINJA_SCOPE does not include 'management'. "
                "Enable the Management scope for this NinjaOne OAuth app and set "
                "NINJA_SCOPE=monitoring management in .env before approving write operations."
            ),
        }
    with _pending_lock:
        action = _pending_actions.get(action_id)
        if not action:
            return {"ok": False, "error": "Approval request not found or already used."}
        if action["session_id"] != session_id:
            return {"ok": False, "error": "Approval request belongs to a different chat session."}
        if time.time() > action["expires_at"]:
            _pending_actions.pop(action_id, None)
            return {"ok": False, "error": "Approval request expired. Ask the agent to prepare it again."}
        _pending_actions.pop(action_id, None)

    try:
        result = http_request(
            action["method"], action["path"], action["query"], action["body"],
            action["media_type"], action.get("headers")
        )
        return {
            "ok": True,
            "operation_id": action["operation_id"],
            "method": action["method"],
            "path": action["path"],
            "result": result,
        }
    except requests.HTTPError as exc:
        response = exc.response
        return {"ok": False, "error": f"NinjaOne HTTP {response.status_code}: {response.text[:3000]}"}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def reject_pending_action(action_id, session_id):
    with _pending_lock:
        action = _pending_actions.get(action_id)
        if not action or action.get("session_id") != session_id:
            return False
        _pending_actions.pop(action_id, None)
        return True


def clear_pending_session(session_id):
    with _pending_lock:
        stale = [key for key, value in _pending_actions.items() if value.get("session_id") == session_id]
        for key in stale:
            _pending_actions.pop(key, None)


def t_call(a):
    op = OPS.get(a.get("operation_id", ""))
    if not op:
        sug = "; ".join(o["id"] for o in IDX.search(a.get("operation_id", ""), 5))
        return err(f"Unknown operation_id. Likely matches: {sug or 'none'}. Use search_api.")
    if is_delete_operation(op["method"], op["path"], op["id"], op.get("summary", "")):
        return err("Delete operations are disabled by policy and cannot be prepared or executed.")

    errors = _argument_errors(op, a)
    if errors:
        return json.dumps({"error": "Request validation failed", "details": errors})

    path = op["path"]
    path_params = a.get("path_params") or {}
    for name in re.findall(r"\{([^}]+)\}", path):
        if name not in path_params:
            return err(f"Missing path_params.{name}")
        path = path.replace("{" + name + "}", quote(str(path_params[name]), safe=""))

    query = a.get("query") or {}
    if op["method"] == "GET":
        rows = fetch(op, path, query, headers)
        result = shape(rows, a.get("where"), a.get("contains"), a.get("group_by"),
                       a.get("fields"), bool(a.get("count_only")), a.get("limit"))
        return json.dumps(result, default=str)[:MAX_CHARS]

    return _prepare_mutation(op, path, query, a.get("body"), a.get("headers"))


DISPATCH = {
    "search_api": t_search,
    "inspect_api": t_inspect,
    "call_api": t_call,
}


ENT = {"organization": "/v2/organizations", "device": "/v2/devices", "end_user": "/v2/user/end-users",
       "technician": "/v2/user/technicians", "user": "/v2/users"}


def t_resolve(a):
    path = ENT.get(a.get("kind"))
    op = op_by_path(path) if path else None
    if not op:
        return err(f"kind must be one of {list(ENT)}")
    name = str(a.get("name", "")).lower().strip()
    scored = []
    for r in fetch(op, path, {}):
        if not isinstance(r, dict):
            continue
        labels = [str(r.get(k, "")) for k in ("name", "systemName", "dnsName", "displayName", "email")]
        labels.append(f'{r.get("firstName", "")} {r.get("lastName", "")}'.strip())
        best = max((difflib.SequenceMatcher(None, name, l.lower()).ratio() + (0.4 if name in l.lower() else 0)
                    for l in labels if l), default=0)
        scored.append((best, r))
    scored.sort(key=lambda x: -x[0])
    matched = [(score, row) for score, row in scored if score > 0.45]
    top = [dict(slim(row), match=round(score, 2)) for score, row in matched[:10]]
    return json.dumps({
        "matches": top,
        "total_matches": len(matched),
        "note": "total_matches counts all matching records; matches contains at most 10 examples"
    }, default=str)


DISPATCH = {"search_api": t_search, "call_api": t_call, "resolve_entity": t_resolve}

TOOLS = [
    {"type": "function", "function": {
        "name": "search_api",
        "description": "Search the NinjaOne OpenAPI catalog for supported GET, POST, PUT and PATCH operations. DELETE and explicit delete operations are excluded.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 12}
        }, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "inspect_api",
        "description": "Inspect an exact operation_id from search_api. Returns path/query parameters, required fields, request-body schema and media type. Call before writes or when arguments are unclear.",
        "parameters": {"type": "object", "properties": {
            "operation_id": {"type": "string"}
        }, "required": ["operation_id"]}}},
    {"type": "function", "function": {
        "name": "call_api",
        "description": "Execute a validated GET request immediately. For POST, PUT or PATCH this ONLY prepares a request for explicit user approval in the chat UI; it never executes the mutation. DELETE and explicit delete actions are blocked. Use inspect_api first for write schemas.",
        "parameters": {"type": "object", "properties": {
            "operation_id": {"type": "string"},
            "path_params": {"type": "object", "additionalProperties": True},
            "query": {"type": "object", "additionalProperties": True},
            "headers": {"type": "object", "additionalProperties": True},
            "body": {},
            "count_only": {"type": "boolean"},
            "group_by": {"type": "string"},
            "where": {"type": "object"},
            "contains": {"type": "string"},
            "fields": {"type": "array", "items": {"type": "string"}},
            "limit": {"type": "integer", "minimum": 1, "maximum": 100}
        }, "required": ["operation_id"]}}},
]


SYSTEM = """You are a reliable NinjaOne API agent. Use the bundled official NinjaOne OpenAPI specification as the source of truth.

For NinjaOne data questions, call tools; never invent records, totals, fields, permissions or endpoint behavior. For greetings and unrelated questions, answer briefly.

API workflow:
- search_api searches the complete supported API catalog, including GET, POST, PUT and PATCH. Use natural-language intent and resource synonyms.
- inspect_api retrieves the exact request schema, required parameters, enums and body fields. Use it whenever a write body or parameter is unclear.
- call_api executes GET operations. It prepares POST, PUT and PATCH operations for user approval; a prepared request has NOT been executed.
- Never claim a mutation succeeded until the UI approval action returns a successful NinjaOne response.
- DELETE operations and any operation explicitly named/described as delete are unavailable. Never attempt to bypass this restriction using POST, PATCH or another method.
- Never call a write operation merely to explore its schema. Use inspect_api first.
- For mutations, resolve target entities and IDs first, validate the exact intended change, then prepare the smallest request body. The user must approve the exact method, path, query and body in the UI.
- If a request is missing required fields or violates an enum/type, correct it using inspect_api. Do not guess.
- GET pagination is automatic only where the OpenAPI parameter schema confirms an integer ID cursor. Do not assume timestamps or opaque cursors are IDs.
- Monitoring scope is read-only. Write operations require the NinjaOne OAuth Management scope and the user’s approval. If the API returns 401/403, explain the actual permission/scope failure.
- Distinguish end-user device access from technician role/organization permissions. Do not equate assigned devices with organization access.
- If a person name resolves ambiguously, show the matching records and ask one concise clarification.
- Report counts only when based on the complete API result or an authoritative total count. State when results are partial or truncated.

Final answers should be concise, factual, and clearly distinguish completed reads, prepared actions awaiting approval, and completed writes."""



def candidates(q):
    hits = IDX.search(q, 3) if IDX else []
    return ("\nLikely endpoints:\n" + "\n".join(describe(o) for o in hits)) if hits else ""


PLAN = re.compile(r"(?m)^\s*\d+[.)]\s")  # numbered list = the model is describing a plan
NUDGE = "Do not describe steps. Call a tool now, using an operation_id from the Likely endpoints list."


# ---------------------------------------------------------------- fast path (no LLM)
FILLER = set("how many count number of total the all we have do are there is in ninjaone ninja ninjarmm "
             "our my a an show me give list what whats tell us".split())
FAST = [  # (noun set, endpoint, label)
    ({"device", "machine", "computer", "endpoint"}, "/v2/devices", "devices"),
    ({"organization", "org", "client", "customer", "company"}, "/v2/organizations", "organizations"),
    ({"technician", "tech"}, "/v2/user/technicians", "technicians"),
    ({"enduser"}, "/v2/user/end-users", "end users"),
    ({"user", "staff", "people"}, "/v2/users", "users"),
]


def row_name(r):
    if not isinstance(r, dict):
        return str(r)
    full = f'{r.get("firstName", "")} {r.get("lastName", "")}'.strip()
    return str(r.get("name") or r.get("systemName") or full or r.get("email") or r.get("id"))


GREET = re.compile(r"^\s*(hi+|hello+|hey+|yo|good (morning|afternoon|evening)|thanks?|thank you|ok(ay)?|help|what can you do)\W*$", re.I)
HELP = ('Hi! I can answer read-only questions about your NinjaOne tenant, for example: "how many devices do we have?", '
        '"list organizations", "how many technicians?", "devices per organization", or "how many devices does Richard have access to?"')

_TXT_CALL = re.compile(r"\b(search_api|call_api|resolve_entity)\s*\(\s*(\{.*\})\s*\)", re.S)
_TAG_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def text_call(txt):
    """Recover structured, JSON-style, or Python-style tool calls printed as plain text."""
    allowed = set(DISPATCH)
    try:
        text = (txt or "").strip()
        # Parse Python-call syntax safely; only literal arguments are accepted.
        for match in re.finditer(r"\b(search_api|call_api|resolve_entity)\s*\(", text):
            start = match.start()
            depth, quote, escaped, end = 0, None, False, None
            for index in range(match.end() - 1, len(text)):
                char = text[index]
                if quote:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == quote:
                        quote = None
                    continue
                if char in ("'", '"'):
                    quote = char
                elif char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                    if depth == 0:
                        end = index + 1
                        break
            if end is None:
                continue
            expression = ast.parse(text[start:end], mode="eval").body
            if not isinstance(expression, ast.Call) or not isinstance(expression.func, ast.Name):
                continue
            name = expression.func.id
            if name not in allowed:
                continue
            if len(expression.args) == 1 and isinstance(expression.args[0], ast.Dict) and not expression.keywords:
                args = ast.literal_eval(expression.args[0])
            elif not expression.args:
                args = {item.arg: ast.literal_eval(item.value) for item in expression.keywords}
            else:
                continue
            if isinstance(args, dict):
                return name, args

        # JSON wrapped in <tool_call>...</tool_call>.
        m = _TAG_CALL.search(text)
        if m:
            data = json.loads(m.group(1))
            name = data.get("name")
            args = data.get("arguments", {})
            if name in allowed and isinstance(args, dict):
                return name, args

        # Legacy JSON-in-parentheses format.
        m = _TXT_CALL.search(text)
        if m:
            args = json.loads(m.group(2))
            if isinstance(args, dict):
                return m.group(1), args
    except (SyntaxError, ValueError, TypeError, json.JSONDecodeError):
        pass
    return None


def exec_tool(name, args):
    try:
        fn = DISPATCH.get(name)
        return fn(args) if fn else err("unknown tool")
    except requests.HTTPError as e:
        return err(f"HTTP {e.response.status_code}: {e.response.text[:200]}")
    except Exception as e:
        return err(f"{type(e).__name__}: {e}")


def text_call_followup(name, out):
    return (f"Result of {name}: {out}\nNow call another tool if needed, otherwise give the final answer "
            "in 1-2 plain sentences, number first.")


NOUN_WORDS = {
    "device", "machine", "computer", "endpoint", "organization", "org", "client", "customer",
    "company", "technician", "tech", "enduser", "user", "staff", "people", "account", "person",
    "role", "alert", "policy", "policies", "patch", "software", "application", "activity",
    "group", "location", "vulnerability", "service", "disk", "volume", "processor",
    "threat", "antivirus", "report", "job", "script",
}
BADCALL = re.compile(r"^\s*(call[\s_]|<tool_call|\{\s*\")", re.I)  # model printed a pseudo tool call
FALLBACK = ("I couldn't turn that into a NinjaOne lookup with the current model. Try a simpler wording, e.g. "
            '"search technician rene", "devices per organization", "how many devices in <organization>", '
            "or switch to a larger model that supports tool calling.")


def _orgs():
    op = op_by_path("/v2/organizations")
    return {o.get("id"): o.get("name") for o in fetch(op, "/v2/organizations", {}) if isinstance(o, dict)} if op else {}


def recipe_per_org(q):
    if not re.search(r"\b(device|machine|computer|endpoint)s?\b.*\b(per|by|each|every)\b.*\b(org|organization|client|customer)", q.lower()):
        return None
    dop = op_by_path("/v2/devices")
    if not dop:
        return None
    devs = [d for d in fetch(dop, "/v2/devices", {}) if isinstance(d, dict)]
    if not devs:
        return "No devices returned by NinjaOne."
    if not any("organizationId" in d for d in devs):
        return "The devices list has no organizationId field, so I can't group it. Fields seen: " + ", ".join(list(devs[0])[:15])
    names, c = _orgs(), Counter(d.get("organizationId") for d in devs)
    lines = [f"{names.get(k) or 'organization ' + str(k)}: {n}" for k, n in c.most_common(40)]
    return f"{len(devs)} devices across {len(c)} organizations:\n" + "\n".join(lines)


KIND_MAP = {"enduser": ["end_user"], "technician": ["technician"], "tech": ["technician"],
            "organization": ["organization"], "org": ["organization"], "client": ["organization"],
            "customer": ["organization"], "device": ["device"], "machine": ["device"],
            "computer": ["device"], "endpoint": ["device"]}


def recipe_list_entity_fields(q):
    """Deterministically list requested name/email fields for NinjaOne people inventories."""
    text = re.sub(r"\s+", " ", q.lower().strip())
    # Correct common typing errors seen in short natural-language requests.
    text = re.sub(r"\b(fisrtname|fristname|firstanme)\b", "firstname", text)
    text = re.sub(r"\b(latsname|lastnamee|lastnme)\b", "lastname", text)
    if not re.search(r"\b(list|show|display|get|fetch|give|provide|who are|what are)\b", text):
        return None

    if re.search(r"\bend[\s-]?users?\b", text):
        kind, path, label = "end_user", "/v2/user/end-users", "end user"
    elif re.search(r"\btechnicians?\b|\btechs?\b", text):
        kind, path, label = "technician", "/v2/user/technicians", "technician"
    elif re.search(r"\b(users?|people|accounts?)\b", text):
        kind, path, label = "user", "/v2/users", "user"
    else:
        return None

    requested = []
    if re.search(r"\b(first\s*name|firstname)\b", text):
        requested.append(("firstName", ("firstName", "first_name", "firstname")))
    if re.search(r"\b(last\s*name|lastname)\b", text):
        requested.append(("lastName", ("lastName", "last_name", "lastname")))
    if re.search(r"\b(email(?:\s+ids?)?s?)\b", text):
        requested.append(("email", ("email", "emailAddress", "email_address")))
    if not requested:
        return None

    op = op_by_path(path)
    if not op:
        return f"The NinjaOne {label}-list endpoint {path} was not found in the loaded API specification."
    rows = fetch(op, path, {})
    output = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        values = []
        for display, aliases in requested:
            value = next((row.get(key) for key in aliases if row.get(key) not in (None, "")), None)
            if value is not None:
                values.append(str(value).strip())
        if values:
            output.append(" ".join(values))
    if not output:
        return (f"NinjaOne returned {len(rows)} {label} record(s), but none contained "
                f"the requested fields ({', '.join(name for name, _ in requested)}).")
    return f"{len(output)} {label}(s):\n" + "\n".join(
        f"{index}. {value}" for index, value in enumerate(output, 1)
    )


def recipe_list_organizations(q):
    """List organization names for natural-language organization inventory requests."""
    text = re.sub(r"\s+", " ", q.lower().strip())
    asks_for_names = bool(re.search(r"\b(name|names)\b", text))
    asks_list = bool(re.search(
        r"\b(list|show|display|get|fetch|give|provide|what are|which are|can i get|could i get)\b",
        text,
    ))
    asks_count = bool(re.search(r"\b(how many|count|number of|total)\b", text))
    has_organization = bool(re.search(r"\b(organizations?|orgs?|clients?|customers?)\b", text))
    if not (has_organization and (asks_list or asks_for_names) and not asks_count):
        return None

    path = "/v2/organizations"
    op = op_by_path(path)
    if not op:
        return "The NinjaOne organization-list endpoint was not found in the loaded API specification."
    rows = fetch(op, path, {})
    names = [
        str(row.get("name")).strip()
        for row in rows
        if isinstance(row, dict) and row.get("name") not in (None, "")
    ]
    if not names:
        return f"NinjaOne returned {len(rows)} organization record(s), but no organization names were present."
    return f"{len(names)} organization(s):\n" + "\n".join(
        f"{index}. {name}" for index, name in enumerate(names, 1)
    )


def recipe_list_technician_names(q):
    """Return technician first/last names directly for simple inventory requests."""
    text = re.sub(r"\s+", " ", q.lower().strip())
    has_technicians = bool(re.search(r"\btechnicians?\b|\btechs?\b", text))
    asks_list = bool(re.search(r"\b(list|show|display|get|fetch|give|provide|what are|who are)\b", text))
    asks_names = bool(
        re.search(r"\b(first\s*name|firstname)\b", text)
        and re.search(r"\b(last\s*name|lastname)\b", text)
    )
    if not (has_technicians and asks_list and asks_names):
        return None

    path = "/v2/user/technicians"
    op = op_by_path(path)
    if not op:
        return "The NinjaOne technician-list endpoint was not found in the loaded API specification."
    rows = fetch(op, path, {})
    names = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        first = row.get("firstName") or row.get("first_name") or row.get("firstname") or ""
        last = row.get("lastName") or row.get("last_name") or row.get("lastname") or ""
        full = " ".join(str(value).strip() for value in (first, last) if value not in (None, ""))
        if full:
            names.append(full)
    if not names:
        return (
            f"NinjaOne returned {len(rows)} technician record(s), but none contained "
            "firstName and lastName fields."
        )
    return f"{len(names)} technician(s):\n" + "\n".join(f"{index}. {name}" for index, name in enumerate(names, 1))


def recipe_count_named(q):
    """Count matching user records for natural questions such as 'how many users with name Rene?'."""
    text = q.lower().strip().rstrip("?.! ")
    match = re.search(
        r"\b(?:users?|people|accounts?)\b.*?\b(?:with|having)\s+(?:the\s+)?name\s+(.+)$",
        text
    )
    if not match:
        return None
    name = match.group(1).strip().strip("\"'")
    if not name or len(name) < 2:
        return None
    result = json.loads(t_resolve({"kind": "user", "name": name}))
    count = result.get("total_matches")
    if count is None:
        count = len(result.get("matches", []))
    examples = result.get("matches", [])
    if count == 0:
        return f'No NinjaOne users matched the name "{name}".'
    details = ", ".join(
        row_name(row) + (f' <{row["email"]}>' if row.get("email") else "")
        for row in examples[:5]
    )
    suffix = f" Examples: {details}." if details else ""
    return f'{count} NinjaOne user record(s) matched the name "{name}".{suffix}'


def recipe_find(q):
    t = re.sub(r"end[\s-]?users?", "enduser", q.lower().strip().rstrip("?.! "))
    m = re.match(r"^(?:please\s+)?(?:search|find|look\s*up|lookup|locate|who\s+is)\s+(?:for\s+)?(?:(?:the|an?|me)\s+)*"
                 r"(?:(enduser|technician|tech|user|account|person|organization|org|client|customer|device|machine|computer|endpoint)s?\s+)?"
                 r"(?:account\s+)?(?:(?:with|by)\s+(?:the\s+)?name\s+|named\s+|called\s+|name\s+)?(.+)$", t)
    if not m:
        return None
    kindw, name = m.group(1), m.group(2).strip()
    if not name or re.search(r"\b(how|many|count|per|each|which|what|all)\b", name):
        return None
    kinds = KIND_MAP.get(kindw, ["technician", "end_user"])
    found = []
    for k in kinds:
        for row in json.loads(t_resolve({"kind": k, "name": name})).get("matches", []):
            found.append((k, row))
    if not found:
        return f'No match for "{name}" among {", ".join(k.replace("_", " ") for k in kinds)}.'
    found.sort(key=lambda x: -x[1].get("match", 0))
    lines = [f'{k.replace("_", " ")}: {row_name(r)}' + (f' <{r["email"]}>' if r.get("email") else "") + f' (id {r.get("id")})'
             for k, r in found[:10]]
    return f'Found {len(found)} match(es) for "{name}":\n' + "\n".join(lines)


def recipe_devices_in(q):
    m = re.search(r"\b(?:device|machine|computer|endpoint)s?\s+(?:in|of|at|under|for)\s+(?:the\s+)?"
                  r"(?:organization\s+|org\s+|client\s+)?(.+?)\s*[?.!]*$", q.lower())
    if not m or re.search(r"\b(how|many|count|per|each|which|what|all)\b", m.group(1)):
        return None
    dop = op_by_path("/v2/devices")
    ms = json.loads(t_resolve({"kind": "organization", "name": m.group(1)})).get("matches", [])
    if not dop or not ms or ms[0].get("match", 0) < 0.9 or (len(ms) > 1 and ms[1]["match"] > ms[0]["match"] - 0.2):
        return None
    devs = [d for d in fetch(dop, "/v2/devices", {}) if isinstance(d, dict)]
    if not any("organizationId" in d for d in devs):
        return None
    n = sum(1 for d in devs if d.get("organizationId") == ms[0].get("id"))
    return f'{n} devices in {ms[0].get("name")}.'


def fast_path(q):
    if GREET.match(q):
        return HELP
    q1 = re.sub(r"end[\s-]?users?", "enduser", q.lower())
    words = re.findall(r"[a-z]+", q1)
    cw = {(w[:-1] if len(w) > 3 and w.endswith("s") else w) for w in words if w not in FILLER}
    for nouns, path, label in FAST:
        if cw and cw <= nouns:
            op = op_by_path(path)
            if not op:
                break
            rows = fetch(op, path, {})
            if set(words) & {"many", "count", "number", "total"}:
                return f"{len(rows)} {label} in NinjaOne."
            names = [row_name(r) for r in rows[:50]]
            return f"{len(rows)} {label}: " + ", ".join(names) + (" ..." if len(rows) > 50 else "")
    try:
        for rec in (recipe_list_entity_fields, recipe_list_organizations, recipe_list_technician_names, recipe_per_org, recipe_count_named, recipe_find, recipe_devices_in):
            out = rec(q)
            if out:
                return out
    except requests.HTTPError as e:
        return f"NinjaOne request failed (HTTP {e.response.status_code}). Check the credentials and NINJA_BASE_URL region."
    return None


# ---------------------------------------------------------------- agent loop
llm = None


def clean(t):
    return re.sub(r"<think>.*?</think>", "", t or "", flags=re.S).strip()


def ask(q, history):
    fp = fast_path(q)
    if fp:
        return fp
    msgs = [{"role": "system", "content": SYSTEM + candidates(q) + ("\n/no_think" if NO_THINK else "")}] + history[-6:]
    msgs.append({"role": "user", "content": q})
    called = nudged = False
    steps_text = 0
    for _ in range(MAX_STEPS):
        m = llm.chat.completions.create(model=LLM_MODEL, messages=msgs, tools=TOOLS, temperature=0).choices[0].message
        if not m.tool_calls:
            tc = text_call(m.content)
            if tc and steps_text < 3:
                steps_text += 1
                called = True
                msgs += [{"role": "assistant", "content": m.content},
                         {"role": "user", "content": text_call_followup(tc[0], exec_tool(*tc))}]
                continue
            txt = clean(m.content)
            tool_intent = bool(re.search(
                r"\b(?:i(?:\s+will|\x27ll)|going\s+to)\s+(?:call|use|invoke)\b|\b(?:retrieve|fetch|look\s+up)\s+(?:the\s+)?(?:list|records|data)\b",
                txt,
                re.I,
            ))
            question_words = set(re.findall(r"[a-z]+", q.lower()))
            lookup_intent = not called and any(
                word in NOUN_WORDS or (word.endswith("s") and word[:-1] in NOUN_WORDS)
                for word in question_words
            )
            bad = tool_intent or (
                not called and bool(PLAN.search(txt) or BADCALL.search(txt) or lookup_intent)
            )
            if bad and not nudged:
                nudged = True
                msgs += [{"role": "assistant", "content": txt}, {"role": "user", "content": NUDGE}]
                continue
            return FALLBACK if bad else txt
        called = True
        msgs.append(m.model_dump(exclude_none=True))
        for tc in m.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
                fn = DISPATCH.get(tc.function.name)
                out = fn(args) if fn else err("unknown tool")
            except requests.HTTPError as e:
                out = err(f"HTTP {e.response.status_code}: {e.response.text[:200]}")
            except Exception as e:  # keep the loop alive; the model sees the hint
                out = err(f"{type(e).__name__}: {e}")
            msgs.append({"role": "tool", "tool_call_id": tc.id, "content": out})
    return "I couldn't finish within the step limit. Try a more specific question."


def init():
    global OPS, IDX, llm
    if not (CID and CSECRET):
        sys.exit("Set NINJA_CLIENT_ID and NINJA_CLIENT_SECRET (and NINJA_BASE_URL for your region).")
    OPS = load_ops()
    IDX = Index(OPS)
    llm = OpenAI(base_url=LLM_URL, api_key=os.getenv("LLM_API_KEY", "local"))
    print(f"Loaded {len(OPS)} operations from spec.", file=sys.stderr)


def main():
    init()
    if len(sys.argv) > 1:
        print(ask(" ".join(sys.argv[1:]), []))
        return
    history = []
    while True:
        try:
            q = input("\nask> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("", "exit", "quit"):
            if q:
                break
            continue
        a = ask(q, history)
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        print(a)


if __name__ == "__main__":
    main()
