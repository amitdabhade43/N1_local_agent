#!/usr/bin/env python3
"""NinjaOne API agent driven by the bundled OpenAPI specification.

GET and spec-confirmed read-only POST queries execute directly. State-changing
POST/PUT/PATCH requests are schema-validated and prepared for explicit UI approval.
DELETE and explicit delete actions are blocked.
The local model selects operations; the OpenAPI catalog validates their contracts.
"""
import ast, difflib, json, math, os, re, sys, time, threading, secrets, tempfile
from collections import Counter
from urllib.parse import quote

import requests
from openai import OpenAI
from ninja_openapi import (build_operations, describe_operation, operation_document, validate_payload, is_delete_operation, operation_field_entries, build_field_catalog, compare_spec_documents, schema_field_entries, expand_refs)

SPEC_URL = os.getenv("NINJA_SPEC_URL", "https://app.ninjarmm.com/apidocs/NinjaRMM-API-v2.json")
BASE = os.getenv("NINJA_BASE_URL", "https://eu.ninjarmm.com").rstrip("/")  # use your region host
CID, CSECRET = os.getenv("NINJA_CLIENT_ID", ""), os.getenv("NINJA_CLIENT_SECRET", "")
SCOPE = os.getenv("NINJA_SCOPE", "monitoring")
TOKEN_URL = os.getenv("NINJA_TOKEN_URL", "").strip()
LLM_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:14b")
NO_THINK = os.getenv("NO_THINK", "0") == "1"  # set 1 for Qwen3 to skip thinking (faster)
MAX_STEPS, MAX_ROWS, MAX_CHARS, TTL = 6, 25, 6000, 60

OPS, IDX, SPEC_DATA, FIELD_INDEX, FIELD_SEARCH_INDEX = {}, None, {}, [], []
SCHEMA_SOURCES, SCHEMA_INDEX, SPEC_SOURCE_REPORT = {}, {}, {}
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


def _spec_counts(spec):
    paths = spec.get("paths") or {}
    methods = {"get", "post", "put", "patch", "delete"}
    operation_count = sum(
        1 for item in paths.values() if isinstance(item, dict)
        for method, value in item.items()
        if method.lower() in methods and isinstance(value, dict)
    )
    schemas = spec.get("components", {}).get("schemas") or {}
    return {
        "operations": operation_count,
        "paths": len(paths),
        "schemas": len(schemas),
    }


def _validate_spec_document(spec, previous=None):
    if not isinstance(spec, dict) or not str(spec.get("openapi", "")).startswith("3."):
        raise ValueError("Downloaded NinjaOne specification is not a valid OpenAPI 3.x document.")
    counts = _spec_counts(spec)
    if counts["operations"] < 100 or counts["schemas"] < 100:
        raise ValueError(
            "Downloaded NinjaOne specification looks incomplete: "
            f'{counts["operations"]} operations, {counts["schemas"]} schemas.'
        )
    if previous:
        old = _spec_counts(previous)
        if counts["operations"] < old["operations"] or counts["schemas"] < old["schemas"]:
            raise ValueError(
                "Refusing to replace the bundled NinjaOne specification with a smaller catalog. "
                f'Current={old}; downloaded={counts}. Review the upstream change manually.'
            )
    return counts


def _prepare_field_search_index(fields):
    """Pre-tokenize schema metadata once so field lookup stays fast during chat."""
    prepared = []
    for field in fields:
        haystack = " ".join(str(field.get(key, "")) for key in (
            "name", "path", "description", "schema_name", "operation_id",
            "operation_path", "location", "enum", "schema_source",
        ))
        prepared.append((
            field,
            " ".join(toks(haystack)),
            set(toks(haystack)),
            " ".join(toks(field.get("name", ""))),
        ))
    return prepared


def load_ops():
    """Load the bundled OpenAPI document and build the complete safe operation catalog."""
    global SPEC_DATA, FIELD_INDEX, FIELD_SEARCH_INDEX, SCHEMA_SOURCES, SCHEMA_INDEX, SPEC_SOURCE_REPORT
    root = os.path.dirname(os.path.abspath(__file__))
    cache = os.path.join(root, "ninja_spec.json")
    yaml_cache = os.path.join(root, "ninjaOne-API-core-resources.yaml")
    refresh_spec = os.getenv("NINJA_SPEC_REFRESH", "0").strip().lower() in {"1", "true", "yes"}
    previous = None
    if os.path.exists(cache):
        with open(cache, "r", encoding="utf-8") as handle:
            previous = json.load(handle)
    if previous is not None and not refresh_spec:
        spec = previous
        _validate_spec_document(spec)
    else:
        response = requests.get(SPEC_URL, timeout=60)
        response.raise_for_status()
        spec = response.json()
        _validate_spec_document(spec, previous)
        fd, temporary_path = tempfile.mkstemp(
            prefix="ninja_spec.", suffix=".tmp", dir=os.path.dirname(cache)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(spec, handle, ensure_ascii=False, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, cache)
        except Exception:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass
            raise
    SPEC_DATA = spec
    operations = build_operations(spec)

    # The JSON document remains authoritative for executable operations and
    # request validation. YAML enriches schema/field discovery only.
    supplemental_specs = []
    yaml_spec = None
    if os.path.isfile(yaml_cache):
        try:
            import yaml
            with open(yaml_cache, "r", encoding="utf-8") as handle:
                yaml_spec = yaml.safe_load(handle)
            if not isinstance(yaml_spec, dict) or not str(yaml_spec.get("openapi", "")).startswith("3."):
                raise ValueError("Supplemental YAML is not a valid OpenAPI 3.x document.")
            supplemental_specs.append(("yaml", yaml_spec))
        except Exception as exc:
            raise RuntimeError(
                f"Could not load supplemental NinjaOne YAML schema catalog ({yaml_cache}): "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    SCHEMA_SOURCES = {"json": spec}
    if yaml_spec is not None:
        SCHEMA_SOURCES["yaml"] = yaml_spec
    SPEC_SOURCE_REPORT = {
        "primary": "ninja_spec.json",
        "supplemental": "ninjaOne-API-core-resources.yaml" if yaml_spec is not None else None,
        "comparison": compare_spec_documents(spec, yaml_spec) if yaml_spec is not None else None,
    }
    SCHEMA_INDEX = {}
    for source_name, document in SCHEMA_SOURCES.items():
        for schema_name, schema in (document.get("components", {}).get("schemas") or {}).items():
            SCHEMA_INDEX[(source_name, schema_name)] = expand_refs(schema, document)

    FIELD_INDEX = build_field_catalog(spec, operations, supplemental_specs=supplemental_specs)
    FIELD_SEARCH_INDEX = _prepare_field_search_index(FIELD_INDEX)
    return operations


SYN = {
    "count": "list", "many": "list", "total": "list", "number": "list",
    "machine": "device", "computer": "device", "endpoint": "device", "laptop": "device",
    "server": "device", "workstation": "device", "asset": "device",
    "client": "organization", "customer": "organization", "org": "organization",
    "company": "organization", "tenant": "organization",
    "tech": "technician", "admin": "technician", "employee": "user",
    "enduser": "end user", "staff": "user", "person": "user",
    "restart": "reboot", "reboot": "restart", "shutdown": "reboot",
    "patching": "patch", "virus": "antivirus", "malware": "antivirus",
    "threat": "antivirus",
    "case": "ticket", "tickets": "ticket", "invoice": "billing",
    "automation": "script", "run": "execute script",
}


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
    access = "read-only" if o.get("read_only") else "approval-gated write"
    return f'{o["id"]}: {o["method"]} {o["path"]} - {o["summary"]} [{access}] | params: {ps or "none"}{body_hint}'


# ---------------------------------------------------------------- HTTP
def token():
    if _tok["v"] and time.time() < _tok["exp"] - 30:
        return _tok["v"]
    token_url = TOKEN_URL or f"{BASE}/oauth/token"
    r = requests.post(token_url, timeout=30, data={
        "grant_type": "client_credentials", "client_id": CID,
        "client_secret": CSECRET, "scope": SCOPE})
    # Keep compatibility with older regional tenants that still expose the
    # legacy token route. The current Public API client-credentials flow uses /oauth/token.
    if r.status_code in {404, 405} and not TOKEN_URL:
        r = requests.post(f"{BASE}/ws/oauth/token", timeout=30, data={
            "grant_type": "client_credentials", "client_id": CID,
            "client_secret": CSECRET, "scope": SCOPE})
    r.raise_for_status()
    j = r.json()
    _tok.update(v=j["access_token"], exp=time.time() + j.get("expires_in", 3600))
    return _tok["v"]


def _serialize_query(query):
    """Serialize JSON tool arguments into HTTP query values consistently."""
    result = {}
    for key, value in (query or {}).items():
        if isinstance(value, bool):
            result[key] = "true" if value else "false"
        elif isinstance(value, list):
            result[key] = [
                ("true" if item else "false") if isinstance(item, bool) else item
                for item in value
            ]
        else:
            result[key] = value
    return result


def http_request(method, path, query=None, body=None, media_type="application/json",
                 extra_headers=None, approved=False, read_only=False):
    """Execute a validated request; state-changing methods require a consumed approval."""
    method = method.upper()
    if method == "DELETE" or re.search(r"(^|[/_-])delete($|[/_-])", path.lower()):
        raise ValueError("Delete execution is disabled by policy.")
    if method not in {"GET", "POST", "PUT", "PATCH"}:
        raise ValueError(f"Unsupported HTTP method: {method}")
    if method in {"POST", "PUT", "PATCH"} and not (approved or read_only):
        raise ValueError("State-changing API requests must pass through explicit user approval.")

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
    kwargs = {"params": _serialize_query(query), "timeout": 90, "headers": headers}
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
        for k in ("results", "data", "items", "content", "records", "tickets", "activities"):
            if isinstance(data.get(k), list):
                return data[k]
        return [data]
    return []


class FetchRows(list):
    def __init__(self, values=(), complete=True, note=""):
        super().__init__(values)
        self.complete = complete
        self.note = note


def _response_list_field(op):
    preferred = ("results", "data", "items", "records", "activities", "tickets")
    for response in (op.get("responses") or {}).values():
        if not isinstance(response, dict):
            continue
        schema = ((response.get("content") or {}).get("application/json") or {}).get("schema") or {}
        properties = schema.get("properties") or {}
        arrays = [name for name, prop in properties.items()
                  if isinstance(prop, dict) and prop.get("type") == "array"]
        for name in preferred:
            if name in arrays:
                return name
        if len(arrays) == 1:
            return arrays[0]
    return None


def _extract_rows(op, response):
    if isinstance(response, list):
        return response
    field = _response_list_field(op)
    if field and isinstance(response, dict) and isinstance(response.get(field), list):
        return response[field]
    return unwrap(response)


def _opaque_cursor_spec(op):
    params = {p["name"]: p for p in op.get("params", [])}
    after = params.get("after") or {}
    after_schema = after.get("schema") or {}
    if after_schema.get("type") != "string" or "nextCursor" not in after.get("description", "") + after_schema.get("description", ""):
        return None
    for response in (op.get("responses") or {}).values():
        if not isinstance(response, dict):
            continue
        schema = ((response.get("content") or {}).get("application/json") or {}).get("schema") or {}
        properties = schema.get("properties") or {}
        page_info = (properties.get("pageInfo") or {}).get("properties") or {}
        if "nextCursor" not in page_info:
            continue
        list_fields = [name for name, prop in properties.items()
                       if isinstance(prop, dict) and prop.get("type") == "array"]
        if list_fields:
            return list_fields[0]
    return None


def _has_unhandled_pagination(op):
    params = {p["name"]: p for p in op.get("params", [])}
    if "pageSize" not in params:
        return False
    cursor_names = {"after", "olderThan", "newerThan", "before"}
    return bool(cursor_names.intersection(params)) and not (
        params.get("after", {}).get("schema", {}).get("type") == "integer"
        and "pageSize" in params
    ) and not _opaque_cursor_spec(op)


def fetch(op, path, query, extra_headers=None):
    """Fetch all pages only when request and response schemas define a safe cursor."""
    params = {p["name"]: p for p in op.get("params", [])}
    after_schema = params.get("after", {}).get("schema", {})
    if "after" in params and "pageSize" in params and after_schema.get("type") == "integer":
        q, rows, last = dict(query), [], None
        q.setdefault("pageSize", 1000)
        for _ in range(200):
            page = _extract_rows(op, http_get(path, dict(q), extra_headers))
            if not page:
                break
            rows += page
            nxt = page[-1].get("id") if isinstance(page[-1], dict) else None
            if nxt is None:
                raise RuntimeError("NinjaOne returned a paginated result without an id cursor; refusing an incomplete total.")
            if nxt == last:
                raise RuntimeError("NinjaOne pagination cursor did not advance; refusing an incomplete total.")
            last, q["after"] = nxt, nxt
        else:
            raise RuntimeError("NinjaOne pagination exceeded 200 pages; refusing to report a possibly incomplete total.")
        return FetchRows(rows)

    list_field = _opaque_cursor_spec(op)
    if list_field:
        q, rows = dict(query), []
        page_schema = params.get("pageSize", {}).get("schema", {})
        q.setdefault("pageSize", page_schema.get("default") or 100)
        cursor = q.get("after")
        for _ in range(200):
            result = http_get(path, dict(q), extra_headers)
            if not isinstance(result, dict) or not isinstance(result.get(list_field), list):
                return FetchRows(unwrap(result), complete=False,
                                 note="The response did not match the documented cursor-paginated schema.")
            page = result[list_field]
            if not page:
                break
            rows.extend(page)
            page_info = result.get("pageInfo") or {}
            next_cursor = page_info.get("nextCursor")
            has_more = page_info.get("hasMore")
            if has_more is False:
                break
            if not next_cursor:
                if has_more is True or len(page) >= int(q["pageSize"]):
                    raise RuntimeError("NinjaOne reported more pages but did not provide nextCursor.")
                break
            if next_cursor == cursor:
                raise RuntimeError("NinjaOne opaque pagination cursor did not advance.")
            cursor = next_cursor
            q["after"] = cursor
        else:
            raise RuntimeError("NinjaOne cursor pagination exceeded 200 pages; refusing an incomplete total.")
        return FetchRows(rows)

    rows = _extract_rows(op, http_get(path, query, extra_headers))
    if _has_unhandled_pagination(op):
        return FetchRows(
            rows,
            complete=False,
            note="This endpoint exposes paging parameters, but the published response schema does not define a safe automatic cursor. The result is the returned page only.",
        )
    return FetchRows(rows)



def op_by_path(path):
    return next((o for o in OPS.values() if o["method"] == "GET" and o["path"] == path), None)


# ---------------------------------------------------------------- shaping
_SENSITIVE_FIELD_RE = re.compile(
    r"(?:password|passwd|secret|token|credential|private.?key|authorization|api.?key|access.?key)",
    re.I,
)


def slim(value, field_name=""):
    """Preserve every documented response field while redacting credential-like values."""
    if _SENSITIVE_FIELD_RE.search(str(field_name or "")):
        return "[redacted]"
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if _SENSITIVE_FIELD_RE.search(str(key)):
                result[key] = "[redacted]"
                continue
            if isinstance(item, list):
                result[key + "_count"] = len(item)
                result[key] = [slim(child, key) for child in item]
            else:
                if isinstance(item, dict) and item.get("name") not in (None, ""):
                    result[key + "_name"] = str(item["name"])
                result[key] = slim(item, key)
        return result
    if isinstance(value, list):
        return [slim(item, field_name) for item in value]
    return value


def _partial_note(rows):
    """Return a user-facing warning when an API result is not known to be complete."""
    if getattr(rows, "complete", True) is False:
        detail = getattr(rows, "note", "") or "The API returned only a partial result."
        return f" Partial data only: {detail}"
    return ""


def _field_value(record, field_path):
    """Read a top-level or dotted nested field path without guessing aliases."""
    value = record
    for part in str(field_path).split("."):
        if isinstance(value, list) and part == "[]":
            continue
        if isinstance(value, dict):
            value = value.get(part)
        elif isinstance(value, list):
            try:
                value = value[int(part)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return value


def shape(rows, where=None, contains=None, group_by=None, fields=None, count_only=False, limit=MAX_ROWS):
    if where:
        rows = [r for r in rows if isinstance(r, dict) and
                all(str(_field_value(r, k)).lower() == str(v).lower() for k, v in where.items())]
    if contains:
        c = str(contains).lower()
        rows = [r for r in rows if c in json.dumps(r).lower()]
    out = {"total": len(rows)}
    if count_only:
        return out
    if group_by:
        def g(r):
            v = _field_value(r, group_by) if isinstance(r, dict) else None
            return str(v.get("name") if isinstance(v, dict) else v)
        out["groups"] = dict(Counter(g(r) for r in rows).most_common(30))
        return out
    limit = min(int(limit or MAX_ROWS), 100)
    out["rows"] = [
        ({field: _field_value(r, field) for field in fields} if fields and isinstance(r, dict) else slim(r))
        for r in rows[:limit]
    ]
    if len(rows) > limit:
        out["truncated"] = f"showing {limit} of {len(rows)}; narrow with where/contains/fields"
    return out


# ---------------------------------------------------------------- tools
def err(msg):
    return json.dumps({"error": msg})


def t_search(a):
    hits = IDX.search(a.get("query", ""), max(1, min(12, int(a.get("limit", 8)))))
    return "\n".join(describe(o) for o in hits) or err("no match; try different keywords")


def t_search_fields(a):
    """Search every field in component schemas and operation request/response contracts."""
    query = str(a.get("query", "")).strip()
    if not query:
        return err("Provide a field name, description, or concept to search.")
    limit = max(1, min(50, int(a.get("limit", 20))))
    schema_filter = str(a.get("schema_name", "")).strip().casefold()
    operation_filter = str(a.get("operation_id", "")).strip().casefold()
    source_filter = str(a.get("schema_source", "")).strip().casefold()
    qnorm = " ".join(toks(query))
    qtokens = set(toks(query))
    ranked = []
    for field, normalized, field_tokens, name_norm in FIELD_SEARCH_INDEX:
        if schema_filter and schema_filter not in str(field.get("schema_name", "")).casefold():
            continue
        if operation_filter and operation_filter not in str(field.get("operation_id", "")).casefold():
            continue
        if source_filter and source_filter != str(field.get("schema_source", "json" if field.get("source") == "component_schema" else "")).casefold():
            continue
        overlap = len(qtokens.intersection(field_tokens))
        score = overlap / max(len(qtokens), 1)
        if qnorm == name_norm:
            score += 5
        elif qnorm and qnorm in normalized:
            score += 2.5
        elif name_norm and (name_norm in qnorm or qnorm in name_norm):
            score += 1.5
        if name_norm:
            score += 0.8 * difflib.SequenceMatcher(None, qnorm, name_norm).ratio()
        if score > 0.25:
            ranked.append((score, field))
    ranked.sort(key=lambda item: (-item[0], item[1].get("name", ""), item[1].get("path", "")))
    comparison = SPEC_SOURCE_REPORT.get("comparison") or {}
    comparison_summary = {
        "primary_version": comparison.get("primary_version"),
        "supplemental_version": comparison.get("supplemental_version"),
        "primary_operations": comparison.get("primary_operations", 0),
        "supplemental_operations": comparison.get("supplemental_operations", 0),
        "matching_operation_ids": comparison.get("matching_operation_ids", 0),
        "operation_route_conflicts": comparison.get("operation_route_conflicts", [])[:25],
        "supplemental_only_schemas": comparison.get("supplemental_only_schemas", [])[:25],
        "primary_only_schemas": comparison.get("primary_only_schemas", [])[:25],
        "schema_definition_conflicts": comparison.get("schema_definition_conflicts", [])[:25],
    } if comparison else None
    return json.dumps({
        "query": query,
        "total_matches": len(ranked),
        "matches": [{**field, "match_score": round(score, 3)} for score, field in ranked[:limit]],
        "schema_sources": {
            "primary": SPEC_SOURCE_REPORT.get("primary"),
            "supplemental": SPEC_SOURCE_REPORT.get("supplemental"),
            "comparison": comparison_summary,
        },
        "note": "These are documented schema definitions, not live record values. JSON is authoritative for executable API contracts; YAML is supplemental schema discovery. Use inspect_schema for a complete component schema, inspect_api for an operation contract, and call_api for live NinjaOne values.",
    }, default=str)


def t_inspect_schema(a):
    """Inspect a complete component schema from the primary or supplemental catalog."""
    name = str(a.get("schema_name", "")).strip()
    source = str(a.get("schema_source", "")).strip().casefold()
    if not name:
        return err("Provide schema_name. Use search_fields to discover schema names.")
    matches = [
        (src, schema) for (src, schema_name), schema in SCHEMA_INDEX.items()
        if schema_name.casefold() == name.casefold() and (not source or src == source)
    ]
    if not matches:
        return err(f"Schema '{name}' was not found in the requested specification source.")
    if len(matches) > 1 and not source:
        return json.dumps({
            "schema_name": name,
            "available_sources": [src for src, _ in matches],
            "note": "This schema exists in multiple specifications. Call inspect_schema with schema_source='json' or 'yaml' to inspect a specific definition.",
        })
    selected_source, schema = matches[0]
    fields = schema_field_entries(schema, name)
    offset = max(0, int(a.get("offset", 0)))
    limit = max(1, min(100, int(a.get("limit", 80))))
    page = fields[offset:offset + limit]
    next_offset = offset + len(page)
    return json.dumps({
        "schema_name": name,
        "schema_source": selected_source,
        "schema_definition": schema,
        "schema_fields_total": len(fields),
        "schema_fields_offset": offset,
        "schema_fields": page,
        "schema_fields_next_offset": next_offset if next_offset < len(fields) else None,
        "schema_fields_complete": next_offset >= len(fields),
        "note": "Schema definitions describe documented fields and constraints, not live tenant values.",
    }, default=str)


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
    details = describe_operation(operation, include_schema=False)
    fields = operation_field_entries(operation)
    field_filter = str(a.get("field_filter", "")).strip().casefold()
    if field_filter:
        fields = [
            field for field in fields
            if field_filter in " ".join(str(field.get(key, "")) for key in (
                "name", "path", "description", "source", "location", "enum",
            )).casefold()
        ]
    offset = max(0, int(a.get("offset", 0)))
    limit = max(1, min(100, int(a.get("limit", 80))))
    details["schema_fields_total"] = len(fields)
    details["schema_fields_offset"] = offset
    details["schema_fields"] = fields[offset:offset + limit]
    next_offset = offset + len(details["schema_fields"])
    details["schema_fields_next_offset"] = next_offset if next_offset < len(fields) else None
    details["schema_fields_complete"] = next_offset >= len(fields)
    details["schema_fields_note"] = (
        "All field definitions include request parameters, request-body properties, and response properties. "
        "If schema_fields_next_offset is not null, call inspect_api again with that offset to retrieve the next page."
    )
    return json.dumps(details, default=str)


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

    for name, value in headers.items():
        definition = header_defs.get(name.lower())
        if not definition:
            errors.append(f"Header '{name}' is not declared by the OpenAPI operation.")
        else:
            errors.extend(
                f"headers.{name}: {e}"
                for e in validate_payload(definition.get("schema") or {}, value, SPEC_DATA)
            )
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

    try:
        result = http_request(
            action["method"], action["path"], action["query"], action["body"],
            action["media_type"], action.get("headers"), approved=True
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


def _has_cursor_post_schema(op):
    body_schema = ((op.get("request_body") or {}).get("schema") or {})
    body_props = body_schema.get("properties") or {}
    if "lastCursorId" not in body_props:
        return False
    for response in (op.get("responses") or {}).values():
        if not isinstance(response, dict):
            continue
        schema = ((response.get("content") or {}).get("application/json") or {}).get("schema") or {}
        props = schema.get("properties") or {}
        metadata_props = (props.get("metadata") or {}).get("properties") or {}
        if "data" in props and "lastCursorId" in metadata_props:
            return True
    return False


def fetch_cursor_post(op, path, query, body, headers):
    """Follow an OpenAPI-described lastCursorId in request and response schemas."""
    request_body = dict(body or {})
    page_size = int(request_body.get("pageSize") or 100)
    request_body["pageSize"] = page_size
    cursor = request_body.get("lastCursorId")
    rows = []
    for _ in range(200):
        result = http_request(
            op["method"], path, query, dict(request_body),
            (op.get("request_body") or {}).get("media_type") or "application/json",
            headers, read_only=True,
        )
        if not isinstance(result, dict) or not isinstance(result.get("data"), list):
            return FetchRows(
                unwrap(result), complete=False,
                note="The response did not match the documented ticket cursor schema."
            )
        page = result["data"]
        if not page:
            break
        rows.extend(page)
        next_cursor = (result.get("metadata") or {}).get("lastCursorId")
        if next_cursor is None:
            if len(page) < page_size:
                break
            raise RuntimeError("NinjaOne ticket pagination returned a full page without lastCursorId.")
        if next_cursor == cursor:
            if len(page) < page_size:
                break
            raise RuntimeError("NinjaOne ticket pagination cursor did not advance.")
        cursor = next_cursor
        request_body["lastCursorId"] = cursor
    else:
        raise RuntimeError("NinjaOne ticket pagination exceeded 200 pages; refusing an incomplete total.")
    return rows


def t_call(a):
    op = OPS.get(a.get("operation_id", ""))
    if not op:
        sug = "; ".join(o["id"] for o in IDX.search(a.get("operation_id", ""), 5))
        return err(f"Unknown operation_id. Likely matches: {sug or 'none'}. Use search_api.")
    if is_delete_operation(op["method"], op["path"], op["id"], op.get("summary", "")):
        return err("Delete operations are disabled by policy and cannot be prepared or executed.")
    if not op.get("read_only"):
        media_type = (op.get("request_body") or {}).get("media_type")
        if media_type and media_type not in {"application/json", "application/x-www-form-urlencoded"}:
            return err(
                f"This operation requires {media_type}; file/multipart transport is not yet enabled. "
                "No request was sent."
            )

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
    headers = a.get("headers") or {}
    if op.get("read_only"):
        if op["method"] == "GET":
            rows = fetch(op, path, query, headers)
        else:
            if _has_cursor_post_schema(op):
                rows = fetch_cursor_post(op, path, query, a.get("body"), headers)
            else:
                media_type = (op.get("request_body") or {}).get("media_type") or "application/json"
                raw_result = http_request(
                    op["method"], path, query, a.get("body"), media_type, headers, read_only=True
                )
                rows = unwrap(raw_result)
        result = shape(rows, a.get("where"), a.get("contains"), a.get("group_by"),
                       a.get("fields"), bool(a.get("count_only")), a.get("limit"))
        if getattr(rows, "complete", True) is False:
            result["partial"] = True
            result["count_scope"] = "current page only"
            result["pagination_note"] = getattr(rows, "note", "The API returned one page only.")
        encoded = json.dumps(result, default=str)
        if len(encoded) > MAX_CHARS and isinstance(result.get("rows"), list):
            result["rows"] = result["rows"][:10]
            result["truncated"] = "Output shortened for model context; use fields or narrower filters for more detail."
            encoded = json.dumps(result, default=str)
        return encoded

    return _prepare_mutation(op, path, query, a.get("body"), a.get("headers"))


ENT = {
    "organization": "/v2/organizations",
    "device": "/v2/devices",
    "end_user": "/v2/user/end-users",
    "technician": "/v2/users",
    "user": "/v2/users",
    "policy": "/v2/policies",
    "location": "/v2/locations",
    "group": "/v2/groups",
    "device_role": "/v2/roles",
    "user_role": "/v2/user/roles",
    "contact": "/v2/contacts",
    "script": "/v2/automation/scripts",
    "alert": "/v2/alerts",
    "software_product": "/v2/software-products",
    "ticket_board": "/v2/ticketing/trigger/boards",
}


def _normalize_person_text(value):
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


def _person_name_match(name, row):
    """Return an exact name-match score; similar spellings are not confirmed matches."""
    query = _normalize_person_text(name)
    if not query or not isinstance(row, dict):
        return 0.0
    first = _normalize_person_text(row.get("firstName") or row.get("first_name"))
    last = _normalize_person_text(row.get("lastName") or row.get("last_name"))
    full = " ".join(part for part in (first, last) if part)
    display = _normalize_person_text(row.get("name") or row.get("displayName"))
    email = _normalize_person_text(row.get("email") or row.get("emailAddress"))
    if query == email:
        return 1.0
    if full and query == full:
        return 1.0
    if display and query == display:
        return 1.0
    # A single name must match a complete first-name or last-name token.
    if len(query.split()) == 1:
        tokens = set((first + " " + last + " " + display).split())
        return 1.0 if query in tokens else 0.0
    # Permit exact contiguous full-name matches, not approximate edit-distance matches.
    for candidate in (full, display):
        if candidate and (candidate.startswith(query + " ") or candidate.endswith(" " + query)):
            return 1.0
    return 0.0


def t_resolve(a):
    path = ENT.get(a.get("kind"))
    op = op_by_path(path) if path else None
    if not op:
        return err(f"kind must be one of {list(ENT)}")
    name = str(a.get("name", "")).strip()
    kind = a.get("kind")
    query = {}
    if kind == "user":
        query["includeRoles"] = True
    elif kind == "technician" and path == "/v2/users":
        query.update({"userType": "TECHNICIAN", "includeRoles": True})
    rows = fetch(op, path, query)
    scored = []
    people_kind = kind in {"user", "end_user", "technician", "contact"}
    for row in rows:
        if not isinstance(row, dict):
            continue
        if people_kind:
            score = _person_name_match(name, row)
        else:
            labels = [
                str(row.get(key, ""))
                for key in (
                    "name", "systemName", "dnsName", "displayName", "email", "title",
                    "subject", "hostname", "serialNumber", "productName", "conditionName",
                    "deviceName", "number", "uid",
                )
            ]
            labels.append(f'{row.get("firstName", "")} {row.get("lastName", "")}'.strip())
            score = max(
                (difflib.SequenceMatcher(None, name.casefold(), label.casefold()).ratio()
                 + (0.4 if name.casefold() in label.casefold() else 0)
                 for label in labels if label),
                default=0,
            )
        if score > 0:
            scored.append((score, row))
    scored.sort(key=lambda item: -item[0])
    matched = [(score, row) for score, row in scored if score >= 1.0] if people_kind else [
        (score, row) for score, row in scored if score > 0.45
    ]
    top = [dict(slim(row), match=round(score, 2)) for score, row in matched[:100 if people_kind else 10]]
    result = {
        "matches": top,
        "total_matches": len(matched),
        "note": "People matches require exact full-name or complete first/last-name token matches; similar spellings are not counted."
                if people_kind else "total_matches counts matching records in the retrieved result; matches contains at most 10 examples",
    }
    if getattr(rows, "complete", True) is False:
        result["partial"] = True
        result["pagination_note"] = getattr(rows, "note", "The API returned one page only.")
    return json.dumps(result, default=str)


DISPATCH = {
    "search_api": t_search,
    "search_fields": t_search_fields,
    "inspect_schema": t_inspect_schema,
    "inspect_api": t_inspect,
    "call_api": t_call,
    "resolve_entity": t_resolve,
}

TOOLS = [
    {"type": "function", "function": {
        "name": "search_api",
        "description": "Search the NinjaOne OpenAPI catalog for supported GET, POST, PUT and PATCH operations. DELETE and explicit delete operations are excluded.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 12}
        }, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "search_fields",
        "description": "Search every NinjaOne OpenAPI component schema and every operation's path/query/header parameters, request-body properties and response properties. Returns exact field names, paths, types, required flags, descriptions, enums and read/write metadata. This returns schema definitions, not live values.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
            "schema_name": {"type": "string"},
            "operation_id": {"type": "string"},
            "schema_source": {"type": "string", "enum": ["json", "yaml"]},
            "limit": {"type": "integer", "minimum": 1, "maximum": 50}
        }, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "inspect_schema",
        "description": "Inspect a complete NinjaOne component schema, including nested properties, array item fields, types, enums and constraints. Supports the primary JSON specification and supplemental YAML specification. Use search_fields first to find the schema name; specify schema_source when a schema exists in both.",
        "parameters": {"type": "object", "properties": {
            "schema_name": {"type": "string"},
            "schema_source": {"type": "string", "enum": ["json", "yaml"]},
            "offset": {"type": "integer", "minimum": 0},
            "limit": {"type": "integer", "minimum": 1, "maximum": 100}
        }, "required": ["schema_name"]}}},
    {"type": "function", "function": {
        "name": "inspect_api",
        "description": "Inspect an exact operation_id from search_api. Returns all documented operation parameters and a paginated, lossless field catalog for request and response schemas. Use offset/limit to retrieve subsequent field pages. Call before writes or when arguments are unclear.",
        "parameters": {"type": "object", "properties": {
            "operation_id": {"type": "string"},
            "field_filter": {"type": "string"},
            "offset": {"type": "integer", "minimum": 0},
            "limit": {"type": "integer", "minimum": 1, "maximum": 100}
        }, "required": ["operation_id"]}}},
    {"type": "function", "function": {
        "name": "resolve_entity",
        "description": "Resolve names for common NinjaOne entities (organization, device, user, technician, policy, location, role, contact, script, alert and more) to matching records and IDs. Use before entity-specific operations and before preparing writes.",
        "parameters": {"type": "object", "properties": {
            "kind": {"type": "string", "enum": ["organization", "device", "end_user", "technician", "user", "policy", "location", "group", "device_role", "user_role", "contact", "script", "alert", "software_product", "ticket_board"]},
            "name": {"type": "string"}
        }, "required": ["kind", "name"]}}},
    {"type": "function", "function": {
        "name": "call_api",
        "description": "Execute validated GET and explicitly read-only POST queries immediately. For state-changing POST, PUT or PATCH this ONLY prepares a request for explicit user approval in the chat UI; it never executes the mutation. DELETE and explicit delete actions are blocked. Use inspect_api first for write schemas.",
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

Execution strategy:
- Prefer deterministic fast paths for simple counts, IDs, exact lookups, small lists and straightforward filters when they can fully answer the request. Do not invoke the model unnecessarily.
- Use model reasoning and the OpenAPI catalog for fuzzy wording, misspellings, multi-condition requests, cross-resource questions and investigations.
- Plan only genuinely multi-step tasks; do not narrate internal reasoning or produce a plan for simple lookups.
- Verify the actual API result, requested filters, pagination completeness and output fields before answering.
- Retry only transient read failures when safe. Never automatically repeat a potentially successful write after a timeout; verify its outcome first.
- Keep simple answers concise and tailor longer results to the requested format.

API workflow:
- search_api searches the complete supported API catalog, including GET, POST, PUT and PATCH. Use natural-language intent and resource synonyms.
- The field index covers all primary JSON component schemas, supplemental YAML component schemas, plus primary operation path/query/header parameters, request bodies and responses. Use search_fields for an exact field name or concept before guessing a field. It returns schema name, field path, source, type, required status, description, enum, nullable/readOnly/writeOnly and other constraints where documented.
- JSON is authoritative for executable endpoint selection, request parameters, request-body validation and live calls. YAML is supplemental for schema discovery only; never execute an endpoint or construct a request solely from a YAML-only operation.
- Use inspect_schema to retrieve the complete nested component schema (including array item fields) after search_fields identifies the schema. If the schema exists in both sources, specify schema_source to inspect each version and compare them.
- inspect_api returns a paginated lossless field catalog for the selected operation. If schema_fields_next_offset is non-null, retrieve additional pages as needed. Do not assume omitted fields do not exist; search_fields can find fields across the entire specification.
- Schema definitions describe possible fields and constraints, not actual tenant values. Use call_api to retrieve live values and report only values present in the response. Do not confuse absent, null, false, empty, and zero values.
- resolve_entity resolves names for organizations, devices, users, technicians, policies, locations, groups, roles, contacts, scripts, alerts, software products and ticket boards. Resolve names to IDs before calling entity-specific endpoints.
- inspect_api retrieves the exact request schema, required parameters, enums and body fields. Use it whenever a write body or parameter is unclear.
- call_api executes GET operations and POST operations that the OpenAPI catalog explicitly identifies as read-only queries. It prepares state-changing POST, PUT and PATCH operations for user approval; a prepared request has NOT been executed.
- Never claim a mutation succeeded until the UI approval action returns a successful NinjaOne response.
- DELETE operations and any operation explicitly named/described as delete are unavailable. Never attempt to bypass this restriction using POST, PATCH or another method.
- Never call a write operation merely to explore its schema. Use inspect_api first.
- For mutations, resolve target entities and IDs first, validate the exact intended change, then prepare the smallest request body. The user must approve the exact method, path, query and body in the UI.
- If a request is missing required fields or violates an enum/type, correct it using inspect_api. Do not guess.
- State-changing operations are supported when the OpenAPI catalog documents a non-delete POST, PUT or PATCH operation. Do not treat writes as unavailable: validate them and prepare an approval card; never execute before explicit UI approval.
- For "create end user" requests, use POST /v2/user/end-users (createEndUser). The required body fields are firstName, lastName and email. Optional body fields are phone, organizationId, fullPortalAccess and managerUid. The optional sendInvitation boolean is a query parameter, not a body field. If any required value is missing, ask the user for the missing information in plain language and wait; do not return a generic write-failure message or guess values. If an organization is supplied by name, resolve it to one unambiguous organization ID before preparing the request. Once required details are available, inspect the operation schema, validate the body, and prepare the POST for approval.
- For explicit delete/erase/purge/destroy requests, refuse clearly and do not search for or substitute a non-DELETE endpoint. This applies to every resource. Requests to revoke access or remove a role/member are not automatically deletes; use only a documented non-delete operation and require approval.
- Pagination is automatic only when the OpenAPI request/response schemas establish the cursor contract: integer ID cursors, documented opaque nextCursor/pageInfo cursors, or the read-only ticket-board lastCursorId body cursor. Do not treat activity timestamps as cursors.
- If an endpoint exposes pageSize/cursor parameters but the spec does not define a safe continuation contract, label the result as the current page only and never report it as a complete tenant-wide count.
- Multipart/file-upload operations are visible in the OpenAPI catalog but are not executable through the current chat transport. Do not fabricate an upload or claim it succeeded; explain the limitation.
- Monitoring scope is read-only. Write operations require the NinjaOne OAuth Management scope and the user’s approval. If the API returns 401/403, explain the actual permission/scope failure.
- Access semantics from the published API spec: resolve end users through GET /v2/user/end-users because it exposes accessibleDeviceIds and assignedDeviceIds. Report accessibleDeviceIds as device access and assignedDeviceIds separately; never treat assignment as equivalent to access. GET /v2/users?includeRoles=true exposes deviceIds, assignedDeviceIds and role names but not accessibleDeviceIds.
- The documented PATCH /v2/user/end-user/{id}/device-access operation adds or removes up to 100 accessible device IDs for an END USER only. For an explicit add request, first resolve the end user, read their current accessibleDeviceIds, resolve/check every requested device, remove IDs already present from the add list, then inspect patchEndUserDeviceAccess and prepare only the remaining IDs using addDeviceIds. If all requested IDs are already accessible, report that no change is needed. If more than 100 new IDs are requested, split into batches of at most 100 and require separate approval for each batch. Never claim access was added until approval succeeds and the API response confirms it. This operation does not grant access to technicians or change assignedDeviceIds.
- Technician records expose assignedDeviceIds and role names. GET /v2/user/roles exposes role members, but this published spec has no GET operation to read a role's organizationPermissions (only a PATCH setter). Never infer a technician's organization-access count from role names; state this API limitation clearly.
- If a person-access question does not distinguish assigned devices from accessible devices, ask one short clarification.
- If a person name resolves ambiguously, show the matching records and ask one concise clarification.
- Report counts only when based on the complete API result or an authoritative total count. State when results are partial or truncated.

Final answers should be concise, factual, and clearly distinguish completed reads, prepared actions awaiting approval, and completed writes.
- Tailor the answer to the user's requested output. If they ask for a list, provide a readable numbered or bulleted list; if they ask for a comparison, use a compact table; if they ask for selected fields, show only those fields; if they ask for a count, give the count first and explain its scope.
- Use Markdown formatting where it improves readability. Do not dump raw JSON unless the user explicitly asks for raw JSON or the data is best represented that way.
- For long inventories, group related results and clearly state the total, displayed count, and whether the result is complete or truncated. Never silently omit records.
- Preserve exact API values and identifiers; do not invent summaries or fields not present in the returned data.

Descriptive summary behavior:
- If the user asks for a summary, overview, analysis, explanation, or report, provide a useful synthesis rather than merely repeating raw API rows.
- Start with a short direct conclusion that answers the question. Then add the most relevant supporting findings, counts, and context from the returned records.
- Organize multi-record results into meaningful groups when the returned fields support grouping (for example, organization, status, role, or device type). Do not infer a group when the field is absent.
- Explain notable patterns only when they are directly demonstrated by the data. Separate observed facts from recommendations or possible next steps.
- Include the scope of the data: filters applied, records returned, total versus displayed count, and whether pagination is complete. Clearly label partial or truncated results.
- For device or user details, prioritize identity, status, organization, and the fields relevant to the user's question. Include IDs and exact values where useful; avoid dumping the entire raw JSON unless requested.
- For comparisons, state the key differences and support them with the exact values. For counts, lead with the number and explain exactly what was counted.
- Match the depth to the request: concise for simple questions; structured and more descriptive when the user explicitly asks for detail or a summary.
- Never add business impact, root cause, risk level, health interpretation, or recommendations unless the returned fields or an explicit documented rule support that conclusion.
- If evidence is missing or ambiguous, say what cannot be established instead of filling the gap with a plausible explanation."""



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

def _detail_value(value, limit=8):
    """Render an API field as readable text without inventing its meaning."""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, dict):
        preferred = ("name", "displayName", "systemName", "id", "value", "version")
        parts = [f"{key}: {value[key]}" for key in preferred if value.get(key) not in (None, "")]
        return "; ".join(parts) if parts else f"{len(value)} field(s) returned"
    if isinstance(value, list):
        if not value:
            return "None"
        if all(not isinstance(item, (dict, list)) for item in value):
            shown = ", ".join(str(item) for item in value[:limit])
            return shown + (f" … (+{len(value) - limit} more)" if len(value) > limit else "")
        return f"{len(value)} item(s)"
    return str(value)


def format_record_summary(record, title="Record", max_fields=40):
    """Group returned record fields into readable facts, preserving source values."""
    if not isinstance(record, dict):
        return f"{title}: {_detail_value(record) or 'No details returned'}"
    groups = (
        ("Identity", re.compile(r"(?:^|_)(?:id|name|hostname|system|display|serial|uid|email)", re.I)),
        ("Status and activity", re.compile(r"(?:status|state|offline|online|last|contact|checkin|seen|health)", re.I)),
        ("Organization and management", re.compile(r"(?:org|organization|location|group|policy|role|site)", re.I)),
        ("System and hardware", re.compile(r"(?:os|operating|version|model|manufacturer|vendor|cpu|processor|memory|ram|disk|volume|ip|mac|hardware|architecture)", re.I)),
    )
    rendered, used = [], set()
    for label, pattern in groups:
        fields = []
        for key, value in record.items():
            if key in used or not pattern.search(key):
                continue
            text_value = _detail_value(value)
            if text_value is not None:
                display_key = re.sub(r"(?<!^)(?=[A-Z])", " ", key).replace("_", " ").strip()
                fields.append(f"- {display_key}: {text_value}")
                used.add(key)
        if fields:
            rendered.append(f"**{label}**\n" + "\n".join(fields))
    other = []
    for key, value in record.items():
        if key in used:
            continue
        text_value = _detail_value(value)
        if text_value is not None:
            display_key = re.sub(r"(?<!^)(?=[A-Z])", " ", key).replace("_", " ").strip()
            other.append(f"- {display_key}: {text_value}")
            used.add(key)
        if len(used) >= max_fields:
            break
    if other:
        rendered.append("**Other returned details**\n" + "\n".join(other))
    omitted = [key for key in record if key not in used and record.get(key) not in (None, "")]
    if omitted:
        rendered.append(f"{len(omitted)} additional field(s) omitted to keep the summary readable: " + ", ".join(omitted[:12]))
    return f"**{title}**\n" + ("\n\n".join(rendered) if rendered else "No populated fields were returned.")


GREET = re.compile(r"^\s*(hi+|hello+|hey+|yo|good (morning|afternoon|evening)|thanks?|thank you|ok(ay)?|help|what can you do)\W*$", re.I)
HELP = ('Hi! I can investigate NinjaOne data and prepare approved changes (delete actions are disabled), for example: "how many devices do we have?", '
        '"list organizations", "how many technicians?", "devices per organization", or "how many devices does Richard have access to?"')

_TXT_CALL = re.compile(r"\b(search_api|search_fields|inspect_schema|inspect_api|call_api|resolve_entity)\s*\(\s*(\{.*\})\s*\)", re.S)
_TAG_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def text_call(txt):
    """Recover structured, JSON-style, or Python-style tool calls printed as plain text."""
    allowed = set(DISPATCH)
    try:
        text = (txt or "").strip()
        # Parse Python-call syntax safely; only literal arguments are accepted.
        for match in re.finditer(r"\b(search_api|search_fields|inspect_schema|inspect_api|call_api|resolve_entity)\s*\(", text):
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
    return (
        f"Result of {name}: {out}\n"
        "Use another tool if required to verify the answer. In the final response, state what was counted, "
        "the source fields or records used, and any ambiguity, partial result, permission issue or API limitation."
    )


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
    fetched = fetch(dop, "/v2/devices", {})
    devs = [d for d in fetched if isinstance(d, dict)]
    if not devs:
        return "No devices returned by NinjaOne." + _partial_note(fetched)
    if not any("organizationId" in d for d in devs):
        return "The devices list has no organizationId field, so I can't group it. Fields seen: " + ", ".join(list(devs[0])[:15]) + _partial_note(fetched)
    names, c = _orgs(), Counter(d.get("organizationId") for d in devs)
    lines = [f"{names.get(k) or 'organization ' + str(k)}: {n}" for k, n in c.most_common(40)]
    partial = _partial_note(fetched)
    heading = (f"Partial breakdown from {len(devs)} returned device record(s); tenant-wide totals are unknown."
               if partial else f"{len(devs)} devices across {len(c)} organizations:")
    return heading + partial + "\n" + "\n".join(lines)


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
    return f"{len(output)} {label}(s) in the returned result:{_partial_note(rows)}\n" + "\n".join(
        f"{index}. {value}" for index, value in enumerate(output, 1)
    )


def recipe_list_organizations(q):
    """List organizations, including IDs when explicitly requested."""
    text = re.sub(r"\s+", " ", q.lower().strip())
    asks_for_names = bool(re.search(r"\b(name|names)\b", text))
    asks_for_ids = bool(re.search(r"\b(ids?|identifiers?)\b", text))
    asks_list = bool(re.search(
        r"\b(list|show|display|get|fetch|give|provide|what are|which are|can i get|could i get)\b",
        text,
    ))
    asks_count = bool(re.search(r"\b(how many|count|number of|total)\b", text))
    has_organization = bool(re.search(r"\b(organizations?|orgs?|clients?|customers?)\b", text))
    if not (has_organization and (asks_list or asks_for_names or asks_for_ids) and not asks_count):
        return None

    path = "/v2/organizations"
    op = op_by_path(path)
    if not op:
        return "The NinjaOne organization-list endpoint was not found in the loaded API specification."
    rows = fetch(op, path, {})
    records = [
        row for row in rows
        if isinstance(row, dict) and row.get("name") not in (None, "")
    ]
    if not records:
        return f"NinjaOne returned {len(rows)} organization record(s), but no organization names were present."

    if asks_for_ids:
        lines = [
            f"{index}. {str(row.get('name')).strip()} — ID: {row.get('id') if row.get('id') is not None else 'not returned'}"
            for index, row in enumerate(records, 1)
        ]
    else:
        lines = [
            f"{index}. {str(row.get('name')).strip()}"
            for index, row in enumerate(records, 1)
        ]
    return f"{len(records)} organization(s) in the returned result:{_partial_note(rows)}\n" + "\n".join(lines)


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
    return f"{len(names)} technician(s) in the returned result:{_partial_note(rows)}\n" + "\n".join(f"{index}. {name}" for index, name in enumerate(names, 1))


def recipe_search_people(q):
    """Search people inventories using the name only, excluding tenant/context words."""
    text = re.sub(r"\s+", " ", str(q).strip())
    match = re.search(
        r"\b(?:users?|people|accounts?|persons?)\b.*?"
        r"(?:\bwith\s+(?:the\s+)?name\s+|\bnamed\s+|\bcalled\s+)(.+?)\s*$",
        text, re.I,
    )
    if not match:
        return None
    name = re.sub(r"\s+(?:in|at)\s+(?:ninjaone|the\s+ninjaone\s+tenant)\s*$", "", match.group(1), flags=re.I)
    name = name.strip(" \t\r\n\"'.,!?")
    if not name or len(name) < 2:
        return None

    inventories = (
        ("user", "Console / NinjaOne user"),
        ("end_user", "End user"),
        ("technician", "Technician"),
        ("contact", "Contact"),
    )
    found, seen = [], set()
    partial_notes = []
    for kind, label in inventories:
        try:
            result = json.loads(t_resolve({"kind": kind, "name": name}))
        except Exception as exc:
            return f"Could not search the NinjaOne {label.lower()} inventory: {type(exc).__name__}: {exc}"
        if result.get("error"):
            return f"Could not search the NinjaOne {label.lower()} inventory: {result['error']}"
        for row in result.get("matches", []):
            # Defensive second check: never trust approximate or inconsistent upstream previews.
            if _person_name_match(name, row) < 1.0:
                continue
            email = str(row.get("email") or row.get("emailAddress") or "").strip().casefold()
            identity = (email,) if email else (kind, str(row.get("id") or ""), _normalize_person_text(row_name(row)))
            if identity in seen:
                continue
            seen.add(identity)
            user_type = str(row.get("userType") or row.get("type") or "").upper()
            category = "Technician" if user_type == "TECHNICIAN" else (
                "End user" if user_type in {"END_USER", "ENDUSER"} or kind == "end_user" else label
            )
            found.append((category, row))
        if result.get("partial"):
            partial_notes.append(result.get("pagination_note") or f"{label} results may be incomplete.")

    if not found:
        suffix = " Some inventory results may be incomplete." if partial_notes else ""
        return f'No exact NinjaOne person-name matches found for "{name}".{suffix}'
    found.sort(key=lambda item: (item[0], row_name(item[1]).casefold()))
    lines = []
    for index, (category, row) in enumerate(found, 1):
        email = row.get("email") or row.get("emailAddress") or "not returned"
        identifier = row.get("id")
        lines.append(
            f"{index}. {row_name(row)} — {category}; Email: {email}; ID: {identifier if identifier is not None else 'not returned'}"
        )
    total = f'{len(found)} exact NinjaOne person record(s) matched "{name}".'
    if partial_notes:
        total += " Results may be incomplete: " + "; ".join(dict.fromkeys(partial_notes))
    return total + "\n" + "\n".join(lines)

def recipe_latest_end_user(q):
    """Identify the highest-ID end-user record only after a complete inventory fetch.

    The published end-user list does not define a creation timestamp. A highest ID
    is therefore reported as an ID-based indicator, never as a verified createdAt.
    """
    text = re.sub(r"\buend\s+users?\b", "end user", str(q or ""), flags=re.I)
    text = re.sub(r"\s+", " ", text.strip().rstrip("?.! "))
    if not re.search(r"\bend[\s-]?users?\b", text, re.I):
        return None
    latest_intent = re.search(
        r"\b(?:last|latest|newest|most\s+recent)\b.{0,60}\b(?:created|added|registered|end\s+user)\b"
        r"|\b(?:created|added|registered)\b.{0,60}\b(?:last|latest|newest|most\s+recent)\b",
        text, re.I,
    )
    if not latest_intent:
        return None

    path = "/v2/user/end-users"
    operation = op_by_path(path)
    if not operation:
        return f"The NinjaOne end-user listing operation GET {path} is not present in the loaded API specification."

    rows = fetch(operation, path, {})
    if getattr(rows, "complete", True) is False:
        return (
            "I cannot safely identify the highest-ID end user because NinjaOne pagination "
            f"did not complete. {_partial_note(rows)}"
        )

    candidates = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw_id = row.get("id")
        if isinstance(raw_id, bool):
            continue
        try:
            numeric_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        candidates.append((numeric_id, row))

    if not candidates:
        return (
            "NinjaOne returned a complete end-user inventory, but none of the records "
            "contained a usable numeric ID. I cannot determine the highest-ID record."
        )

    highest_id = max(item[0] for item in candidates)
    highest_rows = [row for numeric_id, row in candidates if numeric_id == highest_id]
    if len(highest_rows) != 1:
        return (
            f"The complete end-user inventory contains multiple records with ID {highest_id}; "
            "I cannot identify a unique record."
        )

    row = highest_rows[0]
    name = row_name(row)
    email = row.get("email") or row.get("emailAddress")
    details = [f"**{name}** (ID {highest_id})"]
    if email:
        details.append(f"Email: {email}")
    return (
        "The end user with the highest ID in the complete NinjaOne end-user inventory is "
        + " — ".join(details)
        + f". I checked {len(candidates)} records using GET {path} and completed pagination. "
        "The published end-user list does not expose a creation timestamp, so the highest ID "
        "is an ID-based indicator—not independent proof of the most recent creation time."
    )


def recipe_count_named(q):
    """Count matching user records for natural questions such as 'how many users with name Rene?'."""
    text = str(q).strip().rstrip("?.! ")
    match = re.search(
        r"\b(?:users?|people|accounts?)\b.*?\b(?:with|having)\s+(?:the\s+)?name\s+(.+)$",
        text, re.I
    )
    if not match:
        return None
    name = re.sub(r"\s+(?:in|at)\s+(?:ninjaone|the\s+ninjaone\s+tenant)\s*$", "", match.group(1), flags=re.I)
    name = name.strip().strip("\"'")
    if not name or len(name) < 2:
        return None
    result = json.loads(t_resolve({"kind": "user", "name": name}))
    count = result.get("total_matches")
    if count is None:
        count = len(result.get("matches", []))
    examples = result.get("matches", [])
    if count == 0:
        if result.get("partial"):
            return f'No NinjaOne users matched "{name}" in the returned page; the search is incomplete. {result.get("pagination_note", "")}'
        return f'No NinjaOne users matched the name "{name}".'
    details = ", ".join(
        row_name(row) + (f' <{row["email"]}>' if row.get("email") else "")
        for row in examples[:5]
    )
    suffix = f" Examples: {details}." if details else ""
    if result.get("partial"):
        return (f'At least {count} NinjaOne user record(s) matched "{name}" in the returned page; '
                f'the tenant-wide total is unknown. {result.get("pagination_note", "")}{suffix}')
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


def recipe_add_device_access(q):
    """Prepare exact end-user device-access grants as one approval-only PATCH.

    Supports one or many device system names. It never executes the mutation.
    """
    text = re.sub(r"\s+", " ", str(q).strip()).strip(" .!?")
    multi = re.search(
        r"^(?:please\s+)?(?:add|grant|give|provide)\s+"
        r"(?:(?:the\s+)?(?:below|following)\s+)?"
        r"(?:systems|devices|computers|machines|endpoints)\s+to\s+"
        r"(?:the\s+)?accessible\s+device\s+list\s+for\s+(.+)$",
        text, re.I,
    )

    def normalized(value):
        return re.sub(r"[^a-z0-9]", "", str(value or "").lower())

    def exact_user_rows(label):
        result = json.loads(t_resolve({"kind": "end_user", "name": label}))
        return [
            row for row in (result.get("matches") or [])
            if normalized(row_name(row)) == normalized(label)
        ]

    def exact_device_rows(label):
        result = json.loads(t_resolve({"kind": "device", "name": label}))
        exact = []
        for row in (result.get("matches") or []):
            labels = (
                row.get("systemName"), row.get("displayName"), row.get("dnsName"),
                row.get("netbiosName"), row.get("serialNumber"), row.get("name"),
            )
            if any(normalized(value) == normalized(label) for value in labels if value):
                exact.append(row)
        return exact

    user_label = ""
    device_labels = []
    user = None

    if multi:
        # System names are single whitespace-delimited identifiers in NinjaOne.
        # Try each possible user/device boundary, accepting only an exact user
        # name followed by exact device matches. Never use fuzzy matches for writes.
        tail = multi.group(1).strip()
        tokens = [token.strip(" ,;") for token in tail.split() if token.strip(" ,;")]
        if len(tokens) < 2:
            return "Please provide an exact end-user name followed by one or more device system names. No change was prepared."
        partitions = []
        for split_at in range(1, len(tokens)):
            candidate_user = " ".join(tokens[:split_at]).strip(" ,;")
            candidate_devices = [token.strip(" ,;") for token in tokens[split_at:]]
            try:
                users = exact_user_rows(candidate_user)
            except Exception as exc:
                return f"Could not resolve the end user: {type(exc).__name__}: {exc}. No change was prepared."
            if len(users) != 1:
                continue
            resolved_devices = []
            valid = True
            for label in candidate_devices:
                try:
                    matches = exact_device_rows(label)
                except Exception as exc:
                    return f'Could not resolve device "{label}": {type(exc).__name__}: {exc}. No change was prepared.'
                if len(matches) != 1:
                    valid = False
                    break
                resolved_devices.append(matches[0])
            if valid:
                partitions.append((candidate_user, users[0], candidate_devices, resolved_devices))
        if not partitions:
            return (
                "I could not split that request into one exact end-user name and exact device system names. "
                "Please use: add below systems to accessible device list for <end-user name> <device1> <device2>. "
                "No change was prepared."
            )
        if len(partitions) != 1:
            return "More than one exact end-user/device interpretation matched. Please clarify the end-user and device names. No change was prepared."
        user_label, user, device_labels, devices = partitions[0]
    else:
        # Accept conversational variants such as "add X to Y for device access",
        # while retaining the existing explicit access-list forms.
        match = re.search(
            r"\b(?:add|grant|give|provide)\s+(?:the\s+)?(?:device\s+)?"
            r"(?P<device>[A-Za-z0-9][A-Za-z0-9._-]*)\s+to\s+"
            r"(?P<user>.+?)\s+(?:for\s+)?(?:the\s+)?(?:device\s+)?access"
            r"(?:\s+list)?\s*$",
            text, re.I,
        )
        if match:
            device_label, user_label = match.group("device"), match.group("user")
        else:
            match = re.search(
                r"\b(?:add|grant|give|provide)\s+(.+?)\s+(?:device\s+)?access\s+(?:to|for)\s+(.+?)\s*$",
                text, re.I,
            )
            if match:
                device_label, user_label = match.group(1), match.group(2)
            else:
                match = re.search(
                    r"\b(?:add|grant|give|provide)\s+(?:the\s+)?device\s+(.+?)\s+to\s+(.+?)"
                    r"(?:'s)?\s+(?:device\s+)?access(?:\s+list)?\s*$",
                    text, re.I,
                )
                if not match:
                    return None
                device_label, user_label = match.group(1), match.group(2)

        device_label = device_label.strip().strip("\\'").strip('"')
        user_label = user_label.strip().strip(" .")
        email_match = re.search(r"<\s*([^<>@\s]+@[^<>\s]+)\s*>", user_label)
        email = email_match.group(1).strip().lower() if email_match else ""
        user_name = re.sub(r"<[^<>]*>", "", user_label).strip(" \t<>")
        if not device_label or (not user_name and not email):
            return "Please provide both a device name and an end-user name."
        try:
            lookup = email or user_name
            user_result = json.loads(t_resolve({"kind": "end_user", "name": lookup}))
        except Exception as exc:
            return f"Could not resolve the end user: {type(exc).__name__}: {exc}"
        # Resolve the requested identity across every NinjaOne people inventory.
        # /v2/users also contains technicians, so normalize that overlap by stable ID.
        people_records = []
        people_results = [("end_user", user_result)]

        # Cross-category disambiguation is essential for shorthand/first-name
        # requests (for example, "Rene"). An explicit email or full name is an
        # identity-qualified request; keep that fast path to the end-user record
        # and avoid redundant API calls.
        if not email and len(user_name.split()) == 1:
            for kind in ("user", "technician", "contact"):
                try:
                    people_results.append((
                        kind,
                        json.loads(t_resolve({"kind": kind, "name": lookup})),
                    ))
                except Exception as exc:
                    return (
                        f'Could not verify all NinjaOne people records for "{user_label}": '
                        f'{type(exc).__name__}: {exc}. The search is incomplete, so no person '
                        "will be selected and no change was prepared."
                    )

        for kind, result in people_results:
            if not isinstance(result, dict) or result.get("error"):
                error_detail = result.get("error") if isinstance(result, dict) else "invalid API response"
                return (
                    f'I could not verify the {kind.replace("_", " ")} inventory for "{user_label}": '
                    f'{error_detail}. The identity search is incomplete. '
                    "No person was selected. No change was prepared."
                )
            result_matches = result.get("matches") or []
            if result.get("partial") or (
                result.get("total_matches") is not None
                and int(result.get("total_matches") or 0) > len(result_matches)
            ):
                detail = result.get("pagination_note") or (
                    f'{result.get("total_matches")} matching records exist, but only '
                    f'{len(result_matches)} candidate records were returned.'
                )
                return (
                    f'I cannot safely identify "{user_label}" because the {kind.replace("_", " ")} '
                    f'inventory search is incomplete. {detail} '
                    "No person was selected. No change was prepared."
                )
            for row in result_matches:
                if email:
                    matched = str(row.get("email", "")).strip().lower() == email
                else:
                    requested_name = normalized(user_name)
                    matched = (
                        normalized(row_name(row)) == requested_name
                        or (
                            len(user_name.split()) == 1
                            and user_name.casefold() in {
                                token.casefold()
                                for token in re.findall(
                                    r"[A-Za-z0-9]+",
                                    " ".join(str(row.get(key) or "") for key in (
                                        "firstName", "lastName", "name", "displayName"
                                    )),
                                )
                            }
                        )
                    )
                if not matched:
                    continue

                if kind == "end_user":
                    category = "end-user"
                elif kind == "contact":
                    category = "contact"
                elif kind == "technician" or str(row.get("userType", "")).upper() == "TECHNICIAN":
                    category = "technician"
                else:
                    category = "console user"

                record_id = row.get("id")
                email_value = str(row.get("email") or "").strip().lower()
                identity = (
                    (category, str(record_id)) if record_id is not None
                    else (category, email_value or normalized(row_name(row)))
                )
                existing = next((item for item in people_records if item["identity"] == identity), None)
                if existing:
                    # Prefer the richer row when the same /v2/users technician appears
                    # through both user and technician resolver paths.
                    if len(row) > len(existing["row"]):
                        existing["row"] = row
                    continue
                people_records.append({
                    "identity": identity,
                    "category": category,
                    "row": row,
                })

        if not people_records:
            return f'No exact NinjaOne people record matched "{user_label}". No change was prepared.'

        # A first-name match may legitimately identify several distinct records,
        # including records that cannot receive this end-user-only PATCH. Show all.
        if len(people_records) > 1:
            choices = []
            for index, item in enumerate(people_records, 1):
                row = item["row"]
                name = row_name(row) or "Unnamed person"
                email_value = str(row.get("email") or "").strip()
                record_id = row.get("id")
                details = name
                if email_value:
                    details += f" <{email_value}>"
                if record_id is not None:
                    details += f' ({item["category"]} ID {record_id})'
                else:
                    details += f' ({item["category"]})'
                choices.append(f"{index}. {details}")
            return (
                f'Multiple NinjaOne people records matched "{user_label}". '
                f'Which user should receive the device access for "{device_label}"?\n'
                + "\n".join(choices)
                + "\nReply with the option number, full name, or email address. "
                "Only an end-user record supports this device-access operation. "
                "No change has been prepared; I will prepare the approval only after you choose one."
            )

        selected = people_records[0]
        if selected["category"] != "end-user":
            row = selected["row"]
            return (
                f'Matched {row_name(row)} as a NinjaOne {selected["category"]} record, '
                "not an end-user record. The documented direct device-access PATCH supports "
                "end-user records only. No change was prepared."
            )
        user = selected["row"]
        try:
            devices = exact_device_rows(device_label)
        except Exception as exc:
            return f"Could not resolve the device: {type(exc).__name__}: {exc}"
        if not devices:
            return f'No exact device match for "{device_label}". No change was prepared.'
        if len(devices) != 1:
            return f'Multiple exact device records matched "{device_label}". Please clarify before changing access.'
        device_labels = [device_label]

    user_id = user.get("id")
    if not isinstance(user_id, int) and not str(user_id or "").isdigit():
        return "The matched end-user record has no numeric NinjaOne ID. No change was prepared."
    user_id = int(user_id)

    if len(devices) > 100:
        return f"{len(devices)} devices were requested, but NinjaOne allows at most 100 device IDs per PATCH. No change was prepared."

    accessible_ids = user.get("accessibleDeviceIds")
    if not isinstance(accessible_ids, list):
        return "The end-user lookup did not return accessibleDeviceIds. I cannot safely check for duplicate access; no change was prepared."
    existing_ids = {int(value) for value in accessible_ids if str(value).isdigit()}

    unique_devices = {}
    for label, device in zip(device_labels, devices):
        device_id = device.get("id")
        if not isinstance(device_id, int) and not str(device_id or "").isdigit():
            return f'The matched device "{label}" has no numeric NinjaOne ID. No change was prepared.'
        unique_devices[int(device_id)] = (label, device)

    already_accessible = [
        (label, device) for device_id, (label, device) in unique_devices.items()
        if device_id in existing_ids
    ]
    new_devices = [
        (device_id, label, device)
        for device_id, (label, device) in unique_devices.items()
        if device_id not in existing_ids
    ]
    if not new_devices:
        names = ", ".join(row_name(device) for _label, device in already_accessible)
        return f'{row_name(user)} already has access to all requested device(s): {names}. No change is needed.'

    op = next((
        operation for operation in OPS.values()
        if operation.get("method") == "PATCH"
        and operation.get("path") == "/v2/user/end-user/{id}/device-access"
    ), None)
    if not op:
        return "The bundled NinjaOne API catalog does not contain the documented end-user device-access PATCH. No change was prepared."

    path_params = {
        name: user_id for name in re.findall(r"\{([^}]+)\}", op["path"])
    }
    add_ids = [device_id for device_id, _label, _device in new_devices]
    prepared_raw = t_call({
        "operation_id": op["id"],
        "path_params": path_params,
        "body": {"addDeviceIds": add_ids},
    })
    try:
        prepared = json.loads(prepared_raw)
    except (TypeError, ValueError):
        return f"Could not prepare the access change: {prepared_raw}"
    if not prepared.get("approval_required"):
        return f"Could not prepare the access change: {prepared_raw}"

    user_org = user.get("organizationId")
    device_orgs = sorted({
        str(device.get("organizationId"))
        for _device_id, _label, device in new_devices
        if device.get("organizationId") is not None
    })
    target_text = ", ".join(
        f'{row_name(device)} (ID {device_id}, org {device.get("organizationId", "unknown")})'
        for device_id, _label, device in new_devices
    )
    org_note = ""
    if user_org is not None and any(str(user_org) != org for org in device_orgs):
        org_note = f" Note: end user is in organization {user_org}, while one or more target devices are in organization(s) {', '.join(device_orgs)}; verify this cross-organization access is intended before approving."
    already_note = (
        " Already accessible and omitted: " +
        ", ".join(row_name(device) for _label, device in already_accessible) + "."
        if already_accessible else ""
    )
    prepared["summary"] = (
        f'Add device access for {row_name(user)} ({user.get("email", "no email")}) '
        f'to {len(add_ids)} device(s): {target_text}.{already_note}{org_note}'
    )
    return json.dumps(prepared, default=str)

def recipe_person_device_access(q):
    """Resolve a named end user and report the complete accessible/assigned device lists."""
    text = re.sub(r"\s+", " ", q.lower().strip().rstrip("?.! "))
    if not re.search(r"\b(access|accessible|accessibility|assigned|assignment|device access)\b", text):
        return None
    match = re.search(
        r"\b(?:of|for|does|has)\s+(?:the\s+)?(?:user\s+|technician\s+|person\s+)?"
        r"([a-z][a-z .'-]{2,})$",
        text,
    )
    if not match:
        return None
    name = re.sub(r"\s+", " ", match.group(1)).strip(" .'-")
    if not name or re.search(r"\b(devices?|access|assigned|accessible|give|show|list)\b", name):
        return None

    # Only the end-user endpoint exposes accessibleDeviceIds in the published spec.
    resolved = json.loads(t_resolve({"kind": "end_user", "name": name}))
    matches = resolved.get("matches") or []
    if not matches:
        other = json.loads(t_resolve({"kind": "user", "name": name}))
        other_matches = other.get("matches") or []
        if other_matches and other_matches[0].get("match", 0) >= 0.65:
            candidate = other_matches[0]
            return (
                f'Matched {row_name(candidate)} as a NinjaOne {candidate.get("userType", "user")} record, '
                "but no matching end-user record was found. The published API exposes the direct "
                "accessible-device list and device-access update endpoint for end users only. "
                "For this record, I can report assignedDeviceIds/deviceIds where present, but cannot "
                "claim those fields represent end-user access or add direct device access."
            )
        return f'No NinjaOne end user matched "{name}".'
    if len(matches) > 1 and matches[1].get("match", 0) >= matches[0].get("match", 0) - 0.08:
        choices = ", ".join(row_name(row) + (f' <{row["email"]}>' if row.get("email") else "")
                            for row in matches[:5])
        return f'Multiple NinjaOne users (end-user records) closely match "{name}": {choices}. Please confirm which user.'
    user = matches[0]
    if user.get("match", 0) < 0.65:
        return f'No sufficiently close NinjaOne end-user match for "{name}". Closest result: {row_name(user)}.'

    def ids_for(*keys):
        for key in keys:
            value = user.get(key)
            if isinstance(value, list):
                return list(dict.fromkeys(value))
        return []

    accessible = ids_for("accessibleDeviceIds")
    assigned = ids_for("assignedDeviceIds")
    device_ids = ids_for("deviceIds")
    who = row_name(user)
    email = f' <{user["email"]}>' if user.get("email") else ""

    # Count-only questions use the authoritative ID arrays directly. Do not
    # fetch the device inventory or render a device list for these requests.
    count_only = bool(re.search(r"\b(count|how many|total|number of)\b", text))
    if count_only:
        return (
            f"NinjaOne device-access count for {who}{email}:\n"
            f"Accessible devices: {len(accessible)}\n"
            f"Assigned devices (separate from access): {len(assigned)}"
        )

    all_ids = list(dict.fromkeys(accessible + assigned + device_ids))
    devices = []
    device_list_complete = True
    device_list_note = ""
    if all_ids:
        op = op_by_path("/v2/devices")
        if op:
            fetched = fetch(op, "/v2/devices", {})
            device_list_complete = getattr(fetched, "complete", True)
            device_list_note = getattr(fetched, "note", "")
            by_id = {str(row.get("id")): row for row in fetched
                     if isinstance(row, dict) and row.get("id") is not None}
            devices = [by_id.get(str(device_id)) for device_id in all_ids]

    def describe_ids(label, values):
        if not values:
            return f"{label} (0): none returned."
        lines = []
        for index, device_id in enumerate(values, 1):
            row = next((item for item in devices if isinstance(item, dict)
                        and str(item.get("id")) == str(device_id)), None)
            name_value = row_name(row) if row else "device details unavailable"
            lines.append(f"{index}. {name_value} (ID {device_id})")
        sections = []
        for offset in range(0, len(lines), 20):
            batch = lines[offset:offset + 20]
            sections.append(
                f"{label} ({len(lines)}) — devices {offset + 1}-{offset + len(batch)} of {len(lines)}:\n"
                + "\n".join(batch)
            )
        return "\n\n".join(sections)

    who = row_name(user)
    email = f' <{user["email"]}>' if user.get("email") else ""
    sections = [
        f"NinjaOne device-access details for {who}{email}:",
        describe_ids("Accessible devices", accessible),
        describe_ids("Assigned devices", assigned) + " (assignment is separate from device access)",
    ]
    if device_ids and set(map(str, device_ids)) != set(map(str, accessible + assigned)):
        sections.append(describe_ids("Other device IDs on user record", device_ids))
    if not any((accessible, assigned, device_ids)):
        sections.append(
            "The matched end-user record contains no accessibleDeviceIds or assignedDeviceIds. "
            "The API response does not establish device-level access for this user."
        )
    if not device_list_complete:
        sections.append("Device details may be incomplete: " + (device_list_note or "the device endpoint returned a partial page."))
    matched_device_count = sum(1 for row in devices if isinstance(row, dict))
    if matched_device_count < len(all_ids):
        sections.append(
            f"{len(all_ids) - matched_device_count} device ID(s) could not be matched to the device inventory; "
            "those IDs are still shown above."
        )
    return "\n\n".join(sections)

def recipe_device_details(q):
    """Resolve a device name/system name to its numeric ID, then fetch its full detail record."""
    text = re.sub(r"\s+", " ", q.lower().strip().rstrip("?.! "))
    match = re.search(
        r"\b(?:system|device|computer|machine|endpoint)\s+"
        r"(?:details?|information|info|specifications?)\s+"
        r"(?:of|for|about)\s+(.+)$",
        text,
    )
    if not match:
        return None

    identifier = match.group(1).strip().strip(chr(34) + chr(39))
    if not identifier:
        return "Please provide a device name or identifier."

    resolved = json.loads(t_resolve({"kind": "device", "name": identifier}))
    matches = resolved.get("matches") or []
    if not matches:
        return f'No NinjaOne device matched "{identifier}".'

    def norm(value):
        return re.sub(r"[^a-z0-9]", "", str(value or "").lower())

    exact = []
    for row in matches:
        labels = (
            row.get("systemName"), row.get("displayName"), row.get("dnsName"),
            row.get("netbiosName"), row.get("serialNumber"), row.get("uid"),
            row.get("name"),
        )
        if any(norm(label) == norm(identifier) for label in labels if label):
            exact.append(row)

    if len(exact) > 1:
        choices = ", ".join(
            f'{row_name(row)} (device ID {row.get("id")})' for row in exact[:8]
        )
        return f'Multiple NinjaOne devices exactly match "{identifier}": {choices}. Please confirm which device.'

    if not exact:
        choices = ", ".join(
            f'{row_name(row)} (device ID {row.get("id")}, match {row.get("match")})'
            for row in matches[:5]
        )
        return (
            f'I could not verify an exact device match for "{identifier}". '
            f'Closest matches: {choices}. Please provide the exact system name or confirm a match.'
        )

    device = exact[0]
    device_id = device.get("id")
    if isinstance(device_id, bool) or not str(device_id).isdigit():
        return (
            f'The device "{row_name(device)}" resolved, but NinjaOne did not return a numeric device ID. '
            'The device-details endpoint requires the numeric ID; no details request was sent.'
        )

    path = "/v2/device/{id}"
    op = op_by_path(path)
    if not op:
        return f'The bundled NinjaOne API specification does not contain GET {path}. No details request was sent.'

    # Ask the operation schema for its complete top-level response field list.
    fields = []
    for response in (op.get("responses") or {}).values():
        if not isinstance(response, dict):
            continue
        schema = ((response.get("content") or {}).get("application/json") or {}).get("schema") or {}
        properties = schema.get("properties") or {}
        if properties:
            fields = list(properties)
            break

    arguments = {
        "operation_id": op["id"],
        "path_params": {"id": int(device_id)},
    }
    if fields:
        arguments["fields"] = fields

    result_text = t_call(arguments)
    try:
        result = json.loads(result_text)
    except (TypeError, ValueError):
        return "The NinjaOne device-details operation returned an unreadable response."

    if result.get("error"):
        return "NinjaOne device-details lookup failed: " + json.dumps(result, default=str)

    rows = result.get("rows") or []
    if not rows:
        return f'NinjaOne returned no device details for "{identifier}" (device ID {device_id}).'

    summary = format_record_summary(rows[0], title="System summary")
    # Keep exact API field names and values alongside the readable summary.
    # This preserves the established details response for callers verifying source data.
    source_values = json.dumps(rows[0], indent=2, ensure_ascii=False, default=str)
    return (
        f'NinjaOne system details for {row_name(device)} (device ID {device_id}) '
        f'from GET {path.replace("{id}", str(device_id))}.\n\n'
        + summary
        + "\n\n**Exact API response**\n\n```json\n"
        + source_values
        + "\n```"
    )



def recipe_device_summary(q):
    """Build a descriptive, read-only device summary from verified API records."""
    text = re.sub(r"\s+", " ", (q or "").strip().rstrip("?.! "))
    patterns = (
        r"\b(?:descriptive\s+)?(?:summary|overview|report)\s+(?:of|for|about)\s+(?:the\s+)?(?:device|computer|machine|endpoint)\s+(.+)$",
        r"\b(?:device|computer|machine|endpoint)\s+(?:summary|overview|report)\s+(?:of|for|about)\s+(.+)$",
    )
    match = next((re.search(pattern, text, re.I) for pattern in patterns
                  if re.search(pattern, text, re.I)), None)
    if not match:
        return None

    identifier = match.group(1).strip().strip(chr(34) + chr(39))
    if not identifier:
        return "Please provide a device name or identifier."

    # Reuse the exact-match and numeric-ID verification in the established details recipe.
    details = recipe_device_details(f"device details for {identifier}")
    if not details or not details.startswith("NinjaOne system details for "):
        return details

    sections = [details]
    subresources = (
        ("/v2/device/{id}/processors", "Processors"),
        ("/v2/device/{id}/disks", "Disk drives"),
        ("/v2/device/{id}/network-interfaces", "Network interfaces"),
        ("/v2/device/{id}/volumes", "Storage volumes"),
    )
    device_id_match = re.search(r"\(device ID (\d+)\)", details)
    if not device_id_match:
        return details
    device_id = int(device_id_match.group(1))

    for path_template, label in subresources:
        operation = op_by_path(path_template)
        if not operation:
            sections.append(f"**{label}**\nNot available: GET {path_template} is not present in the loaded API specification.")
            continue
        result_text = t_call({
            "operation_id": operation["id"],
            "path_params": {"id": device_id},
        })
        try:
            result = json.loads(result_text)
        except (TypeError, ValueError):
            sections.append(f"**{label}**\nNot available: NinjaOne returned an unreadable response.")
            continue
        if result.get("error"):
            sections.append(f"**{label}**\nNot retrieved: {result.get('error')}.")
            continue
        rows = result.get("rows") or []
        if not rows:
            sections.append(f"**{label}**\nNo records returned.")
            continue
        rendered = [format_record_summary(row, title=f"{label[:-1] if label.endswith('s') else label} {index}")
                    for index, row in enumerate(rows, 1) if isinstance(row, dict)]
        sections.append(f"**{label}**\n" + ("\n\n".join(rendered) if rendered else "No readable records returned."))

    sections.append(
        "This summary is based on the device detail and hardware/network endpoints listed above. "
        "It does not establish a complete health assessment; unavailable endpoints are identified individually."
    )
    return "\n\n".join(sections)

def recipe_devices_in(q):
    m = re.search(r"\b(?:device|machine|computer|endpoint)s?\s+(?:in|of|at|under|for)\s+(?:the\s+)?"
                  r"(?:organization\s+|org\s+|client\s+)?(.+?)\s*[?.!]*$", q.lower())
    if not m or re.search(r"\b(how|many|count|per|each|which|what|all)\b", m.group(1)):
        return None
    dop = op_by_path("/v2/devices")
    ms = json.loads(t_resolve({"kind": "organization", "name": m.group(1)})).get("matches", [])
    if not dop or not ms or ms[0].get("match", 0) < 0.9 or (len(ms) > 1 and ms[1]["match"] > ms[0]["match"] - 0.2):
        return None
    fetched = fetch(dop, "/v2/devices", {})
    devs = [d for d in fetched if isinstance(d, dict)]
    if not any("organizationId" in d for d in devs):
        return None
    n = sum(1 for d in devs if d.get("organizationId") == ms[0].get("id"))
    if getattr(fetched, "complete", True) is False:
        return (f'At least {n} devices in {ms[0].get("name")} were found in the returned page; '
                f'the organization-wide total is unknown. {_partial_note(fetched)}')
    return f'{n} devices in {ms[0].get("name")}.'



def recipe_count_device_prefix(q):
    """Count device names by prefix using the complete spec-aware inventory fetch."""
    text = re.sub(r"""["']""", "", (q or "").strip())
    if not re.search(r"\b(?:count|how many|number of|total)\b", text, re.I):
        return None
    if not re.search(r"\b(?:devices?|machines?|computers?|endpoints?)\b", text, re.I):
        return None

    patterns = (
        r"\b(?:device|machine|computer|endpoint)s?\b.*?\b(?:names?|hostnames?)\s+(?:start|starts|begin|begins)\s+with\s+([a-z0-9._-]+)",
        r"\b(?:devices?|machines?|computers?|endpoints?)\s+(?:that\s+)?(?:start|starts|begin|begins)\s+with\s+([a-z0-9._-]+)",
        r"\b(?:names?|hostnames?)\s+(?:starting|beginning)\s+with\s+([a-z0-9._-]+)",
    )
    match = next((re.search(pattern, text, re.I) for pattern in patterns
                  if re.search(pattern, text, re.I)), None)
    if not match:
        return None

    prefix = match.group(1).rstrip(".,;:!?")
    if not prefix:
        return None
    operation = op_by_path("/v2/devices")
    if not operation:
        return None
    fetched = fetch(operation, "/v2/devices", {})
    name_fields = ("systemName", "name", "displayName", "hostname")
    matched = 0
    for row in fetched:
        if not isinstance(row, dict):
            continue
        device_name = next((str(row.get(field)).strip() for field in name_fields
                            if row.get(field) not in (None, "")), "")
        if device_name.casefold().startswith(prefix.casefold()):
            matched += 1

    if getattr(fetched, "complete", True) is False:
        return (f'At least {matched} devices have names starting with "{prefix}" in the returned page; '
                f'the complete total is unknown.{_partial_note(fetched)}')
    return f'{matched} devices have names starting with "{prefix}".'


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
            partial = _partial_note(rows)
            if set(words) & {"many", "count", "number", "total"}:
                if partial:
                    return f"At least {len(rows)} {label} were returned by NinjaOne; the total is unknown.{partial}"
                return f"{len(rows)} {label} in NinjaOne."
            names = [row_name(r) for r in rows[:50]]
            heading = f"{len(rows)} {label} in the returned result"
            return heading + partial + ": " + ", ".join(names) + (" ..." if len(rows) > 50 else "")
    try:
        for rec in (recipe_latest_end_user, recipe_count_device_prefix, recipe_device_summary, recipe_device_details, recipe_person_device_access, recipe_list_entity_fields, recipe_list_organizations, recipe_list_technician_names, recipe_per_org, recipe_search_people, recipe_count_named, recipe_find, recipe_devices_in):
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
