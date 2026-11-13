#!/usr/bin/env python3
"""NinjaOne fuzzy-question agent.

Read-only, driven at runtime by the NinjaOne OpenAPI spec, tuned for local
~14B models (Qwen) served through an OpenAI-compatible API (Ollama, LM Studio, vLLM).

Design: 3 tools only (search_api, call_api, resolve_entity), a no-LLM fast path for
trivial questions, server-side pagination/filtering/counting, compact tool output.
"""
import difflib, json, math, os, re, sys, time
from collections import Counter

import requests
from openai import OpenAI

SPEC_URL = os.getenv("NINJA_SPEC_URL", "https://app.ninjarmm.com/apidocs/NinjaRMM-API-v2.json")
BASE = os.getenv("NINJA_BASE_URL", "https://app.ninjarmm.com").rstrip("/")  # use your region host
CID, CSECRET = os.getenv("NINJA_CLIENT_ID", ""), os.getenv("NINJA_CLIENT_SECRET", "")
SCOPE = os.getenv("NINJA_SCOPE", "monitoring")
LLM_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:14b")
NO_THINK = os.getenv("NO_THINK", "0") == "1"  # set 1 for Qwen3 to skip thinking (faster)
MAX_STEPS, MAX_ROWS, MAX_CHARS, TTL = 6, 25, 6000, 60

OPS, IDX = {}, None
_tok, _cache = {"v": None, "exp": 0.0}, {}


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
    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ninja_spec.json")
    if os.path.exists(cache) and time.time() - os.path.getmtime(cache) < 86400:
        spec = json.load(open(cache))
    else:
        spec = requests.get(SPEC_URL, timeout=60).json()
        json.dump(spec, open(cache, "w"))
    ops = {}
    for path, item in spec.get("paths", {}).items():
        shared = item.get("parameters", [])
        for method, op in item.items():
            if method not in ("get", "post", "put", "patch", "delete"):
                continue
            ps = [deref(spec, p) for p in shared + op.get("parameters", [])]
            ps = [{"name": p["name"], "in": p.get("in"), "required": bool(p.get("required"))}
                  for p in ps if isinstance(p, dict) and p.get("name")]
            oid = op.get("operationId") or f"{method}_{path}"
            ops[oid] = dict(id=oid, method=method.upper(), path=path,
                            summary=op.get("summary", ""), tags=op.get("tags", []), params=ps)
    return ops


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
        self.docs = {i: set(toks(" ".join([o["id"], o["summary"], " ".join(o["tags"]),
                                           re.sub(r"[{}]", " ", o["path"])]))) for i, o in ops.items()}
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
            if o["method"] != "GET":  # read-only agent
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
    return f'{o["id"]}: GET {o["path"]} - {o["summary"]} | params: {ps or "none"}'


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


def http_get(path, query=None):
    key = (path, json.dumps(query or {}, sort_keys=True))
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < TTL:
        return hit[1]
    for attempt in (0, 1):
        r = requests.get(BASE + path, params=query, timeout=60,
                         headers={"Authorization": "Bearer " + token(), "Accept": "application/json"})
        if r.status_code == 401 and attempt == 0:
            _tok["v"] = None
            continue
        break
    r.raise_for_status()
    data = r.json() if r.text else []
    _cache[key] = (time.time(), data)
    return data


def unwrap(data):
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in ("results", "data", "items", "content", "records"):
            if isinstance(data.get(k), list):
                return data[k]
        return [data]
    return []


def fetch(op, path, query):
    """Fetch all rows; auto-paginates ops that have after+pageSize (cursor = last id)."""
    names = {p["name"] for p in op["params"]}
    if not ("after" in names and "pageSize" in names):
        return unwrap(http_get(path, query))
    q, rows, last = dict(query), [], None
    q.setdefault("pageSize", 1000)
    for _ in range(200):
        page = unwrap(http_get(path, q))
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
    hits = IDX.search(a.get("query", ""), 5)
    return "\n".join(describe(o) for o in hits) or err("no match; try different keywords")


def t_call(a):
    op = OPS.get(a.get("operation_id", ""))
    if not op:
        return err("unknown operation_id; call search_api first")
    if op["method"] != "GET":
        return err("read-only agent: only GET operations are allowed")
    path = op["path"]
    pp = a.get("path_params") or {}
    for name in re.findall(r"\{([^}]+)\}", path):
        if name not in pp:
            return err(f"missing path_params.{name}")
        path = path.replace("{" + name + "}", str(pp[name]))
    q = a.get("query") or {}
    valid = {p["name"] for p in op["params"] if p["in"] == "query"}
    bad = [k for k in q if k not in valid]
    if bad:
        return err(f"unknown query params {bad}; valid: {sorted(valid)}")
    rows = fetch(op, path, q)
    res = shape(rows, a.get("where"), a.get("contains"), a.get("group_by"),
                a.get("fields"), bool(a.get("count_only")), a.get("limit"))
    return json.dumps(res, default=str)[:MAX_CHARS]


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
    top = [dict(slim(r), match=round(s, 2)) for s, r in scored[:5] if s > 0.45]
    return json.dumps({"matches": top, "note": "if several close matches or none, ask the user"}, default=str)


DISPATCH = {"search_api": t_search, "call_api": t_call, "resolve_entity": t_resolve}

TOOLS = [
    {"type": "function", "function": {
        "name": "search_api", "description": "Find NinjaOne GET endpoints by keywords. Returns operation_id, path, params.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "call_api",
        "description": "Call a GET operation. Auto-paginates. Use count_only for counts, group_by for 'per X', "
                       "where for exact field match, contains for text match, fields to pick columns.",
        "parameters": {"type": "object", "properties": {
            "operation_id": {"type": "string"}, "path_params": {"type": "object"}, "query": {"type": "object"},
            "count_only": {"type": "boolean"}, "group_by": {"type": "string"}, "where": {"type": "object"},
            "contains": {"type": "string"}, "fields": {"type": "array", "items": {"type": "string"}},
            "limit": {"type": "integer"}}, "required": ["operation_id"]}}},
    {"type": "function", "function": {
        "name": "resolve_entity", "description": "Fuzzy-match a name to an id. kind: organization|device|end_user|technician|user.",
        "parameters": {"type": "object", "properties": {"kind": {"type": "string"}, "name": {"type": "string"}},
                       "required": ["kind", "name"]}}},
]

SYSTEM = """You answer questions about a NinjaOne tenant using tools. Read-only. Never guess numbers.
Rules:
1. A name (person, organization, device) -> resolve_entity first, then use its id.
2. Otherwise search_api, then call_api with the exact operation_id and parameter names it returned.
3. Counts: call_api with count_only=true (the tool paginates; never count rows yourself). "Per X": group_by.
   Filtering by organization: where={"organizationId": <id>} (check field names in a sample if unsure).
4. Unsure of field names: call once with limit=3 and read them. On an error, fix the call using its hint.
5. "How many devices does PERSON have access to": resolve the person as end_user AND technician. End user =
   assigned devices; technician = devices in organizations they may access; device owner = owner field. If the
   meaning or the match is ambiguous, ask ONE short question. Always say what the number measures.
6. If an API has no way to answer, say so plainly. Do not invent endpoints.
Answer in 1-3 short sentences with the number first. Use at most 5 tool steps."""


# ---------------------------------------------------------------- fast path (no LLM)
FILLER = set("how many count number of total the all we have do are there is in ninjaone ninja ninjarmm "
             "our my a an show me give list what whats".split())
DEV = {"device", "machine", "computer", "endpoint"}
ORG = {"organization", "org", "client", "customer", "company"}


def fast_path(q):
    words = re.findall(r"[a-z]+", q.lower())
    core = {(w[:-1] if len(w) > 3 and w.endswith("s") else w) for w in words if w not in FILLER}
    if not core or not (core <= DEV or core <= ORG):
        return None
    path, label = ("/v2/devices", "devices") if core <= DEV else ("/v2/organizations", "organizations")
    op = op_by_path(path)
    if not op:
        return None
    rows = fetch(op, path, {})
    if set(words) & {"many", "count", "number", "total"}:
        return f"{len(rows)} {label} in NinjaOne."
    names = [str(r.get("name") or r.get("systemName") or r.get("id")) for r in rows[:50]]
    return f"{len(rows)} {label}: " + ", ".join(names) + (" ..." if len(rows) > 50 else "")


# ---------------------------------------------------------------- agent loop
llm = None


def clean(t):
    return re.sub(r"<think>.*?</think>", "", t or "", flags=re.S).strip()


def ask(q, history):
    fp = fast_path(q)
    if fp:
        return fp
    msgs = [{"role": "system", "content": SYSTEM + ("\n/no_think" if NO_THINK else "")}] + history[-6:]
    msgs.append({"role": "user", "content": q})
    for _ in range(MAX_STEPS):
        m = llm.chat.completions.create(model=LLM_MODEL, messages=msgs, tools=TOOLS, temperature=0).choices[0].message
        if not m.tool_calls:
            return clean(m.content)
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
