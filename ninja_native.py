"""Native NinjaOne backend for N1_local_agent's FastAPI UI (no MCP server needed).

The Foundry base URL is discovered by app.py and passed into run_native, so this
backend follows dynamic Foundry port changes just like the MCP backend.

Keeps the UI contract of app.py: returns (answer, activity) where activity is a list of
{"kind":"call","tool","arguments"} / {"kind":"result","tool","result"} items.
Reuses the spec index, tools and fast path from ninja_agent.py.
"""
import asyncio, json, os, re

import requests
from fastapi import HTTPException
from openai import OpenAI

import ninja_agent as core

MAX_STEPS = 10
_ready, _sessions, _clients = False, {}, {}


# Cache clients by endpoint because Foundry can receive a new port after restart.
def _client(base):
    if base not in _clients:
        _clients[base] = OpenAI(base_url=base, api_key=os.getenv("FOUNDRY_API_KEY", "local"),
                                timeout=120, max_retries=0)
    return _clients[base]


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


def _run_inner(session_id, model_id, content, base):
    _ensure()
    q = _text(content)
    fp = core.fast_path(q)  # no-LLM answer for plain "how many devices" style questions
    if fp:
        return fp, [{"kind": "call", "tool": "fast_path", "arguments": {"question": q}}]

    activity = []
    hist = _sessions.get(session_id, [])[-6:]
    system = core.SYSTEM + core.candidates(q) + "\nReply in plain text only (the UI shows raw text): no markdown, no bold, no tables." + ("\n/no_think" if core.NO_THINK else "")
    msgs = [{"role": "system", "content": system}] + hist + [{"role": "user", "content": content}]
    answer = "I couldn't finish within the step limit. Try a more specific question."
    called = nudged = False
    steps_text = 0
    for _ in range(MAX_STEPS):
        m = _client(base).chat.completions.create(
            model=model_id, messages=msgs, tools=core.TOOLS, temperature=0).choices[0].message
        if not m.tool_calls:
            tc = core.text_call(m.content)
            if tc and steps_text < 3:
                steps_text += 1
                called = True
                out = core.exec_tool(*tc)
                activity.append({"kind": "call", "tool": tc[0], "arguments": tc[1]})
                activity.append({"kind": "result", "tool": tc[0], "result": out[:2500]})
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
            bad = tool_intent or (
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
            try:
                args = json.loads(tc.function.arguments or "{}")
                fn = core.DISPATCH.get(tc.function.name)
                out = fn(args) if fn else core.err("unknown tool")
            except requests.HTTPError as e:
                args, out = {}, core.err(f"HTTP {e.response.status_code}: {e.response.text[:200]}")
            except Exception as e:
                args, out = {}, core.err(f"{type(e).__name__}: {e}")
            activity.append({"kind": "call", "tool": tc.function.name, "arguments": args})
            activity.append({"kind": "result", "tool": tc.function.name, "result": out[:2500]})
            try:
                prepared = json.loads(out)
                if prepared.get("approval_required"):
                    activity.append({"kind": "approval_required", "action": prepared})
            except (ValueError, AttributeError):
                pass
            msgs.append({"role": "tool", "tool_call_id": tc.id, "content": out})
    approval_events = [item for item in activity if item.get("kind") == "approval_required"]
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

def _run(session_id, model_id, content, base):
    core.set_active_session(session_id)
    try:
        return _run_inner(session_id, model_id, content, base)
    finally:
        core.clear_active_session()


async def run_native(session_id, model_id, content, base_url):
    try:
        return await asyncio.to_thread(_run, session_id, model_id, content, base_url)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Native NinjaOne agent failed: {exc}")


def clear_session(session_id):
    _sessions.pop(session_id, None)
    core.clear_pending_session(session_id)
