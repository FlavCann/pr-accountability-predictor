"""The tool surface for the verification agent (concept 5).

Design rules applied here, in order of how often they are got wrong:

1. **Return IDs and short spans, never whole documents.** A tool that returns a
   14,000-token transcript fills the agent's context on its third call. Every
   tool here returns bounded excerpts plus IDs the agent can follow.
2. **Narrow and typed.** No mega-`query()` that takes free text and does
   anything -- that just moves the schema problem into prose.
3. **Errors that say how to recover**, not just what failed.
4. **Idempotent**, so retries are free.

These are plain Python functions plus JSON schemas. An MCP server is a thin
wrapper over this module -- see the deferral note in REPORT.md.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional

MAX_EXCERPT_CHARS = 600
MAX_RESULTS = 8


# ----------------------------------------------------------------------
# Implementations
# ----------------------------------------------------------------------

def search_corpus(
    store,
    company: str,
    query: str,
    after: Optional[str] = None,
    before: Optional[str] = None,
) -> Dict[str, Any]:
    """Keyword search over stored sources, returning bounded excerpts.

    Each hit carries `start_char` (where its excerpt begins) and `match_char`
    (where the matched word is), so the agent can pass `start_char` straight to
    `get_source_excerpt` to read around it. Without positions the two tools
    could not be chained -- the agent could only guess offsets.
    """
    terms = [t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) > 2]
    if not terms:
        return {
            "error": "query had no searchable terms",
            "hint": "pass content words, e.g. 'farmers poverty 500000'",
            "results": [],
        }

    hits: List[Dict[str, Any]] = []
    for src in store.list_sources(company):
        if after and src.published and src.published.isoformat() < after:
            continue
        if before and src.published and src.published.isoformat() > before:
            continue

        low = src.text.lower()
        for term in terms:
            pos = low.find(term)
            if pos == -1:
                continue
            start = max(0, pos - MAX_EXCERPT_CHARS // 2)
            end = min(len(src.text), pos + MAX_EXCERPT_CHARS // 2)
            hits.append(
                {
                    "source_id": src.id,
                    "title": src.title,
                    "date": src.published.isoformat() if src.published else None,
                    "url": src.url,
                    "start_char": start,
                    "match_char": pos,
                    "excerpt": src.text[start:end],
                    "matched": term,
                }
            )
            break

    return {"results": hits[:MAX_RESULTS], "total": len(hits)}


def get_promise_thread(store, thread_id: str) -> Dict[str, Any]:
    """A thread with every observation of it, and the silence signal."""
    from .threading_ import silence_signal

    thread = store.get_thread(thread_id)
    if thread is None:
        return {
            "error": "no thread with id {}".format(thread_id),
            "hint": "call list_open_threads to see valid IDs",
        }

    observations = []
    for pid in thread.promise_ids:
        matches = [p for p in store.list_promises(company=thread.company) if p.id == pid]
        if not matches:
            continue
        p = matches[0]
        src = store.get_source(p.source_id)
        observations.append(
            {
                "promise_id": p.id,
                "date": src.published.isoformat() if src and src.published else None,
                "source_title": src.title if src else None,
                "verbatim": p.verbatim(src.text) if src else "",
                "hedge_level": p.hedge_level.value,
                "deadline_raw": p.deadline_raw,
                "video_url": p.video_url(src.url) if src else None,
            }
        )

    return {
        "thread_id": thread.id,
        "claim": thread.canonical_claim,
        "type": thread.promise_type.value,
        "metric": thread.metric,
        "deadline": thread.deadline.isoformat() if thread.deadline else None,
        "first_seen": thread.first_seen.isoformat() if thread.first_seen else None,
        "last_seen": thread.last_seen.isoformat() if thread.last_seen else None,
        "observations": observations,
        "silence": silence_signal(store, thread),
    }


def list_open_threads(store, company: str, limit: int = 20) -> Dict[str, Any]:
    """Threads with no verdict yet."""
    out = []
    for t in store.list_threads(company):
        if store.latest_verdict(t.id) is None:
            out.append(
                {
                    "thread_id": t.id,
                    "claim": t.canonical_claim,
                    "deadline": t.deadline.isoformat() if t.deadline else None,
                    "observations": len(t.promise_ids),
                }
            )
    return {"threads": out[:limit], "total": len(out)}


def get_source_excerpt(
    store, source_id: str, start_char: int = 0, length: int = MAX_EXCERPT_CHARS
) -> Dict[str, Any]:
    """A bounded window into a source. Never the whole document."""
    src = store.get_source(source_id)
    if src is None:
        return {"error": "no source {}".format(source_id), "hint": "use search_corpus first"}
    # A negative offset would make Python slice from the END of the text and
    # silently return the wrong passage.
    start_char = max(0, start_char)
    length = max(0, min(length, MAX_EXCERPT_CHARS * 4))
    return {
        "source_id": src.id,
        "start_char": start_char,
        "title": src.title,
        "date": src.published.isoformat() if src.published else None,
        "excerpt": src.text[start_char : start_char + length],
        "total_chars": len(src.text),
    }


# ----------------------------------------------------------------------
# Schemas + dispatch
# ----------------------------------------------------------------------

def tool_definitions() -> List[Dict[str, Any]]:
    """Anthropic tool definitions. `strict` keeps arguments schema-valid."""
    return [
        {
            "name": "search_corpus",
            "description": (
                "Keyword search across this company's stored sources. Returns "
                "short excerpts with source IDs, dates and each excerpt's "
                "start_char. Use content words, not questions."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Content words to search for"},
                    "after": {"type": "string", "description": "ISO date lower bound"},
                    "before": {"type": "string", "description": "ISO date upper bound"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "name": "get_promise_thread",
            "description": (
                "Fetch one commitment thread: every time it was stated, the "
                "verbatim wording, and whether the company has gone silent on it."
            ),
            "input_schema": {
                "type": "object",
                "properties": {"thread_id": {"type": "string"}},
                "required": ["thread_id"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "name": "get_source_excerpt",
            "description": (
                "Read a bounded window of a source document (up to 2400 "
                "characters). To read around a search_corpus hit, pass that "
                "hit's source_id and start_char; subtract from start_char to "
                "read what came before it."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "source_id": {"type": "string"},
                    "start_char": {"type": "integer"},
                    "length": {"type": "integer"},
                },
                "required": ["source_id"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    ]


def dispatch(store, company: str, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a tool call. Unknown names return a recoverable error."""
    table: Dict[str, Callable] = {
        "search_corpus": lambda a: search_corpus(store, company, **a),
        "get_promise_thread": lambda a: get_promise_thread(store, **a),
        "get_source_excerpt": lambda a: get_source_excerpt(store, **a),
        "list_open_threads": lambda a: list_open_threads(store, company, **a),
    }
    fn = table.get(name)
    if fn is None:
        return {
            "error": "unknown tool {!r}".format(name),
            "hint": "available: {}".format(", ".join(sorted(table))),
        }
    try:
        return fn(args)
    except TypeError as e:
        return {"error": "bad arguments: {}".format(e), "hint": "check the tool schema"}
