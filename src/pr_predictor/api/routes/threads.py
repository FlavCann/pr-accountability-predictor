"""Commitment threads, and each thread's full audit trail."""

from __future__ import annotations

import re
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException

from ...store import Store
from ..deps import get_store, require_tenant
from ..schemas import ThreadOut, promise_out

router = APIRouter(tags=["threads"])


@router.get("/threads", response_model=List[ThreadOut])
def list_threads(
    company: Optional[str] = None,
    silent_only: bool = False,
    store: Store = Depends(get_store),
    tenant: str = Depends(require_tenant),
):
    from ...threading_ import silence_signal

    out = []
    for t in store.list_threads(company):
        signal = silence_signal(store, t)
        if silent_only and not signal["silent"]:
            continue
        v = store.latest_verdict(t.id)
        out.append(
            ThreadOut(
                id=t.id,
                company=t.company,
                claim=t.canonical_claim,
                promise_type=t.promise_type.value,
                deadline=t.deadline.isoformat() if t.deadline else None,
                first_seen=t.first_seen.isoformat() if t.first_seen else None,
                last_seen=t.last_seen.isoformat() if t.last_seen else None,
                observations=len(t.promise_ids),
                restatements=t.restatement_count(),
                silent=signal["silent"],
                verdict=v.final_status().value if v else None,
                confidence=v.confidence if v else None,
            )
        )
    return out


@router.get("/threads/{thread_id}")
def get_thread(thread_id: str, store: Store = Depends(get_store)):
    from ... import tools

    data = tools.get_promise_thread(store, thread_id)
    if "error" in data:
        raise HTTPException(status_code=404, detail=data["error"])
    v = store.latest_verdict(thread_id)
    data["verdict"] = (
        {
            "status": v.final_status().value,
            "rationale": v.rationale,
            "confidence": v.confidence,
            "as_of": v.as_of.isoformat(),
            "escalated": v.escalated,
        }
        if v
        else None
    )
    return data


_EVIDENCE_ID = re.compile(r"evd_[0-9a-f]{12}")


@router.get("/threads/{thread_id}/audit")
def thread_audit(thread_id: str, store: Store = Depends(get_store)):
    """A thread's full trail: every observation with its provenance, then every
    verdict with the evidence it rests on.

    Evidence is flagged `cited` when the rationale names it. A verdict carries
    every evidence row the agent gathered, but usually argues from a few.
    """
    from ...threading_ import silence_signal

    t = store.get_thread(thread_id)
    if t is None:
        raise HTTPException(status_code=404, detail="no thread {}".format(thread_id))

    members = set(t.promise_ids)
    promises = [
        promise_out(store, p)
        for p in store.list_promises(company=t.company)
        if p.id in members
    ]
    promises.sort(key=lambda p: (p.provenance.published or "", p.provenance.start_char))

    sources = {}

    def source_meta(source_id):
        if source_id not in sources:
            src = store.get_source(source_id) if source_id else None
            sources[source_id] = {
                "title": src.title if src else None,
                "published": src.published.isoformat() if src and src.published else None,
                "url": src.url if src else None,
            }
        return sources[source_id]

    evidence = {e.id: e for e in store.list_evidence(thread_id)}
    verdicts = []
    for v in store.list_verdicts(thread_id):
        cited = set(_EVIDENCE_ID.findall(v.rationale))
        verdicts.append({
            "id": v.id,
            "status": v.final_status().value,
            "model_status": v.status.value,
            "initial_status": v.initial_status.value if v.initial_status else None,
            "as_of": v.as_of.isoformat(),
            "confidence": v.confidence,
            "rationale": v.rationale,
            "adjudicated_by": v.adjudicated_by,
            "escalated": v.escalated,
            "advisor_model": v.advisor_model,
            "reviews": [dict(r) for r in store.list_reviews(v.id)],
            "evidence": [
                {
                    "id": e.id,
                    "excerpt": e.excerpt,
                    "polarity": e.polarity,
                    "found_by": e.found_by,
                    "external_url": e.external_url,
                    "source_id": e.source_id,
                    "source": source_meta(e.source_id),
                    "cited": e.id in cited,
                }
                for e in (evidence[i] for i in v.evidence_ids if i in evidence)
            ],
        })

    return {
        "id": t.id,
        "company": t.company,
        "claim": t.canonical_claim,
        "promise_type": t.promise_type.value,
        "metric": t.metric,
        "deadline": t.deadline.isoformat() if t.deadline else None,
        "first_seen": t.first_seen.isoformat() if t.first_seen else None,
        "last_seen": t.last_seen.isoformat() if t.last_seen else None,
        "prompt_version": t.prompt_version,
        "silence": silence_signal(store, t),
        "promises": [p.model_dump() for p in promises],
        "verdicts": verdicts,
    }
