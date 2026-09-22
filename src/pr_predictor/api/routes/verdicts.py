"""Verdicts, and the analyst review that can correct them."""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException

from ...store import Store
from ...taxonomy import VerdictStatus
from ..deps import get_store, require_tenant
from ..schemas import ReviewIn, VerdictOut

router = APIRouter(tags=["verdicts"])


@router.get("/verdicts", response_model=List[VerdictOut])
def list_verdicts(
    company: Optional[str] = None,
    status: Optional[str] = None,
    store: Store = Depends(get_store),
    tenant: str = Depends(require_tenant),
):
    threads = {t.id: t for t in store.list_threads(company)}
    out = []
    for tid in threads:
        for v in store.list_verdicts(tid):
            if status and v.final_status().value != status:
                continue
            out.append(
                VerdictOut(
                    id=v.id,
                    thread_id=v.thread_id,
                    status=v.final_status().value,
                    as_of=v.as_of.isoformat(),
                    rationale=v.rationale,
                    confidence=v.confidence,
                    escalated=v.escalated,
                    adjudicated_by=v.adjudicated_by,
                    review_action=v.review_action,
                    evidence=[
                        {"id": e.id, "excerpt": e.excerpt, "source_id": e.source_id}
                        for e in store.list_evidence(v.thread_id)
                    ],
                )
            )
    return out


@router.post("/verdicts/{verdict_id}/review")
def review_verdict(
    verdict_id: str, body: ReviewIn, store: Store = Depends(get_store)
):
    """The human checkpoint, and the feedback loop into durable memory.

    A correction is not just recorded -- it becomes retrievable context for
    later runs, which is how the product improves from use rather than only
    from engineering.
    """
    v = store.get_verdict(verdict_id)
    if v is None:
        raise HTTPException(status_code=404, detail="no verdict {}".format(verdict_id))
    corrected = None
    if body.corrected_status:
        try:
            corrected = VerdictStatus(body.corrected_status)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="unknown status; valid: {}".format([s.value for s in VerdictStatus]),
            )
    try:
        store.add_review(
            v.id, body.action, reviewer=body.reviewer,
            corrected_status=corrected, note=body.note,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    thread = store.get_thread(v.thread_id)
    if body.note and thread:
        store.memory_put(thread.company, "correction", thread.id[:24], body.note)
    reviewed = store.get_verdict(v.id)
    return {
        "ok": True,
        "status": reviewed.final_status().value,
        "review_history": [dict(r) for r in store.list_reviews(v.id)],
    }
