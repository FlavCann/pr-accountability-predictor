"""Response shapes -- provenance is part of the contract, not an extra."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class Provenance(BaseModel):
    source_id: str
    source_title: Optional[str] = None
    source_url: str
    published: Optional[str] = None
    start_char: int
    end_char: int
    t_start: Optional[float] = None
    video_url: Optional[str] = None
    verbatim: str
    prompt_version: str
    model: str


class PromiseOut(BaseModel):
    id: str
    company: str
    claim: str
    promise_type: str
    hedge_level: str
    deadline: Optional[str] = None
    deadline_raw: Optional[str] = None
    thread_id: Optional[str] = None
    provenance: Provenance


class ThreadOut(BaseModel):
    id: str
    company: str
    claim: str
    promise_type: str
    deadline: Optional[str] = None
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None
    observations: int
    restatements: int
    silent: bool
    verdict: Optional[str] = None
    confidence: Optional[float] = None


class VerdictOut(BaseModel):
    id: str
    thread_id: str
    status: str
    as_of: str
    rationale: str
    confidence: float
    escalated: bool
    adjudicated_by: str
    review_action: Optional[str] = None
    evidence: List[dict] = []


class ReviewIn(BaseModel):
    action: str  # accepted | corrected | rejected
    corrected_status: Optional[str] = None
    note: Optional[str] = None
    reviewer: str = "api"


def provenance(store, p) -> Provenance:
    src = store.get_source(p.source_id)
    return Provenance(
        source_id=p.source_id,
        source_title=src.title if src else None,
        source_url=src.url if src else "",
        published=src.published.isoformat() if src and src.published else None,
        start_char=p.start_char,
        end_char=p.end_char,
        t_start=p.t_start,
        video_url=p.video_url(src.url) if src else None,
        verbatim=p.verbatim(src.text) if src else "",
        prompt_version=p.prompt_version,
        model=p.model,
    )


def promise_out(store, p) -> PromiseOut:
    return PromiseOut(
        id=p.id,
        company=p.company,
        claim=p.normalized_claim,
        promise_type=p.promise_type.value,
        hedge_level=p.hedge_level.value,
        deadline=p.deadline.isoformat() if p.deadline else None,
        deadline_raw=p.deadline_raw,
        thread_id=p.thread_id,
        provenance=provenance(store, p),
    )
