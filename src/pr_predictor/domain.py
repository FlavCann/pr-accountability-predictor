"""The object graph.

Source -> Segment -> Promise -> PromiseThread -> Evidence -> Verdict -> Prediction

Two design decisions carry most of the weight:

1. A Promise stores CHARACTER OFFSETS into its Source, not a retyped snippet.
   Verbatim fidelity then becomes a substring assertion we can run in CI
   (`Promise.verify_span`) rather than an instruction we hope the model
   followed. It also means speech-recognition errors stay in the source where
   they belong and are corrected at display time.

2. A Segment carries a timestamp range, so every claim resolves to a clickable
   moment in the video (`Promise.video_url`). The original pipeline discarded
   timestamps at write time, which made this impossible.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import date, datetime
from typing import List, Optional

from pydantic import BaseModel, Field

from .taxonomy import HedgeLevel, PromiseType, VerdictStatus


def _id(prefix: str) -> str:
    return "{}_{}".format(prefix, uuid.uuid4().hex[:12])


def content_hash(text: str) -> str:
    """Stable hash used for idempotency and the result cache."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class Source(BaseModel):
    """A document we ingested: a video transcript, a filing, a press release."""

    id: str = Field(default_factory=lambda: _id("src"))
    company: str
    kind: str = "video_transcript"
    url: str
    title: Optional[str] = None
    published: Optional[date] = None
    retrieved_at: datetime = Field(default_factory=datetime.utcnow)
    text: str = ""
    text_hash: str = ""
    # Provenance for the ingest itself.
    fetcher: str = "youtube_transcript_api"

    def finalise(self) -> "Source":
        self.text_hash = content_hash(self.text)
        return self


class Segment(BaseModel):
    """A timestamped span of a Source.

    For video transcripts one Segment is one caption cue. `start_char` /
    `end_char` index into `Source.text`, which is what lets a Promise found
    anywhere in the document be resolved back to a moment on the tape.
    """

    source_id: str
    index: int
    start_char: int
    end_char: int
    t_start: float
    t_end: float


class Promise(BaseModel):
    """One forward-looking commitment, observed once, in one source."""

    id: str = Field(default_factory=lambda: _id("prm"))
    source_id: str
    company: str

    # --- provenance: the span, not the text ---
    start_char: int
    end_char: int
    t_start: Optional[float] = None
    t_end: Optional[float] = None

    # --- classification ---
    promise_type: PromiseType
    hedge_level: HedgeLevel
    normalized_claim: str
    """A clean restatement for display and threading. The verbatim text is
    always re-read from the source via the offsets; this is never the citation."""

    metric: Optional[str] = None
    target_value: Optional[str] = None
    unit: Optional[str] = None
    deadline: Optional[date] = None
    deadline_raw: Optional[str] = None
    """What the speaker actually said ('by 2025', 'end of fiscal year'), kept
    because fiscal years do not align with calendar years. See CLAUDE.md."""

    speaker: Optional[str] = None
    speaker_role: Optional[str] = None
    materiality: str = "medium"
    restates_prior_target: bool = False

    # --- lineage: which prompt and model produced this row ---
    prompt_version: str
    model: str
    run_id: str
    confidence: float = 0.0

    thread_id: Optional[str] = None

    def verbatim(self, source_text: str) -> str:
        return source_text[self.start_char : self.end_char]

    def verify_span(self, source_text: str) -> bool:
        """True when the offsets point at real, non-empty text.

        Cheap enough to assert on every row at write time, which is the point
        of storing offsets instead of a retyped snippet.
        """
        if self.start_char < 0 or self.end_char > len(source_text):
            return False
        if self.end_char <= self.start_char:
            return False
        return bool(source_text[self.start_char : self.end_char].strip())

    def video_url(self, source_url: str) -> str:
        """Deep link to the moment the promise was made."""
        if self.t_start is None or "youtube.com" not in source_url:
            return source_url
        sep = "&" if "?" in source_url else "?"
        return "{}{}t={}s".format(source_url, sep, int(self.t_start))


class PromiseThread(BaseModel):
    """One commitment, observed across many sources over time.

    The unit of accountability. 'We stick to our target of half a million
    farmers' (2023) and the original 2016 statement are one thread with two
    observations, not two promises. Verdicts attach here.
    """

    id: str = Field(default_factory=lambda: _id("thr"))
    company: str
    canonical_claim: str
    promise_type: PromiseType
    metric: Optional[str] = None
    deadline: Optional[date] = None
    promise_ids: List[str] = Field(default_factory=list)
    """Hydrated from the `thread_members` records on read; never stored here."""
    prompt_version: str = ""
    """Threads are built from one extraction version. Mixing versions would
    count the same passage, extracted twice, as a restatement."""
    first_seen: Optional[date] = None
    last_seen: Optional[date] = None
    opened_by: str = ""

    def restatement_count(self) -> int:
        return max(0, len(self.promise_ids) - 1)


class Evidence(BaseModel):
    """Something later that bears on a thread."""

    id: str = Field(default_factory=lambda: _id("evd"))
    thread_id: str
    source_id: Optional[str] = None
    external_url: Optional[str] = None
    excerpt: str
    polarity: str  # supports | contradicts | ambiguous
    found_by: str  # which tool surfaced it
    run_id: str = ""


class Verdict(BaseModel):
    """An adjudication of a thread, as of a date, with its evidence chain."""

    id: str = Field(default_factory=lambda: _id("vrd"))
    thread_id: str
    status: VerdictStatus
    as_of: date
    rationale: str
    evidence_ids: List[str] = Field(default_factory=list)
    adjudicated_by: str = ""
    """Model that made the call, e.g. 'claude-sonnet-5' or, when escalated,
    the advisor model."""
    escalated: bool = False
    initial_status: Optional[VerdictStatus] = None
    """When escalated: the executor's call BEFORE consulting the advisor. Kept
    so the audit trail shows whether, and how, the advisor changed the outcome."""
    advisor_model: Optional[str] = None
    confidence: float = 0.0

    # --- human review (concept 10: outcomes come from what analysts accept) ---
    reviewed_by: Optional[str] = None
    review_action: Optional[str] = None  # accepted | corrected | rejected
    corrected_status: Optional[VerdictStatus] = None
    review_note: Optional[str] = None

    def final_status(self) -> VerdictStatus:
        return self.corrected_status or self.status


class Prediction(BaseModel):
    """Likelihood a still-open thread will be kept.

    Gradeable in arrears: when the deadline passes, reality scores it for free.
    """

    id: str = Field(default_factory=lambda: _id("prd"))
    thread_id: str
    p_kept: float
    horizon: date
    features: dict = Field(default_factory=dict)
    model_version: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Filled in once the horizon passes.
    resolved_status: Optional[VerdictStatus] = None
    brier_score: Optional[float] = None

    def resolve(self, actual: VerdictStatus) -> "Prediction":
        """Grade the prediction against what actually happened."""
        self.resolved_status = actual
        outcome = 1.0 if actual == VerdictStatus.KEPT else 0.0
        self.brier_score = (self.p_kept - outcome) ** 2
        return self


class RunRecord(BaseModel):
    """One row per pipeline run. The run ledger (concept 3).

    Without this there is no way to answer 'what did that cost', 'which prompt
    produced these rows', or 'did the change help'.
    """

    id: str = Field(default_factory=lambda: _id("run"))
    stage: str
    started_at: datetime = Field(default_factory=datetime.utcnow)
    finished_at: Optional[datetime] = None
    model: str = ""
    prompt_version: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    items_ok: int = 0
    items_failed: int = 0
    notes: str = ""
