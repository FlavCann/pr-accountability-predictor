"""Stage 4: extraction (concepts 1, 2, 3, 8).

The model does NOT return character offsets -- language models cannot count
characters reliably across a 10,000-character chunk, and a wrong offset that
lands on some other real text would pass any range check. Instead it copies the
first and last few words of each promise exactly (`quote_start`, `quote_end`),
which models do well, and `locate` finds them in the source and computes the
offsets deterministically.

The citation is therefore always the source text between offsets we computed,
never text the model retyped. A promise whose anchors cannot be found is
UNCITABLE: it is counted and reported, and never written to the store.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from . import config, context, prompts
from .domain import Promise, Source
from .llm import LLMClient, LLMError, RunLedger, Usage
from .taxonomy import HedgeLevel, PromiseType


# ----------------------------------------------------------------------
# What we ask the model for
# ----------------------------------------------------------------------

class ExtractedPromise(BaseModel):
    quote_start: str = Field(
        description="The first 5-10 words of the promise, copied EXACTLY from the "
        "excerpt, including any transcription errors"
    )
    quote_end: str = Field(
        description="The last 5-10 words of the promise, copied EXACTLY from the "
        "excerpt. May overlap quote_start for short promises"
    )
    promise_type: PromiseType
    hedge_level: HedgeLevel
    normalized_claim: str
    metric: Optional[str] = None
    target_value: Optional[str] = None
    unit: Optional[str] = None
    deadline: Optional[str] = None
    deadline_raw: Optional[str] = None
    speaker: Optional[str] = None
    speaker_role: Optional[str] = None
    materiality: str = "medium"
    restates_prior_target: bool = False
    confidence: float = 0.5


class ExtractionResult(BaseModel):
    promises: List[ExtractedPromise] = Field(default_factory=list)


# Legacy v1 shape, kept so the eval harness can run the original prompt.
class V1Promise(BaseModel):
    snippet: str


class V1Result(BaseModel):
    promises: List[V1Promise] = Field(default_factory=list)


# ----------------------------------------------------------------------
# Reconciliation
# ----------------------------------------------------------------------

def _overlaps(a: Tuple[int, int], b: Tuple[int, int], tolerance: int = 40) -> bool:
    """True when two spans refer to substantially the same passage."""
    a0, a1 = a
    b0, b1 = b
    inter = min(a1, b1) - max(a0, b0)
    if inter <= 0:
        return False
    shorter = min(a1 - a0, b1 - b0)
    return inter >= max(1, shorter - tolerance) or inter / max(1, shorter) > 0.6


def reconcile(promises: List[Promise]) -> List[Promise]:
    """Collapse duplicates produced by chunk overlap.

    Keeps the highest-confidence observation of each passage, preferring the
    longer span on a tie because a truncated span is the more common failure.
    """
    ordered = sorted(promises, key=lambda p: (-p.confidence, -(p.end_char - p.start_char)))
    kept: List[Promise] = []
    for p in ordered:
        if any(_overlaps((p.start_char, p.end_char), (k.start_char, k.end_char)) for k in kept):
            continue
        kept.append(p)
    return sorted(kept, key=lambda p: p.start_char)


# ----------------------------------------------------------------------
# Extraction
# ----------------------------------------------------------------------

_WORD = re.compile(r"[A-Za-z0-9]+")


def _anchor_pattern(phrase: str) -> Optional["re.Pattern[str]"]:
    """Match a phrase's words in order, ignoring case, punctuation and spacing.

    The model copies anchors "exactly", but routinely normalises case, adds
    punctuation or tidies spacing. Those differences are irrelevant to *where*
    the text is; changed or missing words are not, and do not match.
    """
    words = _WORD.findall(phrase)
    if not words:
        return None
    return re.compile(r"\b" + r"\W+".join(re.escape(w) for w in words) + r"\b", re.IGNORECASE)


def locate(
    text: str, quote_start: str, quote_end: Optional[str] = None, max_len: int = 1500
) -> Optional[Tuple[int, int]]:
    """Find a span in `text` from its opening and closing words.

    Returns (start, end) offsets into `text`, or None if either anchor cannot
    be found. `quote_end` is searched for at or after the opening anchor, within
    `max_len` characters. If `quote_end` is omitted, the span is the opening
    anchor alone -- which is how v1's whole-snippet quotes are located.
    """
    pat_start = _anchor_pattern(quote_start)
    if pat_start is None:
        return None
    m_start = pat_start.search(text)
    if m_start is None:
        return None
    if not quote_end:
        return m_start.start(), m_start.end()

    pat_end = _anchor_pattern(quote_end)
    if pat_end is None:
        return None
    window_end = min(len(text), m_start.start() + max_len)
    m_end = pat_end.search(text, m_start.start(), window_end)
    if m_end is None:
        return None
    return m_start.start(), max(m_start.end(), m_end.end())


def _to_domain(
    raw: ExtractedPromise,
    chunk,
    source: Source,
    prompt_version: str,
    model: str,
    run_id: str,
) -> Optional[Promise]:
    """Locate the anchors in the chunk, then build an absolute-offset Promise."""
    from datetime import date as _date

    found = locate(chunk.text, raw.quote_start, raw.quote_end)
    if found is None:
        return None
    start, end = chunk.offset + found[0], chunk.offset + found[1]

    deadline = None
    if raw.deadline:
        try:
            deadline = _date.fromisoformat(raw.deadline[:10])
        except ValueError:
            deadline = None

    p = Promise(
        source_id=source.id,
        company=source.company,
        start_char=start,
        end_char=end,
        promise_type=raw.promise_type,
        hedge_level=raw.hedge_level,
        normalized_claim=raw.normalized_claim,
        metric=raw.metric,
        target_value=raw.target_value,
        unit=raw.unit,
        deadline=deadline,
        deadline_raw=raw.deadline_raw,
        speaker=raw.speaker,
        speaker_role=raw.speaker_role,
        materiality=raw.materiality,
        restates_prior_target=raw.restates_prior_target,
        prompt_version=prompt_version,
        model=model,
        run_id=run_id,
        confidence=raw.confidence,
    )
    return p if p.verify_span(source.text) else None


def attach_timestamps(store, promises: List[Promise]) -> int:
    """Resolve each promise's start offset to a video timestamp.

    Only possible because ingest kept the segment table. Sources ingested from
    the legacy .txt files have no segments and are skipped.
    """
    n = 0
    for p in promises:
        seg = store.segment_at(p.source_id, p.start_char)
        if seg is not None:
            p.t_start = seg["t_start"]
            end_seg = store.segment_at(p.source_id, max(p.start_char, p.end_char - 1))
            p.t_end = end_seg["t_end"] if end_seg is not None else seg["t_end"]
            n += 1
    return n


def extract_source(
    store,
    client: LLMClient,
    source: Source,
    prompt_version: str = prompts.DEFAULT_EXTRACTION_PROMPT,
    run_id: str = "",
    stable_context: Optional[str] = None,
) -> Tuple[List[Promise], Usage, int]:
    """Extract every promise from one source, over the whole text.

    No truncation. The original pipeline sent the first 1,000 words, which for
    these transcripts is between 8% and 36% of the document.

    Returns (promises, usage, uncitable). `uncitable` counts extractions whose
    anchors could not be found; the eval scores them against span fidelity
    rather than letting them vanish.
    """
    prompt = prompts.get(prompt_version)
    chunks = context.chunk_text(source.text)
    if stable_context is None:
        stable_context = context.build_extraction_context(store, source.company)

    total = Usage()
    found: List[Promise] = []
    uncitable = 0

    for chunk in chunks:
        question = (
            "{}\n\nTRANSCRIPT EXCERPT (chunk {} of {}). quote_start and quote_end "
            "must be copied exactly from THIS excerpt.\n\n"
            "Source: {}\nDate: {}\n\n<<<EXCERPT>>>\n{}\n<<<END EXCERPT>>>".format(
                stable_context,
                chunk.index + 1,
                len(chunks),
                source.title or source.url,
                source.published.isoformat() if source.published else "unknown",
                chunk.text,
            )
        )
        try:
            result = client.structured(
                system=prompt.system,
                question=question,
                output_model=ExtractionResult,
                cached_context=None,
                prompt_version=prompt_version,
            )
        except LLMError as e:
            # Per-item isolation: one bad chunk must not lose the rest.
            print("  chunk {} failed: {}".format(chunk.index, e))
            continue

        total += result.usage
        parsed = result.parsed
        for raw in getattr(parsed, "promises", []) or []:
            p = _to_domain(raw, chunk, source, prompt_version, client.model, run_id)
            if p is None:
                uncitable += 1
            else:
                found.append(p)

    deduped = reconcile(found)
    attach_timestamps(store, deduped)
    return deduped, total, uncitable


def extract_source_v1(
    client: LLMClient,
    source: Source,
    truncate_words: Optional[int] = None,
) -> Tuple[List[Promise], Usage, int]:
    """Reproduce the ORIGINAL pipeline, for use as a measured eval baseline only.

    One call, the v1 prompt, free-text snippets -- exactly as the pre-0.2
    `extract_promises.py` did, optionally with its 1,000-word truncation.

    Snippets are located with the same `locate` used for extract-v3, so the
    two are compared on equal terms. A snippet that cannot be found is
    UNCITABLE: the model paraphrased or garbled what it was told to copy.

    Returns (promises, usage, uncitable_count). Never writes to the store.
    """
    prompt = prompts.get("extract-v1")
    text = source.text
    if truncate_words:
        text = " ".join(text.split()[:truncate_words])

    result = client.structured(
        system=prompt.system,
        question=text,
        output_model=V1Result,
        prompt_version="extract-v1" + ("-trunc{}".format(truncate_words) if truncate_words else ""),
        max_tokens=16000,
    )

    promises: List[Promise] = []
    uncitable = 0
    for item in getattr(result.parsed, "promises", []) or []:
        snippet = (item.snippet or "").strip()
        found = locate(source.text, snippet) if snippet else None
        if found is None:
            uncitable += 1
            continue
        promises.append(
            Promise(
                source_id=source.id,
                company=source.company,
                start_char=found[0],
                end_char=found[1],
                # v1 does not classify. Placeholders only; the eval does not
                # score type or hedge for v1.
                promise_type=PromiseType.PROCEDURAL_COMMITMENT,
                hedge_level=HedgeLevel.FIRM,
                normalized_claim=snippet,
                prompt_version="extract-v1",
                model=client.model,
                run_id="eval",
            )
        )
    return reconcile(promises), result.usage, uncitable


def extract_all(
    store,
    client: LLMClient,
    company: Optional[str] = None,
    prompt_version: str = prompts.DEFAULT_EXTRACTION_PROMPT,
) -> dict:
    """Extract over every source in the store."""
    sources = store.list_sources(company)
    if not sources:
        return {"sources": 0, "promises": 0, "cost_usd": 0.0}

    stats = {
        "sources": 0, "skipped": 0, "promises": 0, "uncitable": 0,
        "failed": 0, "cost_usd": 0.0,
    }
    with RunLedger(
        store, "extract", model=client.model, prompt_version=prompt_version
    ) as run:
        # The stable prefix is identical for every chunk of every source for a
        # company, so build it once and let prompt caching do the rest.
        by_company: Dict[str, str] = {}
        for src in sources:
            # Idempotency: a source already extracted by this prompt and model
            # is skipped, not re-extracted into duplicate rows.
            if store.has_extraction(src.id, prompt_version, client.model):
                stats["skipped"] += 1
                continue
            if src.company not in by_company:
                by_company[src.company] = context.build_extraction_context(store, src.company)

            try:
                promises, usage, uncitable = extract_source(
                    store,
                    client,
                    src,
                    prompt_version=prompt_version,
                    run_id=run.run_id,
                    stable_context=by_company[src.company],
                )
            except Exception as e:  # noqa: BLE001 - per-item isolation
                run.fail(note="{}: {}".format(src.id, type(e).__name__))
                stats["failed"] += 1
                continue

            run.add(usage)
            store.add_promises(promises)
            store.record_extraction(
                src.id, prompt_version, client.model, run.run_id, len(promises)
            )
            run.ok()
            stats["sources"] += 1
            stats["promises"] += len(promises)
            stats["uncitable"] += uncitable
            print(
                "  {:<52} {:>3} promises".format(
                    (src.title or src.url)[:52], len(promises)
                )
            )

        stats["cost_usd"] = run.record.cost_usd
    return stats
