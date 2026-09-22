"""Stage 5: threading promises into commitments.

A PromiseThread is one commitment observed across many sources over time. This
is the unit of accountability: "we stick to our target of half a million
farmers" (2023) and the original 2016 statement are one thread with two
observations, not two unrelated promises.

Matching is deliberately deterministic first. Metric plus deadline plus token
overlap resolves the great majority of cases at zero cost and with reproducible
results; the model is only consulted where the cheap signals are ambiguous.
Reaching for an LLM on a problem that `set.intersection` solves is a common and
expensive mistake.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel

from .domain import Promise, PromiseThread
from .llm import RunLedger, Usage

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "in", "is", "it", "its", "of", "on", "or", "our", "that", "the", "to", "we",
    "will", "with", "this", "these", "those", "be", "been", "being", "us",
}


def _tokens(text: str) -> set:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def similarity(a: str, b: str) -> float:
    """Jaccard overlap of content words. Cheap, deterministic, good enough."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _match_score(p: Promise, t: PromiseThread) -> float:
    """How strongly a promise belongs to an existing thread."""
    if p.promise_type != t.promise_type:
        return 0.0

    # A conflicting deadline is decisive, not a penalty. The same commitment
    # with a new date is a *revised* target and must stay a separate thread so
    # the revision stays visible -- Barry Callebaut moved its 100% sustainable
    # ingredients target from 2025 to 2030, and merging those would erase one of
    # the most newsworthy facts in the corpus. A weighted penalty is not enough:
    # with identical claim wording the similarity term overwhelms it.
    if p.deadline and t.deadline and p.deadline != t.deadline:
        return 0.0

    score = similarity(p.normalized_claim, t.canonical_claim)

    # A shared metric is a strong signal.
    if p.metric and t.metric and similarity(p.metric, t.metric) > 0.5:
        score += 0.25
    if p.deadline and t.deadline:
        score += 0.15

    # The model told us this is a restatement; trust it as a tiebreaker only.
    if p.restates_prior_target:
        score += 0.10

    return score


MATCH_THRESHOLD = 0.45
AMBIGUOUS_BAND = (0.15, 0.45)
"""Scores in this band are genuinely uncertain and go to the model.

The band is wide on purpose. Token overlap handles rewordings of the same
phrase well ("500,000 farmers" vs "half a million farmers" scores 0.56) but
badly when the same commitment is expressed in entirely different words -- "in
Q1 we will give an update on midterm guidance" against "at the end of the first
quarter we will give you our new midterm guidance" scores only 0.25 despite
being the same promise restated minutes later. Those are exactly the cases
worth spending a model call on, and there are few of them.
"""


class ThreadMatch(BaseModel):
    same_commitment: bool
    reason: str = ""


THREAD_MATCH_PROMPT = "thread-match-v1"

TIEBREAK_SYSTEM = (
    "You decide whether two statements refer to the SAME underlying "
    "commitment, restated, or to two different commitments. A "
    "restatement may use entirely different words. Two commitments "
    "that share a topic but differ in what is promised, in the "
    "quantity, or in the deadline are DIFFERENT -- a revised target "
    "must stay separate from the original so the revision is visible."
)


def _resolve_ambiguous(
    client, promise: Promise, thread: PromiseThread
) -> Tuple[bool, Usage, Optional[str]]:
    """Ask the model whether an uncertain pair is really the same commitment.

    Uses the pipeline's executor model: a narrow yes/no judgement, not an
    advisor consultation. Returns (same_commitment, usage, error). Usage is
    returned rather than dropped so the link stage's spend reaches the ledger.
    """
    if client is None or getattr(client, "dry_run", False):
        return False, Usage(), None
    try:
        result = client.structured(
            system=TIEBREAK_SYSTEM,
            question='EXISTING COMMITMENT:\n"{}"\n\nNEW STATEMENT:\n"{}"\n\n'
            "Same commitment restated?".format(
                thread.canonical_claim, promise.normalized_claim
            ),
            output_model=ThreadMatch,
            prompt_version=THREAD_MATCH_PROMPT,
        )
    except Exception as e:  # noqa: BLE001 - a failed tie-break just means "don't merge"
        return False, Usage(), type(e).__name__
    return bool(getattr(result.parsed, "same_commitment", False)), result.usage, None


def build_threads(
    store,
    company: str,
    prompt_version: Optional[str] = None,
    client=None,
) -> Dict[str, object]:
    """Assign every not-yet-threaded promise of one extraction version to a thread.

    Idempotent by construction: only promises with no membership record are
    considered, and `thread_members` is keyed on the promise, so a second run
    finds nothing to do rather than linking everything again.

    Scoped to one `prompt_version`. Threading v1 and v3 extractions together
    would count the same passage, extracted twice, as a restatement.

    Processing in chronological order matters: the earliest statement of a
    commitment becomes the thread's canonical claim, so a thread reads as
    "stated in 2016, restated in 2021 and 2023" rather than the reverse.

    Recorded in the run ledger like every other stage that can spend money:
    tie-break tokens and cost land in a `link` row, and every membership
    carries that run's ID. A run with nothing to link writes no ledger row.
    """
    from . import prompts

    version = prompt_version or prompts.DEFAULT_EXTRACTION_PROMPT
    promises = store.list_promises(
        company=company, prompt_version=version, unthreaded_only=True
    )
    threads: List[PromiseThread] = list(store.list_threads(company, prompt_version=version))
    stats: Dict[str, object] = {
        "threads": len(threads), "new_threads": 0, "linked": 0, "ambiguous": 0,
        "escalated_merges": 0, "tiebreak_errors": 0, "cost_usd": 0.0,
    }
    if not promises:
        stats["restated"] = sum(1 for t in threads if t.restatement_count() > 0)
        return stats

    pub: Dict[str, object] = {src.id: src.published for src in store.list_sources(company)}
    promises.sort(
        key=lambda p: (pub.get(p.source_id) is None, pub.get(p.source_id), p.start_char)
    )

    with RunLedger(
        store, "link",
        model=getattr(client, "model", "") or "",
        prompt_version=THREAD_MATCH_PROMPT,
    ) as run:
        run.record.notes = "threading {} extractions".format(version)
        for p in promises:
            thread = _link_one(store, p, threads, pub, version, client, run, stats)
            store.put_thread(thread)  # summary first, so the membership has a home
            store.add_member(thread.id, p.id, run_id=run.run_id)
            run.ok()

    stats["threads"] = len(threads)
    stats["linked"] = len(promises)
    stats["restated"] = sum(1 for t in threads if t.restatement_count() > 0)
    stats["cost_usd"] = round(run.record.cost_usd, 6)
    return stats


def _link_one(store, p, threads, pub, version, client, run, stats) -> PromiseThread:
    """Find or create the thread for one promise, updating its summary."""
    best: Tuple[float, Optional[PromiseThread]] = (0.0, None)
    for t in threads:
        s = _match_score(p, t)
        if s > best[0]:
            best = (s, t)

    score, thread = best
    merge = thread is not None and score >= MATCH_THRESHOLD

    # Cheap signals were inconclusive -- spend a model call rather than guess.
    # This is the only place threading costs anything, so it is metered.
    if thread is not None and AMBIGUOUS_BAND[0] <= score < AMBIGUOUS_BAND[1]:
        stats["ambiguous"] += 1
        same, usage, error = _resolve_ambiguous(client, p, thread)
        run.add(usage)
        if error:
            stats["tiebreak_errors"] += 1
            # The promise is still linked (as its own thread), so this is a
            # note on the run, not a failed item.
            run.fail(n=0, note="tie-break {}: {}".format(p.id, error))
        if same:
            merge = True
            stats["escalated_merges"] += 1

    published = pub.get(p.source_id)
    if merge:
        if published:
            if thread.first_seen is None or published < thread.first_seen:
                thread.first_seen = published
            if thread.last_seen is None or published > thread.last_seen:
                thread.last_seen = published
        # A restatement can supply a deadline the original lacked.
        if thread.deadline is None and p.deadline:
            thread.deadline = p.deadline
        if thread.metric is None and p.metric:
            thread.metric = p.metric
    else:
        thread = PromiseThread(
            company=p.company,
            canonical_claim=p.normalized_claim,
            promise_type=p.promise_type,
            metric=p.metric,
            deadline=p.deadline,
            prompt_version=version,
            first_seen=published,
            last_seen=published,
            opened_by=p.id,
        )
        threads.append(thread)
        stats["new_threads"] += 1

    thread.promise_ids.append(p.id)
    return thread


def silence_signal(store, thread: PromiseThread) -> dict:
    """Evidence for the `quietly_dropped` verdict.

    A commitment that stops being mentioned while the company keeps publishing
    is the thing a human analyst is worst at spotting, because noticing an
    absence requires remembering everything. It is trivial for a threaded
    corpus.

    Returns the count of sources published after the thread was last mentioned,
    which the adjudicator weighs. Two or more is the threshold for considering
    `quietly_dropped` at all.
    """
    if thread.last_seen is None:
        return {"sources_since": 0, "silent": False, "last_mentioned": None}

    later = [
        s
        for s in store.list_sources(thread.company)
        if s.published and s.published > thread.last_seen
    ]
    return {
        "sources_since": len(later),
        "silent": len(later) >= 2,
        "last_mentioned": thread.last_seen.isoformat(),
        "later_sources": [
            {"title": s.title, "date": s.published.isoformat()} for s in later
        ],
    }
