"""Product outcomes and the predictor (concept 10).

Evals measure the component in a lab. Outcomes measure whether the customer got
value in production. They are different numbers with different audiences and
this module keeps them apart.

The unusual property of this product: **prediction hit rate is measurable in
arrears**. When a 2025 deadline passes, every prediction made about it is
graded by reality for free. Very few AI products are self-grading, so the
telemetry is instrumented from the start -- retrofitting it is painful and the
deadlines only pass once.
"""

from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional

from . import rating
from .domain import Prediction, PromiseThread
from .taxonomy import HedgeLevel, VerdictStatus

PREDICTOR_VERSION = "heuristic-v0"

# Base likelihood by how firmly the commitment was stated. These are priors,
# not learned weights: there is no resolved outcome data yet. Once enough
# deadlines have passed, `calibration()` below will show whether they hold and
# the dreaming job (concept 13) can fit real coefficients to replace them.
HEDGE_PRIOR = {
    HedgeLevel.FIRM: 0.70,
    HedgeLevel.INTENDED: 0.55,
    HedgeLevel.CONDITIONAL: 0.40,
    HedgeLevel.ASPIRATIONAL: 0.30,
}


def predict_thread(store, thread: PromiseThread) -> Optional[Prediction]:
    """Estimate the likelihood an open commitment will be kept.

    Deliberately a transparent heuristic rather than an opaque model call. Every
    feature is inspectable and the rationale is auditable, which matters for a
    product whose output is a claim about a named company. Replace with a fitted
    model once `calibration()` has enough resolved rows to justify one.
    """
    if thread.deadline is None:
        return None

    promises = [
        p for p in store.list_promises(company=thread.company) if p.id in thread.promise_ids
    ]
    if not promises:
        return None

    firmest = min(
        promises,
        key=lambda p: list(HEDGE_PRIOR).index(p.hedge_level)
        if p.hedge_level in HEDGE_PRIOR
        else 99,
    )
    p_kept = HEDGE_PRIOR.get(firmest.hedge_level, 0.5)

    features: Dict[str, object] = {"hedge_level": firmest.hedge_level.value}

    # Repeatedly re-affirming a commitment is a credible signal of intent.
    restatements = thread.restatement_count()
    features["restatements"] = restatements
    if restatements >= 2:
        p_kept += 0.10
    elif restatements == 1:
        p_kept += 0.05

    # Going quiet is the strongest negative signal available, and is the
    # mechanism behind the `quietly_dropped` verdict.
    from .threading_ import silence_signal

    silence = silence_signal(store, thread)
    features["sources_since_last_mention"] = silence["sources_since"]
    if silence["silent"]:
        p_kept -= 0.25

    # A deadline already in the past with no verdict is not a good sign.
    if thread.deadline < date.today():
        features["deadline_passed"] = True
        p_kept -= 0.10

    p_kept = max(0.02, min(0.98, p_kept))

    return Prediction(
        thread_id=thread.id,
        p_kept=round(p_kept, 3),
        horizon=thread.deadline,
        features=features,
        model_version=PREDICTOR_VERSION,
    )


def predict_all(store, company: str) -> Dict[str, int]:
    """Predict every dated thread -- appending only when something changed.

    Predictions are records: a new one is written when the inputs moved (a new
    source, a restatement, a deadline passing), never overwritten. Re-running
    on unchanged data writes nothing, which keeps the stage idempotent and
    stops calibration from counting the same judgement twice.
    """
    made = unchanged = 0
    for t in store.list_threads(company):
        pred = predict_thread(store, t)
        if pred is None:
            continue
        prior = store.latest_prediction(t.id)
        if (
            prior is not None
            and prior.model_version == pred.model_version
            and prior.p_kept == pred.p_kept
            and prior.features == pred.features
        ):
            unchanged += 1
            continue
        store.put_prediction(pred)
        made += 1
    return {"predictions": made, "unchanged": unchanged}


def resolve_predictions(store) -> Dict[str, object]:
    """Grade past predictions against verdicts that have since landed.

    This is the loop that turns the passage of time into a free eval.
    """
    resolved = 0
    for pred in store.list_predictions():
        if pred.resolved_status is not None:
            continue
        verdict = store.latest_verdict(pred.thread_id)
        if verdict is None:
            continue
        status = verdict.final_status()
        if status in (VerdictStatus.TOO_EARLY, VerdictStatus.NO_EVIDENCE):
            continue
        pred.resolve(status)
        store.resolve_prediction(pred.id, status, pred.brier_score)
        resolved += 1
    return {"resolved": resolved}


def calibration(store, bins: int = 5) -> List[Dict[str, object]]:
    """Are we right 80% of the time when we say 80%?

    A model that is accurate but badly calibrated is unusable for a product
    that reports confidence to analysts.
    """
    graded = [p for p in store.list_predictions() if p.resolved_status is not None]
    return calibration_bins(graded, bins)


def calibration_bins(graded: List[Prediction], bins: int = 5) -> List[Dict[str, object]]:
    """Predicted vs actual kept-rate per probability bucket."""
    out = []
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        rows = [p for p in graded if lo <= p.p_kept < hi or (i == bins - 1 and p.p_kept == 1.0)]
        if not rows:
            continue
        actual = sum(1 for p in rows if p.resolved_status == VerdictStatus.KEPT) / len(rows)
        out.append(
            {
                "bucket": "{:.0%}-{:.0%}".format(lo, hi),
                "n": len(rows),
                "predicted": round(sum(p.p_kept for p in rows) / len(rows), 3),
                "actual": round(actual, 3),
            }
        )
    return out


def report(store, company: Optional[str] = None) -> Dict[str, object]:
    """The product-level dashboard.

    These are outcome metrics, not eval metrics: they answer "is the customer
    getting value", not "is the extractor accurate".
    """
    threads = store.list_threads(company)
    verdicts = [v for t in threads for v in store.list_verdicts(t.id)]
    reviewed = [v for v in verdicts if v.review_action]
    current = store.list_predictions(current_only=True)
    graded = [p for p in store.list_predictions() if p.brier_score is not None]

    by_status: Dict[str, int] = {}
    for v in verdicts:
        key = v.final_status().value
        by_status[key] = by_status.get(key, 0) + 1

    accepted = sum(1 for v in reviewed if v.review_action == "accepted")
    promises = store.list_promises(company=company)

    return {
        "company": company or "all",
        "coverage": {
            "sources": len(store.list_sources(company)),
            "promises": len(promises),
            "threads": len(threads),
            "threads_with_verdict": sum(1 for t in threads if store.latest_verdict(t.id)),
            "restated_threads": sum(1 for t in threads if t.restatement_count() > 0),
        },
        "verdicts_by_status": by_status,
        "analyst_review": {
            "reviewed": len(reviewed),
            "accepted": accepted,
            "acceptance_rate": round(accepted / len(reviewed), 3) if reviewed else None,
            "corrections_per_100_promises": (
                round(100 * (len(reviewed) - accepted) / len(promises), 2)
                if promises and reviewed
                else None
            ),
        },
        "prediction": {
            "model_version": PREDICTOR_VERSION,
            "outstanding": sum(1 for p in current if p.resolved_status is None),
            "graded": len(graded),
            "mean_brier": (
                round(sum(p.brier_score for p in graded) / len(graded), 4) if graded else None
            ),
            "calibration": calibration(store),
        },
        "spend_usd": round(store.total_cost(), 4),
    }


# Statuses that settle a commitment one way or the other. TOO_EARLY and
# NO_EVIDENCE are verdicts too, but they say nothing about the company's record.
RESOLVED_STATUSES = (
    VerdictStatus.KEPT,
    VerdictStatus.PARTIAL,
    VerdictStatus.MISSED,
    VerdictStatus.QUIETLY_DROPPED,
)


def accountability_profile(store, company: str) -> Dict[str, object]:
    """One company's investor-relations track record, for the dashboard.

    Counted on the *latest* verdict per thread, with analyst corrections
    applied -- unlike `report`, which counts every verdict ever issued. Threads
    with no verdict are reported as such rather than left out, so a record
    built on a handful of adjudications cannot pass for a full one.
    """
    from .threading_ import silence_signal

    sources = store.list_sources(company)
    promises = store.list_promises(company=company)
    threads = store.list_threads(company)

    latest = {t.id: store.latest_verdict(t.id) for t in threads}
    status_counts: Dict[str, int] = {s.value: 0 for s in VerdictStatus}
    by_type: Dict[str, Dict[str, int]] = {}
    for t in threads:
        v = latest[t.id]
        key = v.final_status().value if v else "unadjudicated"
        if v:
            status_counts[key] += 1
        row = by_type.setdefault(t.promise_type.value, {"threads": 0})
        row["threads"] += 1
        row[key] = row.get(key, 0) + 1

    adjudicated = [v for v in latest.values() if v]
    resolved = sum(status_counts[s.value] for s in RESOLVED_STATUSES)
    kept = status_counts[VerdictStatus.KEPT.value]

    hedge_mix: Dict[str, int] = {h.value: 0 for h in HedgeLevel}
    for p in promises:
        hedge_mix[p.hedge_level.value] += 1

    silent = [t for t in threads if silence_signal(store, t)["silent"]]
    dated = [s.published for s in sources if s.published]

    profile: Dict[str, object] = {
        "company": company,
        "coverage": {
            "sources": len(sources),
            "first_source": min(dated).isoformat() if dated else None,
            "last_source": max(dated).isoformat() if dated else None,
            "promises": len(promises),
            "threads": len(threads),
            "restated_threads": sum(1 for t in threads if t.restatement_count() > 0),
            "adjudicated_threads": len(adjudicated),
            "unadjudicated_threads": len(threads) - len(adjudicated),
        },
        "verdicts": status_counts,
        "record": {
            "resolved": resolved,
            "kept": kept,
            "kept_share": round(kept / resolved, 3) if resolved else None,
        },
        "silence": {
            "silent_threads": len(silent),
            "silent_unadjudicated": sum(1 for t in silent if latest[t.id] is None),
        },
        "by_type": by_type,
        "hedge_mix": hedge_mix,
        "escalated": sum(1 for v in adjudicated if v.escalated),
        "reviewed": sum(1 for v in adjudicated if v.review_action),
    }
    profile["rating"] = rating.rate(profile)
    return profile
