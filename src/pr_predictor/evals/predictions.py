"""Prediction benchmark: graded by reality, in arrears.

The one benchmark that cannot be overfitted. A prediction is recorded before
its deadline and scored after the verdict lands. The only way to cheat is to
score a prediction made after the fact, so those are excluded and counted:
a prediction counts only if it was created before its horizon.

Brier skill is reported against the base rate of the same graded set -- the
score a predictor would get by always saying "the usual fraction are kept".
Below zero means the predictor is worse than that constant.
"""

from __future__ import annotations

from typing import Dict

from ..outcomes import PREDICTOR_VERSION, calibration_bins
from ..taxonomy import VerdictStatus


def score_predictions(store) -> Dict[str, object]:
    graded = [p for p in store.list_predictions() if p.resolved_status is not None]
    honest = [p for p in graded if p.created_at.date() < p.horizon]
    out: Dict[str, object] = {
        "predictor_version": PREDICTOR_VERSION,
        "graded": len(graded),
        "scored": len(honest),
        "excluded_made_after_horizon": len(graded) - len(honest),
    }
    if not honest:
        out["note"] = "No prediction made before its deadline has resolved yet."
        return out

    outcomes = [1.0 if p.resolved_status == VerdictStatus.KEPT else 0.0 for p in honest]
    brier = sum((p.p_kept - y) ** 2 for p, y in zip(honest, outcomes)) / len(honest)
    base = sum(outcomes) / len(outcomes)
    brier_ref = sum((base - y) ** 2 for y in outcomes) / len(outcomes)
    out.update(
        {
            "mean_brier": round(brier, 4),
            "base_rate_kept": round(base, 3),
            "brier_skill": round(1 - brier / brier_ref, 3) if brier_ref else None,
            "calibration": calibration_bins(honest),
            "by_version": _by_version(honest),
        }
    )
    return out


def _by_version(preds) -> Dict[str, Dict[str, float]]:
    groups: Dict[str, list] = {}
    for p in preds:
        groups.setdefault(p.model_version or "unknown", []).append(p)
    out = {}
    for version, ps in sorted(groups.items()):
        ys = [1.0 if p.resolved_status == VerdictStatus.KEPT else 0.0 for p in ps]
        out[version] = {
            "n": len(ps),
            "mean_brier": round(sum((p.p_kept - y) ** 2 for p, y in zip(ps, ys)) / len(ps), 4),
        }
    return out
