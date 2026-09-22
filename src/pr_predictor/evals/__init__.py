"""The eval benchmark (concept 9).

Without this, every prompt change is a coin flip judged by reading eight
paragraphs of output and forming an impression. With it, "did that help?" has
an answer -- with an interval, for one change at a time, at every stage.

Principles:

* **Push work to deterministic checks.** Span fidelity is a substring test,
  deadlines are compared exactly, threading is scored pairwise -- no judge, no
  cost, no noise. Only semantic faithfulness needs a model.
* **Match on spans, not strings.** A predicted promise matches a gold one when
  their character ranges overlap, which is robust to the model choosing
  slightly different boundaries.
* **Isolate the stages.** Threading and adjudication are fed gold input, so an
  upstream miss never shows up as a downstream error.
* **Splits keep it honest.** Examples come only from the `examples` split;
  iterate on `dev`; promote on `test`.
* **Report uncertainty.** Repeats per config, bootstrap intervals over
  documents, paired comparisons, and an UNDERPOWERED label when the data
  cannot tell.
* **The judge is a third model**, checked against human labels before its
  number means anything.

Modules:

    gold          gold set: load, fingerprint, validate, splits, leakage
    scoring       extraction metrics and slices
    stats         bootstrap intervals, paired comparison, kappa
    runner        run_eval, manifest, champion, gate
    replay        GoldReplayClient (offline stand-in for the model)
    threads       threading benchmark
    adjudication  verdict benchmark
    predictions   Brier and calibration, graded by reality
    judge         claim faithfulness, and the judge's own calibration
    agreement     inter-annotator agreement (the ceiling)
"""

from __future__ import annotations

from .gold import (
    SCORED_SPLITS,
    SPLITS,
    GoldDoc,
    GoldPromise,
    _norm,
    find_leakage,
    gold_fingerprint,
    load_gold,
    load_splits,
    validate_gold,
)
from .replay import GoldReplayClient
from .runner import (
    CHAMPION_FILE,
    build_manifest,
    changed_axes,
    gate,
    harness_overrides,
    load_champion,
    run_eval,
    save_champion,
)
from .scoring import (
    DocScore,
    Metrics,
    boundary_tags,
    score_document,
    score_document_detailed,
    slice_report,
    spans_match,
)

__all__ = [
    "CHAMPION_FILE", "DocScore", "GoldDoc", "GoldPromise", "GoldReplayClient",
    "Metrics", "SCORED_SPLITS", "SPLITS", "boundary_tags", "build_manifest",
    "changed_axes", "find_leakage", "gate", "gold_fingerprint", "harness_overrides",
    "load_champion", "load_gold", "load_splits", "run_eval", "save_champion",
    "score_document", "score_document_detailed", "slice_report", "spans_match",
    "validate_gold",
]
