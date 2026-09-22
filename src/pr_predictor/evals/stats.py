"""Uncertainty for eval scores.

A point estimate on a handful of documents is mostly noise: with nine labelled
promises, one extraction more or less moves recall by eleven points. Every
comparison the gate makes therefore goes through a bootstrap over documents.

The unit of resampling is the document, not the promise. Promises in one
transcript share a speaker, a topic and an ASR quality, so they are not
independent; resampling them individually would report intervals that are too
narrow. Repeats of the same config are averaged within a document first, which
removes sampling noise from the point estimate but not from the interval --
`repeat_spread` in the eval result reports that separately.
"""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple

Counts = Dict[str, float]


def micro_f1(docs: Sequence[Counts]) -> float:
    tp = sum(d.get("true_positives", 0) for d in docs)
    fp = sum(d.get("false_positives", 0) for d in docs)
    fn = sum(d.get("false_negatives", 0) for d in docs)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def _percentiles(values: List[float], alpha: float) -> Tuple[float, float]:
    values = sorted(values)
    n = len(values)
    lo = values[max(0, int((alpha / 2) * n))]
    hi = values[min(n - 1, int((1 - alpha / 2) * n))]
    return lo, hi


def bootstrap_ci(
    docs: Sequence[Counts],
    stat: Callable[[Sequence[Counts]], float] = micro_f1,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Optional[Tuple[float, float]]:
    """Percentile interval for `stat`, resampling documents. None below 2 docs."""
    if len(docs) < 2:
        return None
    rng = random.Random(seed)
    n = len(docs)
    samples = [stat([docs[rng.randrange(n)] for _ in range(n)]) for _ in range(n_resamples)]
    lo, hi = _percentiles(samples, alpha)
    return round(lo, 3), round(hi, 3)


def paired_delta_ci(
    baseline: Sequence[Counts],
    candidate: Sequence[Counts],
    stat: Callable[[Sequence[Counts]], float] = micro_f1,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, Optional[float], Optional[float]]:
    """Candidate minus baseline, with a paired bootstrap interval.

    `baseline[i]` and `candidate[i]` must be the same document. Pairing is what
    makes a small benchmark usable at all: document difficulty varies far more
    than the difference between two prompts, and pairing cancels it out.
    """
    if len(baseline) != len(candidate):
        raise ValueError("paired comparison needs the same documents on both sides")
    point = round(stat(candidate) - stat(baseline), 3)
    n = len(baseline)
    if n < 2:
        return point, None, None
    rng = random.Random(seed)
    deltas = []
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        deltas.append(stat([candidate[i] for i in idx]) - stat([baseline[i] for i in idx]))
    lo, hi = _percentiles(deltas, alpha)
    return point, round(lo, 3), round(hi, 3)


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> Optional[float]:
    """Agreement beyond chance between two raters on the same items."""
    if len(a) != len(b):
        raise ValueError("raters must label the same items")
    n = len(a)
    if n == 0:
        return None
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    labels = set(a) | set(b)
    expected = sum((list(a).count(l) / n) * (list(b).count(l) / n) for l in labels)
    if expected >= 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return round((observed - expected) / (1 - expected), 3)
