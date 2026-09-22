"""Extraction scoring: span matching, metrics and slices.

Deterministic throughout -- no model is involved in deciding whether an
extraction is right, so a score can be recomputed from stored rows forever.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Dict, List, Optional, Tuple

from ..domain import Promise
from .gold import GoldDoc, GoldPromise


def spans_match(a: Tuple[int, int], b: Tuple[int, int], min_overlap: float = 0.5) -> bool:
    """Overlap-based match, tolerant of different boundary choices."""
    inter = min(a[1], b[1]) - max(a[0], b[0])
    if inter <= 0:
        return False
    shorter = min(a[1] - a[0], b[1] - b[0])
    return shorter > 0 and inter / shorter >= min_overlap


@dataclass
class Metrics:
    """Counts for one document, or summed over many.

    Counts are floats because repeated runs are averaged per document before
    documents are summed.
    """

    true_positives: float = 0
    false_positives: float = 0
    false_negatives: float = 0
    unjudged: float = 0
    forbidden_extracted: float = 0
    ambiguous_matched: float = 0
    masked_ignored: float = 0
    span_fidelity_ok: float = 0
    span_fidelity_bad: float = 0
    type_correct: float = 0
    type_total: float = 0
    hedge_correct: float = 0
    hedge_total: float = 0
    deadline_correct: float = 0
    deadline_total: float = 0

    @property
    def precision(self) -> float:
        """Judged precision: TP / (TP + FP), ignoring unjudged extractions.

        Only trustworthy when `unjudged` is zero -- which is why the gate
        refuses to promote while any remain.
        """
        d = self.true_positives + self.false_positives
        return self.true_positives / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.true_positives + self.false_negatives
        return self.true_positives / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def span_fidelity(self) -> float:
        d = self.span_fidelity_ok + self.span_fidelity_bad
        return self.span_fidelity_ok / d if d else 1.0

    def counts(self) -> Dict[str, float]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_counts(cls, counts: Dict[str, float]) -> "Metrics":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in counts.items() if k in known})

    def as_dict(self) -> Dict[str, object]:
        def _rate(n: float, d: float) -> Optional[float]:
            return round(n / d, 3) if d else None

        return {
            "precision": round(self.precision, 3),
            "recall": round(self.recall, 3),
            "f1": round(self.f1, 3),
            "span_fidelity": round(self.span_fidelity, 3),
            "type_accuracy": _rate(self.type_correct, self.type_total),
            "hedge_accuracy": _rate(self.hedge_correct, self.hedge_total),
            "deadline_accuracy": _rate(self.deadline_correct, self.deadline_total),
            "true_positives": _num(self.true_positives),
            "false_positives": _num(self.false_positives),
            "false_negatives": _num(self.false_negatives),
            "unjudged": _num(self.unjudged),
            "forbidden_extracted": _num(self.forbidden_extracted),
            "ambiguous_matched": _num(self.ambiguous_matched),
            "masked_ignored": _num(self.masked_ignored),
            "uncitable": _num(self.span_fidelity_bad),
        }

    def __iadd__(self, other: "Metrics") -> "Metrics":
        for f in fields(self):
            setattr(self, f.name, getattr(self, f.name) + getattr(other, f.name))
        return self


def _num(x: float) -> float:
    """Ints stay ints in reports; averaged counts keep two decimals."""
    return int(x) if float(x).is_integer() else round(x, 2)


@dataclass
class DocScore:
    metrics: Metrics
    unjudged: List[Promise] = field(default_factory=list)
    pairs: List[Tuple[GoldPromise, Optional[Promise]]] = field(default_factory=list)
    """Every non-ambiguous gold promise with the prediction that matched it, or
    None. The raw material for slice reports and the judge."""


def _deadline(p: Promise) -> Optional[str]:
    return p.deadline.isoformat() if p.deadline else None


def score_document_detailed(
    predicted: List[Promise],
    gold: GoldDoc,
    source_text: str,
    uncitable: int = 0,
    score_fields: bool = True,
) -> DocScore:
    """Score one document.

    An extraction is:
      * a true positive if it overlaps a labelled promise;
      * a false positive if it overlaps a must-not-extract passage, duplicates
        an already-matched promise, or if the document is exhaustively
        labelled and it matches nothing;
      * `ambiguous_matched` if it overlaps a promise annotators disagreed on --
        neither right nor wrong;
      * `masked_ignored` if it touches a masked span (a prompt example) --
        not scored at all;
      * otherwise UNJUDGED -- possibly a real promise nobody labelled. These
        are returned so a person can label them and re-score (pooling).

    `uncitable` counts extractions whose text could not be located in the
    source at all. They count against span fidelity.
    """
    m = Metrics()
    m.span_fidelity_bad += uncitable
    for p in predicted:
        if p.verify_span(source_text):
            m.span_fidelity_ok += 1
        else:
            m.span_fidelity_bad += 1

    firm = [g for g in gold.promises if not g.ambiguous]
    contested = [g for g in gold.promises if g.ambiguous]
    matched: Dict[int, Promise] = {}
    unjudged: List[Promise] = []

    for p in predicted:
        span = (p.start_char, p.end_char)
        if any(p.start_char < x.end_char and x.start_char < p.end_char for x in gold.masked):
            m.masked_ignored += 1
            continue
        hit = next(
            (i for i, g in enumerate(firm) if i not in matched and spans_match(span, g.span)),
            None,
        )
        if hit is not None:
            matched[hit] = p
            m.true_positives += 1
            g = firm[hit]
            if score_fields and g.promise_type:
                m.type_total += 1
                m.type_correct += int(p.promise_type.value == g.promise_type)
            if score_fields and g.hedge_level:
                m.hedge_total += 1
                m.hedge_correct += int(p.hedge_level.value == g.hedge_level)
            if score_fields and g.deadline_labelled:
                m.deadline_total += 1
                m.deadline_correct += int(_deadline(p) == g.deadline_resolved)
        elif any(spans_match(span, n.span) for n in gold.must_not_extract):
            m.false_positives += 1
            m.forbidden_extracted += 1
        elif any(spans_match(span, g.span) for g in contested):
            m.ambiguous_matched += 1
        elif any(spans_match(span, g.span) for g in firm):
            # A second extraction of a promise already matched: a duplicate.
            m.false_positives += 1
        elif gold.exhaustive:
            m.false_positives += 1
        else:
            m.unjudged += 1
            unjudged.append(p)

    m.false_negatives += len(firm) - len(matched)
    pairs = [(g, matched.get(i)) for i, g in enumerate(firm)]
    return DocScore(metrics=m, unjudged=unjudged, pairs=pairs)


def score_document(
    predicted: List[Promise],
    gold: GoldDoc,
    source_text: str,
    uncitable: int = 0,
    score_fields: bool = True,
) -> Tuple[Metrics, List[Promise]]:
    """Score one document. Returns (metrics, unjudged extractions)."""
    s = score_document_detailed(predicted, gold, source_text, uncitable, score_fields)
    return s.metrics, s.unjudged


def slice_report(
    pairs: List[Tuple[GoldPromise, Optional[Promise], List[str]]],
) -> Dict[str, Dict[str, object]]:
    """Recall and field accuracy per slice tag.

    Each entry is (gold, matched prediction or None, tags). Precision is not
    reported per slice: a slice is a property of gold promises, and a false
    positive has no gold promise to carry the tag.
    """
    acc: Dict[str, Dict[str, float]] = {}
    for g, p, tags in pairs:
        for tag in tags:
            s = acc.setdefault(tag, {"n": 0, "found": 0, "type_ok": 0, "hedge_ok": 0,
                                     "deadline_ok": 0, "deadline_n": 0})
            s["n"] += 1
            if p is None:
                continue
            s["found"] += 1
            s["type_ok"] += int(bool(g.promise_type) and p.promise_type.value == g.promise_type)
            s["hedge_ok"] += int(bool(g.hedge_level) and p.hedge_level.value == g.hedge_level)
            if g.deadline_labelled:
                s["deadline_n"] += 1
                s["deadline_ok"] += int(_deadline(p) == g.deadline_resolved)

    out: Dict[str, Dict[str, object]] = {}
    for tag, s in sorted(acc.items()):
        found = s["found"]
        out[tag] = {
            "gold_observations": int(s["n"]),
            "recall": round(found / s["n"], 3) if s["n"] else None,
            "type_accuracy": round(s["type_ok"] / found, 3) if found else None,
            "hedge_accuracy": round(s["hedge_ok"] / found, 3) if found else None,
            "deadline_accuracy": (
                round(s["deadline_ok"] / s["deadline_n"], 3) if s["deadline_n"] else None
            ),
        }
    return out


def boundary_tags(g: GoldPromise, chunks) -> List[str]:
    """Tags describing how a gold promise sits relative to the chunking.

    Computed at score time from the harness settings actually used, so a run
    with different chunk sizes gets a truthful slice. `chunk_boundary` means a
    chunk edge falls inside the promise; `unseen_whole` means no single chunk
    contains all of it -- the overlap failed to protect it.
    """
    crosses = any(
        c.offset < g.end_char and c.end > g.start_char
        and not (c.offset <= g.start_char and g.end_char <= c.end)
        for c in chunks
    )
    whole = any(c.offset <= g.start_char and g.end_char <= c.end for c in chunks)
    tags = []
    if crosses:
        tags.append("chunk_boundary")
    if not whole:
        tags.append("unseen_whole")
    return tags
