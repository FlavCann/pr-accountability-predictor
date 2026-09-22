"""Inter-annotator agreement: the ceiling on every extraction score.

If two careful people labelling the same transcript agree on 80% of promise
spans, a model scoring 0.9 F1 against one of them is not better than a person
-- it has learned that person's habits. Agreement is measured by labelling the
same documents twice, independently, into two gold directories built by
`evals/build_gold.py --labels ... --out ...`, then comparing them here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from .gold import GoldDoc, load_gold
from .scoring import spans_match
from .stats import cohen_kappa


def _match(a: GoldDoc, b: GoldDoc):
    """Greedy one-to-one span matching of B's promises onto A's."""
    used = set()
    pairs = []
    for ga in a.promises:
        hit = next(
            (i for i, gb in enumerate(b.promises) if i not in used and spans_match(ga.span, gb.span)),
            None,
        )
        if hit is not None:
            used.add(hit)
            pairs.append((ga, b.promises[hit]))
    return pairs


def agreement(dir_a: Path, dir_b: Path) -> Dict[str, object]:
    """Span F1 between annotators, plus kappa on type and hedge where both agree
    something is a promise."""
    a_docs = {d.source_url: d for d in load_gold(dir_a)}
    b_docs = {d.source_url: d for d in load_gold(dir_b)}
    shared = sorted(set(a_docs) & set(b_docs))
    if not shared:
        raise SystemExit("The two gold directories share no documents.")

    per_doc: Dict[str, Dict[str, object]] = {}
    n_a = n_b = n_match = 0
    types_a: List[str] = []
    types_b: List[str] = []
    hedges_a: List[str] = []
    hedges_b: List[str] = []
    for url in shared:
        a, b = a_docs[url], b_docs[url]
        pairs = _match(a, b)
        n_a += len(a.promises)
        n_b += len(b.promises)
        n_match += len(pairs)
        denom = len(a.promises) + len(b.promises)
        per_doc[a.name] = {
            "a": len(a.promises),
            "b": len(b.promises),
            "matched": len(pairs),
            "span_f1": round(2 * len(pairs) / denom, 3) if denom else None,
        }
        for ga, gb in pairs:
            types_a.append(str(ga.promise_type))
            types_b.append(str(gb.promise_type))
            hedges_a.append(str(ga.hedge_level))
            hedges_b.append(str(gb.hedge_level))

    denom = n_a + n_b
    return {
        "documents": len(shared),
        "span_f1": round(2 * n_match / denom, 3) if denom else None,
        "type_kappa": cohen_kappa(types_a, types_b),
        "hedge_kappa": cohen_kappa(hedges_a, hedges_b),
        "per_document": per_doc,
    }
