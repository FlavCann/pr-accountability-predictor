"""Threading benchmark, run in isolation from extraction.

The input is the gold promises themselves, so an extraction miss can never
show up here as a threading error. Each gold promise carries a `thread_key`;
two promises with the same key are one commitment restated. The score is
pairwise: over every pair of promises, did threading put them together exactly
when the labels do?

One pair class is a hard failure rather than a statistic: a promise merged
into the thread it *revises* (CLAUDE.md invariant 6). Losing a moved deadline
loses the story.

Uses a throwaway store. The production store is never written to.
"""

from __future__ import annotations

import tempfile
from datetime import date
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..domain import Promise
from ..taxonomy import HedgeLevel, PromiseType
from .gold import SCORED_SPLITS, GoldDoc, GoldPromise, load_gold

PROMPT_VERSION = "gold-threads"


def pairwise(pred: Dict[str, str], gold: Dict[str, str]) -> Dict[str, object]:
    """Pairwise precision/recall of a clustering against gold clusters."""
    items = sorted(set(pred) & set(gold))
    tp = fp = fn = 0
    for x, y in combinations(items, 2):
        same_pred = pred[x] == pred[y]
        same_gold = gold[x] == gold[y]
        tp += same_pred and same_gold
        fp += same_pred and not same_gold
        fn += same_gold and not same_pred
    p = tp / (tp + fp) if (tp + fp) else 1.0
    r = tp / (tp + fn) if (tp + fn) else 1.0
    return {
        "items": len(items),
        "pair_precision": round(p, 3),
        "pair_recall": round(r, 3),
        "pair_f1": round(2 * p * r / (p + r), 3) if (p + r) else 0.0,
        "false_merges": fp,
        "missed_links": fn,
    }


def revision_merges(
    pred: Dict[str, str], gold: Dict[str, str], revises: Dict[str, str]
) -> List[Tuple[str, str]]:
    """Pairs where a revision was merged into the thread it revises."""
    bad = []
    for x, target_key in revises.items():
        for y, key in gold.items():
            if key == target_key and x in pred and y in pred and pred[x] == pred[y]:
                bad.append((x, y))
    return bad


def _deadline(g: GoldPromise) -> Optional[date]:
    if g.deadline_labelled and g.deadline_resolved:
        return date.fromisoformat(g.deadline_resolved)
    return None


def run_threading_suite(
    store,
    client=None,
    gold_dir: Optional[Path] = None,
    splits=SCORED_SPLITS,
) -> Dict[str, object]:
    """Thread every labelled gold promise and score the result.

    `client=None` measures the deterministic matcher alone; pass a client to
    include model tie-breaks on ambiguous pairs (and pay for them).
    """
    from .. import threading_
    from ..store import Store

    docs: List[GoldDoc] = [
        d for d in load_gold(gold_dir, splits=splits) if any(g.thread_key for g in d.promises)
    ]
    if not docs:
        return {"skipped": "no gold promises carry a thread_key"}

    gold: Dict[str, str] = {}
    revises: Dict[str, str] = {}
    labels: Dict[str, str] = {}
    with tempfile.TemporaryDirectory() as tmp:
        scratch = Store(Path(tmp) / "threads.db")
        try:
            companies = set()
            for doc in docs:
                src = store.source_by_url(doc.source_url)
                if src is None:
                    continue
                scratch.put_source(src)
                companies.add(src.company)
                promises = []
                for g in doc.promises:
                    if not g.thread_key or g.ambiguous:
                        continue
                    verbatim = src.text[g.start_char : g.end_char]
                    p = Promise(
                        source_id=src.id,
                        company=src.company,
                        start_char=g.start_char,
                        end_char=g.end_char,
                        promise_type=PromiseType(g.promise_type or "procedural_commitment"),
                        hedge_level=HedgeLevel(g.hedge_level or "firm"),
                        normalized_claim=g.claim or verbatim,
                        deadline=_deadline(g),
                        deadline_raw=g.deadline_raw,
                        prompt_version=PROMPT_VERSION,
                        model="gold",
                        run_id="eval-threads",
                    )
                    promises.append(p)
                    gold[p.id] = g.thread_key
                    labels[p.id] = "{}@{}".format(g.thread_key, doc.name)
                    if g.revision_of:
                        revises[p.id] = g.revision_of
                scratch.add_promises(promises)

            cost = 0.0
            for company in sorted(companies):
                stats = threading_.build_threads(
                    scratch, company, prompt_version=PROMPT_VERSION, client=client
                )
                cost += float(stats.get("cost_usd", 0.0))
            pred = {
                p.id: p.thread_id
                for p in scratch.list_promises(prompt_version=PROMPT_VERSION)
                if p.thread_id
            }
        finally:
            scratch.close()

    out = pairwise(pred, gold)
    bad = revision_merges(pred, gold, revises)
    out.update(
        {
            "mode": "deterministic" if client is None else "with tie-breaks ({})".format(client.model),
            "gold_threads": len(set(gold.values())),
            "predicted_threads": len(set(pred.values())),
            "revision_merges": [(labels[x], labels[y]) for x, y in bad],
            "passed": not bad,
            "cost_usd": round(cost, 4),
        }
    )
    return out
