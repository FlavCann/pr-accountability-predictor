"""Adjudication benchmark: given a thread and its evidence, is the verdict right?

Evidence gathering is not in scope -- each case supplies the evidence -- so a
retrieval miss cannot show up here as an adjudication error. Each case pins its
own `as_of` date, so "has the deadline passed?" means the same thing every time
the case is run.

Case files (evals/adjudication/*.json), one case per file, labelled by a person:

    {
      "id": "bc-farmers-2025",
      "company": "Barry Callebaut",
      "as_of": "2026-06-30",
      "labelled_by": "who, when",
      "thread": {"canonical_claim": "...", "promise_type": "quantified_target",
                 "metric": null, "deadline": "2025-12-31",
                 "first_seen": "2016-11-15", "last_seen": "2023-05-10"},
      "later_sources": [{"title": "...", "published": "2024-11-06"}],
      "evidence": [{"excerpt": "...", "polarity": "supports"}],
      "expected_status": "partial",
      "also_acceptable": [],
      "note": "why"
    }
"""

from __future__ import annotations

import json
import tempfile
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

from .. import config
from ..domain import Evidence, PromiseThread, Source
from ..taxonomy import PromiseType, VerdictStatus

CASES_DIR = "adjudication"


def load_cases(gold_dir: Optional[Path] = None) -> List[Dict[str, object]]:
    directory = Path(gold_dir or config.GOLD_DIR).parent / CASES_DIR
    return [json.loads(p.read_text()) for p in sorted(directory.glob("*.json"))]


def _d(value) -> Optional[date]:
    return date.fromisoformat(value) if value else None


def run_adjudication_suite(
    client, gold_dir: Optional[Path] = None, allow_escalation: bool = True
) -> Dict[str, object]:
    from ..store import Store
    from ..verify import adjudicate

    cases = load_cases(gold_dir)
    if not cases:
        return {"skipped": "no cases in evals/{}/".format(CASES_DIR)}

    rows = []
    confusion: Dict[str, Dict[str, int]] = {}
    cost = 0.0
    with tempfile.TemporaryDirectory() as tmp:
        scratch = Store(Path(tmp) / "adjudication.db")
        try:
            for case in cases:
                t = case["thread"]
                thread = PromiseThread(
                    company=case["company"],
                    canonical_claim=t["canonical_claim"],
                    promise_type=PromiseType(t["promise_type"]),
                    metric=t.get("metric"),
                    deadline=_d(t.get("deadline")),
                    first_seen=_d(t.get("first_seen")),
                    last_seen=_d(t.get("last_seen")),
                    prompt_version="gold-adjudication",
                )
                scratch.put_thread(thread)
                for s in case.get("later_sources", []):
                    scratch.put_source(
                        Source(
                            company=case["company"], url="case:{}:{}".format(case["id"], s["title"]),
                            title=s["title"], published=_d(s["published"]), fetcher="eval_case",
                        ).finalise()
                    )
                evidence = [
                    Evidence(
                        thread_id=thread.id, excerpt=e["excerpt"],
                        polarity=e.get("polarity", "ambiguous"), found_by="eval_case",
                    )
                    for e in case.get("evidence", [])
                ]
                verdict, usage = adjudicate(
                    scratch, client, thread, evidence,
                    allow_escalation=allow_escalation, as_of=_d(case["as_of"]),
                )
                cost += usage.cost_usd
                expected = VerdictStatus(case["expected_status"]).value
                ok_set = {expected} | {VerdictStatus(s).value for s in case.get("also_acceptable", [])}
                got = verdict.status.value
                confusion.setdefault(expected, {}).setdefault(got, 0)
                confusion[expected][got] += 1
                rows.append(
                    {
                        "id": case["id"], "expected": expected, "got": got,
                        "correct": got in ok_set, "escalated": verdict.escalated,
                        "changed_by_advisor": bool(
                            verdict.escalated and verdict.initial_status
                            and verdict.initial_status != verdict.status
                        ),
                    }
                )
        finally:
            scratch.close()

    n = len(rows)
    escalated = [r for r in rows if r["escalated"]]
    return {
        "cases": n,
        "accuracy": round(sum(r["correct"] for r in rows) / n, 3),
        "escalation_rate": round(len(escalated) / n, 3),
        "advisor_changed_outcome": sum(r["changed_by_advisor"] for r in rows),
        "confusion": confusion,
        "wrong": [r for r in rows if not r["correct"]],
        "cost_usd": round(cost, 4),
    }
