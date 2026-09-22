"""Model evaluation: the benchmark's scoreboard, for the dashboard's second tab.

Scores only -- extracted text (unjudged passages) stays out of the API, since
transcripts are not ours to redistribute.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from ...store import Store
from ..deps import get_store

router = APIRouter(tags=["evals"])

_RUN_SUMMARY_KEYS = (
    "run_id", "recorded_at", "prompt_version", "model", "documents", "overall",
    "f1_ci95", "underpowered", "repeat_f1", "splits", "cost_usd", "latency_s",
    "not_human_labelled", "not_scored_labelling_incomplete", "gold_fingerprint",
    "baseline_truncate_words",
)


def _run_summary(run: dict, champion_id: Optional[str]) -> dict:
    out = {k: run.get(k) for k in _RUN_SUMMARY_KEYS}
    manifest = run.get("manifest") or {}
    out["harness"] = manifest.get("harness")
    out["repeats"] = manifest.get("repeats", len(run.get("repeat_f1") or []) or 1)
    out["unjudged"] = len(run.get("unjudged_extractions") or [])
    out["is_champion"] = run.get("run_id") == champion_id
    return out


@router.get("/evals")
def list_evals(limit: int = Query(200, le=1000), store: Store = Depends(get_store)):
    """Every recorded eval run, oldest first, and which one is the champion."""
    from ... import evals

    champion = evals.load_champion()
    champion_id = champion.get("run_id") if champion else None
    runs = list(reversed(store.list_eval_runs(limit=limit)))
    return {
        "champion_run_id": champion_id,
        "runs": [_run_summary(r, champion_id) for r in runs],
    }


@router.get("/evals/{run_id}")
def get_eval(run_id: str, store: Store = Depends(get_store)):
    """One run in full: per-document scores, slices and its manifest."""
    from ... import evals

    champion = evals.load_champion()
    champion_id = champion.get("run_id") if champion else None
    run = next((r for r in store.list_eval_runs(limit=1000) if r.get("run_id") == run_id), None)
    if run is None and champion and champion_id == run_id:
        run = champion
    if run is None:
        raise HTTPException(status_code=404, detail="no eval run {}".format(run_id))
    out = _run_summary(run, champion_id)
    out["manifest"] = run.get("manifest")
    out["slices"] = run.get("slices")
    out["per_document"] = [
        dict({k: v for k, v in d.items() if k != "counts"}, source_url=url)
        for url, d in (run.get("per_document") or {}).items()
    ]
    return out
