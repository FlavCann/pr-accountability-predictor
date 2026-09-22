"""Service health, the product report and the run ledger."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends

from ... import outcomes
from ...store import Store
from ..deps import get_store

router = APIRouter(tags=["ops"])


@router.get("/health")
def health(store: Store = Depends(get_store)) -> dict:
    return {"status": "ok", "sources": len(store.list_sources())}


@router.get("/report")
def report(company: Optional[str] = None, store: Store = Depends(get_store)):
    return outcomes.report(store, company)


@router.get("/ledger")
def ledger(limit: int = 25, store: Store = Depends(get_store)):
    return {
        "runs": [r.model_dump(mode="json") for r in store.list_runs(limit)],
        "total_cost_usd": round(store.total_cost(), 4),
    }
