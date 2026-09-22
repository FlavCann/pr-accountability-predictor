"""Individual promises, each with its provenance."""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, Query

from ...store import Store
from ..deps import get_store, require_tenant
from ..schemas import PromiseOut, promise_out

router = APIRouter(tags=["promises"])


@router.get("/promises", response_model=List[PromiseOut])
def list_promises(
    company: Optional[str] = None,
    promise_type: Optional[str] = None,
    hedge_level: Optional[str] = None,
    limit: int = Query(default=100, le=1000),
    store: Store = Depends(get_store),
    tenant: str = Depends(require_tenant),
):
    rows = store.list_promises(company=company)
    if promise_type:
        rows = [p for p in rows if p.promise_type.value == promise_type]
    if hedge_level:
        rows = [p for p in rows if p.hedge_level.value == hedge_level]

    return [promise_out(store, p) for p in rows[:limit]]
