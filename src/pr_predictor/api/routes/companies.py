"""Companies and their accountability profile."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ... import company_info, outcomes
from ...store import Store
from ..deps import get_store

router = APIRouter(tags=["companies"])


@router.get("/companies")
def companies(store: Store = Depends(get_store)) -> dict:
    names = sorted({s.company for s in store.list_sources()})
    return {"companies": names}


@router.get("/companies/{company}/profile")
def company_profile(company: str, store: Store = Depends(get_store)):
    if not store.list_sources(company):
        raise HTTPException(status_code=404, detail="no sources for {}".format(company))
    profile = outcomes.accountability_profile(store, company)
    profile["about"] = company_info.about(company)
    return profile
