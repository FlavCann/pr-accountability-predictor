"""Dependencies injected into every route."""

from __future__ import annotations

from typing import Iterator, Optional

from fastapi import Header

from ..store import Store


def get_store() -> Iterator[Store]:
    store = Store()
    try:
        yield store
    finally:
        store.close()


def require_tenant(x_api_key: Optional[str] = Header(default=None)) -> str:
    """PLACEHOLDER. Not authentication.

    Before this is exposed to anyone, replace with real key verification and
    scope the store per tenant. Left obviously incomplete on purpose.
    """
    return x_api_key or "public"
