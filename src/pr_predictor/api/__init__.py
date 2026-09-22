"""The HTTP service (Part 4 of the architecture).

    uvicorn pr_predictor.api:app --reload

The API is the product and the store is the asset, so the design constraint
that dominates is the **audit trail**: every response that asserts something
about a company carries its provenance down to a timestamped span in a named
source. That is the difference between something a compliance team can use and
something their legal department will not let them buy.

Auth, tenancy and rate limiting are deliberately stubbed rather than faked --
see `deps.require_tenant`. Shipping a plausible-looking but non-functional auth
layer is worse than an obvious placeholder.

Layout: `app` assembles the service, `deps` holds the injected store and tenant,
`schemas` the response shapes, and `routes/` one router per resource. The
dashboard is static files under `pr_predictor/static/`, served at /dashboard/.
"""

from __future__ import annotations

try:
    import fastapi  # noqa: F401
except ImportError:  # pragma: no cover
    raise SystemExit(
        "The API needs FastAPI. Install it with:  pip install -e '.[api]'"
    )

from .app import app
from .deps import get_store, require_tenant

__all__ = ["app", "get_store", "require_tenant"]
