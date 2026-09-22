"""Assembles the service: one router per resource, plus the dashboard."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from .routes import companies, evals, ops, promises, threads, verdicts

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(
    title="PR Accountability Predictor",
    version="0.3.0",
    description=(
        "Track what companies promise in investor communications, and whether "
        "they keep it. Every claim traces to a timestamped span in a named source."
    ),
)

for module in (ops, companies, promises, threads, verdicts, evals):
    app.include_router(module.router)


@app.get("/dashboard", include_in_schema=False)
def dashboard() -> RedirectResponse:
    # The page loads its assets by relative path, so it must be served from a
    # directory URL.
    return RedirectResponse("/dashboard/")


app.mount("/dashboard", StaticFiles(directory=str(STATIC_DIR), html=True), name="dashboard")
