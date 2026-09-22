"""Settings, model routing and the cost table.

Model IDs and prices were verified against the bundled Claude API reference on
2026-09-21. They move -- re-check before relying on the cost figures.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv(path: Path) -> None:
    """Read KEY=VALUE lines from a git-ignored .env into the environment.

    Deliberately tiny and dependency-free. Never overrides a variable that is
    already set, so an exported key always wins over the file.
    """
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv(ROOT / ".env")

DATA_DIR = Path(os.environ.get("PRP_DATA_DIR", ROOT / "data"))
DB_PATH = DATA_DIR / "ledger.db"
TRANSCRIPTS_DIR = ROOT / "transcripts"
PAGES_DIR = ROOT / "pages"
TO_SCRAPE = ROOT / "to_scrape.json"
GOLD_DIR = ROOT / "evals" / "gold"

# --- Model routing (concept 11) ------------------------------------------
# Route by decision value x volume, not by "hard vs easy".

EXECUTOR_MODEL = os.environ.get("PRP_EXECUTOR_MODEL", "claude-sonnet-5")
"""High volume, schema-constrained: extraction over every chunk of every
transcript, evidence retrieval, field normalisation."""

ADVISOR_MODEL = os.environ.get("PRP_ADVISOR_MODEL", "claude-opus-4-8")
"""Low volume, high consequence: adjudicating contested verdicts.

Deliberately NOT claude-fable-5-1 by default. Fable returns
`advisor_redacted_result` -- encrypted advice we can replay but not read --
and this product sells a traceable rationale, so the advice has to be
auditable. Opus 4.8 returns plaintext `advisor_result`. Set
PRP_ADVISOR_MODEL=claude-fable-5-1 if you want the stronger advisor and can
accept an unreadable audit trail. See verify.py.
"""

JUDGE_MODEL = os.environ.get("PRP_JUDGE_MODEL", "claude-opus-5")
"""Eval judge. Kept distinct from both of the above so nothing grades its own
homework (concept 9)."""

ADVISOR_TOOL_TYPE = "advisor_20260301"
ADVISOR_BETA = "advisor-tool-2026-03-01"

# --- Cost table, USD per million tokens ----------------------------------
PRICING = {
    "claude-fable-5-1": {"in": 10.00, "out": 50.00, "cache_read_mult": 0.025},
    "claude-opus-5": {"in": 5.00, "out": 25.00, "cache_read_mult": 0.1},
    "claude-opus-4-8": {"in": 5.00, "out": 25.00, "cache_read_mult": 0.1},
    "claude-sonnet-5": {"in": 2.00, "out": 10.00, "cache_read_mult": 0.1},
    "claude-haiku-4-5": {"in": 1.00, "out": 5.00, "cache_read_mult": 0.1},
}

CACHE_WRITE_MULT_5M = 1.25
CACHE_WRITE_MULT_1H = 2.0

# Minimum cacheable prefix, in tokens. Below this a cache_control marker is
# silently ignored -- no error, just cache_creation_input_tokens: 0.
MIN_CACHEABLE_PREFIX = {
    "claude-fable-5-1": 512,
    "claude-opus-5": 512,
    "claude-opus-4-8": 1024,
    "claude-sonnet-5": 1024,
    "claude-haiku-4-5": 4096,
}


def estimate_cost(
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    cache_ttl: str = "5m",
) -> float:
    """Dollar cost of one call. Used by the run ledger."""
    p = PRICING.get(model)
    if p is None:
        return 0.0
    write_mult = CACHE_WRITE_MULT_1H if cache_ttl == "1h" else CACHE_WRITE_MULT_5M
    per_m = 1_000_000.0
    return (
        input_tokens * p["in"] / per_m
        + output_tokens * p["out"] / per_m
        + cache_read_tokens * p["in"] * p["cache_read_mult"] / per_m
        + cache_write_tokens * p["in"] * write_mult / per_m
    )


# --- Harness limits (concept 3) ------------------------------------------
MAX_RETRIES = 4
MAX_CONCURRENCY = 4
REQUEST_TIMEOUT_S = 600.0

# --- Agent loop bounds (concept 6) ---------------------------------------
VERIFY_MAX_STEPS = 12
VERIFY_MAX_COST_USD = 0.50
"""Per-thread ceiling. Exceeding it is not an error -- the agent returns
NO_EVIDENCE, which is a legitimate terminal outcome."""

# --- Chunking (concept 2) ------------------------------------------------
CHUNK_WORDS = 1800
CHUNK_OVERLAP_WORDS = 250
"""Overlap exists so a promise split across a chunk boundary is seen whole by
at least one call. Duplicates are reconciled by span overlap in extract.py."""


# --- Eval benchmark (concept 9) ------------------------------------------
EVAL_REPEATS = 3
"""Samples per config. Repeat 0 may come from the result cache; the rest are
fresh, so `repeat_f1` in a result shows run-to-run noise."""
EVAL_MIN_DOCS_FOR_CI = 5
"""Below this many scored documents a bootstrap interval is not reported and
the gate labels its verdict UNDERPOWERED."""
EVAL_MIN_EFFECT = 0.0
"""The F1 gain a promotion must clear -- by the lower bound of its paired
interval once the benchmark is powered, by the point estimate before then."""
EVAL_BOOTSTRAP_RESAMPLES = 2000
JUDGE_MIN_AGREEMENT = 0.85
JUDGE_MIN_CALIBRATION_ITEMS = 20
"""The judge's faithfulness score is reported as validated only after it agrees
with human labels at this rate over at least this many items."""


def api_key_present() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
