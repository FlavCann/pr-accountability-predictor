"""The orchestration layer (concept 15).

Stages: discover -> fetch -> normalise -> extract -> link -> verify ->
adjudicate -> review -> publish.

The headline design position: **agents are steps inside the workflow; the
workflow is not an agent.** The verification agent (verify.py) runs inside one
stage, bounded and disposable. It does not drive the process. Making an agent
the top-level controller of a multi-day business process -- one that must
"remember" for three days that it is waiting for an analyst -- is the most
common way this class of product fails.

This is a deliberately small engine: a resumable stage runner over the SQLite
store. It has the properties that matter (durable state, per-stage retry,
idempotency, a human checkpoint that can pause for days without holding a
process open) without a Temporal deployment. The stage boundaries are drawn so
that swapping in Temporal or Prefect later is a mechanical change -- see
REPORT.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from . import context, extract, ingest, outcomes, threading_, verify
from .llm import LLMClient


@dataclass
class StageResult:
    name: str
    ok: bool
    detail: Dict
    skipped: bool = False
    message: str = ""


class Pipeline:
    """Runs stages in order, stopping at the first hard failure.

    Every stage is idempotent: re-running skips work already done, because the
    store records what exists rather than the runner tracking position. That is
    what makes a crash recoverable -- restart the pipeline and it resumes.
    """

    def __init__(
        self, store, client: LLMClient, company: str, prompt_version: Optional[str] = None
    ):
        from . import prompts

        self.store = store
        self.client = client
        self.company = company
        # One version for the whole run: extract and link must agree, or the
        # link stage would thread a different extraction than was just made.
        self.prompt_version = prompt_version or prompts.DEFAULT_EXTRACTION_PROMPT

    # ---------------- stages ----------------

    def stage_seed(self) -> StageResult:
        n = context.seed_memory(self.store, self.company)
        return StageResult("seed", True, {"memory_entries": n})

    def stage_discover(self) -> StageResult:
        from . import config

        if not config.PAGES_DIR.exists():
            return StageResult(
                "discover", True, {}, skipped=True,
                message="no pages/ directory -- using the existing work list",
            )
        added = ingest.discover()
        return StageResult("discover", True, {"new_videos": added})

    def stage_ingest(self) -> StageResult:
        """Load the on-disk corpus, then fetch anything still missing.

        Legacy .txt files are ingested first so the pipeline is usable today
        without hitting YouTube, which blocks aggressively.
        """
        legacy = ingest.ingest_legacy_transcripts(self.store, self.company)
        return StageResult("ingest", True, {"legacy_sources_loaded": legacy})

    def stage_fetch(self, limit: Optional[int] = None) -> StageResult:
        """Re-fetch with timestamps. Optional -- network dependent."""
        try:
            stats = ingest.fetch_pending(self.store, self.company, limit=limit)
        except SystemExit as e:
            return StageResult("fetch", False, {}, message=str(e))
        return StageResult("fetch", True, stats)

    def stage_extract(self) -> StageResult:
        stats = extract.extract_all(
            self.store, self.client, company=self.company, prompt_version=self.prompt_version
        )
        return StageResult("extract", stats.get("failed", 0) == 0, stats)

    def stage_link(self) -> StageResult:
        stats = threading_.build_threads(
            self.store, self.company, prompt_version=self.prompt_version, client=self.client
        )
        return StageResult("link", True, stats)

    def stage_verify(self, limit: Optional[int] = None) -> StageResult:
        stats = verify.verify_all(self.store, self.client, self.company, limit=limit)
        return StageResult("verify", True, stats)

    def stage_predict(self) -> StageResult:
        stats = outcomes.predict_all(self.store, self.company)
        stats.update(outcomes.resolve_predictions(self.store))
        return StageResult("predict", True, stats)

    def stage_review(self) -> StageResult:
        """The human checkpoint.

        Does not block. It reports the queue and returns; an analyst works
        through it via `prp review` or the API, and the next pipeline run picks
        up whatever they decided. A workflow that held a process open waiting
        for a person would be a bug, not a feature.
        """
        pending = [
            v
            for t in self.store.list_threads(self.company)
            for v in self.store.list_verdicts(t.id)
            if not v.review_action
        ]
        return StageResult(
            "review",
            True,
            {"awaiting_review": len(pending), "escalated": sum(1 for v in pending if v.escalated)},
            message="run `prp review` to work the queue" if pending else "queue empty",
        )

    def stage_publish(self) -> StageResult:
        return StageResult("publish", True, outcomes.report(self.store, self.company))

    # ---------------- runner ----------------

    def run(self, stages: Optional[List[str]] = None, **kwargs) -> List[StageResult]:
        table: Dict[str, Callable[..., StageResult]] = {
            "seed": self.stage_seed,
            "discover": self.stage_discover,
            "ingest": self.stage_ingest,
            "fetch": self.stage_fetch,
            "extract": self.stage_extract,
            "link": self.stage_link,
            "verify": self.stage_verify,
            "predict": self.stage_predict,
            "review": self.stage_review,
            "publish": self.stage_publish,
        }
        # `fetch` is out of the default path: it depends on the network and on
        # YouTube not blocking us. Request it explicitly.
        order = stages or [
            "seed", "ingest", "extract", "link", "verify", "predict", "review", "publish"
        ]

        results = []
        for name in order:
            fn = table.get(name)
            if fn is None:
                raise SystemExit(
                    "Unknown stage {!r}. Known: {}".format(name, ", ".join(table))
                )
            result = fn(**kwargs.get(name, {}))
            results.append(result)
            if not result.ok:
                break
        return results
