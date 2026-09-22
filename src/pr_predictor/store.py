"""The store: SQLite, append-only for evidence and decisions (concepts 3 and 8).

Two classes of table, and the distinction is the design:

**Records -- insert-only.** Sources, segments, promises, extraction runs,
thread memberships, evidence, verdicts, analyst reviews, predictions and their
resolutions, and the run ledger. These are what an auditor would ask to see.
No `UPDATE`, no `REPLACE`: inserting a duplicate ID raises. A change of mind is
a *new* row -- a review is appended rather than written over its verdict, a
re-fetched source supersedes the old one rather than replacing it. So the full
history of every claim is always recoverable.

**Derived state -- mutable, and labelled as such.** Thread summaries (claim,
first/last seen, deadline) are recomputed from records as new restatements
arrive; the LLM result cache and durable memory are key-value stores. None of
these is evidence. A thread's *membership* is a record (`thread_members`); only
its summary is derived.

`tests/test_pipeline.py::test_records_are_insert_only` enforces the split by
reading this schema.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

from . import config
from .domain import (
    Evidence,
    Prediction,
    Promise,
    PromiseThread,
    RunRecord,
    Source,
    Verdict,
)
from .taxonomy import VerdictStatus

SCHEMA_VERSION = 3

RECORD_TABLES = (
    "sources", "source_supersessions", "segments", "promises", "extractions",
    "thread_members", "evidence", "verdicts", "reviews", "predictions",
    "prediction_resolutions", "runs", "eval_runs",
)
DERIVED_TABLES = ("threads", "llm_cache", "memory")

SCHEMA = """
-- ---------------- records: insert-only ----------------
CREATE TABLE IF NOT EXISTS sources (
    id TEXT PRIMARY KEY, company TEXT, kind TEXT, url TEXT, title TEXT,
    published TEXT, retrieved_at TEXT, text TEXT, text_hash TEXT, fetcher TEXT
);
CREATE TABLE IF NOT EXISTS source_supersessions (
    old_id TEXT PRIMARY KEY, new_id TEXT, at TEXT
);
CREATE TABLE IF NOT EXISTS segments (
    source_id TEXT, idx INTEGER, start_char INTEGER, end_char INTEGER,
    t_start REAL, t_end REAL,
    PRIMARY KEY (source_id, idx)
);
CREATE TABLE IF NOT EXISTS promises (
    id TEXT PRIMARY KEY, source_id TEXT, company TEXT, payload TEXT,
    prompt_version TEXT, model TEXT, run_id TEXT
);
CREATE TABLE IF NOT EXISTS extractions (
    source_id TEXT, prompt_version TEXT, model TEXT, run_id TEXT,
    n_promises INTEGER, at TEXT,
    PRIMARY KEY (source_id, prompt_version, model)
);
CREATE TABLE IF NOT EXISTS thread_members (
    promise_id TEXT PRIMARY KEY, thread_id TEXT, run_id TEXT, at TEXT
);
CREATE TABLE IF NOT EXISTS evidence (
    id TEXT PRIMARY KEY, thread_id TEXT, payload TEXT
);
CREATE TABLE IF NOT EXISTS verdicts (
    id TEXT PRIMARY KEY, thread_id TEXT, as_of TEXT, payload TEXT
);
CREATE TABLE IF NOT EXISTS reviews (
    verdict_id TEXT, action TEXT, corrected_status TEXT, note TEXT,
    reviewer TEXT, at TEXT
);
CREATE TABLE IF NOT EXISTS predictions (
    id TEXT PRIMARY KEY, thread_id TEXT, payload TEXT
);
CREATE TABLE IF NOT EXISTS prediction_resolutions (
    prediction_id TEXT PRIMARY KEY, status TEXT, brier REAL, at TEXT
);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY, stage TEXT, payload TEXT
);
-- Added without a schema bump: new table only, so older stores gain it on open.
CREATE TABLE IF NOT EXISTS eval_runs (
    id TEXT PRIMARY KEY, prompt_version TEXT, model TEXT, gold_fingerprint TEXT,
    payload TEXT, at TEXT
);

-- ---------------- derived state: mutable ----------------
CREATE TABLE IF NOT EXISTS threads (
    id TEXT PRIMARY KEY, company TEXT, prompt_version TEXT, payload TEXT
);
CREATE TABLE IF NOT EXISTS llm_cache (
    key TEXT PRIMARY KEY, response TEXT, created_at TEXT
);
CREATE TABLE IF NOT EXISTS memory (
    company TEXT, kind TEXT, key TEXT, value TEXT, updated_at TEXT,
    PRIMARY KEY (company, kind, key)
);

CREATE INDEX IF NOT EXISTS idx_promises_source ON promises(source_id);
CREATE INDEX IF NOT EXISTS idx_promises_version ON promises(prompt_version);
CREATE INDEX IF NOT EXISTS idx_members_thread ON thread_members(thread_id);
CREATE INDEX IF NOT EXISTS idx_verdicts_thread ON verdicts(thread_id);
CREATE INDEX IF NOT EXISTS idx_reviews_verdict ON reviews(verdict_id);
CREATE INDEX IF NOT EXISTS idx_predictions_thread ON predictions(thread_id);
"""


def _json(model) -> str:
    return model.model_dump_json()


def _now() -> str:
    return datetime.utcnow().isoformat()


class StoreVersionError(RuntimeError):
    pass


class Store:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path or config.DB_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        existed = self.path.exists() and self.path.stat().st_size > 0
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row

        version = self.conn.execute("PRAGMA user_version").fetchone()[0]
        if existed and version != SCHEMA_VERSION:
            self.conn.close()
            raise StoreVersionError(
                "{} was written by an older version (schema {}, need {}). Pre-0.3 "
                "stores held only offline replay data: delete the file and re-run."
                .format(self.path, version, SCHEMA_VERSION)
            )
        self.conn.executescript(SCHEMA)
        self.conn.execute("PRAGMA user_version = {}".format(SCHEMA_VERSION))
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _insert(self, sql: str, rows: List[tuple]) -> int:
        if not rows:
            return 0
        self.conn.executemany(sql, rows)
        self.conn.commit()
        return len(rows)

    # ---------------- sources, supersession, segments ----------------

    def put_source(self, src: Source) -> None:
        self._insert(
            "INSERT INTO sources VALUES (?,?,?,?,?,?,?,?,?,?)",
            [(
                src.id, src.company, src.kind, src.url, src.title,
                src.published.isoformat() if src.published else None,
                src.retrieved_at.isoformat(), src.text, src.text_hash, src.fetcher,
            )],
        )

    def supersede_source(self, old_id: str, new_id: str) -> None:
        """Record that a re-fetch replaced a source. The old row stays."""
        self._insert(
            "INSERT INTO source_supersessions VALUES (?,?,?)", [(old_id, new_id, _now())]
        )

    _LIVE_SOURCES = "id NOT IN (SELECT old_id FROM source_supersessions)"

    def get_source(self, source_id: str) -> Optional[Source]:
        """Any source by ID, superseded or not -- old citations must still resolve."""
        row = self.conn.execute("SELECT * FROM sources WHERE id=?", (source_id,)).fetchone()
        return self._row_to_source(row) if row else None

    def source_by_url(self, url: str) -> Optional[Source]:
        row = self.conn.execute(
            "SELECT * FROM sources WHERE url=? AND {} ORDER BY retrieved_at DESC LIMIT 1"
            .format(self._LIVE_SOURCES),
            (url,),
        ).fetchone()
        return self._row_to_source(row) if row else None

    def list_sources(self, company: Optional[str] = None) -> List[Source]:
        sql = "SELECT * FROM sources WHERE {}".format(self._LIVE_SOURCES)
        args: tuple = ()
        if company:
            sql += " AND company=?"
            args = (company,)
        sql += " ORDER BY published"
        return [self._row_to_source(r) for r in self.conn.execute(sql, args)]

    @staticmethod
    def _row_to_source(row: sqlite3.Row) -> Source:
        return Source(
            id=row["id"], company=row["company"], kind=row["kind"], url=row["url"],
            title=row["title"],
            published=date.fromisoformat(row["published"]) if row["published"] else None,
            retrieved_at=datetime.fromisoformat(row["retrieved_at"]),
            text=row["text"], text_hash=row["text_hash"], fetcher=row["fetcher"],
        )

    def put_segments(self, segments: Iterable) -> None:
        self._insert(
            "INSERT INTO segments VALUES (?,?,?,?,?,?)",
            [(s.source_id, s.index, s.start_char, s.end_char, s.t_start, s.t_end)
             for s in segments],
        )

    def segment_at(self, source_id: str, char_pos: int):
        """The timestamped cue containing a character position.

        The join that turns an extracted span into a moment in the video.
        """
        return self.conn.execute(
            "SELECT * FROM segments WHERE source_id=? AND start_char<=? AND end_char>=? "
            "ORDER BY idx LIMIT 1",
            (source_id, char_pos, char_pos),
        ).fetchone()

    # ---------------- promises & extraction runs ----------------

    def add_promises(self, promises: Iterable[Promise]) -> int:
        return self._insert(
            "INSERT INTO promises VALUES (?,?,?,?,?,?,?)",
            [(p.id, p.source_id, p.company, _json(p), p.prompt_version, p.model, p.run_id)
             for p in promises],
        )

    def record_extraction(
        self, source_id: str, prompt_version: str, model: str, run_id: str, n: int
    ) -> None:
        """Mark a source as extracted by a prompt/model pair.

        What makes the extract stage idempotent: a second run skips it instead
        of writing the same promises again under new IDs.
        """
        self._insert(
            "INSERT INTO extractions VALUES (?,?,?,?,?,?)",
            [(source_id, prompt_version, model, run_id, n, _now())],
        )

    def has_extraction(self, source_id: str, prompt_version: str, model: str) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM extractions WHERE source_id=? AND prompt_version=? AND model=?",
            (source_id, prompt_version, model),
        ).fetchone() is not None

    def list_promises(
        self,
        company: Optional[str] = None,
        source_id: Optional[str] = None,
        prompt_version: Optional[str] = None,
        unthreaded_only: bool = False,
    ) -> List[Promise]:
        """Promises from live (non-superseded) sources, with thread_id hydrated."""
        sql = (
            "SELECT p.payload, m.thread_id FROM promises p "
            "LEFT JOIN thread_members m ON m.promise_id = p.id "
            "WHERE p.source_id NOT IN (SELECT old_id FROM source_supersessions)"
        )
        args: List = []
        if company:
            sql += " AND p.company=?"
            args.append(company)
        if source_id:
            sql += " AND p.source_id=?"
            args.append(source_id)
        if prompt_version:
            sql += " AND p.prompt_version=?"
            args.append(prompt_version)
        if unthreaded_only:
            sql += " AND m.thread_id IS NULL"
        sql += " ORDER BY p.rowid"
        out = []
        for r in self.conn.execute(sql, args):
            p = Promise(**json.loads(r["payload"]))
            p.thread_id = r["thread_id"]
            out.append(p)
        return out

    def prompt_versions(self) -> List[str]:
        return [
            r["prompt_version"]
            for r in self.conn.execute(
                "SELECT DISTINCT prompt_version FROM promises ORDER BY prompt_version"
            )
        ]

    # ---------------- threads: membership is a record ----------------

    def add_member(self, thread_id: str, promise_id: str, run_id: str = "") -> None:
        """Assign a promise to a thread. Once.

        `promise_id` is the primary key, so a promise can never be linked twice
        -- a second link stage run raises rather than silently duplicating.
        """
        self._insert(
            "INSERT INTO thread_members VALUES (?,?,?,?)",
            [(promise_id, thread_id, run_id, _now())],
        )

    def _members(self, thread_id: str) -> List[str]:
        return [
            r["promise_id"]
            for r in self.conn.execute(
                "SELECT m.promise_id FROM thread_members m JOIN promises p "
                "ON p.id = m.promise_id WHERE m.thread_id=? AND p.source_id NOT IN "
                "(SELECT old_id FROM source_supersessions) ORDER BY m.rowid",
                (thread_id,),
            )
        ]

    def put_thread(self, t: PromiseThread) -> None:
        """Upsert the thread's derived summary. Membership is not stored here."""
        summary = t.model_copy(update={"promise_ids": []})
        self.conn.execute(
            "INSERT OR REPLACE INTO threads VALUES (?,?,?,?)",
            (t.id, t.company, t.prompt_version, _json(summary)),
        )
        self.conn.commit()

    def _hydrate_thread(self, payload: str) -> PromiseThread:
        t = PromiseThread(**json.loads(payload))
        t.promise_ids = self._members(t.id)
        return t

    def get_thread(self, thread_id: str) -> Optional[PromiseThread]:
        row = self.conn.execute(
            "SELECT payload FROM threads WHERE id=?", (thread_id,)
        ).fetchone()
        return self._hydrate_thread(row["payload"]) if row else None

    def list_threads(
        self, company: Optional[str] = None, prompt_version: Optional[str] = None
    ) -> List[PromiseThread]:
        sql = "SELECT payload FROM threads WHERE 1=1"
        args: List = []
        if company:
            sql += " AND company=?"
            args.append(company)
        if prompt_version:
            sql += " AND prompt_version=?"
            args.append(prompt_version)
        sql += " ORDER BY rowid"
        return [self._hydrate_thread(r["payload"]) for r in self.conn.execute(sql, args)]

    # ---------------- evidence ----------------

    def add_evidence(self, items: Iterable[Evidence]) -> None:
        self._insert(
            "INSERT INTO evidence VALUES (?,?,?)",
            [(e.id, e.thread_id, _json(e)) for e in items],
        )

    def list_evidence(self, thread_id: str) -> List[Evidence]:
        return [
            Evidence(**json.loads(r["payload"]))
            for r in self.conn.execute(
                "SELECT payload FROM evidence WHERE thread_id=? ORDER BY rowid", (thread_id,)
            )
        ]

    # ---------------- verdicts and reviews ----------------

    def put_verdict(self, v: Verdict) -> None:
        """Record a verdict as the model made it. Reviews never overwrite it."""
        clean = v.model_copy(
            update={"reviewed_by": None, "review_action": None,
                    "corrected_status": None, "review_note": None}
        )
        self._insert(
            "INSERT INTO verdicts VALUES (?,?,?,?)",
            [(v.id, v.thread_id, v.as_of.isoformat(), _json(clean))],
        )

    def add_review(
        self,
        verdict_id: str,
        action: str,
        reviewer: str,
        corrected_status: Optional[VerdictStatus] = None,
        note: Optional[str] = None,
    ) -> None:
        """Append an analyst's decision. The full review history is kept."""
        if action not in ("accepted", "corrected", "rejected"):
            raise ValueError("review action must be accepted, corrected or rejected")
        self._insert(
            "INSERT INTO reviews VALUES (?,?,?,?,?,?)",
            [(verdict_id, action,
              corrected_status.value if corrected_status else None,
              note, reviewer, _now())],
        )

    def list_reviews(self, verdict_id: str) -> List[sqlite3.Row]:
        return list(self.conn.execute(
            "SELECT * FROM reviews WHERE verdict_id=? ORDER BY rowid", (verdict_id,)
        ))

    def _hydrate_verdict(self, payload: str) -> Verdict:
        v = Verdict(**json.loads(payload))
        reviews = self.list_reviews(v.id)
        if reviews:
            last = reviews[-1]
            v.review_action = last["action"]
            v.reviewed_by = last["reviewer"]
            v.review_note = last["note"]
            v.corrected_status = (
                VerdictStatus(last["corrected_status"]) if last["corrected_status"] else None
            )
        return v

    def get_verdict(self, verdict_id: str) -> Optional[Verdict]:
        row = self.conn.execute(
            "SELECT payload FROM verdicts WHERE id=?", (verdict_id,)
        ).fetchone()
        return self._hydrate_verdict(row["payload"]) if row else None

    def list_verdicts(self, thread_id: Optional[str] = None) -> List[Verdict]:
        """Newest first. Each carries its latest review, if any."""
        sql = "SELECT payload FROM verdicts"
        args: tuple = ()
        if thread_id:
            sql += " WHERE thread_id=?"
            args = (thread_id,)
        sql += " ORDER BY as_of DESC, rowid DESC"
        return [self._hydrate_verdict(r["payload"]) for r in self.conn.execute(sql, args)]

    def latest_verdict(self, thread_id: str) -> Optional[Verdict]:
        vs = self.list_verdicts(thread_id)
        return vs[0] if vs else None

    # ---------------- predictions ----------------

    def put_prediction(self, p: Prediction) -> None:
        clean = p.model_copy(update={"resolved_status": None, "brier_score": None})
        self._insert(
            "INSERT INTO predictions VALUES (?,?,?)", [(p.id, p.thread_id, _json(clean))]
        )

    def resolve_prediction(self, prediction_id: str, status: VerdictStatus, brier: float) -> None:
        """Grade a prediction once its outcome is known. Terminal: once per prediction."""
        self._insert(
            "INSERT INTO prediction_resolutions VALUES (?,?,?,?)",
            [(prediction_id, status.value, brier, _now())],
        )

    def list_predictions(self, current_only: bool = False) -> List[Prediction]:
        """All predictions, or only the latest per thread."""
        rows = self.conn.execute(
            "SELECT p.payload, r.status, r.brier FROM predictions p "
            "LEFT JOIN prediction_resolutions r ON r.prediction_id = p.id "
            "ORDER BY p.rowid"
        ).fetchall()
        out: List[Prediction] = []
        for r in rows:
            pred = Prediction(**json.loads(r["payload"]))
            if r["status"]:
                pred.resolved_status = VerdictStatus(r["status"])
                pred.brier_score = r["brier"]
            out.append(pred)
        if not current_only:
            return out
        latest: Dict[str, Prediction] = {}
        for pred in out:
            latest[pred.thread_id] = pred
        return list(latest.values())

    def latest_prediction(self, thread_id: str) -> Optional[Prediction]:
        preds = [p for p in self.list_predictions() if p.thread_id == thread_id]
        return preds[-1] if preds else None

    # ---------------- run ledger ----------------

    def put_run(self, r: RunRecord) -> None:
        self._insert("INSERT INTO runs VALUES (?,?,?)", [(r.id, r.stage, _json(r))])

    def list_runs(self, limit: int = 50) -> List[RunRecord]:
        return [
            RunRecord(**json.loads(r["payload"]))
            for r in self.conn.execute(
                "SELECT payload FROM runs ORDER BY rowid DESC LIMIT ?", (limit,)
            )
        ]

    def total_cost(self) -> float:
        return sum(
            json.loads(r["payload"]).get("cost_usd", 0.0)
            for r in self.conn.execute("SELECT payload FROM runs")
        )

    # ---------------- eval results ----------------

    def add_eval_run(self, result: Dict) -> None:
        """Record a benchmark result. Every run is kept, so any past score can
        be re-examined against the manifest that produced it."""
        self._insert(
            "INSERT INTO eval_runs VALUES (?,?,?,?,?,?)",
            [(result["run_id"], result.get("prompt_version"), result.get("model"),
              result.get("gold_fingerprint"), json.dumps(result, default=str), _now())],
        )

    def list_eval_runs(self, limit: int = 20) -> List[Dict]:
        """Newest first, each carrying `recorded_at`."""
        out = []
        for r in self.conn.execute(
            "SELECT payload, at FROM eval_runs ORDER BY rowid DESC LIMIT ?", (limit,)
        ):
            run = json.loads(r["payload"])
            run.setdefault("recorded_at", r["at"])
            out.append(run)
        return out

    # ---------------- derived: result cache ----------------

    def cache_get(self, key: str) -> Optional[str]:
        row = self.conn.execute(
            "SELECT response FROM llm_cache WHERE key=?", (key,)
        ).fetchone()
        return row["response"] if row else None

    def cache_put(self, key: str, response: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO llm_cache VALUES (?,?,?)", (key, response, _now())
        )
        self.conn.commit()

    # ---------------- derived: durable memory ----------------

    def memory_put(self, company: str, kind: str, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO memory VALUES (?,?,?,?,?)",
            (company, kind, key, value, _now()),
        )
        self.conn.commit()

    def memory_list(self, company: str, kind: Optional[str] = None) -> List[sqlite3.Row]:
        sql = "SELECT * FROM memory WHERE company=?"
        args: List = [company]
        if kind:
            sql += " AND kind=?"
            args.append(kind)
        return list(self.conn.execute(sql, args))


@contextmanager
def open_store(path: Optional[Path] = None) -> Iterator[Store]:
    s = Store(path)
    try:
        yield s
    finally:
        s.close()
