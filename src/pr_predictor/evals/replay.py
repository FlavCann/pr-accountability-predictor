"""Offline stand-in for the extraction model: replays the gold labels."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from .gold import GoldDoc, load_gold


class GoldReplayClient:
    """Replays gold labels through the real extraction code path.

    Not a mock of the pipeline -- a stand-in for the *model only*. Everything
    downstream still runs for real: chunk-local offsets are translated to
    absolute, spans are validated against the source, overlapping duplicates
    are reconciled, timestamps are attached.

    Its purpose is to exercise and test the risky mechanics offline, with no
    API key and no spend. It cannot tell you whether the prompt is any good --
    only `run_eval` against a live model does that. Scoring a GoldReplayClient
    run would trivially produce a perfect score, which is why `cmd_eval`
    refuses to use it.
    """

    is_replay = True

    def __init__(self, store, gold_dir: Optional[Path] = None, model: str = "gold-replay"):
        self.store = store
        self.model = model
        self.dry_run = False
        self.gold_dir = gold_dir
        self.docs = {d.source_url: d for d in load_gold(gold_dir)}
        self._resolved: Optional[Dict[str, GoldDoc]] = None

    @property
    def _by_text(self) -> Dict[str, GoldDoc]:
        """Resolve gold URLs to stored source IDs, lazily.

        Lazy because the pipeline constructs the client before the ingest stage
        has run, so the sources do not exist yet at construction time.
        """
        if self._resolved is None:
            self._resolved = {}
            for url, doc in self.docs.items():
                src = self.store.source_by_url(url)
                if src is not None:
                    self._resolved[src.id] = doc
        return self._resolved

    def structured(
        self,
        system: str,
        question: str,
        output_model,
        cached_context: Optional[str] = None,
        prompt_version: str = "",
        max_tokens: int = 8000,
        use_cache: bool = True,
    ):
        from ..llm import Result, Usage

        # Only extraction is replayable from gold labels. Any other call
        # (adjudication, thread tie-breaks) gets an unvalidated empty object;
        # callers read those fields with getattr defaults.
        if "promises" not in getattr(output_model, "model_fields", {}):
            return Result(parsed=output_model.model_construct(), usage=Usage())

        excerpt = self._excerpt(question)
        if excerpt is None:
            return Result(parsed=output_model(promises=[]), usage=Usage())

        for source_id, doc in self._by_text.items():
            src = self.store.get_source(source_id)
            if src is None or excerpt not in src.text:
                continue
            offset = src.text.index(excerpt)
            end = offset + len(excerpt)

            promises = []
            for g in doc.promises:
                if g.start_char >= offset and g.end_char <= end:
                    words = src.text[g.start_char : g.end_char].split()
                    promises.append(
                        {
                            "quote_start": " ".join(words[:8]),
                            "quote_end": " ".join(words[-8:]),
                            "promise_type": g.promise_type or "procedural_commitment",
                            "hedge_level": g.hedge_level or "firm",
                            "normalized_claim": src.text[g.start_char : g.end_char][:200],
                            "deadline": g.deadline_resolved if g.deadline_labelled else None,
                            "deadline_raw": g.deadline_raw,
                            "confidence": 0.9,
                        }
                    )
            return Result(parsed=output_model(promises=promises), usage=Usage())

        return Result(parsed=output_model(promises=[]), usage=Usage())

    @staticmethod
    def _excerpt(question: str) -> Optional[str]:
        start = question.find("<<<EXCERPT>>>\n")
        end = question.find("\n<<<END EXCERPT>>>")
        if start == -1 or end == -1:
            return None
        return question[start + len("<<<EXCERPT>>>\n") : end]
