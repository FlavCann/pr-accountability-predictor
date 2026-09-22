"""The LLM harness (concepts 3 and 8).

Everything that stands between "call the model" and "it works at volume":

* retries with exponential backoff on 429/5xx, and *not* on 4xx
* schema validation with one repair attempt before giving up
* prompt caching, with breakpoints placed so the transcript is paid for once
  across the several passes we make over it
* a content-hash result cache, so an unchanged re-run costs nothing
* per-call token and cost accounting into the run ledger
* per-item failure isolation: one bad transcript never loses the others

It also runs without an API key. `FixtureClient` replays recorded responses so
the pipeline, the evals and the tests are all exercisable offline.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, ValidationError

from . import config
from .domain import RunRecord, content_hash

try:  # the SDK is optional for offline work
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None  # type: ignore


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    cached_locally: bool = False
    unpriced_models: List[str] = field(default_factory=list)
    """Models that ran but are missing from `config.PRICING`. Their tokens are
    counted but cost $0, so any entry here means `cost_usd` is too low."""

    def __iadd__(self, other: "Usage") -> "Usage":
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cache_write_tokens += other.cache_write_tokens
        self.cost_usd += other.cost_usd
        for m in other.unpriced_models:
            if m not in self.unpriced_models:
                self.unpriced_models.append(m)
        return self


def usage_from_response(response, default_model: str) -> Usage:
    """Tokens and dollars for one API response, priced per model.

    A response can contain several sub-inferences at different prices -- with
    the advisor tool, the executor's turns and the advisor's consultation are
    separate iterations, and the advisor is the expensive one. When the API
    reports `usage.iterations`, each is priced at its own `model`'s rates and
    the top-level totals are not used. Without iterations, the top-level totals
    are priced at `default_model`.
    """
    u = getattr(response, "usage", None)
    if u is None:
        return Usage()

    def _tokens(obj) -> Dict[str, int]:
        return {
            "input_tokens": getattr(obj, "input_tokens", 0) or 0,
            "output_tokens": getattr(obj, "output_tokens", 0) or 0,
            "cache_read_tokens": getattr(obj, "cache_read_input_tokens", 0) or 0,
            "cache_write_tokens": getattr(obj, "cache_creation_input_tokens", 0) or 0,
        }

    parts = [
        (str(getattr(it, "model", "") or default_model), _tokens(it))
        for it in (getattr(u, "iterations", None) or [])
    ] or [(default_model, _tokens(u))]

    total = Usage()
    for model, tok in parts:
        piece = Usage(**tok)
        if model in config.PRICING:
            piece.cost_usd = config.estimate_cost(model, **tok)
        else:
            piece.unpriced_models.append(model)
        total += piece
    return total


@dataclass
class Result:
    parsed: Any
    usage: Usage = field(default_factory=Usage)
    raw: str = ""


class LLMError(RuntimeError):
    pass


class Refusal(LLMError):
    pass


def _cache_key(model: str, prompt_version: str, system: str, user: str) -> str:
    return content_hash("|".join([model, prompt_version, system, user]))


class LLMClient:
    """Wraps the Anthropic SDK with the harness above.

    Parameters
    ----------
    store
        Used for the result cache and the run ledger. Optional -- pass None to
        disable both (the eval harness does this when measuring raw model
        behaviour).
    dry_run
        Never call the API. Returns an empty parse and zero usage. Lets the
        whole pipeline be walked for wiring errors without spending anything.
    """

    def __init__(self, store=None, model: Optional[str] = None, dry_run: bool = False):
        self.store = store
        self.model = model or config.EXECUTOR_MODEL
        self.dry_run = dry_run
        self._client = None

    # -- lazily constructed so importing this module never needs credentials --
    @property
    def client(self):
        if self._client is None:
            if anthropic is None:
                raise LLMError(
                    "The `anthropic` package is not installed. "
                    "pip install -r requirements.txt"
                )
            if not config.api_key_present():
                raise LLMError(
                    "No ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN) in the "
                    "environment. Export one, or run with --dry-run, or point "
                    "the pipeline at fixtures with --fixtures."
                )
            self._client = anthropic.Anthropic(timeout=config.REQUEST_TIMEOUT_S)
        return self._client

    # ------------------------------------------------------------------

    def structured(
        self,
        system: str,
        question: str,
        output_model: Type[BaseModel],
        cached_context: Optional[str] = None,
        prompt_version: str = "",
        max_tokens: int = 8000,
        use_cache: bool = True,
    ) -> Result:
        """One structured call, with caching and retries.

        `cached_context` is the large, stable payload -- the transcript. It is
        placed in its own content block with a cache breakpoint, ahead of
        `question`, which varies per pass. Render order is tools -> system ->
        messages, so this keeps the expensive bytes in the cached prefix while
        the varying part sits after the last breakpoint.
        """
        if self.dry_run:
            return Result(parsed=output_model.model_construct(), usage=Usage())

        key = _cache_key(self.model, prompt_version, system, (cached_context or "") + question)
        if use_cache and self.store is not None:
            hit = self.store.cache_get(key)
            if hit is not None:
                return Result(
                    parsed=output_model(**json.loads(hit)),
                    usage=Usage(cached_locally=True),
                    raw=hit,
                )

        result = self._call_with_retries(
            system=system,
            question=question,
            cached_context=cached_context,
            output_model=output_model,
            max_tokens=max_tokens,
        )

        if use_cache and self.store is not None and result.raw:
            self.store.cache_put(key, result.raw)
        return result

    def _call_with_retries(
        self,
        system: str,
        question: str,
        cached_context: Optional[str],
        output_model: Type[BaseModel],
        max_tokens: int,
    ) -> Result:
        # Stable content first, volatile last -- see the docstring above.
        system_blocks = [
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
        ]
        content: List[Dict[str, Any]] = []
        if cached_context:
            content.append(
                {
                    "type": "text",
                    "text": cached_context,
                    "cache_control": {"type": "ephemeral"},
                }
            )
        content.append({"type": "text", "text": question})

        last_exc: Optional[Exception] = None
        for attempt in range(config.MAX_RETRIES):
            try:
                response = self.client.messages.parse(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system_blocks,
                    messages=[{"role": "user", "content": content}],
                    output_format=output_model,
                )
                return self._interpret(response, output_model)

            except Refusal:
                raise
            except ValidationError as e:
                # Schema repair: tell the model exactly what failed, once.
                last_exc = e
                if attempt == 0:
                    content.append(
                        {
                            "type": "text",
                            "text": (
                                "Your previous response did not validate against "
                                "the schema:\n{}\nReturn valid output.".format(e)
                            ),
                        }
                    )
                    continue
                raise LLMError("schema validation failed after repair: {}".format(e))
            except Exception as e:  # noqa: BLE001 - narrowed below
                if not self._retryable(e) or attempt == config.MAX_RETRIES - 1:
                    raise
                last_exc = e
                delay = min(2**attempt + random.uniform(0, 1), 30.0)
                time.sleep(delay)

        raise LLMError("exhausted retries: {}".format(last_exc))

    @staticmethod
    def _retryable(e: Exception) -> bool:
        """429, 408, 409 and 5xx are worth retrying. 400/401/404 are not."""
        if anthropic is None:
            return False
        if isinstance(e, (anthropic.RateLimitError, anthropic.APIConnectionError)):
            return True
        if isinstance(e, anthropic.APIStatusError):
            return e.status_code in (408, 409, 429) or e.status_code >= 500
        return False

    def _interpret(self, response, output_model: Type[BaseModel]) -> Result:
        if getattr(response, "stop_reason", None) == "refusal":
            details = getattr(response, "stop_details", None)
            raise Refusal(
                "model refused ({})".format(getattr(details, "category", "unknown"))
            )

        parsed = getattr(response, "parsed_output", None)
        if parsed is None:
            raise LLMError(
                "no parsed output (stop_reason={})".format(
                    getattr(response, "stop_reason", "?")
                )
            )

        usage = usage_from_response(response, self.model)
        return Result(parsed=parsed, usage=usage, raw=parsed.model_dump_json())


class FixtureClient(LLMClient):
    """Replays recorded responses. Lets the pipeline and evals run offline.

    Keyed the same way as the result cache, so a fixture file captured from a
    real run can be replayed byte-for-byte.
    """

    def __init__(self, fixtures: Dict[str, str], store=None, model: Optional[str] = None):
        super().__init__(store=store, model=model)
        self.fixtures = fixtures
        self.misses: List[str] = []

    def structured(
        self,
        system: str,
        question: str,
        output_model: Type[BaseModel],
        cached_context: Optional[str] = None,
        prompt_version: str = "",
        max_tokens: int = 8000,
        use_cache: bool = True,
    ) -> Result:
        key = _cache_key(self.model, prompt_version, system, (cached_context or "") + question)
        if key in self.fixtures:
            raw = self.fixtures[key]
            return Result(parsed=output_model(**json.loads(raw)), usage=Usage(), raw=raw)
        self.misses.append(key)
        return Result(parsed=output_model.model_construct(promises=[]), usage=Usage())


class RunLedger:
    """Context manager that opens a run, accumulates usage, and records it.

    Usage:
        with RunLedger(store, "extract", model=..., prompt_version=...) as run:
            run.add(usage); run.ok(); run.fail()
    """

    def __init__(self, store, stage: str, model: str = "", prompt_version: str = ""):
        self.store = store
        self.record = RunRecord(stage=stage, model=model, prompt_version=prompt_version)

    def __enter__(self) -> "RunLedger":
        return self

    def add(self, usage: Usage) -> None:
        self.record.input_tokens += usage.input_tokens
        self.record.output_tokens += usage.output_tokens
        self.record.cache_read_tokens += usage.cache_read_tokens
        self.record.cache_write_tokens += usage.cache_write_tokens
        self.record.cost_usd += usage.cost_usd
        for m in usage.unpriced_models:
            flag = "UNPRICED MODEL {} -- cost under-reported".format(m)
            if flag not in self.record.notes:
                self.record.notes = (self.record.notes + " | " + flag).strip(" |")

    def ok(self, n: int = 1) -> None:
        self.record.items_ok += n

    def fail(self, n: int = 1, note: str = "") -> None:
        self.record.items_failed += n
        if note:
            self.record.notes = (self.record.notes + " | " + note).strip(" |")

    @property
    def run_id(self) -> str:
        return self.record.id

    def __exit__(self, *exc) -> None:
        from datetime import datetime

        self.record.finished_at = datetime.utcnow()
        if self.store is not None:
            self.store.put_run(self.record)
