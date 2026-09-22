# CLAUDE.md

Context for agents working on this repo. Everything here is something you
cannot learn by reading the code, or something that will otherwise be
rediscovered painfully. Anything derivable from the source does not belong
here — see `README.md` for the layout and flow diagram.

## What this is

A pipeline that extracts forward-looking commitments from corporate investor
transcripts, threads restatements across years, verifies them against later
evidence, and predicts which open commitments will be kept. The output is a
claim about a named public company, so **provenance is a product requirement,
not hygiene**.

## Invariants

Break these and the product stops being defensible:

1. **Records are insert-only.** Sources, promises, thread memberships,
   evidence, verdicts, analyst reviews, predictions and the run ledger are
   never updated or replaced (`store.RECORD_TABLES`; a test reads the schema).
   A change of mind is a new row: a review is appended rather than written over
   its verdict, a re-fetched source supersedes the old one. Only *derived*
   state -- thread summaries, the LLM cache, durable memory -- is mutable.
2. **Every row records `prompt_version`, `model` and `run_id`.** No exceptions.
3. **The model never supplies offsets, and never supplies the citation.** It
   copies the first and last few words of each promise; `extract.locate` finds
   them in the source and computes the offsets. The citation is the source text
   between them. Models cannot count characters reliably, and a wrong offset
   that lands on other real text passes any range check -- so do not "simplify"
   this back to model-generated offsets. Unlocatable promises are counted as
   uncitable and never stored.
4. **Never truncate the input.** The original pipeline's central defect was
   `MAX_WORDS = 1000`, which sent 8–36% of each transcript. If something does
   not fit, chunk it — `context.chunk_text` preserves absolute offsets.
5. **Timestamps survive ingestion.** `ingest.build_source` writes a `Segment`
   per caption cue so any character position resolves to a moment on the video.
   Do not add a code path that flattens a transcript to a bare string.
6. **Every stage is idempotent.** Running the pipeline twice must change
   nothing. Extraction is recorded per (source, prompt, model) and skipped
   thereafter; `thread_members` is keyed on the promise, so it cannot join two
   threads. `test_whole_pipeline_is_idempotent` runs the real corpus twice.
7. **A revised deadline is a new thread.** A target moved from 2025 to 2030
   must not merge into the original — losing the revision loses the story.
   `threading_._match_score` returns 0.0 on a deadline conflict.

## The eval gate

**Do not change the extraction prompt, the taxonomy, or `extract.py` without
running the evals.**

```bash
python -m pr_predictor eval --prompt-version extract-v3
```

The gate blocks on a span-fidelity regression or more forbidden extractions
*regardless of F1*: an unverifiable citation is not a tradeable quantity here.
It also blocks when the model, prompt and harness differ from the champion on
more than one axis, when the scoring code or gold set changed, and — once there
are `EVAL_MIN_DOCS_FOR_CI` scored documents — when the paired bootstrap
interval of the gain does not clear zero. Below that it passes on a point
estimate and says **UNDERPOWERED**: with two dev documents, today, every
verdict is.

`extract-v1` in `prompts.py` is the original prompt, preserved as the baseline.
Do not delete it.

The gold set is generated from labelled phrases by `evals/build_gold.py`, not
hand-typed offsets. Edit the phrases and re-run it. Labelling rules are in
`evals/GUIDELINES.md`.

**Splits (`evals/splits.json`) are assigned before labelling and never moved.**
Every non-dev transcript supplies an extract-v3 worked example. Four are
`examples` and never scored. Two (`S07gnpUx15s`, `1FUvRp9MGpU`) are the `test`
set: each supplies exactly one example, and that passage is **masked** in
`evals/labels/<id>.json` -- never labelled, never scored. Test labels are made
blind: never show model output for a test transcript before its labelling is
complete. Do not draw a new prompt example from
a `dev`/`test`/`fresh` transcript — `find_leakage` refuses the eval, even for
transcripts not yet labelled.

The judge (`--judge`) is reported, never gated on, and marked `validated:
false` until `eval-judge-check` passes against human pairs in
`evals/judge_calibration/`. Adjudication cases and judge calibration pairs are
human judgements about a named company: never generate them.

## Traps specific to this domain

- **Barry Callebaut's fiscal year ends 31 August.** "FY23" means Sep 2022 – Aug
  2023, *not* calendar 2023. A speaker saying "this fiscal year" in a November
  presentation means something ending the following August. Never map a
  fiscal-year phrase to a calendar date without this. `Promise.deadline_raw`
  exists to preserve what was actually said; leave `deadline` null rather than
  guessing.
- **The transcripts are machine-generated and garble proper nouns.** The
  company's own name appears as `calabroth` and `calibot`; COVID-19 as
  `kovac 19`; Swiss francs as `Nets`. Corrections live in durable memory
  (`context.BARRY_CALLEBAUT_SEED`) and are injected into the prompt. Correct
  them in `normalized_claim`, never in the stored source text — the source is
  evidence and must stay as fetched.
- **YouTube blocks aggressively.** `RequestBlocked` stops the fetch stage
  deliberately rather than retrying, so the block is not extended. Set
  `TRANSCRIPT_PROXY_URL` (residential proxies; datacentre IPs are usually
  blocked too) or switch network.
- **The files in `transcripts/` have no timestamps.** They were written by the
  pre-0.2 pipeline, which discarded them. They ingest fine and are marked
  `fetcher="legacy_txt_no_timestamps"`; promises from them have no video deep
  link. Re-fetch to upgrade.
- **Results presentations are mostly not promises.** A 12,000-word transcript
  typically contains a handful of real commitments surrounded by Q&A, catering
  banter and macro commentary. Precision matters more than recall: a false
  promise attributed to a company is worse than a missed one.

## Model routing

- Executor (`config.EXECUTOR_MODEL`, default `claude-sonnet-5`) does the
  high-volume schema-constrained work.
- Advisor (`config.ADVISOR_MODEL`, default `claude-opus-4-8`) is consulted only
  on contested adjudications, via the advisor tool.
- **The advisor default is deliberately not `claude-fable-5-1`.** Fable returns
  `advisor_redacted_result` — encrypted advice that can be replayed but not
  read. Since a verdict's rationale is part of an audit trail, the default is
  Opus 4.8, which returns plaintext `advisor_result`. If you change this,
  understand you are trading auditability for capability.
- When reading advisor output, switch on the *content* type of the
  `advisor_tool_result` block, never on the block type. Code that reads `.text`
  unconditionally gets nothing back from an encrypted advisor.
- **Escalation must fail safe.** A contested verdict is re-adjudicated by the
  executor with the advisor tool attached, and that answer becomes the verdict
  (the first call is kept as `initial_status`). Only a validated structured
  answer, after a real consultation, may change the outcome: on an error, no
  consultation, or no parsed output, the initial verdict stands and the
  rationale says why. Do not relax this.
- **Price every response with `llm.usage_from_response`.** One response can
  mix models: the advisor's tokens arrive as separate `usage.iterations`
  entries and cost more. Pricing the top-level total at the executor's rate
  under-reports. Add any new model to `config.PRICING`, or the ledger flags it
  as unpriced.
- The eval judge (`config.JUDGE_MODEL`) is a third model on purpose, so nothing
  grades its own homework.

## Environment

- The venv is Python **3.9**, which is end-of-life. Code must stay 3.9-compatible:
  use `Optional[X]`, not `X | None`, and keep `from __future__ import annotations`
  at the top of every module. The official `mcp` package needs 3.10+, which is
  one reason the MCP server is deferred.
- **Apple's Python 3.9 caches bytecode outside the repo**, in
  `~/Library/Caches/com.apple.python/<path>`. Deleting `__pycache__` does
  nothing. If a test fails in a way that contradicts the source on disk --
  especially `test_prompt_versions_are_pinned` right after an edit -- clear
  that directory for this project.
- The API key lives in a git-ignored `.env` at the repo root
  (`ANTHROPIC_API_KEY=...`), loaded by `config.py`. Never commit it and never
  paste it into a chat.
- Costs are estimated locally by `config.estimate_cost` from a hard-coded price
  table verified on 2026-09-21. **Prices and model IDs move** — re-check before
  trusting the ledger's dollar figures.

## Working offline

There is no API key in this environment by default. Two modes keep the whole
pipeline exercisable:

- `--dry-run` walks the wiring and calls nothing.
- `--replay` replays the gold labels through the *real* extraction path —
  chunking, quote location, span validation, reconciliation — with no key
  and no spend. It is a stand-in for the model only, not a mock of the pipeline.

`--replay` is refused by `eval` on purpose: scoring it would trivially produce
a perfect result.
