# Rebuild report

What changed, what it enables, and what is still unproven.

The starting point was 251 lines across four scripts: scrape YouTube
transcripts, send the first 1,000 words of each to Claude, print the results to
a text file. The result (v0.3) is 4,925 lines across 19 modules plus 656 lines
of tests, organised as a pipeline with a durable store, an eval gate, a bounded
verification agent and an HTTP service.

Line count is not the point. The point is that the original could not answer
"is this right?", "what did it cost?", "did that change help?" or "was the
promise kept?" — and none of those were prompt problems.

---

## 1. Where the old code went

| Was | Now | What changed |
| --- | --- | --- |
| `find_videos.py` (60) | `ingest.discover` | Unchanged logic, moved behind the pipeline. |
| `transcript.py` (57) | `ingest.fetch_cues`, `build_source` | **Returns timestamped cues, not a flat string.** |
| `scrape_all.py` (42) | `ingest.fetch_pending` | Writes to the store with a `Segment` per caption cue, instead of a `.txt` file with the timestamps thrown away. |
| `extract_promises.py` (92) | `extract.py` + `prompts.py` + `llm.py` + `store.py` | Split into prompt, harness and persistence. No truncation, located quotes instead of retyped citations, versioned, cached, ledgered. |
| `promises.txt` | `evals/v1_baseline_output.txt` | Kept as the "before" to measure against. Its false positives became the prompt's negative examples. |

New, with no predecessor: `taxonomy.py`, `domain.py`, `config.py`, `store.py`,
`llm.py`, `context.py`, `threading_.py`, `tools.py`, `verify.py`, `evals.py`,
`outcomes.py`, `pipeline.py`, `api.py`, `cli.py`, `CLAUDE.md`.

The flow diagram is in [README.md](README.md#flow).

---

## 2. What each concept turned into

### Implemented and verified offline

**1 · Prompt engineering** → `prompts.py`, `taxonomy.py`
A versioned registry. `extract-v1` is the original prompt, preserved as the
eval baseline. `extract-v3` adds a four-way taxonomy with decision rules, five
negative examples, four of them real false positives produced by v1, worked
positives — all drawn from transcripts *outside* the gold set (the catering
joke itself is a gold test case, so a near-identical remark from another call
stands in for it) — a separate `hedge_level` field, and — the
structural change that matters most — **the model never writes the citation**.
It copies the first and last few words of each promise; `extract.locate` finds
them in the source and computes the offsets. (The first build asked the model
for raw character offsets; see section 6 for why that was replaced.)
*Enables:* every stored citation is source text at offsets we computed, and
anything the model paraphrased or garbled is counted as uncitable instead of
stored. ASR garbling stops being a data-integrity problem.

**2 · Context engineering** → `context.py`
Overlapping chunks that preserve absolute character offsets; a company brief,
an ASR glossary and prior threads assembled from durable memory.
*Enables:* full-document coverage (verified: 100% of a 65,761-character
transcript, 8 chunks, zero offset drift) and a model that knows Forever
Chocolate is a 2016 programme.

**3 · Harness engineering** → `llm.py`, `store.py`
Retries with backoff on 429/5xx and not on 4xx, one schema-repair attempt,
per-item failure isolation, idempotency by content hash, insert-only records,
and a run ledger recording tokens and cost per stage.
*Enables:* `prp ledger`. One bad chunk no longer loses the transcript.

**5 · Tools** → `tools.py`
Three narrow typed tools offered to the model, with `strict: true`, returning
IDs and ≤600-character
excerpts rather than documents, and errors that say how to recover.
*Enables:* the agent can search the corpus without its context filling with
transcripts on the third call.

**6 · Loops** → `verify.gather_evidence`, `pipeline.py`, `prp eval`
The three loops are now separate things: a bounded agent loop (step cap, cost
cap, `NO_EVIDENCE` as a legitimate terminal outcome), a pipeline loop of
idempotent resumable stages, and an improvement loop via the eval gate.

**7 · Memory** → `store.memory_*`, `context.build_extraction_context`
Three kinds, seeded with twelve entries learned from this corpus: facts
(fiscal year end, programme history), ASR corrections, and analyst corrections.
*Enables:* the feedback loop — correcting a verdict in `prp review` or
`POST /verdicts/{id}/review` writes memory that is retrieved into every later
extraction. This is the dotted line in the flow diagram.

**8 · Caching** → `llm.py`, `store.llm_cache`
Prompt caching with breakpoints placed stable-first (system prompt and
taxonomy cached; the varying chunk after the last breakpoint), plus a
content-hash result cache keyed on `(content, prompt_version, model)`.
*Enables:* unchanged re-runs cost nothing. Cost arithmetic is unit-tested
against published prices.

**9 · Evals** → `evals.py`, `evals/build_gold.py`, `evals/gold/`
A real gold set: **10 labelled promises and 3 labelled must-not-extract
passages across 2 transcripts**, generated from labelled *phrases* so the
offsets cannot silently go stale — `build_gold.py` fails loudly if a phrase is
missing or ambiguous. Metrics are span-overlap precision/recall, span fidelity,
type and hedge accuracy, with precision computed over judged extractions only
(see section 6). The champion gate blocks a span-fidelity regression or
extra forbidden extractions *regardless of F1*.
*Enables:* the question "did that change help?" now has an answer.

**10 · Outcomes** → `outcomes.py`
Kept deliberately separate from evals. Coverage, analyst acceptance rate,
corrections per 100 promises, and the self-grading part: predictions resolve
against verdicts as deadlines pass, scored by Brier and bucketed into a
calibration table.

**11 · Executor / advisory** → `verify.py`, `config.py`
The advisor tool (`advisor_20260301`, beta `advisor-tool-2026-03-01`), invoked
only when the adjudicator marks a verdict `contested`.
*Finding that changed the default:* a `claude-fable-5-1` advisor returns
`advisor_redacted_result` — encrypted advice you can replay but not read. Since
a verdict's rationale is an audit artifact, the default advisor is
`claude-opus-4-8`, which returns plaintext. `read_advisor_blocks` switches on
the content type, not the block type, which is the trap that silently returns
nothing.

**14 · CLAUDE.md** → [CLAUDE.md](CLAUDE.md)
Seven invariants, the eval rules, and the traps: the August fiscal year
end, the ASR glossary, YouTube blocking, the timestamp-less legacy files, and
the 3.9 constraint.

**15 · Orchestration** → `pipeline.py`
Ten stages, each idempotent -- verified by running the real corpus twice --
and resumable because the store records what exists rather than the runner
tracking position. `stage_review` reports the queue and
returns rather than blocking — an analyst taking three days is a normal state,
not a stalled process.
*The design position made concrete:* the verification agent runs bounded
inside `stage_verify`. It does not drive the workflow.

**Service surface** → `api.py`
13 routes. `Provenance` is a required field on every promise response — source,
span, timestamp, video deep link, prompt version and model — so a claim cannot
be served without its evidence chain.

### Implemented, correctness not yet provable

**Domain: threading** → `threading_.py`
Deterministic scoring first (Jaccard on content words, metric and deadline
signals), with model escalation only for the ambiguous band. This works on
rewordings ("500,000 farmers" vs "half a million farmers" scores 0.56) but is
weak where the same commitment is stated in entirely different words (0.25) —
which is why the escalation band is wide. **Threading has no eval yet and is
the weakest link in the pipeline.** Thread purity should be the next gold set.

**Predictor** → `outcomes.predict_thread`
A transparent heuristic over hedge level, restatement count and silence — not a
fitted model, because there is no resolved outcome data yet. Deliberately
inspectable rather than opaque. `calibration()` will show whether the priors
hold once deadlines pass.

### Deliberately deferred

**4 · MCP.** Consistent with the advice in the architecture doc: defer until
there is a second consumer. `tools.py` is written so an MCP server is a thin
wrapper over `tool_definitions()` and `dispatch()`. Also blocked practically —
the official `mcp` package requires Python 3.10+ and the venv is 3.9.

**12 · Managed Agents.** The architecture doc argued these largely overlap with
a real orchestration layer, and that the multi-day human checkpoint makes a
workflow engine the stronger spine. Having built `pipeline.py`, that still
holds. The place CMA would earn its keep is the bounded research task inside
`stage_verify` — a sandbox for running analysis over filings — not as the
top-level controller.

**13 · Dreaming.** Worthless before evals produce a signal. The foundations are
in place: prompt versioning and insert-only records mean a backfill can be
diffed against prior extractions. The four jobs (backfill, eval-set growth,
hedge-pattern mining, deadline sweeps) are all Batch API work at 50% cost.

---

## 3. Defects found and fixed during the build

**Revised targets were being silently merged.** A test written to assert the
opposite failed: with identical claim wording, the −0.35 deadline-mismatch
penalty was swamped by the similarity term (0.65 > 0.45 threshold), so a target
moved from 2025 to 2030 would be absorbed into the original thread. Barry
Callebaut actually did this with its 100%-sustainable-ingredients target, and
merging them would erase one of the most newsworthy facts in the corpus. A
conflicting deadline is now decisive, not weighted.

**Threading under-merged genuine restatements.** The two midterm-guidance
statements in the FY21/22 call — the same commitment restated minutes apart in
Q&A — score only 0.25 on token overlap. The docstring claimed the model
resolved ambiguous cases; it did not. The escalation is now built, with the
band widened to 0.15–0.45.

**The replay client resolved sources too early.** It looked up sources at
construction, but the pipeline builds the client before the ingest stage runs,
so it silently found nothing. Now lazy.

---

## 4. Verification status

Be clear about this: **nothing below has been run against a live model.** An
API key is now configured (in the git-ignored `.env`), but no call has been
made yet.

### Verified

- 80 tests pass, no live API calls.
- Chunk offsets exact and coverage complete over a real 65,761-character
  transcript.
- All 10 gold promises extract through the real path — chunking, locating the
  opening and closing quotes, span validation, reconciliation — with 0
  uncitable and every span verifying against the source.
- The pipeline is idempotent: a second run over the real corpus skips all 8
  sources, links nothing and leaves every table's row count unchanged.
- Gold set offsets resolve against the shipped corpus (guarded by a test, so it
  cannot go stale silently).
- Full pipeline runs end to end: 8 sources → 10 promises → 10 threads → 10
  verdicts → review queue, with 4 threads correctly flagged silent.
- All 13 API routes serve, with the provenance chain intact.
- Cost arithmetic matches published prices ($2/$10 per MTok Sonnet 5; cache
  reads 0.1×; cache writes 1.25× at 5-minute TTL).

### Not verified — needs a key

- **Extraction quality.** Whether `extract-v3` actually beats `extract-v1` is
  exactly what the eval exists to answer, and it has not been run. Do this
  first; the three runs in section 5 are estimated at about $0.50 in total.
- **Prompt cache hit rates.** Breakpoint placement follows the documented
  render order, but `cache_read_input_tokens` has never been observed non-zero.
- **The advisor tool call.** Built from the API reference and the installed
  SDK's types, never executed. In particular, whether the API accepts the
  advisor tool and structured output in the same call is undocumented. If it
  does not, escalation fails safe: the executor's verdict stands and the
  rationale records the rejection.
- **The agent tool loop**, adjudication, and the retry/backoff paths.

---

## 5. What to do next, in order

1. **Clear the offline replay data first.** `data/ledger.db` currently holds
   10 promises, 10 threads and 10 verdicts produced by the gold-replay stand-in
   (`model: gold-replay`), not by a model. `prp report` and the API would serve
   them as findings. Delete `data/` before the first live run; it is
   git-ignored local state.
2. **Run the three evals** (the commands are in the README): extract-v1 on the
   first 1,000 words (the original pipeline), extract-v1 on the full text, and
   extract-v3. About $0.50 in total. Comparing them separates the effect of
   removing truncation from the effect of the new prompt.
3. **Label the unjudged extractions** they report, rebuild the gold set, and
   re-run all three. Only then will the gate promote a winner with
   `--promote`.
4. **Re-fetch the transcripts** (`run --stages fetch`) so promises get video
   deep links. The current corpus was ingested from timestamp-less legacy files.
5. **Grow the gold set to a third transcript** and add thread-purity labels —
   threading is the weakest link and currently unmeasured.
6. **Then** run the full pipeline with verification and work the review queue,
   which starts generating the outcome metrics that matter commercially.

One environment change was made: pip and setuptools in `.venv` were upgraded
(21.2.4 → 26.0.1, 58.0.4 → 82.0.1) because the 2021 toolchain could not perform
a PEP 660 editable install. The venv is git-ignored and reproducible.
Python itself is untouched at 3.9 — note it is end-of-life, and moving to 3.12
would remove the MCP blocker.

---

## 6. Second pass: defects found in review (v0.3)

A review of the first build, done before any live run, found nine defects.
Four of them would have made the headline eval number meaningless. All are
fixed, and each has a regression test.

**1. The eval was contaminated.** Three of the 13 gold labels were copied into
the v2 prompt as worked examples: the catering joke, the "half a million
farmers" line and the forest-positive line. Any v2-versus-v1 comparison would
partly have measured whether the model remembered what it had just been shown.
*Fix:* `extract-v2` was retired without ever running live. `extract-v3` has
the same structure, with every example taken from transcripts outside the gold
set. `evals.find_leakage` refuses to run a contaminated eval, and a test checks
every registered prompt against the real corpus. A related gap is also closed:
prompts are partly generated from the taxonomy, so editing the taxonomy used to
silently change what a version name meant. Each version is now pinned to a
fingerprint of its text, and a test fails if the text drifts.

**2. "Append-only" was claimed but not true.** `store.py` used
`INSERT OR REPLACE` throughout, and one `UPDATE`. *Fix:* the claim is now true
for everything an auditor would ask to see: sources, promises, memberships,
evidence, verdicts, reviews, predictions and runs are insert-only. Analyst
reviews are appended rather than written over the verdict, so the model's
original call and every later change of mind are all kept. A re-fetched
transcript supersedes the old one instead of replacing it. Derived state
(thread summaries, cache, memory) is labelled as mutable. A test reads the
schema to enforce the split.

**3. The pipeline wasn't idempotent.** Running the link stage twice put every
promise into its thread a second time and reported 10 restatements that didn't
exist. That would also have inflated the predictor, because restatements raise
`p_kept`. Extraction had the same problem, writing duplicate rows on every
re-run. *Fix:* extraction is recorded per (source, prompt, model) and skipped
once done; `thread_members` is keyed on the promise, so a promise cannot join
two threads; threading is scoped to one prompt version; predictions are only
appended when their inputs change. `test_whole_pipeline_is_idempotent` runs the
real corpus twice and checks that the row counts don't change.

**4. The model was trusted to count characters.** Language models can't
reliably produce character offsets into a 10,000-character chunk, and a wrong
offset that lands on some other real text passes a range check. Worse, spans
that failed were dropped *before* scoring, so v3's span fidelity was
automatically perfect while v1's retyped snippets were penalised. *Fix:* the
anchor-and-locate design described in section 2. Unlocatable extractions are
now counted as uncitable and scored against span fidelity, and v1 snippets go
through the same locator, so the two are compared on equal terms.

**5. The gold set was partial but scored as if complete.** Only some promises
in each gold transcript are labelled. Counting every unlabelled extraction as
a false positive would have punished the model for finding real promises.
*Fix:* unlabelled extractions are reported as `unjudged`, precision is computed
over judged extractions only, and the gate won't promote a prompt while any
remain. The workflow is standard pooling: run every system, label the pooled
unjudged extractions, rebuild, re-score. Scores also carry a gold-set
fingerprint, and the gate refuses to compare numbers computed against
different label sets.

**6. The link stage spent money off the books.** Thread tie-breaks call the
model, but `build_threads` had no ledger, so their cost never appeared in
`prp ledger`, and every `thread_members` row had a blank `run_id`. *Fix:*
`link` now opens a `RunLedger` like `extract` and `verify`, adds up tie-break
tokens and cost, and stamps its run ID on every membership. A tie-break that
errors is noted on the run and the promise gets its own thread. Four tests
cover it.

**7. The advisor could not change a verdict.** Escalation consulted the
advisor and appended its advice to the rationale as text, but the status
always came from the executor's first call. The second opinion was paid for
and then ignored. *Fix:* a contested verdict is re-adjudicated by the executor
*with the advisor tool attached*, and that validated, advisor-informed answer
becomes the verdict. The first call is kept as `initial_status` for the audit
trail. Anything short of a validated, advisor-informed answer keeps the
original: an error, the executor not actually consulting the advisor, or no
structured output.

**8. The advisor was priced as if it were the executor.** Its tokens were
billed at Sonnet's rate; Opus 4.8 costs 2.5x as much. *Fix:* one shared
function, `llm.usage_from_response`, prices every sub-call from the response's
`usage.iterations` at that sub-call's own model rate. A model missing from the
price table is flagged in the run's notes instead of silently costing $0.

**9. The most expensive loop had no prompt caching.** The evidence-gathering
agent resent its whole growing conversation at full price on each of up to 12
turns. *Fix:* automatic caching, whose breakpoint moves forward with the
conversation, so each turn pays full price only for what is new.

Ten tests cover 7-9.

### Still open

- **Nothing has run against a live model.** The fixes above were made so that
  the first live number means something.
- **The verifier can only search the company's own videos.** Judging "kept"
  or "missed" needs outcome sources: sustainability progress reports, annual
  reports, third parties. Until one is added, most verdicts will be
  `no_evidence`.
- **Ten labels on two transcripts is a small gold set**, and threading still
  has no gold labels of its own.
- The predictor is a transparent heuristic, not a fitted model.
