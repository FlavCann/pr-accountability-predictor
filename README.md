# PR Accountability Predictor

Track what companies promise in investor communications, and hold them to it.

The pipeline extracts forward-looking commitments from investor transcripts,
threads restatements together across years, gathers evidence about whether each
was met, issues a verdict, and predicts which open commitments will be kept.
Every claim traces back to a timestamped span in a named source.

Example corpus: eight Barry Callebaut investor presentations, 2021–2023.

---

## Flow

```mermaid
flowchart TD
    subgraph ingest["INGEST — ingest.py"]
        A[pages/*.html] --> B[discover]
        B --> C[to_scrape.json]
        C --> D[fetch]
        D --> E[("Source + Segment<br/>text + char offsets + timestamps")]
    end

    subgraph extraction["EXTRACT — context.py, extract.py"]
        E --> F[chunk with overlap]
        F --> G[assemble context<br/>brief · glossary · prior threads]
        G --> H[extract: model quotes<br/>opening + closing words]
        H --> I[locate quotes, compute offsets<br/>reconcile duplicates<br/>attach timestamps]
        I --> J[("Promise")]
    end

    subgraph linking["LINK — threading_.py"]
        J --> K{deterministic<br/>match score}
        K -->|clear| M[("PromiseThread")]
        K -->|ambiguous| L[model tie-break]
        L --> M
    end

    subgraph verification["VERIFY — tools.py, verify.py"]
        M --> N[bounded agent loop<br/>step cap · cost cap]
        N --> O[("Evidence")]
        O --> P[adjudicate]
        P -->|contested| Q[advisor tool]
        Q --> R
        P --> R[("Verdict")]
    end

    subgraph out["OUTCOMES — outcomes.py"]
        M --> S[("Prediction")]
        R --> T[analyst review]
        T -->|correction| U[("Memory")]
        U -.feeds context.-> G
        R --> V[resolve predictions<br/>Brier · calibration]
        S --> V
    end

    V --> W[report / API]
    T --> W

    style E fill:#e8eef2,stroke:#1F4E5F,color:#111
    style J fill:#e8eef2,stroke:#1F4E5F,color:#111
    style M fill:#e8eef2,stroke:#1F4E5F,color:#111
    style R fill:#e8eef2,stroke:#1F4E5F,color:#111
    style U fill:#f1ecf6,stroke:#6B4E8A,color:#111
```

The dotted line is the loop that makes the product improve from use: an analyst
correcting a verdict writes durable memory, which is retrieved into the context
of every later extraction.

`pipeline.py` orchestrates these stages. **Agents are steps inside the
workflow, not the workflow** — the verification agent runs bounded and
disposable inside one stage; it does not drive the process.

---

## Layout

| Layer | Module | Job |
| --- | --- | --- |
| Definitions | `taxonomy.py` | What counts as a promise. Single source of truth, shared by the prompt, the evals and the adjudicator. |
| | `domain.py` | The object graph: Source → Segment → Promise → Thread → Evidence → Verdict → Prediction. |
| | `config.py` | Settings, model routing, cost table, loop bounds. |
| Infrastructure | `store.py` | SQLite. Evidence and decisions are insert-only; derived state is labelled mutable. |
| | `llm.py` | The harness: retries, prompt caching, schema repair, cost accounting. |
| | `prompts.py` | Versioned prompt registry. Every row records the prompt that made it. |
| Pipeline | `ingest.py` | Discover → fetch → normalise. Keeps timestamps. |
| | `context.py` | Chunking and context assembly. |
| | `extract.py` | Extraction. The model quotes each promise's first and last words; we locate them and compute the offsets. |
| | `threading_.py` | Restatements grouped into commitment threads. |
| | `tools.py` | The verification agent's tool surface. |
| | `verify.py` | Bounded agent loop, adjudication, advisor escalation. |
| Product | `evals/` | The benchmark: gold set, splits, scoring, intervals, the gate, and suites for threading, adjudication, predictions and the judge. |
| | `outcomes.py` | Product telemetry, the predictor, and a company's accountability profile. |
| | `rating.py` | The headline PR-accountability level. Currently a placeholder: every company is `pr_med_low`. |
| | `pipeline.py` | Stage orchestration. |
| Interfaces | `cli.py` | Command line. |
| | `api/` | HTTP service: `app.py` assembles it, `routes/` has one router per resource. |
| | `static/` | The dashboard: `index.html`, `css/`, `js/` (ES modules: `views/`, `components/`, `lib/`), `img/`. |

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[api,dev]"
```

Python 3.9 or later. Calling a model needs an API key. Put it in a `.env` file
at the repo root, which is git-ignored and loaded automatically:

```bash
echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env
```

Everything else runs without one.

---

## Usage

**Run the pipeline.** Ingests the transcripts already in `transcripts/`, then
extracts, threads, verifies, predicts and reports.

```bash
python -m pr_predictor run              # reads the key from .env
```

**Without an API key**, two offline modes:

```bash
python -m pr_predictor --dry-run run    # walk the wiring, call nothing
python -m pr_predictor --replay run     # replay gold labels through the real
                                        # extraction path: no key, no spend
```

**Inspect what it found.**

```bash
python -m pr_predictor promises         # verbatim text + video deep links
python -m pr_predictor threads          # commitments, restatements, silence
python -m pr_predictor ledger           # tokens and dollars per stage
python -m pr_predictor report           # product outcome metrics
```

**Measure a prompt change.** Never change the extraction prompt without this.

```bash
# The original pipeline, reproduced: v1 prompt on the first 1,000 words
python -m pr_predictor eval --prompt-version extract-v1 --baseline-truncate 1000
# The original prompt on the full text (isolates the prompt from the truncation)
python -m pr_predictor eval --prompt-version extract-v1
# The current prompt
python -m pr_predictor eval --prompt-version extract-v3
```

Each eval lists **unjudged** extractions: passages the model called promises
that the partial gold set doesn't label either way. Label them with the label
desk (`python evals/label_server.py`), and re-run all three. The gate will not promote
a prompt (`--promote`) while any remain.

Each run samples every config `--repeats` times (default 3), reports F1 with a
bootstrap interval once there are enough documents, and is recorded in the
store (`eval-history`). Change one thing at a time — the gate refuses a
comparison where the model, the prompt and the harness differ in more than one:

```bash
python -m pr_predictor eval --models claude-sonnet-5,claude-opus-5          # model matrix
python -m pr_predictor eval --harness chunk_words=900,chunk_overlap_words=150  # harness change
python -m pr_predictor eval --judge                                         # + claim faithfulness
python -m pr_predictor eval-threads                                         # threading, deterministic
python -m pr_predictor eval-adjudication                                    # verdicts on labelled cases
python -m pr_predictor eval-predictions                                     # Brier, graded by reality
python -m pr_predictor eval-history
```

**Work the review queue** — the human checkpoint. Corrections become durable
memory.

```bash
python -m pr_predictor review
```

**Serve the API.**

```bash
uvicorn pr_predictor.api:app --reload
# http://127.0.0.1:8000/docs
```

The dashboard is at `http://127.0.0.1:8000/dashboard`: a company's
PR-accountability rating (revealed by a spin; a placeholder until `rating.py`
reads the record), its accountability profile, its commitment threads, and for each thread the
promises (verbatim, with video deep links), the verdict, and the evidence the
rationale cites.

Its **Model evaluation** tab (`/dashboard#view=evals`) shows the benchmark: the
champion's scores against the previous run, F1 / precision / recall across every
recorded run, and the champion's scores per transcript and per slice. It reads
`GET /evals` and `GET /evals/{run_id}`, which serve scores only, never
transcript text.

**Re-fetch transcripts with timestamps.** The files in `transcripts/` were
written by the old pipeline without timestamps; that information can only be
recovered from YouTube.

```bash
python -m pr_predictor run --stages fetch
```

---

## Evals

New to evals? Open `evals/GUIDE.html` in a browser: a plain-language walkthrough
of how the benchmark works, with diagrams, a worked example and a command cheat
sheet.

To label transcripts, run `python evals/label_server.py`: a local page where you
highlight a passage and classify it. Rules are in `evals/GUIDELINES.md`.

The gold set lives in `evals/gold/*.json` as character offsets, generated from
labelled phrases by `evals/build_gold.py`. Edit the phrases, not the offsets:

```bash
python evals/build_gold.py
```

It fails loudly if a labelled phrase is missing or ambiguous in the transcript,
so the gold set cannot silently go stale against the corpus.

Every transcript has a split in `evals/splits.json`, assigned before it is
labelled: `examples` (prompt examples come from here; never scored), `dev`
(iterate), `test` (held out; promotion is decided here when it has documents)
and `fresh` (published after the model's training cutoff). The eval refuses to
run if a prompt example appears in any scored transcript, labelled or not.

How to label — spans, hedges, fiscal-year deadlines, thread keys, slices,
disagreement — is in `evals/GUIDELINES.md`. See `CLAUDE.md` for the rest of the
eval rules.

`evals/v1_baseline_output.txt` is what the original pipeline produced, kept for
comparison.

---

## Tests

```bash
python -m pytest
```

108 tests covering offset arithmetic, quote location, duplicate reconciliation,
timestamp resolution, threading, cost accounting, the eval gate and its
refusals, benchmark statistics, eval contamination, insert-only records and whole-pipeline idempotency. No live API
calls.

---

## Notes

- Only English transcripts are requested.
- If fetching stops with `IpBlocked`, YouTube has flagged your network. Wait,
  switch network, or set `TRANSCRIPT_PROXY_URL=http://user:pass@host:port`
  (residential proxies work; datacentre IPs are usually blocked too).
- Transcripts, saved pages and the SQLite store are git-ignored. Respect the
  terms of the sites and videos you collect from.
- Barry Callebaut's fiscal year ends 31 August, so "FY23" is not calendar 2023.
  See `CLAUDE.md`.
