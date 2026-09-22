# Labelling guidelines

These rules decide what goes into `evals/build_gold.py`. Every score the
benchmark reports can only be as good as the agreement between two people
following them. Where a rule leaves you unsure, write the case down here rather
than deciding it privately.

Definitions of promise types and hedge levels live in
`src/pr_predictor/taxonomy.py`. This file covers how to apply them to a
transcript.

## Before you read a transcript

1. **Check its split in `evals/splits.json`.** Assign one before reading if it
   has none. Never move a transcript between splits after seeing how a model
   scores on it.
2. **Never label an `examples` transcript.** The prompt's worked examples come
   from those transcripts, and `build_gold.py` refuses them.
3. **Decide up front whether you will label exhaustively.** An exhaustive
   document (`"exhaustive": true`) turns every unmatched extraction into a
   false positive. Only exhaustive documents give a real precision. Set it
   only after reading the whole transcript, Q&A included.

## Where labels go, and how to read the transcript

- **Use the label desk:** `python evals/label_server.py` opens a local page in
  your browser. Highlight a passage, choose what it is, and it is saved to
  `evals/labels/<video id>.json` and the answer key is rebuilt. It runs only on
  your machine and never shows model output.
- Every transcript's labels live in `evals/labels/<video id>.json`, dev and
  test alike. You can edit these by hand too; `python evals/build_gold.py`
  checks them.
- Without the tool, `python evals/show_transcript.py <video id>` writes a
  wrapped, readable copy to `data/labelling/<video id>.txt`. Phrases copied from
  it may include its line breaks; they still resolve.
- **Test documents are labelled blind.** Do not run `python -m pr_predictor
  promises`, do not open the API, and do not read an eval's unjudged list for
  a test transcript until its labels are complete. After that, candidate misses
  can be proposed from model output and accepted or rejected one by one. Tag
  each one you accept with `"slices": ["pooled"]`, so blind recall can still be
  told apart from pooled recall.
- When a document is done, set `labelling_complete` and (if true) `exhaustive`,
  fill in `labelled_by`, and run `python evals/build_gold.py`. Only then is it
  scored.

## What counts as a promise

A forward-looking statement the **company** can be shown to have kept or
missed. Ask: *could a later document prove this wrong?* If nothing could,
it is not a promise.

The following are not promises. Each is listed as `must_not_extract` when it
is tempting:
- event logistics ("we will be your moderators")
- jokes and catering
- macro expectations ("markets should recover")
- statements of character ("we are a company that will support its customers")
- "we will continue to monitor"
- personnel decisions already taken

## Span boundaries

- **Start** at the clause that carries the commitment. **End** where the
  commitment ends.
- Include the number, the metric and the deadline if they are in the same
  sentence. Do not swallow the neighbouring sentence.
- If one sentence holds two commitments that share a clause, label it once and
  say so in the note.
- Copy the phrase exactly from the transcript, including speech-recognition
  errors. `build_gold.py` fails if the phrase is missing or not unique; make
  it longer rather than editing it.

## Hedge

Grade the clause that carries the checkable content, not the loudest verb.
"We will double down our efforts and **aim to** become forest positive by
2025" is `intended`.

## Deadlines

- **`deadline_raw`**: what was said, verbatim ("later in the fiscal year").
- **`deadline_resolved`** is three-valued:
  - **omitted**: not labelled; not scored.
  - **`None`**: the correct output is *no date*. Use this for anything relative
    to a **fiscal year**. Barry Callebaut's fiscal year ends 31 August, and the
    extraction contract says leave `deadline` null rather than guess.
  - **`"YYYY-MM-DD"`**: only when the speaker gave an unambiguous calendar
    date.
- **"By 2025" is not unambiguous** for a company with a non-calendar fiscal
  year. Leave it omitted unless the context settles it.
- Tag fiscal cases with the `fiscal_deadline` slice.

## Threads

- **`thread_key`**: promises that are the same commitment restated, in this
  transcript or another, share a key. Use a readable key
  (`bc-new-midterm-guidance`). Without one, every promise is its own thread.
- **`revision_of`**: set it when a promise *moves* an earlier target (2025 →
  2030). Give it its own `thread_key`, and set `revision_of` to the key of the
  target it revises. Merging the two is a hard failure of the threading suite.

## Slices

Tag anything that is known to be hard, so a regression there is visible even
when the average does not move:

| slice | when |
|---|---|
| `fiscal_deadline` | deadline relative to a fiscal year or quarter |
| `asr_garble` | speech recognition garbled a word that matters to the claim |
| `qa` | made in Q&A rather than the prepared presentation |
| `restatement` | re-affirms an earlier commitment |
| `moderator` | stated by a moderator or third party on the company's behalf |
| `revised_deadline` | moves an earlier target |
| `pooled` | added after the blind pass, from a proposed miss |

`chunk_boundary` and `unseen_whole` are added automatically at score time from
the chunking actually used. Do not label them.

## Disagreement

When two annotators disagree about whether a passage is a promise, keep it
with `"ambiguous": True` and a note on both readings. Do not delete it.
Ambiguous promises are reported (`ambiguous_matched`) but never count as
right or wrong.

## Adjudication cases

Adjudication cases go in `evals/adjudication/`, one JSON file per case. The
schema is in `src/pr_predictor/evals/adjudication.py`.
- Pin `as_of` to the date the verdict should be judged at.
- Supply the evidence yourself; the suite does not search for it.
- Use `also_acceptable` only when the rubric in `prompts.py` genuinely admits
  two statuses.

## Judge calibration

Judge calibration pairs go in `evals/judge_calibration/*.json`. Each file
holds a list of items of the form
`{"verbatim": ..., "claim": ..., "faithful": true|false, "note": ...}`.
- Label from the rules in `JUDGE_SYSTEM` (`src/pr_predictor/evals/judge.py`).
- Include unfaithful examples on purpose: added dates, fiscal years mapped to
  calendar dates, "aim to" turned into "will".
