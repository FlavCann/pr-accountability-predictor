"""The extraction benchmark: run, record, compare, gate.

A result is only worth comparing if you know exactly what produced it, so every
run carries a manifest -- model, prompt text, harness settings and code
fingerprints -- and is appended to the store like every other record. The gate
then refuses comparisons that change more than one axis at a time, and refuses
promotions that the data cannot distinguish from noise.
"""

from __future__ import annotations

import json
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from .. import config, prompts
from ..domain import _id, content_hash
from .gold import SCORED_SPLITS, find_leakage, gold_fingerprint, load_gold, validate_gold
from .scoring import Metrics, boundary_tags, score_document_detailed, slice_report
from .stats import bootstrap_ci, micro_f1, paired_delta_ci

CHAMPION_FILE = "champion.json"

# ----------------------------------------------------------------------
# Manifest: what produced this score
# ----------------------------------------------------------------------

HARNESS_SETTINGS = {
    "chunk_words": "CHUNK_WORDS",
    "chunk_overlap_words": "CHUNK_OVERLAP_WORDS",
}
"""Harness knobs a run may override, mapped to their `config` attribute."""

AXES = {
    "model": ("executor_model",),
    "prompt": ("prompt_version", "prompt_fingerprint"),
    "harness": ("harness", "pipeline_code"),
}
"""The three things an iteration changes. A comparison is only attributable
when exactly one of them differs."""

_PKG = Path(__file__).resolve().parents[1]


def harness_settings() -> Dict[str, int]:
    return {k: getattr(config, attr) for k, attr in HARNESS_SETTINGS.items()}


@contextmanager
def harness_overrides(overrides: Optional[Dict[str, int]]) -> Iterator[None]:
    """Temporarily change harness settings for one run, then restore them."""
    overrides = overrides or {}
    unknown = set(overrides) - set(HARNESS_SETTINGS)
    if unknown:
        raise SystemExit(
            "Unknown harness setting(s): {}. Known: {}".format(
                ", ".join(sorted(unknown)), ", ".join(sorted(HARNESS_SETTINGS))
            )
        )
    saved = harness_settings()
    try:
        for k, v in overrides.items():
            setattr(config, HARNESS_SETTINGS[k], int(v))
        yield
    finally:
        for k, v in saved.items():
            setattr(config, HARNESS_SETTINGS[k], v)


def _files_fingerprint(paths: Sequence[Path]) -> str:
    return content_hash("|".join(p.name + ":" + p.read_text() for p in sorted(paths)))


PIPELINE_MODULES = ("extract.py", "context.py", "llm.py", "domain.py")
"""The code between the prompt and the scored spans. Deliberately excludes
`prompts.py` and `taxonomy.py` (the prompt axis, via its fingerprint) and
`config.py` (the model and harness settings, recorded by value)."""


def pipeline_code_fingerprint() -> str:
    return _files_fingerprint([_PKG / name for name in PIPELINE_MODULES])


def scoring_code_fingerprint() -> str:
    """The code that turns predictions into numbers. If it changes, old scores
    are not comparable with new ones even on the same predictions."""
    here = Path(__file__).resolve().parent
    return _files_fingerprint([here / "gold.py", here / "scoring.py"])


def _git(*args: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", *args], cwd=str(config.ROOT), capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def build_manifest(client, prompt_version: str, repeats: int) -> Dict[str, object]:
    prompt = prompts.get(prompt_version)
    status = _git("status", "--porcelain")
    return {
        "executor_model": client.model,
        "prompt_version": prompt_version,
        "prompt_fingerprint": prompt.fingerprint,
        "harness": harness_settings(),
        "pipeline_code": pipeline_code_fingerprint(),
        "scoring_code": scoring_code_fingerprint(),
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "repeats": repeats,
    }


def changed_axes(a: Dict[str, object], b: Dict[str, object]) -> List[str]:
    return [axis for axis, keys in AXES.items() if any(a.get(k) != b.get(k) for k in keys)]


# ----------------------------------------------------------------------
# Running
# ----------------------------------------------------------------------

class _FreshSamples:
    """Forwards to a client but bypasses the result cache.

    Repeat 0 may come from the cache -- it is a genuine earlier sample of the
    same prompt and model. Later repeats must be new samples, or the spread
    across repeats would be a cache artefact reading as zero variance.
    """

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def structured(self, *args, **kwargs):
        kwargs["use_cache"] = False
        return self._inner.structured(*args, **kwargs)


def _mean_counts(runs: List[Dict[str, float]]) -> Dict[str, float]:
    keys = runs[0].keys()
    return {k: sum(r[k] for r in runs) / len(runs) for k in keys}


def run_eval(
    store,
    client,
    prompt_version: str = prompts.DEFAULT_EXTRACTION_PROMPT,
    gold_dir: Optional[Path] = None,
    baseline_truncate_words: Optional[int] = None,
    repeats: int = 1,
    splits: Optional[Sequence[str]] = None,
    harness: Optional[Dict[str, int]] = None,
    judge_client=None,
    record: bool = True,
) -> Dict:
    """Extract over every scored gold document, `repeats` times, and score it.

    `baseline_truncate_words` exists only to reproduce the ORIGINAL pipeline
    (extract-v1 on the first 1,000 words) as a measured baseline. It violates
    the no-truncation invariant on purpose and is accepted only for extract-v1.

    `judge_client`, when given, grades each true positive's `normalized_claim`
    for faithfulness to its span. Reported, never gated on -- see judge.py.
    """
    from ..context import chunk_text
    from ..extract import extract_source, extract_source_v1

    if repeats < 1:
        raise SystemExit("repeats must be at least 1")
    all_docs = load_gold(gold_dir)
    wanted = tuple(splits or SCORED_SPLITS)
    in_split = [d for d in all_docs if d.split in wanted and d.split in SCORED_SPLITS]
    docs = [d for d in in_split if d.labelling_complete]
    not_ready = [d.name for d in in_split if not d.labelling_complete]
    if not docs:
        raise SystemExit(
            "No scored gold documents in {} for splits {}. Run `python evals/build_gold.py`."
            .format(gold_dir or config.GOLD_DIR, ", ".join(wanted))
        )

    problems = validate_gold(store, all_docs)
    if problems:
        raise SystemExit("Gold set is invalid:\n  " + "\n  ".join(problems))

    prompt = prompts.get(prompt_version)
    leaks = find_leakage(store, all_docs, prompt.system)
    if leaks:
        raise SystemExit(
            "Refusing to run a contaminated eval -- the model would be tested on "
            "examples it was shown:\n  " + "\n  ".join(leaks)
        )

    is_v1 = prompt_version == "extract-v1"
    if baseline_truncate_words and not is_v1:
        raise SystemExit("Truncation is only allowed to reproduce the extract-v1 baseline.")

    cost = 0.0
    started = time.monotonic()
    doc_runs: Dict[str, List[Dict[str, float]]] = {d.source_url: [] for d in docs}
    repeat_counts: List[List[Dict[str, float]]] = []
    slice_rows = []
    judge_pairs = []
    to_label: Dict[Tuple[str, int, int], Dict[str, object]] = {}

    with harness_overrides(harness):
        manifest = build_manifest(client, prompt_version, repeats)
        for r in range(repeats):
            sampler = client if r == 0 else _FreshSamples(client)
            this_repeat = []
            for doc in docs:
                src = store.source_by_url(doc.source_url)
                uncitable = 0
                if is_v1:
                    predicted, usage, uncitable = extract_source_v1(
                        sampler, src, truncate_words=baseline_truncate_words
                    )
                else:
                    predicted, usage, uncitable = extract_source(
                        store, sampler, src, prompt_version=prompt_version, run_id="eval"
                    )
                cost += usage.cost_usd
                scored = score_document_detailed(
                    predicted, doc, src.text, uncitable=uncitable, score_fields=not is_v1
                )
                counts = scored.metrics.counts()
                doc_runs[doc.source_url].append(counts)
                this_repeat.append(counts)

                chunks = chunk_text(src.text)
                for g, p in scored.pairs:
                    tags = ["all", "split:" + doc.split] + g.slices + boundary_tags(g, chunks)
                    slice_rows.append((g, p, tags))
                    if r == 0 and p is not None:
                        judge_pairs.append((src.text[g.start_char : g.end_char], p))
                for p in scored.unjudged:
                    to_label.setdefault(
                        (doc.name, p.start_char, p.end_char),
                        {
                            "gold_file": doc.name,
                            "start_char": p.start_char,
                            "end_char": p.end_char,
                            "text": src.text[p.start_char : p.end_char],
                            "model_said": "{} / {}".format(
                                p.promise_type.value, p.hedge_level.value
                            ),
                            "seen_in_repeats": 0,
                        },
                    )["seen_in_repeats"] += 1
            repeat_counts.append(this_repeat)

    per_doc: Dict[str, Dict[str, object]] = {}
    total = Metrics()
    by_split: Dict[str, List[Dict[str, float]]] = {}
    for doc in docs:
        mean = _mean_counts(doc_runs[doc.source_url])
        m = Metrics.from_counts(mean)
        total += m
        by_split.setdefault(doc.split, []).append(mean)
        per_doc[doc.source_url] = dict(
            m.as_dict(), title=doc.source_title, split=doc.split, counts=mean
        )

    split_report = {
        name: {
            "documents": len(rows),
            "f1": round(micro_f1(rows), 3),
            "f1_ci95": bootstrap_ci(
                rows, micro_f1, n_resamples=config.EVAL_BOOTSTRAP_RESAMPLES
            ) if len(rows) >= config.EVAL_MIN_DOCS_FOR_CI else None,
        }
        for name, rows in sorted(by_split.items())
    }
    all_rows = [row for rows in by_split.values() for row in rows]
    repeat_f1 = [round(micro_f1(rows), 3) for rows in repeat_counts]

    result = {
        "run_id": _id("evl"),
        "prompt_version": prompt_version,
        "prompt_fingerprint": prompt.fingerprint,
        "gold_fingerprint": gold_fingerprint(all_docs),
        "baseline_truncate_words": baseline_truncate_words,
        "model": client.model,
        "manifest": manifest,
        "documents": len(docs),
        "not_scored_labelling_incomplete": not_ready,
        "not_human_labelled": [d.name for d in docs if d.provenance != "human"],
        "overall": total.as_dict(),
        "f1_ci95": bootstrap_ci(
            all_rows, micro_f1, n_resamples=config.EVAL_BOOTSTRAP_RESAMPLES
        ) if len(all_rows) >= config.EVAL_MIN_DOCS_FOR_CI else None,
        "underpowered": len(all_rows) < config.EVAL_MIN_DOCS_FOR_CI,
        "repeat_f1": repeat_f1,
        "splits": split_report,
        "slices": slice_report(slice_rows),
        "per_document": per_doc,
        "unjudged_extractions": list(to_label.values()),
        "cost_usd": round(cost, 4),
        "latency_s": round(time.monotonic() - started, 2),
    }

    if judge_client is not None and not is_v1:
        from .judge import judge_pairs as _judge

        result["judge"] = _judge(judge_client, judge_pairs, gold_dir=gold_dir)
        result["cost_usd"] = round(result["cost_usd"] + result["judge"]["cost_usd"], 4)

    if record and hasattr(store, "add_eval_run"):
        store.add_eval_run(result)
    return result


# ----------------------------------------------------------------------
# Champion + gate
# ----------------------------------------------------------------------

def load_champion(gold_dir: Optional[Path] = None) -> Optional[Dict]:
    path = Path(gold_dir or config.GOLD_DIR).parent / CHAMPION_FILE
    return json.loads(path.read_text()) if path.exists() else None


def save_champion(result: Dict, gold_dir: Optional[Path] = None) -> Path:
    """Write the incumbent. The full run also lives in the store under run_id."""
    path = Path(gold_dir or config.GOLD_DIR).parent / CHAMPION_FILE
    keep = {k: v for k, v in result.items() if k != "unjudged_extractions"}
    path.write_text(json.dumps(keep, indent=2, default=str) + "\n")
    return path


def _paired_rows(
    champion: Dict, result: Dict
) -> Tuple[str, List[Dict[str, float]], List[Dict[str, float]]]:
    """Per-document counts on both sides, over the documents they share.

    Compared on `test` when the candidate has any test documents, otherwise on
    everything scored.
    """
    old_docs = champion.get("per_document") or {}
    new_docs = result.get("per_document") or {}
    has_test = any(d.get("split") == "test" for d in new_docs.values())
    split = "test" if has_test else "all scored"
    old_rows, new_rows = [], []
    for url, d in sorted(new_docs.items()):
        if has_test and d.get("split") != "test":
            continue
        o = old_docs.get(url)
        if o and "counts" in o and "counts" in d:
            old_rows.append(o["counts"])
            new_rows.append(d["counts"])
    return split, old_rows, new_rows


def gate(
    result: Dict, champion: Optional[Dict], min_effect: Optional[float] = None
) -> Tuple[bool, str]:
    passed, why = _gate(result, champion, min_effect)
    drafted = result.get("not_human_labelled") or []
    if drafted:
        why += (
            " NOTE: {} of the scored documents have labels not made by a person ({}); "
            "this score shows the pipeline runs, not that it is good.".format(
                len(drafted), ", ".join(drafted)
            )
        )
    return passed, why


def _gate(
    result: Dict, champion: Optional[Dict], min_effect: Optional[float] = None
) -> Tuple[bool, str]:
    """Does this beat the incumbent?

    Hard blocks first, regardless of F1:
      * unjudged extractions -- precision is not yet known;
      * a different gold set or different scoring code -- numbers not comparable;
      * more than one axis changed -- the gain cannot be attributed;
      * a span-fidelity regression -- an unverifiable citation is not a
        tradeable quantity in a product whose whole claim is traceability;
      * more forbidden extractions.

    Then F1. With enough documents, the paired bootstrap interval of the gain
    must clear `min_effect`. Below that, the comparison is a point estimate and
    the verdict says so -- it is labelled UNDERPOWERED rather than dressed up.
    """
    if min_effect is None:
        min_effect = config.EVAL_MIN_EFFECT
    unjudged = result["overall"].get("unjudged", 0)
    if unjudged:
        return False, (
            "{} unjudged extraction(s). Precision is not trustworthy until they are "
            "labelled: add each as a promise or a not-a-promise with the label "
            "desk (python evals/label_server.py), and re-run.".format(unjudged)
        )
    if champion is None:
        return True, "No champion yet -- this becomes the baseline."
    if champion.get("gold_fingerprint") != result.get("gold_fingerprint"):
        return False, (
            "The champion was scored on a different gold set, so the numbers are "
            "not comparable. Re-run the champion's prompt ({}) on the current gold "
            "set first.".format(champion.get("prompt_version"))
        )

    old_m, new_m = champion.get("manifest"), result.get("manifest")
    if old_m and new_m:
        if old_m.get("scoring_code") != new_m.get("scoring_code"):
            return False, (
                "The champion was scored by different scoring code, so the numbers "
                "are not comparable. Re-run the champion's config first."
            )
        axes = changed_axes(old_m, new_m)
        if len(axes) > 1:
            return False, (
                "This run changes {} at once, so any difference cannot be attributed. "
                "Change one axis at a time.".format(" and ".join(axes))
            )

    new, old = result["overall"], champion["overall"]
    if new["span_fidelity"] < old["span_fidelity"]:
        return False, "Span fidelity regressed ({} -> {}). Blocked regardless of F1.".format(
            old["span_fidelity"], new["span_fidelity"]
        )
    if new["forbidden_extracted"] > old["forbidden_extracted"]:
        return False, "Extracted more explicitly-forbidden passages ({} -> {}).".format(
            old["forbidden_extracted"], new["forbidden_extracted"]
        )

    split, old_rows, new_rows = _paired_rows(champion, result)
    if len(new_rows) >= config.EVAL_MIN_DOCS_FOR_CI:
        delta, lo, hi = paired_delta_ci(
            old_rows, new_rows, micro_f1, n_resamples=config.EVAL_BOOTSTRAP_RESAMPLES
        )
        if lo is not None and lo > min_effect:
            return True, "F1 improved on {} by {} (95% CI {} to {}, {} documents).".format(
                split, delta, lo, hi, len(new_rows)
            )
        return False, (
            "Improvement not established on {}: F1 change {} (95% CI {} to {}, {} "
            "documents). The lower bound must exceed {}.".format(
                split, delta, lo, hi, len(new_rows), min_effect
            )
        )

    note = (
        " UNDERPOWERED: {} paired document(s), below the {} needed for an interval; "
        "this is a point estimate, not a measured gain.".format(
            len(new_rows), config.EVAL_MIN_DOCS_FOR_CI
        )
        if "per_document" in result
        else ""
    )
    if new["f1"] - old["f1"] > min_effect:
        return True, "F1 improved ({} -> {}).{}".format(old["f1"], new["f1"], note)
    return False, "F1 did not improve ({} -> {}).{}".format(old["f1"], new["f1"], note)
