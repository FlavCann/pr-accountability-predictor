"""Build the gold set from phrase labels.

Gold files store character offsets, but offsets are unreadable and unauditable
by hand. This script is the source of truth: it holds the labelled *phrases*,
resolves each to an exact offset in the stored source text, and fails loudly if
a phrase is missing or ambiguous.

Re-run it whenever the ingestion changes, so the gold set cannot silently go
stale against the corpus:

    python evals/build_gold.py

Labels live in evals/labels/<video id>.json. Ambiguous cases were
deliberately excluded -- a gold set that encodes a coin-flip judgement makes
every later measurement noisier, not better. Passages such as "we will continue
to drive growth on the sound Foundation of our four strategic pillars" are
genuinely arguable and are left out rather than guessed at. When a second
annotator disagrees about a passage, keep it with `"ambiguous": True` rather
than deleting it: it is then reported, but never scored as right or wrong.

Each promise label is (phrase, type, hedge, note[, extras]); `extras` may set
deadline_raw, deadline_resolved, thread_key, revision_of, claim, slices and
ambiguous. See evals/GUIDELINES.md for what each means and how to decide.

A second annotator labels the same transcripts independently into JSON and
builds a parallel gold directory, for `python -m pr_predictor eval-agreement`:

    python evals/build_gold.py --labels labels_b.json --out evals/gold_b --labeller "name, date"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pr_predictor.domain import content_hash  # noqa: E402
from pr_predictor.ingest import parse_legacy_file  # noqa: E402

LABELLER = "flavio+claude, 2026-09-21 (needs analyst review before use as a release gate)"

GOLD: dict = {}
"""Labels used to live here as Python tuples. They now live in
evals/labels/<video id>.json -- one file per transcript, dev and test alike --
so labelling never means editing code. Anything added here is still merged."""


def locate(text: str, phrase: str, path: Path) -> tuple:
    """Exact words, in order; any run of whitespace between them matches any
    other, so a phrase copied across a line break of the wrapped labelling view
    still resolves."""
    pattern = re.compile(r"\s+".join(re.escape(w) for w in phrase.split()))
    hits = [m.span() for m in pattern.finditer(text)]
    count = len(hits)
    if count == 0:
        raise SystemExit(
            "PHRASE NOT FOUND in {}:\n  {!r}\n"
            "The transcript may have changed. Fix the phrase, do not fudge offsets.".format(
                path.name, phrase[:90]
            )
        )
    if count > 1:
        raise SystemExit(
            "PHRASE AMBIGUOUS ({} matches) in {}:\n  {!r}\n"
            "Lengthen the phrase until it is unique.".format(count, path.name, phrase[:90])
        )
    return hits[0]


LABEL_EXTRAS = {
    "deadline_raw", "deadline_resolved", "thread_key", "revision_of", "claim",
    "slices", "ambiguous",
}
"""Optional fifth element of a promise label. `deadline_resolved` is written
only when given -- absent means "not labelled", None means "the right answer
is null" (see src/pr_predictor/evals/gold.py)."""


LABELS_DIR = ROOT / "evals" / "labels"


def load_label_files() -> dict:
    """Labels kept as JSON in evals/labels/<video id>.json, merged into GOLD.

    The same structure as a GOLD entry; tuples become lists. This is where
    labels made outside Python go -- the held-out test transcripts in
    particular. A file with `"labelling_complete": false` is built (so its
    masks are known to the leakage check) but not scored.
    """
    out = {}
    for path in sorted(LABELS_DIR.glob("*.json")):
        raw = json.loads(path.read_text())
        raw.pop("_instructions", None)
        out[path.stem + ".txt"] = raw
    return out


def load_splits() -> dict:
    raw = json.loads((ROOT / "evals" / "splits.json").read_text())
    return {vid: split for split, ids in raw["splits"].items() for vid in ids}


def build(labels_by_file: dict, out_dir: Path, labeller: str) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = load_splits()
    written = 0

    for filename, labels in labels_by_file.items():
        path = ROOT / "transcripts" / filename
        if not path.exists():
            print("skip {} (not on disk)".format(filename))
            continue
        stem = Path(filename).stem
        if stem not in splits:
            raise SystemExit(
                "{} has no split in evals/splits.json. Assign it one BEFORE labelling, "
                "so its split cannot be chosen after seeing how it scores.".format(stem)
            )
        if splits[stem] == "examples":
            raise SystemExit(
                "{} is in the `examples` split: prompt examples come from it, so it "
                "can never be scored. Label a dev/test/fresh transcript instead.".format(stem)
            )

        meta, body = parse_legacy_file(path)
        # Must match exactly what ingest.ingest_legacy_transcripts stores.
        text = body.strip()

        doc = {
            "source_url": meta.get("url", "file://{}".format(filename)),
            "source_title": meta.get("title"),
            "source_text_hash": content_hash(text),
            "split": splits[stem],
            "exhaustive": bool(labels.get("exhaustive", False)),
            "labelled_by": labels.get("labelled_by") or labeller,
            "labelling_complete": bool(labels.get("labelling_complete", True)),
            "provenance": labels.get("provenance", "human"),
            "promises": [],
            "must_not_extract": [],
            "masked": [],
        }

        for phrase, why in labels.get("masked", []):
            start, end = locate(text, phrase, path)
            doc["masked"].append({"start_char": start, "end_char": end, "note": why})

        def check_unmasked(start: int, end: int, phrase: str) -> None:
            for m in doc["masked"]:
                if start < m["end_char"] and m["start_char"] < end:
                    raise SystemExit(
                        "{}: label overlaps a masked prompt-example passage and would "
                        "leak it into scoring:\n  {!r}".format(stem, phrase[:90])
                    )

        for i, label in enumerate(labels["promises"]):
            phrase, ptype, hedge, note = label[:4]
            extras = dict(label[4]) if len(label) > 4 else {}
            unknown = set(extras) - LABEL_EXTRAS
            if unknown:
                raise SystemExit("{}: unknown label field(s) {}".format(stem, sorted(unknown)))
            start, end = locate(text, phrase, path)
            check_unmasked(start, end, phrase)
            item = {
                "start_char": start,
                "end_char": end,
                "promise_type": ptype,
                "hedge_level": hedge,
                "note": note,
                # Every promise is its own commitment unless a key says otherwise.
                "thread_key": extras.pop("thread_key", "{}#{}".format(stem, i)),
            }
            item.update(extras)
            doc["promises"].append(item)

        for phrase, why in labels["must_not_extract"]:
            start, end = locate(text, phrase, path)
            check_unmasked(start, end, phrase)
            doc["must_not_extract"].append(
                {"start_char": start, "end_char": end, "note": why}
            )

        out = out_dir / (stem + ".json")
        out.write_text(json.dumps(doc, indent=2) + "\n")
        written += 1
        print(
            "wrote {}  [{}]  ({} promises, {} negatives, {} masked){}".format(
                out.name, doc["split"], len(doc["promises"]), len(doc["must_not_extract"]),
                len(doc["masked"]), "" if doc["labelling_complete"] else "  LABELLING IN PROGRESS",
            )
        )
    return written


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--labels", type=Path, default=None,
        help="JSON file with the same structure as GOLD, for a second annotator. "
        "Tuples become lists; the optional fifth element is an object.",
    )
    ap.add_argument("--out", type=Path, default=ROOT / "evals" / "gold")
    ap.add_argument("--labeller", default=LABELLER)
    args = ap.parse_args(argv)

    labels = json.loads(args.labels.read_text()) if args.labels else dict(GOLD, **load_label_files())
    written = build(labels, args.out, args.labeller)
    print("\n{} gold document(s). Total labelled promises: {}".format(
        written, sum(len(v["promises"]) for v in labels.values())
    ))


if __name__ == "__main__":
    main()
