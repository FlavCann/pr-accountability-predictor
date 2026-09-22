"""Write a transcript as readable, wrapped text for labelling.

    python evals/show_transcript.py S07gnpUx15s

The stored transcripts are one line of 50-70k characters. This wraps them at
word boundaries into data/labelling/<id>.txt (git-ignored, like the transcripts
themselves), with a character offset every 20 lines and masked prompt-example
passages marked so they are skipped. Phrases copied from it -- line breaks
included -- resolve in build_gold.py.

It shows the transcript only. It never shows model output: labels for held-out
documents are made blind.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pr_predictor.ingest import parse_legacy_file  # noqa: E402

sys.path.insert(0, str(ROOT / "evals"))
from build_gold import locate  # noqa: E402


def main(video_id: str) -> Path:
    path = ROOT / "transcripts" / (video_id + ".txt")
    meta, body = parse_legacy_file(path)
    text = body.strip()

    labels = ROOT / "evals" / "labels" / (video_id + ".json")
    masks = []
    if labels.exists():
        for phrase, _ in json.loads(labels.read_text()).get("masked", []):
            masks.append(locate(text, phrase, path))
    for start, end in sorted(masks, reverse=True):
        text = text[:start] + " [[MASKED PROMPT EXAMPLE -- DO NOT LABEL: " + text[start:end] + " ]] " + text[end:]

    lines = textwrap.wrap(text, width=100, break_long_words=False, break_on_hyphens=False)
    out_lines = ["{}  --  {}  --  {}".format(meta.get("title"), meta.get("date"), meta.get("url")), ""]
    for i, line in enumerate(lines):
        if i % 20 == 0:
            out_lines.append("---- line {} ----".format(i + 1))
        out_lines.append(line)

    out = ROOT / "data" / "labelling" / (video_id + ".txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(out_lines) + "\n")
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    print("wrote {}".format(main(sys.argv[1])))
