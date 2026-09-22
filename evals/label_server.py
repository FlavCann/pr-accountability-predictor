"""The label desk: a local tool for labelling transcripts in the browser.

    python evals/label_server.py            # opens http://127.0.0.1:8765

Highlight a passage, choose what it is, and the label is written to
evals/labels/<video id>.json -- the same files build_gold.py reads -- and the
answer key is rebuilt. Nothing leaves this machine: the server binds to
localhost only, and the transcripts are not ours to redistribute.

It shows the transcript and your own labels, never model output, so it is safe
for blind labelling of test transcripts.

Every save is checked the way build_gold.py checks it -- the phrase must occur
exactly once, must not touch a masked prompt example, and type and hedge must
be real values -- and nothing invalid is written.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evals"))

import build_gold  # noqa: E402
from pr_predictor.ingest import parse_legacy_file  # noqa: E402
from pr_predictor.taxonomy import HedgeLevel, PromiseType  # noqa: E402

PAGE = ROOT / "evals" / "label_tool.html"
TRANSCRIPTS = ROOT / "transcripts"
LABELS = build_gold.LABELS_DIR
LABELLABLE = ("dev", "test", "fresh")
KEY_ORDER = (
    "_instructions", "provenance", "labelling_complete", "exhaustive", "labelled_by",
    "masked", "promises", "must_not_extract",
)
_VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{6,20}$")


def hits(text: str, phrase: str) -> List[Tuple[int, int]]:
    """Every place `phrase` occurs, matched the way build_gold.locate does."""
    words = phrase.split()
    if not words:
        return []
    pattern = re.compile(r"\s+".join(re.escape(w) for w in words))
    return [m.span() for m in pattern.finditer(text)]


def labellable_ids() -> Dict[str, str]:
    """Video ID -> split, for transcripts on disk in a split that is scored."""
    splits = build_gold.load_splits()
    return {
        vid: split for vid, split in sorted(splits.items())
        if split in LABELLABLE and (TRANSCRIPTS / (vid + ".txt")).exists()
    }


def load_doc(vid: str) -> Dict[str, object]:
    meta, body = parse_legacy_file(TRANSCRIPTS / (vid + ".txt"))
    text = body.strip()
    path = LABELS / (vid + ".json")
    labels = json.loads(path.read_text()) if path.exists() else {
        "labelling_complete": False, "exhaustive": False, "labelled_by": "",
        "masked": [], "promises": [], "must_not_extract": [],
    }
    masks = []
    for phrase, note in labels.get("masked", []):
        found = hits(text, phrase)
        if found:
            masks.append({"start": found[0][0], "end": found[0][1], "note": note})
    return {
        "id": vid,
        "split": labellable_ids().get(vid),
        "title": meta.get("title"),
        "date": meta.get("date"),
        "url": meta.get("url"),
        "text": text,
        "labels": labels,
        "masks": masks,
    }


def validate(labels: Dict[str, object], text: str) -> List[str]:
    """Everything build_gold.py would reject, as readable messages."""
    errors: List[str] = []
    types = {t.value for t in PromiseType}
    hedges = {h.value for h in HedgeLevel}

    masks = []
    for phrase, _ in labels.get("masked", []):
        found = hits(text, phrase)
        if len(found) != 1:
            errors.append("A masked passage no longer resolves; do not edit `masked`.")
        else:
            masks.append(found[0])

    def check_phrase(where: str, phrase: object) -> None:
        if not isinstance(phrase, str) or not phrase.strip():
            errors.append("{}: the phrase is empty.".format(where))
            return
        found = hits(text, phrase)
        short = " ".join(phrase.split()[:8])
        if not found:
            errors.append("{}: “{}…” is not in the transcript.".format(where, short))
        elif len(found) > 1:
            errors.append(
                "{}: “{}…” appears {} times. Select a longer passage.".format(where, short, len(found))
            )
        elif any(found[0][0] < m1 and m0 < found[0][1] for m0, m1 in masks):
            errors.append("{}: “{}…” overlaps a masked prompt example.".format(where, short))

    for i, p in enumerate(labels.get("promises", [])):
        where = "Promise {}".format(i + 1)
        if not isinstance(p, list) or len(p) < 4:
            errors.append("{}: needs phrase, type, hedge and note.".format(where))
            continue
        check_phrase(where, p[0])
        if p[1] not in types:
            errors.append("{}: unknown type {!r}.".format(where, p[1]))
        if p[2] not in hedges:
            errors.append("{}: unknown hedge {!r}.".format(where, p[2]))
        if len(p) > 4:
            extras = p[4]
            if not isinstance(extras, dict):
                errors.append("{}: the fifth element must be an object.".format(where))
            else:
                unknown = set(extras) - build_gold.LABEL_EXTRAS
                if unknown:
                    errors.append("{}: unknown field(s) {}.".format(where, sorted(unknown)))
                dr = extras.get("deadline_resolved")
                if dr is not None and not re.match(r"^\d{4}-\d{2}-\d{2}$", str(dr)):
                    errors.append("{}: deadline must be YYYY-MM-DD, or null.".format(where))

    for i, n in enumerate(labels.get("must_not_extract", [])):
        where = "Not-a-promise {}".format(i + 1)
        if not isinstance(n, list) or len(n) != 2:
            errors.append("{}: needs a phrase and a reason.".format(where))
            continue
        check_phrase(where, n[0])

    spans = []
    for kind, items in (("Promise", labels.get("promises", [])),
                        ("Not-a-promise", labels.get("must_not_extract", []))):
        for i, item in enumerate(items):
            if isinstance(item, list) and item and isinstance(item[0], str):
                found = hits(text, item[0])
                if len(found) == 1:
                    spans.append((found[0], "{} {}".format(kind, i + 1)))
    spans.sort()
    for (a, name_a), (b, name_b) in zip(spans, spans[1:]):
        if b[0] < a[1]:
            errors.append("{} and {} overlap.".format(name_a, name_b))

    if labels.get("labelling_complete") and not str(labels.get("labelled_by") or "").strip():
        errors.append("Say who labelled it (labelled by) before marking it complete.")
    return errors


def save(vid: str, labels: Dict[str, object]) -> Dict[str, object]:
    text = load_doc(vid)["text"]
    errors = validate(labels, text)
    if errors:
        return {"ok": False, "errors": errors}
    ordered = {k: labels[k] for k in KEY_ORDER if k in labels}
    ordered.update({k: v for k, v in labels.items() if k not in ordered})
    (LABELS / (vid + ".json")).write_text(json.dumps(ordered, indent=2, ensure_ascii=False) + "\n")
    return {"ok": True, "rebuild": rebuild()}


def rebuild() -> str:
    """Regenerate evals/gold from every label file, as build_gold.py does."""
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            build_gold.build(
                dict(build_gold.GOLD, **build_gold.load_label_files()),
                ROOT / "evals" / "gold",
                build_gold.LABELLER,
            )
    except SystemExit as e:
        return "Labels saved, but the answer key was not rebuilt: {}".format(e)
    return "Answer key rebuilt."


def docs_summary() -> List[Dict[str, object]]:
    out = []
    for vid, split in labellable_ids().items():
        meta, _ = parse_legacy_file(TRANSCRIPTS / (vid + ".txt"))
        path = LABELS / (vid + ".json")
        labels = json.loads(path.read_text()) if path.exists() else {}
        out.append({
            "id": vid, "split": split, "title": meta.get("title"), "date": meta.get("date"),
            "promises": len(labels.get("promises", [])),
            "negatives": len(labels.get("must_not_extract", [])),
            "complete": bool(labels.get("labelling_complete")),
            "exhaustive": bool(labels.get("exhaustive")),
        })
    return out


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: object) -> None:
        self._send(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _vid(self) -> Optional[str]:
        vid = (parse_qs(urlparse(self.path).query).get("id") or [""])[0]
        if not _VIDEO_ID.match(vid) or vid not in labellable_ids():
            self._json(404, {"ok": False, "errors": ["Unknown or unlabellable transcript."]})
            return None
        return vid

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        route = urlparse(self.path).path
        if route in ("/", "/index.html"):
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif route == "/api/docs":
            self._json(200, docs_summary())
        elif route == "/api/doc":
            vid = self._vid()
            if vid:
                self._json(200, load_doc(vid))
        else:
            self._json(404, {"ok": False, "errors": ["Not found."]})

    def do_PUT(self) -> None:  # noqa: N802 - http.server API
        if urlparse(self.path).path != "/api/doc":
            self._json(404, {"ok": False, "errors": ["Not found."]})
            return
        vid = self._vid()
        if not vid:
            return
        try:
            labels = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        except (ValueError, json.JSONDecodeError):
            self._json(400, {"ok": False, "errors": ["The request was not valid JSON."]})
            return
        result = save(vid, labels)
        self._json(200 if result["ok"] else 400, result)

    def log_message(self, fmt: str, *args) -> None:
        if not str(args[1] if len(args) > 1 else "").startswith("2"):
            sys.stderr.write("{} {}\n".format(self.address_string(), fmt % args))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Label transcripts in the browser.")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = "http://127.0.0.1:{}/".format(server.server_address[1])
    print("Label desk running at {}  (Ctrl+C to stop)".format(url))
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
