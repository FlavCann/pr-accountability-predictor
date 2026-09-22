"""Stage 1-3: discover, fetch, normalise.

Replaces `find_videos.py`, `transcript.py` and `scrape_all.py`.

The one substantive change from the originals: **timestamps survive**. The old
`scrape_all.py` called `get_transcript(url, timestamps=False)`, flattening the
caption cues into a single string and throwing away the only thing that could
link a claim back to the moment it was made. Here we build the flat text *and*
a `Segment` per cue carrying both the character range and the time range, so
every extracted span resolves to a deep link into the video.
"""

from __future__ import annotations

import json
import re
import sys
import time
from datetime import date
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from . import config
from .domain import Segment, Source

# Matches watch?v=ID, youtu.be/ID, /embed/ID, /shorts/ID, /live/ID. Channel
# links and player scripts don't match because a video ID is exactly 11 chars.
VIDEO_RE = re.compile(
    r"(?:youtube(?:-nocookie)?\.com/(?:watch\?(?:[^\"'\s<>]*?&(?:amp;)?)?v=|embed/|shorts/|live/)"
    r"|youtu\.be/)([A-Za-z0-9_-]{11})(?![A-Za-z0-9_-])"
)


# ----------------------------------------------------------------------
# Discover
# ----------------------------------------------------------------------

def extract_video_id(url: str) -> str:
    parsed = urlparse(url)
    if parsed.hostname == "youtu.be":
        return parsed.path.lstrip("/")
    if parsed.hostname and "youtube.com" in parsed.hostname:
        if parsed.path == "/watch":
            return parse_qs(parsed.query)["v"][0]
        if parsed.path.startswith(("/shorts/", "/embed/", "/live/")):
            return parsed.path.split("/")[2]
    raise ValueError("Could not find a video ID in: {}".format(url))


def extract_videos(html: str) -> List[dict]:
    """{url, title, date} per YouTube link, taking title/date from its card."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    videos = []
    for a in soup.find_all("a", href=True):
        match = VIDEO_RE.search(a["href"])
        if not match:
            continue
        card = a.find_parent(class_="overview-teaser")
        title = card.select_one(".overview-teaser__title") if card else None
        t = card.find("time") if card else None
        videos.append(
            {
                "url": "https://www.youtube.com/watch?v={}".format(match.group(1)),
                "title": title.get_text(" ", strip=True) if title else None,
                "date": t["datetime"][:10] if t and t.has_attr("datetime") else None,
            }
        )
    return videos


def discover(pages_dir: Optional[Path] = None) -> int:
    """Scan saved IR pages, append new videos to the work list."""
    pages_dir = Path(pages_dir or config.PAGES_DIR)
    found = {}
    for path in sorted(pages_dir.glob("*.htm*")):
        vids = extract_videos(path.read_text(encoding="utf-8", errors="ignore"))
        print("{}: {} video(s)".format(path.name, len(vids)), file=sys.stderr)
        for v in vids:
            found.setdefault(v["url"], v)

    existing = (
        json.loads(config.TO_SCRAPE.read_text()) if config.TO_SCRAPE.exists() else []
    )
    known = {v["url"] for v in existing}
    new = [v for v in found.values() if v["url"] not in known]
    config.TO_SCRAPE.write_text(
        json.dumps(existing + new, indent=2, ensure_ascii=False) + "\n"
    )
    return len(new)


# ----------------------------------------------------------------------
# Fetch
# ----------------------------------------------------------------------

def _make_api():
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api.proxies import GenericProxyConfig

    import os

    proxy = os.environ.get("TRANSCRIPT_PROXY_URL")
    if proxy:
        return YouTubeTranscriptApi(
            proxy_config=GenericProxyConfig(http_url=proxy, https_url=proxy)
        )
    return YouTubeTranscriptApi()


def fetch_cues(url: str) -> List[Tuple[str, float, float]]:
    """(text, t_start, t_end) per caption cue."""
    snippets = _make_api().fetch(extract_video_id(url), languages=["en"])
    out = []
    for s in snippets:
        text = " ".join(s.text.split())
        if text:
            out.append((text, float(s.start), float(s.start) + float(s.duration)))
    return out


def build_source(
    url: str,
    cues: List[Tuple[str, float, float]],
    company: str,
    title: Optional[str] = None,
    published: Optional[date] = None,
) -> Tuple[Source, List[Segment]]:
    """Assemble flat text plus a char-offset-indexed, timestamped segment list.

    This is the function that keeps provenance intact. Joining cues with a
    single space and recording each cue's char range means any character
    position in `source.text` can be resolved to a timestamp in O(1) via
    `Store.segment_at`.
    """
    parts: List[str] = []
    segments: List[Segment] = []
    cursor = 0
    src = Source(company=company, url=url, title=title, published=published)

    for i, (text, t0, t1) in enumerate(cues):
        if i > 0:
            parts.append(" ")
            cursor += 1
        start = cursor
        parts.append(text)
        cursor += len(text)
        segments.append(
            Segment(
                source_id=src.id,
                index=i,
                start_char=start,
                end_char=cursor,
                t_start=t0,
                t_end=t1,
            )
        )

    src.text = "".join(parts)
    src.finalise()
    return src, segments


# --- reading the legacy transcripts/*.txt files -----------------------

HEADER_SEPARATOR = "-" * 40


def parse_legacy_file(path: Path) -> Tuple[dict, str]:
    """Split a file written by the old `scrape_all.py` into header and body.

    These files have no timestamps -- that information was destroyed at write
    time and cannot be recovered without re-fetching. We ingest them so the
    corpus is usable immediately, and mark them so a later re-fetch can upgrade
    them in place.
    """
    raw = path.read_text()
    header_text, _, body = raw.partition(HEADER_SEPARATOR + "\n")
    meta = {}
    for line in header_text.strip().splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            meta[k.strip().lower()] = v.strip()
    return meta, body.strip()


def ingest_legacy_transcripts(store, company: str, directory: Optional[Path] = None) -> int:
    """Load the existing transcripts/*.txt corpus into the store.

    Lets the whole pipeline run today against the eight transcripts already on
    disk, without re-hitting YouTube (which blocks aggressively).
    """
    directory = Path(directory or config.TRANSCRIPTS_DIR)
    n = 0
    for path in sorted(directory.glob("*.txt")):
        meta, body = parse_legacy_file(path)
        url = meta.get("url", "file://{}".format(path.name))
        if store.source_by_url(url):
            continue

        published = None
        if meta.get("date") and meta["date"] != "None":
            try:
                published = date.fromisoformat(meta["date"])
            except ValueError:
                published = None

        src = Source(
            company=company,
            url=url,
            title=meta.get("title"),
            published=published,
            text=body,
            fetcher="legacy_txt_no_timestamps",
        ).finalise()
        store.put_source(src)
        n += 1
    return n


def fetch_pending(store, company: str, delay: float = 5.0, limit: Optional[int] = None) -> dict:
    """Fetch transcripts for everything in the work list not already stored."""
    from youtube_transcript_api import RequestBlocked

    work = json.loads(config.TO_SCRAPE.read_text()) if config.TO_SCRAPE.exists() else []
    stats = {"fetched": 0, "skipped": 0, "failed": 0}

    for i, video in enumerate(work):
        if limit is not None and stats["fetched"] >= limit:
            break
        url = video["url"]
        existing = store.source_by_url(url)
        if existing and existing.fetcher != "legacy_txt_no_timestamps":
            stats["skipped"] += 1
            continue

        try:
            cues = fetch_cues(url)
        except RequestBlocked:
            raise SystemExit(
                "YouTube is blocking this IP. Stopping so the block isn't "
                "extended. Switch network or set TRANSCRIPT_PROXY_URL, then re-run."
            )
        except Exception as e:  # noqa: BLE001 - per-item isolation
            print("fail  {}: {}".format(url, type(e).__name__), file=sys.stderr)
            stats["failed"] += 1
            continue

        published = None
        if video.get("date"):
            try:
                published = date.fromisoformat(video["date"])
            except ValueError:
                pass

        src, segments = build_source(
            url, cues, company=company, title=video.get("title"), published=published
        )
        store.put_source(src)
        store.put_segments(segments)
        if existing is not None:
            # A timestamped re-fetch supersedes the legacy text. The old row and
            # anything cited from it stay in the store; they simply stop being
            # listed, so nothing is extracted or counted twice.
            store.supersede_source(existing.id, src.id)
        stats["fetched"] += 1
        print("saved {} ({} cues)".format(url, len(segments)), file=sys.stderr)

        if i < len(work) - 1:
            time.sleep(delay)

    return stats
