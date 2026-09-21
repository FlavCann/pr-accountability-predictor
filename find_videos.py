import json
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup

PAGES_DIR = Path("pages")
TO_SCRAPE = Path("to_scrape.json")

# Matches watch?v=ID, youtu.be/ID, /embed/ID, /shorts/ID, /live/ID. Channel links and
# player scripts don't match because a video ID is exactly 11 characters.
VIDEO_RE = re.compile(
    r"(?:youtube(?:-nocookie)?\.com/(?:watch\?(?:[^\"'\s<>]*?&(?:amp;)?)?v=|embed/|shorts/|live/)"
    r"|youtu\.be/)([A-Za-z0-9_-]{11})(?![A-Za-z0-9_-])"
)


def extract_videos(html: str) -> list:
    """Return {url, title, date} for each YouTube link, taking title/date from its enclosing card."""
    soup = BeautifulSoup(html, "html.parser")
    videos = []
    for a in soup.find_all("a", href=True):
        match = VIDEO_RE.search(a["href"])
        if not match:
            continue
        card = a.find_parent(class_="overview-teaser")
        title = card.select_one(".overview-teaser__title") if card else None
        time = card.find("time") if card else None
        videos.append({
            "url": f"https://www.youtube.com/watch?v={match.group(1)}",
            "title": title.get_text(" ", strip=True) if title else None,
            "date": time["datetime"][:10] if time and time.has_attr("datetime") else None,
        })
    return videos


def find_youtube_videos(pages_dir: Path) -> list:
    videos = {}
    for path in sorted(pages_dir.glob("*.htm*")):
        found = extract_videos(path.read_text(encoding="utf-8", errors="ignore"))
        print(f"{path.name}: {len(found)} video(s)", file=sys.stderr)
        for v in found:
            videos.setdefault(v["url"], v)
    return list(videos.values())


def add_to_scrape_list(videos: list) -> int:
    existing = json.loads(TO_SCRAPE.read_text()) if TO_SCRAPE.exists() else []
    known = {v["url"] for v in existing}
    new = [v for v in videos if v["url"] not in known]
    TO_SCRAPE.write_text(json.dumps(existing + new, indent=2, ensure_ascii=False) + "\n")
    return len(new)


if __name__ == "__main__":
    pages_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else PAGES_DIR
    videos = find_youtube_videos(pages_dir)
    added = add_to_scrape_list(videos)
    print(f"Found {len(videos)} video(s), added {added} new to {TO_SCRAPE}")
