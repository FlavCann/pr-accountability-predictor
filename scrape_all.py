import json
import time
from pathlib import Path

from youtube_transcript_api import RequestBlocked

from transcript import extract_video_id, get_transcript

TO_SCRAPE = Path("to_scrape.json")
OUT_DIR = Path("transcripts")
DELAY_SECONDS = 5


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    for video in json.loads(TO_SCRAPE.read_text()):
        url = video["url"]
        out = OUT_DIR / f"{extract_video_id(url)}.txt"
        if out.exists():
            print(f"skip  {url} (already done)")
            continue
        try:
            header = (
                f"Title: {video.get('title')}\n"
                f"Date: {video.get('date')}\n"
                f"URL: {url}\n"
                f"{'-' * 40}\n"
            )
            out.write_text(header + get_transcript(url, timestamps=False) + "\n")
            print(f"saved {url} -> {out}")
        except RequestBlocked:
            raise SystemExit(
                f"fail  {url}: YouTube is blocking this IP. Stopping so the block "
                "isn't extended. Switch network or set TRANSCRIPT_PROXY_URL (see README), then re-run."
            )
        except Exception as e:
            print(f"fail  {url}: {type(e).__name__}")
        time.sleep(DELAY_SECONDS)


if __name__ == "__main__":
    main()
