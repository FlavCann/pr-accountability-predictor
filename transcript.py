import os
import sys
from urllib.parse import urlparse, parse_qs

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    NoTranscriptFound,
    TranscriptsDisabled,
)
from youtube_transcript_api.proxies import GenericProxyConfig


def extract_video_id(url: str) -> str:
    parsed = urlparse(url)
    if parsed.hostname == "youtu.be":
        return parsed.path.lstrip("/")
    if parsed.hostname and "youtube.com" in parsed.hostname:
        if parsed.path == "/watch":
            return parse_qs(parsed.query)["v"][0]
        if parsed.path.startswith(("/shorts/", "/embed/", "/live/")):
            return parsed.path.split("/")[2]
    raise ValueError(f"Could not find a video ID in: {url}")


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def make_api() -> YouTubeTranscriptApi:
    # YouTube blocks many home/cloud IPs. Set TRANSCRIPT_PROXY_URL
    # (e.g. http://user:pass@host:port) to route requests through a proxy.
    proxy = os.environ.get("TRANSCRIPT_PROXY_URL")
    if proxy:
        return YouTubeTranscriptApi(
            proxy_config=GenericProxyConfig(http_url=proxy, https_url=proxy)
        )
    return YouTubeTranscriptApi()


def get_transcript(url: str, timestamps: bool = True) -> str:
    video_id = extract_video_id(url)
    snippets = make_api().fetch(video_id, languages=["en"])
    if timestamps:
        return "\n".join(f"[{fmt_time(s.start)}] {s.text}" for s in snippets)
    return " ".join(" ".join(s.text.split()) for s in snippets)


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else input("YouTube URL: ").strip()
    try:
        print(get_transcript(url))
    except (TranscriptsDisabled, NoTranscriptFound):
        sys.exit("No transcript available for this video.")
    except ValueError as e:
        sys.exit(str(e))
