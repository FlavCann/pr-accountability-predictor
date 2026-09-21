# PR Accountability Predictor

Tools for collecting the raw text of corporate communications so they can be analysed later. The repo currently covers the data-collection step: finding YouTube videos on saved investor-relations pages and downloading their transcripts. The example data comes from Barry Callebaut's "Results & Publications" page (results presentations, capital markets days, and similar).

## How it works

```
saved HTML pages ──► find_videos.py ──► to_scrape.json ──► scrape_all.py ──► transcripts/*.txt
   (pages/)                                                     │
                                                          transcript.py
```

| File | Purpose |
| --- | --- |
| `find_videos.py` | Parses the HTML files in `pages/`, extracts YouTube links plus each card's title and date, and appends new ones to `to_scrape.json`. Videos already listed are skipped. |
| `to_scrape.json` | The list of videos to fetch (`url`, `title`, `date`). Currently holds 8 Barry Callebaut videos from 2021-2023. |
| `scrape_all.py` | Downloads the English transcript for each entry in `to_scrape.json` and writes it to `transcripts/<video_id>.txt` with a title/date/URL header. Skips videos that already have a file and waits 5 seconds between requests. |
| `transcript.py` | Fetches a single transcript. Works as a module (used by `scrape_all.py`) or as a CLI. |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

**1. Find videos on saved pages**

The `pages/` folder is git-ignored, so create it yourself and save the pages you want to scan as `.html` files (for example with your browser's "Save Page As").

```bash
python find_videos.py            # scans ./pages
python find_videos.py some/dir   # or scan another folder
```

`find_videos.py` picks up the title and date from cards with the `overview-teaser` CSS class, which is specific to the Barry Callebaut site. On other sites it still finds the YouTube links, but the title and date will be `null` unless you adapt `extract_videos()`.

**2. Download transcripts**

```bash
python scrape_all.py
```

Transcripts are saved to `transcripts/`. Videos with no English transcript are reported as `fail` and skipped, so you can re-run the script safely.

**3. Fetch one transcript (optional)**

```bash
python transcript.py "https://www.youtube.com/watch?v=tbovqMt-PJQ"
```

This prints the transcript with `[mm:ss]` timestamps.

## Requirements

- Python 3
- [`youtube-transcript-api`](https://pypi.org/project/youtube-transcript-api/)
- [`beautifulsoup4`](https://pypi.org/project/beautifulsoup4/)

## Notes

- Only English transcripts are requested.
- YouTube may rate-limit or block repeated requests. The 5-second delay in `scrape_all.py` is there to reduce that.
- Transcripts and source pages are not committed to the repo. Please respect the terms of the sites and videos you collect from.
