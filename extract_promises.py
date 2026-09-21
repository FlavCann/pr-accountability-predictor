import sys
from pathlib import Path

import anthropic
from pydantic import BaseModel

TRANSCRIPTS_DIR = Path("transcripts")
OUT_FILE = Path("promises.txt")
MODEL = "claude-sonnet-5"
# Testing mode: only send the start of each transcript to the API.
MAX_WORDS = 1000
HEADER_SEPARATOR = "-" * 40

SYSTEM_PROMPT = (
    "You analyse transcripts of corporate presentations. Extract every future "
    "promise the speaker makes: commitments, targets, guidance, or plans "
    "stated as something the company will do or achieve (e.g. 'we will reach "
    "X by 2025'). Ignore statements about the past or present, and vague "
    "sentiment with no commitment. For each promise, 'snippet' must be copied "
    "verbatim from the transcript, with enough surrounding words to be "
    "understood on its own. Return an empty list if there are none."
)


class Promise(BaseModel):
    snippet: str


class Promises(BaseModel):
    promises: list[Promise]


def read_transcript(path: Path) -> tuple[str, str]:
    """Return (header, body) of a file written by scrape_all.py."""
    header, _, body = path.read_text().partition(HEADER_SEPARATOR + "\n")
    return header.strip(), body


def first_words(text: str, limit: int) -> str:
    return " ".join(text.split()[:limit])


def extract_promises(client: anthropic.Anthropic, text: str) -> list[str]:
    response = client.messages.parse(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": text}],
        output_format=Promises,
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("model refused the request")
    if response.parsed_output is None:
        raise RuntimeError(f"no parsed output (stop_reason={response.stop_reason})")
    return [p.snippet for p in response.parsed_output.promises]


def main() -> None:
    files = sorted(TRANSCRIPTS_DIR.glob("*.txt"))
    if not files:
        sys.exit(f"No transcripts found in {TRANSCRIPTS_DIR}/ - run scrape_all.py first.")

    client = anthropic.Anthropic()
    with OUT_FILE.open("w") as out:

        def log(line: str = "") -> None:
            print(line)
            out.write(line + "\n")

        for path in files:
            header, body = read_transcript(path)
            log(f"=== {path.name} ===")
            log(header)
            try:
                snippets = extract_promises(client, first_words(body, MAX_WORDS))
            except anthropic.APIError as e:
                log(f"fail: {type(e).__name__}: {e}")
                log()
                continue
            except RuntimeError as e:
                log(f"fail: {e}")
                log()
                continue
            if not snippets:
                log("(no promises found)")
            for i, snippet in enumerate(snippets, 1):
                log(f"{i}. {snippet}")
            log()


if __name__ == "__main__":
    main()
