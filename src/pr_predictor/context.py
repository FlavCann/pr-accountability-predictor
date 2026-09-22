"""Context assembly (concept 2).

What goes into the window, and why. The original pipeline put 1,000 words of
transcript in and nothing else -- no company background, no prior targets, no
glossary -- so the model could not know that "Forever Chocolate" is a 2016
programme or that "calabroth" is the company's own name.

On chunking: a full transcript here is ~14k tokens against a 1M-token context
window, so it fits hundreds of times over. Chunking is therefore NOT about the
size limit. It is about attention quality and recall across a long, rambling,
low-information-density document. Conflating "it fits" with "it will be
attended to evenly" is what produces pipelines that technically read everything
and still miss most of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from . import config


@dataclass
class Chunk:
    """A window over a source, carrying its absolute offset.

    `offset` is what lets the model's chunk-local character positions be
    translated back to absolute positions in the full source text.
    """

    index: int
    offset: int
    text: str

    @property
    def end(self) -> int:
        return self.offset + len(self.text)


def chunk_text(
    text: str,
    chunk_words: Optional[int] = None,
    overlap_words: Optional[int] = None,
) -> List[Chunk]:
    """Split into overlapping windows, preserving absolute character offsets.

    The overlap exists so a promise straddling a boundary is seen whole by at
    least one call. Duplicates created by the overlap are reconciled by span
    intersection in `extract.reconcile`.
    """
    chunk_words = chunk_words or config.CHUNK_WORDS
    overlap_words = overlap_words or config.CHUNK_OVERLAP_WORDS
    if overlap_words >= chunk_words:
        raise ValueError("overlap must be smaller than the chunk")

    # Walk words while tracking exact character positions, so offsets stay true
    # even where the source has irregular whitespace.
    spans: List[tuple] = []
    pos = 0
    for word in text.split(" "):
        if word:
            start = text.index(word, pos)
            spans.append((start, start + len(word)))
            pos = start + len(word)
        else:
            pos += 1

    if not spans:
        return []

    chunks: List[Chunk] = []
    step = chunk_words - overlap_words
    i = 0
    idx = 0
    while i < len(spans):
        window = spans[i : i + chunk_words]
        start_char = window[0][0]
        end_char = window[-1][1]
        chunks.append(Chunk(index=idx, offset=start_char, text=text[start_char:end_char]))
        idx += 1
        if i + chunk_words >= len(spans):
            break
        i += step
    return chunks


# ----------------------------------------------------------------------
# Company brief
# ----------------------------------------------------------------------

def company_brief(store, company: str) -> str:
    """Background the model needs to read a transcript correctly.

    Assembled from durable memory (concept 7) rather than hard-coded, so
    analyst corrections and newly learned facts flow into every later call.
    """
    lines = ["COMPANY CONTEXT: {}".format(company)]

    facts = store.memory_list(company, kind="fact")
    if facts:
        lines.append("")
        lines.append("Known facts:")
        for row in facts:
            lines.append("- {}: {}".format(row["key"], row["value"]))

    glossary = store.memory_list(company, kind="asr")
    if glossary:
        lines.append("")
        lines.append(
            "Speech-recognition corrections (these transcripts are machine-generated; "
            "the left-hand form is a mis-transcription of the right-hand one):"
        )
        for row in glossary:
            lines.append('- "{}" -> "{}"'.format(row["key"], row["value"]))

    corrections = store.memory_list(company, kind="correction")
    if corrections:
        lines.append("")
        lines.append("Analyst corrections from previous runs -- follow these:")
        for row in corrections:
            lines.append("- {}: {}".format(row["key"], row["value"]))

    return "\n".join(lines)


def prior_threads_brief(store, company: str, limit: int = 40) -> str:
    """Existing commitments, so a restatement is linked rather than duplicated.

    Without this the model cannot know that "we stick to our target of half a
    million farmers" refers to a commitment first made in 2016.
    """
    threads = store.list_threads(company)[:limit]
    if not threads:
        return ""
    lines = [
        "",
        "EXISTING COMMITMENT THREADS for this company. If the speaker is "
        "re-affirming one of these rather than making a new commitment, set "
        "restates_prior_target=true and make normalized_claim match the "
        "existing wording closely so we can link them:",
        "",
    ]
    for t in threads:
        deadline = t.deadline.isoformat() if t.deadline else (t.metric or "no deadline")
        lines.append("- [{}] {} ({})".format(t.id, t.canonical_claim, deadline))
    return "\n".join(lines)


def build_extraction_context(store, company: str) -> str:
    """The stable prefix for extraction calls.

    Everything here is identical across every chunk of every transcript for a
    company, which is exactly what makes it worth a cache breakpoint.
    """
    parts = [company_brief(store, company)]
    threads = prior_threads_brief(store, company)
    if threads:
        parts.append(threads)
    return "\n".join(parts)


# ----------------------------------------------------------------------
# Seed memory
# ----------------------------------------------------------------------

BARRY_CALLEBAUT_SEED = {
    "fact": {
        "fiscal_year_end": "31 August. 'FY23' means Sep 2022 - Aug 2023, NOT calendar "
        "2023. Never map a fiscal-year deadline to a calendar date without this.",
        "forever_chocolate": "Sustainability programme launched 2016 with four 2025 "
        "ambitions: lift 500,000 farmers out of poverty; eradicate child labour from "
        "the supply chain; become carbon and forest positive; 100% sustainable "
        "ingredients in all products. Re-sharpened in May 2023 with targets extended "
        "to 2030.",
        "bc_next_level": "Strategic investment programme announced 6 Sep 2023: ~CHF 500m "
        "investment, ~CHF 250m expected annual cost savings.",
        "ceo_transition": "Antoine de Saint-Affrique handed over to Peter Boone during "
        "FY2020/21; Boone became CEO 1 September 2021.",
        "reporting_currency": "CHF. Speakers sometimes say 'francs' or, via ASR, 'Nets'.",
    },
    "asr": {
        "calabroth": "Callebaut",
        "calibot": "Callebaut",
        "barry calabroth": "Barry Callebaut",
        "kovac 19": "COVID-19",
        "kovid": "COVID",
        "Nets": "CHF (Swiss francs)",
        "atlantic stark": "(unverified ASR garble - check against the source audio)",
    },
}


def seed_memory(store, company: str = "Barry Callebaut") -> int:
    """Populate durable memory with what we already know.

    Everything here was learned by reading this repo's own transcripts and
    output. It is exactly the context whose absence caused the original
    pipeline's failures.
    """
    n = 0
    for kind, entries in BARRY_CALLEBAUT_SEED.items():
        for key, value in entries.items():
            store.memory_put(company, kind, key, value)
            n += 1
    return n
