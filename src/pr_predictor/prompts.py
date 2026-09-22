"""Versioned prompt registry (concept 1).

Every prompt has an ID that is written onto every row it produces. That is what
makes "did this change help?" answerable: you can diff v1 output against v3
output over the same sources, and the eval harness can pin a version.

v1 is the original single-paragraph prompt, kept verbatim so the evals can
measure against the starting point. Do not delete it.
"""

from __future__ import annotations

from typing import Dict, NamedTuple

from .taxonomy import rubric


class Prompt(NamedTuple):
    version: str
    system: str
    notes: str

    @property
    def fingerprint(self) -> str:
        """Hash of the exact prompt text.

        A version name is only meaningful if it names one fixed text. Prompts
        here are partly generated from the taxonomy, so an edit to
        `taxonomy.py` silently changes what a version *means*. `PINNED` below
        and `tests/test_pipeline.py::test_prompt_versions_are_pinned` catch
        that: change the text without bumping the version and the test fails.
        """
        from .domain import content_hash

        return content_hash(self.system)


# --------------------------------------------------------------------------
# v1 -- the original. Preserved as the eval baseline.
# --------------------------------------------------------------------------

V1_SYSTEM = (
    "You analyse transcripts of corporate presentations. Extract every future "
    "promise the speaker makes: commitments, targets, guidance, or plans "
    "stated as something the company will do or achieve (e.g. 'we will reach "
    "X by 2025'). Ignore statements about the past or present, and vague "
    "sentiment with no commitment. For each promise, 'snippet' must be copied "
    "verbatim from the transcript, with enough surrounding words to be "
    "understood on its own. Return an empty list if there are none."
)

# --------------------------------------------------------------------------
# The structured template -- taxonomy, negative examples, and anchor quotes
# we locate ourselves instead of a retyped citation. First registered as
# extract-v2 (retired), now extract-v3.
# --------------------------------------------------------------------------

STRUCTURED_SYSTEM = """You extract forward-looking commitments from transcripts of corporate \
investor presentations. Your output is used to hold public companies to what they \
said, so precision matters more than recall: a false promise attributed to a company \
is worse than a missed one.

{rubric}

OUTPUT RULES

1. Mark each promise with `quote_start` (its first 5-10 words) and
   `quote_end` (its last 5-10 words), copied EXACTLY from the excerpt --
   including transcription errors such as misspelled names. Do not correct,
   paraphrase or retype anything else. We find these words in the source
   ourselves and cite the text between them; a promise whose words cannot be
   found is discarded as uncitable.

2. Span boundaries: start at the beginning of the clause that carries the
   commitment and end where it finishes. Include enough for the claim to stand
   alone, but do not swallow neighbouring sentences. Keep a promise under
   about 80 words.

3. `normalized_claim` is a clean one-sentence restatement for display and for
   matching against prior commitments. This is NOT the citation -- the citation
   is always the span. Correct obvious speech-recognition errors here (the
   transcripts are machine-generated and garble names).

4. `deadline_raw` is what the speaker actually said ("by 2025", "end of the
   fiscal year", "in the second half"). `deadline` is your best absolute date.
   If the phrasing is relative to a fiscal year, put the phrase in
   `deadline_raw` and leave `deadline` null rather than guessing -- this
   company's fiscal year does not align with the calendar year.

5. `restates_prior_target` is true when the speaker is re-affirming a
   commitment rather than making a new one ("we stick to our target of...",
   "as we said last year..."). These are threaded to the original.

6. `hedge_level` grades the clause that carries the checkable content, not the
   loudest verb in the sentence.

7. Return an empty list if the text contains no qualifying promises. An empty
   list is a valid and common answer for introductory or Q&A passages."""


REGISTRY: Dict[str, Prompt] = {}


def _register(version: str, system: str, notes: str) -> None:
    REGISTRY[version] = Prompt(version=version, system=system, notes=notes)


_register(
    "extract-v1",
    V1_SYSTEM,
    "Original prompt. Free-text snippets, no taxonomy, no negative examples. "
    "Kept as the eval baseline -- this is what we have to beat.",
)

# extract-v2 was retired before it was ever run against a live model: three of
# its worked examples were passages from gold-set transcripts, so any eval of it
# would have been contaminated. extract-v3 is the same structure with every
# example drawn from non-gold transcripts. See REPORT.md, section 6.

_register(
    "extract-v3",
    STRUCTURED_SYSTEM.format(rubric=rubric()),
    "Taxonomy + negative examples + opening/closing anchor quotes that we "
    "locate in the source ourselves. All worked examples come from non-gold "
    "transcripts.",
)

DEFAULT_EXTRACTION_PROMPT = "extract-v3"

# Fingerprint of each version's exact text. If a test fails against this, you
# changed a prompt's content: register a NEW version rather than updating the
# hash, unless the old version has never produced a stored row.
PINNED: Dict[str, str] = {
    "extract-v1": "d9d13ca341e1ec62",
    "extract-v3": "2287121d2cddf943",
}


def get(version: str) -> Prompt:
    if version not in REGISTRY:
        raise KeyError(
            "Unknown prompt version {!r}. Known: {}".format(
                version, ", ".join(sorted(REGISTRY))
            )
        )
    return REGISTRY[version]


# --------------------------------------------------------------------------
# Adjudication (concept 11). Shares the taxonomy so a verdict is graded
# against the same definitions the extraction used.
# --------------------------------------------------------------------------

ADJUDICATION_SYSTEM = """You adjudicate whether a company kept a commitment.

You are given one promise thread (a commitment and every restatement of it) and \
the evidence gathered about it. Decide the status and explain why, citing the \
evidence you relied on by ID.

{rubric}

VERDICT RULES

- kept: evidence shows the commitment was met by its deadline.
- missed: the deadline passed and evidence shows it was not met.
- partial: materially advanced but not met as stated.
- quietly_dropped: the company stopped mentioning it while continuing to report
  on adjacent topics, and no evidence shows it was met or formally abandoned.
  Requires at least two later sources that would plausibly have mentioned it.
- too_early: the deadline has not passed. This is a normal outcome, not a failure.
- no_evidence: the deadline passed but nothing available bears on it.

Do not infer success from the absence of bad news. If the evidence does not
support a finding, say no_evidence. Your rationale is shown to analysts and is
part of an audit trail, so cite specifically and never assert more than the
evidence carries."""


def adjudication_system() -> str:
    return ADJUDICATION_SYSTEM.format(rubric=rubric())
