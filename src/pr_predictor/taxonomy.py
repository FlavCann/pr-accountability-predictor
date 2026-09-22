"""The promise taxonomy.

This is the single source of truth for what counts as a promise. It is imported
by the extraction prompt (`prompts.py`), the eval metrics (`evals.py`) and the
adjudication rubric (`verify.py`), so the definition can never drift between
what we ask for, what we grade, and what we ship.

If you change anything here, you must re-run the evals. See CLAUDE.md.
"""

from __future__ import annotations

from enum import Enum


class PromiseType(str, Enum):
    """What kind of forward-looking statement this is.

    The distinction matters because they are held to different standards. A
    QUANTIFIED_TARGET can be checked against a reported number. A
    SCHEDULING_ANNOUNCEMENT is kept or missed by a calendar. Grading them with
    one rule is what produced the original pipeline's false positives.
    """

    QUANTIFIED_TARGET = "quantified_target"
    """A measurable commitment with a number and usually a date.
    'lift 500,000 farmers out of poverty by 2025'"""

    DIRECTIONAL_GUIDANCE = "directional_guidance"
    """A stated direction of travel without a specific number.
    'we expect margins to improve in the second half'"""

    PROCEDURAL_COMMITMENT = "procedural_commitment"
    """A promise to do something, where doing it is the outcome.
    'we will bring our supply chain under human rights due diligence'"""

    SCHEDULING_ANNOUNCEMENT = "scheduling_announcement"
    """A statement that an event will occur on a date.
    'our full year results will be available on the 1st of November'"""


class HedgeLevel(str, Enum):
    """How firmly the commitment is stated.

    Kept separate from the promise itself so that 'we will' is never silently
    flattened into 'we aim to'. This field is the raw material for the
    predictor: hedge language is the strongest available signal of whether a
    commitment will be met.
    """

    FIRM = "firm"
    """'we will', 'we commit to', 'we are going to'"""

    INTENDED = "intended"
    """'we aim to', 'we intend to', 'we target', 'we plan to'"""

    CONDITIONAL = "conditional"
    """'we should', 'we expect to', 'assuming X, we would'"""

    ASPIRATIONAL = "aspirational"
    """'our ambition is', 'we would like to', 'ideally'"""


class VerdictStatus(str, Enum):
    """The outcome of adjudicating a promise thread against evidence."""

    KEPT = "kept"
    MISSED = "missed"
    PARTIAL = "partial"

    QUIETLY_DROPPED = "quietly_dropped"
    """Stated once, then never mentioned again while the company kept
    reporting. Invisible in any single document and detectable only across a
    threaded multi-year corpus. This is the product's differentiating verdict."""

    TOO_EARLY = "too_early"
    """The deadline has not passed. A legitimate terminal state, not a failure."""

    NO_EVIDENCE = "no_evidence"
    """The deadline passed but nothing in the corpus bears on it. Distinct from
    TOO_EARLY (timing) and from MISSED (which is a positive finding)."""


# ---------------------------------------------------------------------------
# Worked examples fed to the model.
#
# RULE: never take an example from a transcript that is in the gold set
# (evals/gold/). An example the model has been shown cannot also be used to
# test it -- the eval would be measuring memory, not judgement. This is
# enforced twice: `evals.find_leakage` refuses to run a contaminated eval, and
# a test fails if any example appears in a gold transcript.
#
# Current gold transcripts: _pPsXMY2oTE (Forever Chocolate, May 2023) and
# ZcTyUa8q71E (Full-Year Results 2021/22). Do not draw examples from them.
# ---------------------------------------------------------------------------

# Statements that look forward but are NOT promises by the company. Four of the
# five are real false positives produced by the original prompt (extract-v1)
# over this repo's transcripts.
EXCLUSIONS = [
    (
        "Jokes and social pleasantries",
        "we invite you and and have some of the chocolate on and the latest "
        "innovation to taste",
        "An invitation to the tasting after a results presentation. The original "
        "prompt recorded a near-identical catering remark as the only promise in "
        "a 12,000-word results call.",
    ),
    (
        "Observations about the world",
        "should continue with the further easing of kovac 19 measures globally",
        "A macro expectation about external conditions, not something the company "
        "undertakes to do. (Also shows ASR garbling of 'COVID-19'.)",
    ),
    (
        "Generic statements of character",
        "we are a company that will support its customers",
        "Sentiment with no checkable content. No metric, no date, no action.",
    ),
    (
        "Process commentary with no commitment",
        "we will continue to monitor closely and assess the situation",
        "Monitoring is not an outcome. Nothing could falsify this.",
    ),
    (
        "Third-party actions",
        "steve will be succeeding peter as president americas as of september 1st",
        "Personnel announcements are facts about a decision already taken, not "
        "forward commitments the company can fail to meet.",
    ),
]


# Worked positives, drawn from non-gold transcripts.
INCLUSIONS = [
    (
        "we confirmed today our new midterm guidance for the three-year period "
        "2023-2024 to 2025 2026",
        PromiseType.DIRECTIONAL_GUIDANCE,
        HedgeLevel.FIRM,
        "Re-affirms guidance given earlier, so restates_prior_target=true. The "
        "period is in FISCAL years ending in August: leave `deadline` null and "
        "put the phrase in `deadline_raw`.",
    ),
    (
        "we will invest 500 million Nets in core customer areas and efficiency "
        "measures and in turn receive synergies of about 250 million annual cost "
        "reductions",
        PromiseType.QUANTIFIED_TARGET,
        HedgeLevel.FIRM,
        "Two checkable numbers: the investment and the synergy.",
    ),
    (
        "we continue to focus on diligently managing the working capital and aim "
        "to grow it equal or less than our top line",
        PromiseType.DIRECTIONAL_GUIDANCE,
        HedgeLevel.INTENDED,
        "'aim to' makes it INTENDED. Checkable without a number: working capital "
        "growth against revenue growth. 'we continue to focus on' alone would "
        "not qualify -- the hedge attaches to the checkable clause.",
    ),
    (
        "our full year results will be available on the 1st of November",
        PromiseType.SCHEDULING_ANNOUNCEMENT,
        HedgeLevel.FIRM,
        "Low materiality but genuinely checkable. Kept, and scored separately.",
    ),
]


def rubric() -> str:
    """Render the taxonomy as prompt text.

    Used by `prompts.py` for extraction and by `verify.py` for adjudication, so
    both see identical definitions.
    """
    lines = ["PROMISE TYPES", ""]
    for t in PromiseType:
        doc = " ".join((t.__doc__ or "").split())
        lines.append("- {}: {}".format(t.value, doc))

    lines += ["", "HEDGE LEVELS (grade the checkable clause, not the sentence)", ""]
    for h in HedgeLevel:
        doc = " ".join((h.__doc__ or "").split())
        lines.append("- {}: {}".format(h.value, doc))

    lines += ["", "DO NOT EXTRACT", ""]
    for name, example, why in EXCLUSIONS:
        lines.append('- {}. Example: "{}" -- {}'.format(name, example, why))

    lines += ["", "WORKED EXAMPLES", ""]
    for text, ptype, hedge, why in INCLUSIONS:
        lines.append(
            '- "{}"\n  -> type={} hedge={} ({})'.format(text, ptype.value, hedge.value, why)
        )

    return "\n".join(lines)
