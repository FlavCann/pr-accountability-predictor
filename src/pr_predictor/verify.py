"""Stages 6-7: evidence gathering and adjudication (concepts 6 and 11).

Two loops live here.

The **agent loop** gathers evidence about a thread. It is bounded on three
axes -- steps, dollars, and whether it is still making progress -- and
"no evidence found" is a legitimate terminal outcome mapping to a NO_EVIDENCE
verdict, not an error to retry. Agent loops that treat an honest negative as a
failure burn budget rediscovering nothing.

The **adjudication** turns evidence into a verdict, and is where the
executor/advisory split earns its place (concept 11). Sonnet adjudicates
everything; a contested call escalates to the advisor tool, where a stronger
model is consulted mid-generation.

A note on the advisor and this product's audit requirement: a
`claude-fable-5-1` advisor returns `advisor_redacted_result` -- encrypted
advice we can replay but cannot read. Since a verdict's rationale is part of an
audit trail, the default advisor is `claude-opus-4-8`, which returns plaintext
`advisor_result`. See config.ADVISOR_MODEL.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from . import config, prompts, tools
from .domain import Evidence, PromiseThread, Verdict
from .llm import LLMClient, LLMError, RunLedger, Usage, usage_from_response
from .taxonomy import VerdictStatus


class Adjudication(BaseModel):
    status: VerdictStatus
    rationale: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence_excerpts: List[str] = Field(default_factory=list)
    contested: bool = Field(
        default=False,
        description="True when the evidence genuinely supports more than one "
        "reading and a second opinion would change the answer.",
    )


# ----------------------------------------------------------------------
# Evidence gathering: the bounded agent loop
# ----------------------------------------------------------------------

def gather_evidence(
    store,
    client: LLMClient,
    thread: PromiseThread,
    run_id: str = "",
    max_steps: Optional[int] = None,
    max_cost: Optional[float] = None,
) -> Tuple[List[Evidence], Usage, str]:
    """Let the model search the corpus until it has enough, or hits a bound.

    Returns (evidence, usage, stop_reason). `stop_reason` is one of
    'satisfied', 'step_cap', 'cost_cap', 'no_tools_available'.
    """
    max_steps = max_steps or config.VERIFY_MAX_STEPS
    max_cost = max_cost if max_cost is not None else config.VERIFY_MAX_COST_USD

    if client.dry_run:
        return [], Usage(), "dry_run"
    if getattr(client, "is_replay", False):
        # The gold-replay stand-in has no tool loop. Evidence gathering is
        # the one stage that genuinely needs a live model.
        return [], Usage(), "replay_no_tools"

    import anthropic  # local import: offline work must not need the SDK

    thread_brief = tools.get_promise_thread(store, thread.id)
    system = (
        "You gather evidence about whether a company kept a commitment. Search "
        "the corpus for later statements that bear on it -- progress updates, "
        "restatements, revisions, or conspicuous silence. Stop as soon as you "
        "have enough to decide, or can show nothing relevant exists. Finding "
        "nothing is a valid and useful result: say so plainly rather than "
        "searching indefinitely."
    )
    messages: List[Dict[str, Any]] = [
        {
            "role": "user",
            "content": (
                "Commitment to investigate:\n{}\n\nToday's date is {}. Gather "
                "evidence, then summarise what you found in a sentence.".format(
                    json.dumps(thread_brief, indent=2, default=str),
                    date.today().isoformat(),
                )
            ),
        }
    ]

    usage = Usage()
    evidence: List[Evidence] = []
    stop_reason = "step_cap"

    for _ in range(max_steps):
        if usage.cost_usd >= max_cost:
            stop_reason = "cost_cap"
            break

        try:
            response = client.client.messages.create(
                model=client.model,
                max_tokens=4000,
                system=system,
                tools=tools.tool_definitions(),
                messages=messages,
                # Automatic prompt caching: the breakpoint sits on the last
                # cacheable block and moves forward as the conversation grows,
                # so each turn pays full price only for what is new. Without
                # it, every one of up to 12 turns re-paid the whole history.
                cache_control={"type": "ephemeral"},
            )
        except anthropic.APIStatusError as e:
            raise LLMError("evidence gathering failed: {}".format(e))

        usage += usage_from_response(response, client.model)

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            stop_reason = "satisfied"
            break

        # Execute every tool call in the turn, and return ALL results in ONE
        # user message -- splitting them teaches the model to stop batching.
        results = []
        for block in response.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            out = tools.dispatch(store, thread.company, block.name, dict(block.input))
            is_error = "error" in out
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(out, default=str)[:6000],
                    "is_error": is_error,
                }
            )
            for hit in out.get("results", []) or []:
                evidence.append(
                    Evidence(
                        thread_id=thread.id,
                        source_id=hit.get("source_id"),
                        excerpt=hit.get("excerpt", "")[:800],
                        polarity="ambiguous",
                        found_by=block.name,
                        run_id=run_id,
                    )
                )
        messages.append({"role": "user", "content": results})

    # Deduplicate: the agent often re-finds the same passage.
    seen = set()
    unique = []
    for e in evidence:
        k = (e.source_id, e.excerpt[:120])
        if k not in seen:
            seen.add(k)
            unique.append(e)
    return unique, usage, stop_reason


# ----------------------------------------------------------------------
# Adjudication, with escalation
# ----------------------------------------------------------------------

def _advisor_tool() -> Dict[str, Any]:
    return {
        "type": config.ADVISOR_TOOL_TYPE,
        "name": "advisor",
        "model": config.ADVISOR_MODEL,
    }


def read_advisor_blocks(response) -> List[Dict[str, str]]:
    """Extract advisor advice, handling both payload shapes.

    The response block is always `advisor_tool_result`; its `content` is a
    discriminated union. Code that reads `.text` unconditionally gets nothing
    back from a Fable or Opus 5 advisor, because that payload is encrypted
    under `encrypted_content` instead.
    """
    out = []
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) != "advisor_tool_result":
            continue
        content = getattr(block, "content", None)
        ctype = getattr(content, "type", None)
        if ctype == "advisor_result":
            out.append({"kind": "plaintext", "text": getattr(content, "text", "")})
        elif ctype == "advisor_redacted_result":
            out.append(
                {
                    "kind": "redacted",
                    "text": "(advisor advice returned encrypted; consultation is "
                    "recorded but its content is not readable for the audit trail)",
                }
            )
        elif ctype == "advisor_tool_result_error":
            out.append(
                {"kind": "error", "text": getattr(content, "error_code", "unknown")}
            )
    return out


def _adjudication_question(
    store, thread: PromiseThread, evidence: List[Evidence], as_of: Optional[date] = None
) -> str:
    from .threading_ import silence_signal

    brief = tools.get_promise_thread(store, thread.id)
    silence = silence_signal(store, thread)
    return (
        "COMMITMENT\n{}\n\n"
        "SILENCE SIGNAL\n{}\n\n"
        "EVIDENCE GATHERED\n{}\n\n"
        "Today's date is {}. Adjudicate."
    ).format(
        json.dumps(brief, indent=2, default=str),
        json.dumps(silence, indent=2, default=str),
        json.dumps(
            [{"id": e.id, "excerpt": e.excerpt, "source_id": e.source_id} for e in evidence],
            indent=2,
        )
        or "(none)",
        (as_of or date.today()).isoformat(),
    )


def adjudicate(
    store,
    client: LLMClient,
    thread: PromiseThread,
    evidence: List[Evidence],
    run_id: str = "",
    allow_escalation: bool = True,
    as_of: Optional[date] = None,
) -> Tuple[Verdict, Usage]:
    """Decide a thread's status, escalating contested calls to the advisor.

    The executor adjudicates first. If it marks its own call `contested`, the
    same question goes to the executor again *with the advisor tool attached*,
    and that second, advisor-informed structured answer becomes the verdict.
    The first call is kept as `initial_status`, so the audit trail shows
    whether the advisor changed the outcome.

    Fail-safe by design: if the escalation errors, returns no structured
    answer, or the executor never actually consults the advisor, the initial
    verdict stands and the rationale says why. The outcome is never changed by
    anything short of a validated, advisor-informed answer.

    `as_of` pins "today" for the question and the verdict. The adjudication
    benchmark sets it so a stored case means the same thing every year.
    """
    question = _adjudication_question(store, thread, evidence, as_of=as_of)
    result = client.structured(
        system=prompts.adjudication_system(),
        question=question,
        output_model=Adjudication,
        prompt_version="adjudicate-v1",
    )
    first: Adjudication = result.parsed
    usage = result.usage

    status = getattr(first, "status", VerdictStatus.NO_EVIDENCE)
    confidence = getattr(first, "confidence", 0.0)
    rationale = getattr(first, "rationale", "") or ""
    escalated, initial_status, advisor_model = False, None, None

    if (
        allow_escalation
        and getattr(first, "contested", False)
        and not client.dry_run
        and not getattr(client, "is_replay", False)
    ):
        outcome = _adjudicate_with_advisor(client, question, first)
        usage += outcome["usage"]
        if outcome["final"] is not None:
            final: Adjudication = outcome["final"]
            escalated, initial_status, advisor_model = True, status, config.ADVISOR_MODEL
            status, confidence = final.status, final.confidence
            rationale = (
                "{}\n\n[Escalated. Executor's initial call: {} (confidence {}). "
                "Advisor {} advised: {}]"
            ).format(
                final.rationale, initial_status.value, getattr(first, "confidence", 0.0),
                config.ADVISOR_MODEL, outcome["advice"],
            )
        else:
            rationale += "\n\n[Escalation attempted; initial verdict kept: {}]".format(
                outcome["why_kept"]
            )

    verdict = Verdict(
        thread_id=thread.id,
        status=status,
        as_of=as_of or date.today(),
        rationale=rationale,
        evidence_ids=[e.id for e in evidence],
        adjudicated_by=client.model,
        escalated=escalated,
        initial_status=initial_status,
        advisor_model=advisor_model,
        confidence=confidence,
    )
    return verdict, usage


ESCALATION_SYSTEM = (
    "{}\n\nESCALATION\n\nYour first adjudication of this commitment was "
    "marked contested. Before answering, consult the advisor tool on the "
    "specific point of doubt. Then give your final adjudication. You may keep "
    "or change your first call; say which, and why, in the rationale."
)


def _adjudicate_with_advisor(client: LLMClient, question: str, first: Adjudication) -> Dict[str, Any]:
    """Re-adjudicate with the advisor tool attached; return the final answer.

    Returns a dict: `final` (a validated Adjudication, or None), `advice` (what
    the advisor said, or why it is unreadable), `usage` (priced per model --
    the advisor's tokens at the advisor's rates), and `why_kept` when `final`
    is None.
    """
    out: Dict[str, Any] = {"final": None, "advice": "", "usage": Usage(), "why_kept": ""}
    draft = "FIRST ADJUDICATION (contested)\nstatus: {}\nconfidence: {}\nrationale: {}".format(
        getattr(first, "status", VerdictStatus.NO_EVIDENCE).value,
        getattr(first, "confidence", 0.0),
        getattr(first, "rationale", ""),
    )
    try:
        response = client.client.beta.messages.parse(
            model=client.model,
            max_tokens=8000,
            betas=[config.ADVISOR_BETA],
            tools=[_advisor_tool()],
            system=ESCALATION_SYSTEM.format(prompts.adjudication_system()),
            messages=[{"role": "user", "content": "{}\n\n{}".format(question, draft)}],
            output_format=Adjudication,
        )
    except Exception as e:  # noqa: BLE001 - fail safe: keep the initial verdict
        # Includes the API rejecting advisor + structured output together,
        # which the documentation does not settle and has not been tested live.
        out["why_kept"] = "escalation call failed ({}: {})".format(type(e).__name__, str(e)[:200])
        return out

    out["usage"] = usage_from_response(response, client.model)
    blocks = read_advisor_blocks(response)
    out["advice"] = " ".join(b["text"] for b in blocks) or "(none)"

    if getattr(response, "stop_reason", None) == "refusal":
        out["why_kept"] = "escalation refused by the model"
    elif not any(b["kind"] in ("plaintext", "redacted") for b in blocks):
        out["why_kept"] = "the executor did not consult the advisor"
    elif getattr(response, "parsed_output", None) is None:
        out["why_kept"] = "no structured answer (stop_reason={})".format(
            getattr(response, "stop_reason", "?")
        )
    else:
        out["final"] = response.parsed_output
    return out


def verify_all(store, client: LLMClient, company: str, limit: Optional[int] = None) -> dict:
    """Gather evidence and adjudicate every unresolved thread."""
    threads = [t for t in store.list_threads(company) if store.latest_verdict(t.id) is None]
    if limit:
        threads = threads[:limit]

    stats = {"threads": 0, "verdicts": 0, "escalated": 0, "failed": 0, "cost_usd": 0.0}
    with RunLedger(store, "verify", model=client.model, prompt_version="adjudicate-v1") as run:
        for t in threads:
            stats["threads"] += 1
            try:
                evidence, eu, stop = gather_evidence(store, client, t, run_id=run.run_id)
                run.add(eu)
                store.add_evidence(evidence)

                verdict, au = adjudicate(store, client, t, evidence, run_id=run.run_id)
                run.add(au)
                store.put_verdict(verdict)

                stats["verdicts"] += 1
                if verdict.escalated:
                    stats["escalated"] += 1
                run.ok()
            except Exception as e:  # noqa: BLE001 - per-item isolation
                run.fail(note="{}: {}".format(t.id, type(e).__name__))
                stats["failed"] += 1
        stats["cost_usd"] = run.record.cost_usd
    return stats
