"""The eval judge: is a `normalized_claim` faithful to its span?

Everything that can be checked deterministically is checked deterministically
elsewhere. What is left is semantic: the claim is shown to analysts and used
for threading, so it must say what the speaker said -- ASR garbles corrected,
nothing added, nothing dropped that changes the commitment.

Two rules keep the judge honest:

* **It is a third model.** `config.JUDGE_MODEL` must differ from the executor,
  so nothing grades its own homework.
* **It is checked before it is trusted.** `check_judge` scores it against
  human-labelled pairs in `evals/judge_calibration/`, and writes the outcome to
  `evals/judge_check.json`. Until that shows agreement above
  `config.JUDGE_MIN_AGREEMENT` on enough items, eval results mark the judge's
  number `validated: false`. The gate never uses it either way.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel

from .. import config
from ..domain import Promise
from .stats import cohen_kappa

JUDGE_PROMPT_VERSION = "judge-claim-v1"
CALIBRATION_DIR = "judge_calibration"
CHECK_FILE = "judge_check.json"

JUDGE_SYSTEM = """You check whether a one-sentence restatement of a corporate \
commitment is faithful to the verbatim transcript passage it was drawn from.

The passage is a machine-generated transcript and garbles names and numbers \
("calabroth" for Barry Callebaut, "kovac 19" for COVID-19, "Nets" for Swiss \
francs). A restatement that CORRECTS such garbling is faithful.

The restatement is UNFAITHFUL if it:
- adds a number, date, metric or scope the passage does not state;
- drops or changes a number, date or condition that the passage does state;
- makes the commitment firmer or softer than the passage ("aim to" -> "will");
- maps a fiscal-year phrase to a calendar date.

Judge only faithfulness, not whether the passage is a promise at all."""


class ClaimJudgement(BaseModel):
    faithful: bool
    issues: str = ""


def make_judge_client(store, executor_model: str):
    """The judge's LLM client. Refuses to let a model grade itself."""
    from ..llm import LLMClient

    if config.JUDGE_MODEL == executor_model:
        raise SystemExit(
            "The judge model ({}) is the executor under test. Set PRP_JUDGE_MODEL to a "
            "different model.".format(config.JUDGE_MODEL)
        )
    return LLMClient(store=store, model=config.JUDGE_MODEL)


def judge_claim(client, verbatim: str, claim: str):
    result = client.structured(
        system=JUDGE_SYSTEM,
        question="PASSAGE:\n\"{}\"\n\nRESTATEMENT:\n\"{}\"\n\nIs the restatement faithful?"
        .format(verbatim, claim),
        output_model=ClaimJudgement,
        prompt_version=JUDGE_PROMPT_VERSION,
        max_tokens=1000,
    )
    parsed = result.parsed
    return bool(getattr(parsed, "faithful", False)), getattr(parsed, "issues", ""), result.usage


def _check_path(gold_dir: Optional[Path]) -> Path:
    return Path(gold_dir or config.GOLD_DIR).parent / CHECK_FILE


def judge_validated(model: str, gold_dir: Optional[Path] = None) -> bool:
    """True when the last calibration check of this judge and prompt passed."""
    path = _check_path(gold_dir)
    if not path.exists():
        return False
    c = json.loads(path.read_text())
    return (
        c.get("judge_model") == model
        and c.get("prompt_version") == JUDGE_PROMPT_VERSION
        and c.get("n", 0) >= config.JUDGE_MIN_CALIBRATION_ITEMS
        and (c.get("accuracy") or 0) >= config.JUDGE_MIN_AGREEMENT
    )


def judge_pairs(
    client, pairs: Sequence[Tuple[str, Promise]], gold_dir: Optional[Path] = None
) -> Dict[str, object]:
    """Grade each (verbatim span, extracted promise) pair's claim."""
    faithful, cost, failures, errors = 0, 0.0, [], 0
    for verbatim, p in pairs:
        try:
            ok, issues, usage = judge_claim(client, verbatim, p.normalized_claim)
        except Exception as e:  # noqa: BLE001 - one failed judgement is not a failed eval
            errors += 1
            failures.append({"claim": p.normalized_claim, "issues": "judge error: " + type(e).__name__})
            continue
        cost += usage.cost_usd
        if ok:
            faithful += 1
        else:
            failures.append({"claim": p.normalized_claim, "verbatim": verbatim[:200], "issues": issues})
    judged = len(pairs) - errors
    return {
        "judge_model": client.model,
        "prompt_version": JUDGE_PROMPT_VERSION,
        "validated": judge_validated(client.model, gold_dir),
        "judged": judged,
        "errors": errors,
        "claim_faithfulness": round(faithful / judged, 3) if judged else None,
        "unfaithful": failures,
        "cost_usd": round(cost, 4),
    }


def load_calibration(gold_dir: Optional[Path] = None) -> List[Dict[str, object]]:
    """Human-labelled pairs: [{"verbatim", "claim", "faithful", "note"}, ...]."""
    directory = Path(gold_dir or config.GOLD_DIR).parent / CALIBRATION_DIR
    items: List[Dict[str, object]] = []
    for path in sorted(directory.glob("*.json")):
        items += json.loads(path.read_text())
    return items


def check_judge(client, gold_dir: Optional[Path] = None) -> Dict[str, object]:
    """Score the judge against human labels and record the outcome."""
    items = load_calibration(gold_dir)
    if not items:
        raise SystemExit(
            "No human-labelled pairs in evals/{}/. Add some before trusting the judge."
            .format(CALIBRATION_DIR)
        )
    human, model, disagreements, cost = [], [], [], 0.0
    for item in items:
        ok, issues, usage = judge_claim(client, str(item["verbatim"]), str(item["claim"]))
        cost += usage.cost_usd
        human.append(str(bool(item["faithful"])))
        model.append(str(ok))
        if ok != bool(item["faithful"]):
            disagreements.append({"claim": item["claim"], "human": item["faithful"], "judge_issues": issues})
    n = len(items)
    out = {
        "judge_model": client.model,
        "prompt_version": JUDGE_PROMPT_VERSION,
        "n": n,
        "accuracy": round(sum(h == m for h, m in zip(human, model)) / n, 3),
        "kappa": cohen_kappa(human, model),
        "disagreements": disagreements,
        "cost_usd": round(cost, 4),
    }
    out["validated"] = (
        n >= config.JUDGE_MIN_CALIBRATION_ITEMS and out["accuracy"] >= config.JUDGE_MIN_AGREEMENT
    )
    _check_path(gold_dir).write_text(json.dumps(out, indent=2) + "\n")
    return out
