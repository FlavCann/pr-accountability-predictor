"""Command line entry point.

    python -m pr_predictor <command>

Commands map onto pipeline stages, plus the eval loop and the analyst review
queue. Every command that spends money reports what it spent.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from . import config, evals, outcomes, prompts
from .llm import LLMClient
from .pipeline import Pipeline
from .store import Store

DEFAULT_COMPANY = "Barry Callebaut"


def _client(args, store):
    if getattr(args, "replay", False):
        from .evals import GoldReplayClient

        return GoldReplayClient(store)
    return LLMClient(
        store=store,
        model=getattr(args, "model", None) or config.EXECUTOR_MODEL,
        dry_run=getattr(args, "dry_run", False),
    )


def _print_stage(r) -> None:
    mark = "--" if r.skipped else ("ok" if r.ok else "FAIL")
    print("[{}] {:<9} {}".format(mark, r.name, json.dumps(r.detail, default=str)))
    if r.message:
        print("     {}".format(r.message))


def cmd_run(args) -> int:
    with Store() as store:
        client = _client(args, store)
        pipe = Pipeline(store, client, args.company, prompt_version=args.prompt_version)
        stages = args.stages.split(",") if args.stages else None
        kwargs = {}
        if args.limit:
            kwargs["verify"] = {"limit": args.limit}

        results = pipe.run(stages, **kwargs)
        for r in results:
            _print_stage(r)
        print("\nTotal spend recorded in ledger: ${:.4f}".format(store.total_cost()))
        return 0 if all(r.ok for r in results) else 1


def _parse_harness(spec: Optional[str]) -> dict:
    """'chunk_words=600,chunk_overlap_words=50' -> {'chunk_words': 600, ...}"""
    out = {}
    for part in (spec or "").split(","):
        if part.strip():
            key, _, value = part.partition("=")
            out[key.strip()] = int(value)
    return out


def _refuse_replay(args) -> bool:
    if getattr(args, "replay", False):
        print(
            "Refusing to eval against --replay: it returns the gold labels, "
            "so the score would be meaningless.\n"
            "Set ANTHROPIC_API_KEY and run without --replay."
        )
        return True
    return False


def _summary(result: dict) -> dict:
    """The eval result without the bulky per-document counts."""
    out = {k: v for k, v in result.items() if k != "unjudged_extractions"}
    out["per_document"] = {
        url: {k: v for k, v in d.items() if k != "counts"}
        for url, d in result.get("per_document", {}).items()
    }
    return out


def cmd_eval(args) -> int:
    from .evals.judge import make_judge_client

    with Store() as store:
        if _refuse_replay(args):
            return 2
        versions = (args.prompt_version or prompts.DEFAULT_EXTRACTION_PROMPT).split(",")
        models = (args.models or args.model or config.EXECUTOR_MODEL).split(",")
        if len(versions) > 1 and len(models) > 1:
            print("A matrix varies one axis at a time: several models OR several prompts.")
            return 2
        splits = args.split.split(",") if args.split else None
        harness = _parse_harness(args.harness)

        results = []
        for model in models:
            for version in versions:
                client = LLMClient(store=store, model=model, dry_run=args.dry_run)
                judge = make_judge_client(store, model) if args.judge else None
                results.append(
                    evals.run_eval(
                        store, client, prompt_version=version,
                        baseline_truncate_words=args.baseline_truncate,
                        repeats=args.repeats, splits=splits, harness=harness,
                        judge_client=judge,
                    )
                )

        if len(results) > 1:
            print("{:<22} {:<16} {:>6} {:>15} {:>9} {:>9} {:>8}".format(
                "MODEL", "PROMPT", "F1", "95% CI", "FIDELITY", "UNJUDGED", "USD"))
            for r in results:
                print("{:<22} {:<16} {:>6} {:>15} {:>9} {:>9} {:>8.4f}".format(
                    r["model"][:22], r["prompt_version"][:16], r["overall"]["f1"],
                    str(r["f1_ci95"] or "underpowered"), r["overall"]["span_fidelity"],
                    r["overall"]["unjudged"], r["cost_usd"]))
            print("\nMatrix runs are recorded but never promoted. Re-run the winner alone to gate it.")
            return 0

        result = results[0]
        unjudged = result["unjudged_extractions"]
        print(json.dumps(_summary(result), indent=2, default=str))
        if unjudged:
            print("\nUNJUDGED -- label each with the label desk (python evals/label_server.py):")
            for u in unjudged:
                print('  [{}] {} x{}  "{}"'.format(
                    u["gold_file"], u["model_said"], u["seen_in_repeats"], u["text"][:140]))

        champion = evals.load_champion()
        passed, why = evals.gate(result, champion)
        print("\nGATE: {} -- {}".format("PASS" if passed else "BLOCK", why))
        if passed and args.promote:
            path = evals.save_champion(result)
            print("Promoted to champion: {}".format(path))
        return 0 if passed else 1


def cmd_eval_threads(args) -> int:
    from .evals.threads import run_threading_suite

    with Store() as store:
        client = None
        if args.tiebreak:
            if _refuse_replay(args):
                return 2
            client = LLMClient(store=store, model=args.model or config.EXECUTOR_MODEL)
        result = run_threading_suite(store, client=client)
        print(json.dumps(result, indent=2, default=str))
        return 0 if result.get("passed", True) else 1


def cmd_eval_adjudication(args) -> int:
    from .evals.adjudication import run_adjudication_suite

    with Store() as store:
        if _refuse_replay(args):
            return 2
        client = LLMClient(store=store, model=args.model or config.EXECUTOR_MODEL,
                           dry_run=args.dry_run)
        result = run_adjudication_suite(client, allow_escalation=not args.no_escalation)
        print(json.dumps(result, indent=2, default=str))
    return 0


def cmd_eval_predictions(args) -> int:
    from .evals.predictions import score_predictions

    with Store() as store:
        print(json.dumps(score_predictions(store), indent=2, default=str))
    return 0


def cmd_eval_judge_check(args) -> int:
    from .evals.judge import check_judge, make_judge_client

    with Store() as store:
        if _refuse_replay(args):
            return 2
        result = check_judge(make_judge_client(store, args.model or config.EXECUTOR_MODEL))
        print(json.dumps(result, indent=2, default=str))
        return 0 if result["validated"] else 1


def cmd_eval_agreement(args) -> int:
    from pathlib import Path

    from .evals.agreement import agreement

    print(json.dumps(agreement(Path(args.a), Path(args.b)), indent=2))
    return 0


def cmd_eval_history(args) -> int:
    with Store() as store:
        runs = store.list_eval_runs(limit=args.limit)
        if not runs:
            print("No eval runs recorded yet.")
            return 0
        print("{:<18} {:<20} {:<14} {:>3} {:>6} {:>15} {:>9} {:>8}".format(
            "RUN", "MODEL", "PROMPT", "N", "F1", "95% CI", "HARNESS", "USD"))
        for r in runs:
            h = (r.get("manifest") or {}).get("harness") or {}
            print("{:<18} {:<20} {:<14} {:>3} {:>6} {:>15} {:>9} {:>8.4f}".format(
                r["run_id"], r["model"][:20], r["prompt_version"][:14],
                (r.get("manifest") or {}).get("repeats", 1), r["overall"]["f1"],
                str(r.get("f1_ci95") or "underpowered"),
                "{}/{}".format(h.get("chunk_words", "?"), h.get("chunk_overlap_words", "?")),
                r.get("cost_usd", 0.0)))
    return 0


def cmd_report(args) -> int:
    with Store() as store:
        print(json.dumps(outcomes.report(store, args.company), indent=2, default=str))
    return 0


def cmd_promises(args) -> int:
    """Show extracted promises with their verbatim text and video deep links."""
    with Store() as store:
        promises = store.list_promises(company=args.company)
        if not promises:
            print("No promises. Run `python -m pr_predictor run` first.")
            return 1
        for p in promises[: args.limit]:
            src = store.get_source(p.source_id)
            text = p.verbatim(src.text) if src else "(source missing)"
            print("\n[{}] {} / {}".format(p.id, p.promise_type.value, p.hedge_level.value))
            print("  claim  : {}".format(p.normalized_claim))
            print('  verbatim: "{}"'.format(text[:200]))
            if src:
                print("  source : {} ({})".format(src.title, p.video_url(src.url)))
            if p.deadline_raw:
                print("  by     : {} [{}]".format(p.deadline_raw, p.deadline or "unresolved"))
        print("\n{} promise(s) total.".format(len(promises)))
    return 0


def cmd_threads(args) -> int:
    from .threading_ import silence_signal

    with Store() as store:
        for t in store.list_threads(args.company):
            v = store.latest_verdict(t.id)
            silence = silence_signal(store, t)
            print("\n[{}] {}".format(t.id, t.canonical_claim[:100]))
            print(
                "  observed {}x  first {}  last {}  deadline {}".format(
                    len(t.promise_ids), t.first_seen, t.last_seen, t.deadline
                )
            )
            if silence["silent"]:
                print(
                    "  SILENT: {} later source(s) with no mention".format(
                        silence["sources_since"]
                    )
                )
            if v:
                print("  verdict: {} (conf {})".format(v.final_status().value, v.confidence))
    return 0


def cmd_review(args) -> int:
    """Work the analyst review queue -- the human checkpoint."""
    from .taxonomy import VerdictStatus

    with Store() as store:
        pending = [
            v
            for t in store.list_threads(args.company)
            for v in store.list_verdicts(t.id)
            if not v.review_action
        ]
        if not pending:
            print("Review queue is empty.")
            return 0

        for v in pending[: args.limit]:
            thread = store.get_thread(v.thread_id)
            print("\n" + "=" * 70)
            print("Thread : {}".format(thread.canonical_claim if thread else v.thread_id))
            print("Verdict: {}  (confidence {})".format(v.status.value, v.confidence))
            print("Why    : {}".format(v.rationale[:600]))
            if v.escalated:
                print("(escalated to advisor)")

            if args.accept_all:
                store.add_review(v.id, "accepted", reviewer="bulk-accept")
                print("-> accepted")
                continue

            answer = input("accept / correct / reject / skip > ").strip().lower()
            if answer.startswith("a"):
                store.add_review(v.id, "accepted", reviewer="cli")
            elif answer.startswith("c"):
                print("statuses: {}".format(", ".join(s.value for s in VerdictStatus)))
                try:
                    corrected = VerdictStatus(input("correct status > ").strip())
                except ValueError:
                    print("unknown status, skipping")
                    continue
                note = input("why (stored as durable memory) > ").strip()
                store.add_review(
                    v.id, "corrected", reviewer="cli", corrected_status=corrected, note=note
                )
                if note and thread:
                    # An overturned verdict becomes retrievable context for
                    # future runs. This is concept 7's feedback loop.
                    store.memory_put(thread.company, "correction", thread.id[:24], note)
            elif answer.startswith("r"):
                store.add_review(v.id, "rejected", reviewer="cli")
    return 0


def cmd_ledger(args) -> int:
    with Store() as store:
        runs = store.list_runs(limit=args.limit)
        if not runs:
            print("No runs recorded yet.")
            return 0
        print(
            "{:<10} {:<18} {:>8} {:>8} {:>9} {:>9} {:>8}".format(
                "STAGE", "PROMPT", "IN", "OUT", "CACHE_R", "CACHE_W", "USD"
            )
        )
        for r in runs:
            print(
                "{:<10} {:<18} {:>8} {:>8} {:>9} {:>9} {:>8.4f}".format(
                    r.stage, r.prompt_version[:18], r.input_tokens, r.output_tokens,
                    r.cache_read_tokens, r.cache_write_tokens, r.cost_usd,
                )
            )
        print("\nTOTAL ${:.4f}".format(store.total_cost()))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pr_predictor", description=__doc__)
    p.add_argument("--company", default=DEFAULT_COMPANY)
    p.add_argument("--model", default=None, help="override the executor model")
    p.add_argument("--dry-run", action="store_true", help="walk the wiring, call nothing")
    p.add_argument(
        "--replay",
        action="store_true",
        help="replay the gold labels instead of calling a model -- exercises the "
        "real extraction path offline with no API key and no spend",
    )
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run the pipeline")
    r.add_argument("--stages", default=None, help="comma-separated subset")
    r.add_argument("--prompt-version", default=None)
    r.add_argument("--limit", type=int, default=None, help="cap threads verified")
    r.set_defaults(func=cmd_run)

    e = sub.add_parser("eval", help="score extraction against the gold set")
    e.add_argument("--prompt-version", default=None,
                   help="one version, or several comma-separated for a matrix")
    e.add_argument("--models", default=None,
                   help="several comma-separated executor models for a matrix")
    e.add_argument("--repeats", type=int, default=config.EVAL_REPEATS,
                   help="samples per config (default %(default)s)")
    e.add_argument("--split", default=None,
                   help="comma-separated splits to score (default: dev,test,fresh)")
    e.add_argument("--harness", default=None, metavar="K=V,...",
                   help="harness overrides, e.g. chunk_words=600,chunk_overlap_words=50")
    e.add_argument("--judge", action="store_true",
                   help="grade claim faithfulness with config.JUDGE_MODEL (reported, never gated)")
    e.add_argument("--promote", action="store_true", help="save as champion if it passes")
    e.add_argument(
        "--baseline-truncate",
        type=int,
        default=None,
        metavar="WORDS",
        help="extract-v1 only: reproduce the original pipeline's truncation "
        "(it sent the first 1000 words) as a measured baseline",
    )
    e.set_defaults(func=cmd_eval)

    et = sub.add_parser("eval-threads", help="score threading on gold promises")
    et.add_argument("--tiebreak", action="store_true",
                    help="include model tie-breaks on ambiguous pairs (spends money)")
    et.set_defaults(func=cmd_eval_threads)

    ea = sub.add_parser("eval-adjudication", help="score verdicts on labelled cases")
    ea.add_argument("--no-escalation", action="store_true", help="never consult the advisor")
    ea.set_defaults(func=cmd_eval_adjudication)

    ep = sub.add_parser("eval-predictions", help="Brier and calibration on resolved predictions")
    ep.set_defaults(func=cmd_eval_predictions)

    ej = sub.add_parser("eval-judge-check", help="score the judge against human labels")
    ej.set_defaults(func=cmd_eval_judge_check)

    eg = sub.add_parser("eval-agreement", help="inter-annotator agreement of two gold dirs")
    eg.add_argument("--a", required=True, help="first annotator's gold directory")
    eg.add_argument("--b", required=True, help="second annotator's gold directory")
    eg.set_defaults(func=cmd_eval_agreement)

    eh = sub.add_parser("eval-history", help="recorded eval runs, newest first")
    eh.add_argument("--limit", type=int, default=20)
    eh.set_defaults(func=cmd_eval_history)

    rep = sub.add_parser("report", help="product outcome metrics")
    rep.set_defaults(func=cmd_report)

    pr = sub.add_parser("promises", help="list extracted promises")
    pr.add_argument("--limit", type=int, default=50)
    pr.set_defaults(func=cmd_promises)

    th = sub.add_parser("threads", help="list commitment threads")
    th.set_defaults(func=cmd_threads)

    rv = sub.add_parser("review", help="work the analyst review queue")
    rv.add_argument("--limit", type=int, default=20)
    rv.add_argument("--accept-all", action="store_true")
    rv.set_defaults(func=cmd_review)

    lg = sub.add_parser("ledger", help="token and cost ledger")
    lg.add_argument("--limit", type=int, default=25)
    lg.set_defaults(func=cmd_ledger)

    return p


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
