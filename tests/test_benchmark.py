"""Tests for the benchmark itself: splits, statistics, the gate's refusals and
the per-stage suites. A benchmark that silently mis-scores is worse than none,
so these cover the ways it could lie."""

from __future__ import annotations

import json
from datetime import date, datetime

import pytest

from pr_predictor import config, evals
from pr_predictor.domain import Prediction, Promise, Source
from pr_predictor.evals import stats
from pr_predictor.evals.threads import pairwise, revision_merges
from pr_predictor.llm import Result, Usage
from pr_predictor.store import Store
from pr_predictor.taxonomy import HedgeLevel, PromiseType, VerdictStatus


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "test.db")
    yield s
    s.close()


def _promise(start, end, **kw):
    defaults = dict(
        source_id="src_1", company="Acme", promise_type=PromiseType.QUANTIFIED_TARGET,
        hedge_level=HedgeLevel.FIRM, normalized_claim="claim", prompt_version="v3",
        model="test", run_id="r1",
    )
    defaults.update(kw)
    return Promise(start_char=start, end_char=end, **defaults)


def _counts(tp, fp=0, fn=0):
    return {"true_positives": tp, "false_positives": fp, "false_negatives": fn}


def _result(f1_rows, manifest=None, **overall):
    """A minimal eval result with per-document counts."""
    base = {"f1": stats.micro_f1(f1_rows), "span_fidelity": 1.0,
            "forbidden_extracted": 0, "unjudged": 0}
    base.update(overall)
    return {
        "gold_fingerprint": "g",
        "manifest": manifest or {},
        "overall": base,
        "per_document": {
            "u{}".format(i): {"split": "dev", "counts": c} for i, c in enumerate(f1_rows)
        },
    }


# ---- scoring: ambiguity and deadlines ------------------------------------

def test_ambiguous_gold_is_neither_right_nor_wrong():
    gold = evals.GoldDoc(
        source_url="u",
        promises=[evals.GoldPromise(0, 50), evals.GoldPromise(100, 150, ambiguous=True),
                  evals.GoldPromise(200, 250, ambiguous=True)],
    )
    m, _ = evals.score_document([_promise(0, 50), _promise(100, 150)], gold, "x" * 300)
    assert (m.true_positives, m.false_positives, m.false_negatives) == (1, 0, 0)
    assert m.ambiguous_matched == 1


def test_a_fiscal_deadline_guessed_into_a_calendar_date_is_wrong():
    gold = evals.GoldDoc(source_url="u", promises=[
        evals.GoldPromise(0, 50, deadline_resolved=None, deadline_labelled=True),
    ])
    guessed = _promise(0, 50, deadline=date(2023, 12, 31))
    left_null = _promise(0, 50)
    assert evals.score_document([guessed], gold, "x" * 60)[0].deadline_correct == 0
    assert evals.score_document([left_null], gold, "x" * 60)[0].deadline_correct == 1


def test_unlabelled_deadlines_are_not_scored():
    gold = evals.GoldDoc(source_url="u", promises=[evals.GoldPromise(0, 50)])
    m, _ = evals.score_document([_promise(0, 50, deadline=date(2030, 1, 1))], gold, "x" * 60)
    assert m.deadline_total == 0


def test_boundary_tags_follow_the_chunking():
    from pr_predictor.context import Chunk

    chunks = [Chunk(0, 0, "x" * 100), Chunk(1, 80, "x" * 100)]
    straddles_but_covered = evals.GoldPromise(90, 110)
    never_whole = evals.GoldPromise(50, 150)
    assert evals.boundary_tags(straddles_but_covered, chunks) == ["chunk_boundary"]
    assert evals.boundary_tags(never_whole, chunks) == ["chunk_boundary", "unseen_whole"]
    assert evals.boundary_tags(evals.GoldPromise(10, 20), chunks) == []


# ---- gold integrity -----------------------------------------------------

def test_a_retyped_transcript_invalidates_the_gold_set(store):
    src = Source(company="Acme", url="u", text="we will lift farmers out of poverty").finalise()
    store.put_source(src)
    doc = evals.GoldDoc(source_url="u", promises=[evals.GoldPromise(0, 10)],
                        source_text_hash="not-the-hash")
    problems = evals.validate_gold(store, [doc])
    assert problems and "source text changed" in problems[0]


def test_fingerprint_changes_when_a_label_type_changes():
    a = evals.GoldDoc(source_url="u", promises=[evals.GoldPromise(0, 10, "quantified_target")])
    b = evals.GoldDoc(source_url="u", promises=[evals.GoldPromise(0, 10, "procedural_commitment")])
    assert evals.gold_fingerprint([a]) != evals.gold_fingerprint([b])


def test_examples_split_is_exempt_but_a_scored_unlabelled_transcript_is_not(store, tmp_path):
    from pr_predictor.taxonomy import EXCLUSIONS

    example = EXCLUSIONS[0][1]
    for vid in ("EXAMPLES01", "TEST000001"):
        store.put_source(Source(company="Acme", url="https://www.youtube.com/watch?v=" + vid,
                                text="preamble " + example).finalise())
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()
    (tmp_path / "splits.json").write_text(json.dumps(
        {"splits": {"examples": ["EXAMPLES01"], "test": ["TEST000001"]}}))
    doc = evals.GoldDoc(source_url="https://www.youtube.com/watch?v=EXAMPLES01",
                        split="examples", path=gold_dir / "EXAMPLES01.json")

    leaks = evals.find_leakage(store, [doc], "prompt text")
    assert len(leaks) == 1 and "TEST000001" in leaks[0]


# ---- statistics ----------------------------------------------------------

def test_no_interval_from_a_single_document():
    assert stats.bootstrap_ci([_counts(3, 1, 1)]) is None


def test_paired_interval_excludes_zero_for_a_consistent_gain():
    base = [_counts(5, 2, 3)] * 8
    better = [_counts(7, 1, 1)] * 8
    delta, lo, hi = stats.paired_delta_ci(base, better)
    assert delta > 0 and lo > 0


def test_paired_interval_spans_zero_for_mixed_results():
    base = [_counts(5, 0, 5), _counts(5, 0, 5)] * 3
    mixed = [_counts(9, 0, 1), _counts(1, 0, 9)] * 3
    _, lo, hi = stats.paired_delta_ci(base, mixed)
    assert lo < 0 < hi or (lo <= 0 <= hi)


def test_kappa_is_zero_for_chance_agreement_and_one_for_perfect():
    assert stats.cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
    assert stats.cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == 0.0


# ---- the gate ------------------------------------------------------------

M = {"executor_model": "m1", "prompt_version": "p1", "prompt_fingerprint": "f1",
     "harness": {"chunk_words": 1800}, "pipeline_code": "c1", "scoring_code": "s1"}


def test_gate_refuses_to_attribute_a_two_axis_change():
    rows = [_counts(5, 1, 1)] * 2
    champion = _result(rows, manifest=M)
    candidate = _result([_counts(9, 0, 0)] * 2, manifest=dict(M, executor_model="m2", prompt_fingerprint="f2"))
    passed, why = evals.gate(candidate, champion)
    assert not passed and "model and prompt" in why


def test_gate_refuses_results_from_different_scoring_code():
    rows = [_counts(5, 1, 1)] * 2
    passed, why = evals.gate(_result(rows, manifest=dict(M, scoring_code="s2")),
                             _result(rows, manifest=M))
    assert not passed and "scoring code" in why


def test_gate_labels_a_small_comparison_underpowered():
    champion = _result([_counts(5, 1, 1)] * 2, manifest=M)
    candidate = _result([_counts(6, 0, 0)] * 2, manifest=dict(M, prompt_fingerprint="f2"))
    passed, why = evals.gate(candidate, champion)
    assert passed and "UNDERPOWERED" in why


def test_gate_promotes_a_gain_the_interval_supports():
    n = config.EVAL_MIN_DOCS_FOR_CI
    champion = _result([_counts(5, 2, 3)] * n, manifest=M)
    candidate = _result([_counts(8, 1, 0)] * n, manifest=dict(M, prompt_fingerprint="f2"))
    passed, why = evals.gate(candidate, champion)
    assert passed and "95% CI" in why


def test_gate_blocks_a_gain_that_one_document_carries():
    n = config.EVAL_MIN_DOCS_FOR_CI + 1
    old = [_counts(5, 0, 5)] * n
    new = [_counts(10, 0, 0)] + [_counts(4, 0, 6)] * (n - 1)
    champion = _result(old, manifest=M)
    candidate = _result(new, manifest=dict(M, prompt_fingerprint="f2"))
    passed, why = evals.gate(candidate, champion)
    assert not passed and "not established" in why


# ---- the runner ----------------------------------------------------------

def test_harness_overrides_apply_and_restore():
    before = config.CHUNK_WORDS
    with evals.harness_overrides({"chunk_words": 300}):
        assert config.CHUNK_WORDS == 300
    assert config.CHUNK_WORDS == before
    with pytest.raises(SystemExit):
        with evals.harness_overrides({"temperature": 1}):
            pass


def test_repeats_after_the_first_bypass_the_result_cache():
    from pr_predictor.evals.runner import _FreshSamples

    seen = {}

    class Client:
        model = "m"

        def structured(self, **kw):
            seen.update(kw)

    _FreshSamples(Client()).structured(system="s")
    assert seen["use_cache"] is False


def test_eval_runs_are_recorded_and_kept(store):
    store.add_eval_run({"run_id": "evl_1", "prompt_version": "p", "model": "m",
                        "gold_fingerprint": "g", "overall": {"f1": 0.5}})
    store.add_eval_run({"run_id": "evl_2", "prompt_version": "p", "model": "m",
                        "gold_fingerprint": "g", "overall": {"f1": 0.6}})
    assert [r["run_id"] for r in store.list_eval_runs()] == ["evl_2", "evl_1"]


# ---- threading suite -----------------------------------------------------

def test_pairwise_threading_scores():
    gold = {"a": "t1", "b": "t1", "c": "t2"}
    assert pairwise({"a": "x", "b": "x", "c": "y"}, gold)["pair_f1"] == 1.0
    over = pairwise({"a": "x", "b": "x", "c": "x"}, gold)
    assert over["false_merges"] == 2 and over["pair_recall"] == 1.0


def test_a_merged_revision_is_caught():
    gold = {"orig": "target-2025", "rev": "target-2030"}
    revises = {"rev": "target-2025"}
    assert revision_merges({"orig": "x", "rev": "x"}, gold, revises) == [("rev", "orig")]
    assert revision_merges({"orig": "x", "rev": "y"}, gold, revises) == []


# ---- adjudication suite --------------------------------------------------

def test_adjudication_case_is_judged_as_of_its_pinned_date(tmp_path, monkeypatch):
    from pr_predictor.evals.adjudication import run_adjudication_suite
    from pr_predictor.verify import Adjudication

    (tmp_path / "adjudication").mkdir()
    (tmp_path / "adjudication" / "c.json").write_text(json.dumps({
        "id": "c1", "company": "Acme", "as_of": "2020-01-01",
        "thread": {"canonical_claim": "Net zero by 2050", "promise_type": "quantified_target",
                   "deadline": "2050-12-31", "last_seen": "2019-01-01"},
        "evidence": [], "expected_status": "too_early",
    }))
    questions = []

    class Client:
        model = "m"
        dry_run = False

        def structured(self, **kw):
            questions.append(kw["question"])
            return Result(parsed=Adjudication(status=VerdictStatus.TOO_EARLY, rationale="r"),
                          usage=Usage(cost_usd=0.01))

    out = run_adjudication_suite(Client(), gold_dir=tmp_path / "gold")
    assert out["accuracy"] == 1.0 and out["cases"] == 1
    assert "Today's date is 2020-01-01" in questions[0]


# ---- predictions suite ---------------------------------------------------

def test_predictions_made_after_the_deadline_are_not_scored(store):
    from pr_predictor.evals.predictions import score_predictions

    horizon = date(2024, 1, 1)
    early = Prediction(thread_id="t1", p_kept=0.8, horizon=horizon,
                       created_at=datetime(2023, 1, 1))
    late = Prediction(thread_id="t2", p_kept=0.99, horizon=horizon,
                      created_at=datetime(2024, 6, 1))
    for p in (early, late):
        store.put_prediction(p)
        store.resolve_prediction(p.id, VerdictStatus.KEPT, (p.p_kept - 1) ** 2)
    out = score_predictions(store)
    assert out["scored"] == 1 and out["excluded_made_after_horizon"] == 1
    assert out["mean_brier"] == round((0.8 - 1) ** 2, 4)


# ---- judge ---------------------------------------------------------------

def test_judge_refuses_to_grade_its_own_model(store):
    from pr_predictor.evals.judge import make_judge_client

    with pytest.raises(SystemExit):
        make_judge_client(store, config.JUDGE_MODEL)


def test_judge_is_unvalidated_without_a_calibration_check(tmp_path):
    from pr_predictor.evals.judge import judge_validated

    assert judge_validated(config.JUDGE_MODEL, gold_dir=tmp_path / "gold") is False


# ---- inter-annotator agreement -------------------------------------------

def test_agreement_between_two_annotators(tmp_path):
    from pr_predictor.evals.agreement import agreement

    def write(name, promises):
        d = tmp_path / name
        d.mkdir()
        (d / "doc.json").write_text(json.dumps({"source_url": "u", "promises": promises}))
        return d

    a = write("a", [{"start_char": 0, "end_char": 50, "promise_type": "quantified_target", "hedge_level": "firm"},
                    {"start_char": 100, "end_char": 150, "promise_type": "procedural_commitment", "hedge_level": "firm"}])
    b = write("b", [{"start_char": 5, "end_char": 50, "promise_type": "quantified_target", "hedge_level": "firm"}])
    out = agreement(a, b)
    assert out["span_f1"] == round(2 * 1 / 3, 3)
    assert out["type_kappa"] == 1.0


# ---- over the real corpus ------------------------------------------------

def test_the_gate_never_promotes_a_config_over_itself():
    r = _result([_counts(5, 1, 1)] * config.EVAL_MIN_DOCS_FOR_CI, manifest=M)
    assert not evals.gate(r, r)[0]


def test_replay_benchmark_is_perfect_and_threading_runs(store):
    """Sanity floor for the whole benchmark: gold replayed through the real
    extraction path must score 1.0 on every metric it can be scored on."""
    from pr_predictor import context
    from pr_predictor.evals.threads import run_threading_suite
    from pr_predictor.ingest import ingest_legacy_transcripts

    if not config.TRANSCRIPTS_DIR.exists() or not evals.load_gold():
        pytest.skip("corpus or gold set not present")
    ingest_legacy_transcripts(store, "Barry Callebaut")
    context.seed_memory(store, "Barry Callebaut")

    r = evals.run_eval(store, evals.GoldReplayClient(store), repeats=2)
    o = r["overall"]
    assert (o["f1"], o["span_fidelity"], o["deadline_accuracy"], o["unjudged"]) == (1.0, 1.0, 1.0, 0)
    assert r["repeat_f1"] == [1.0, 1.0] and r["underpowered"]
    assert store.list_eval_runs()[0]["run_id"] == r["run_id"]

    t = run_threading_suite(store)
    assert t["passed"] and t["false_merges"] == 0


# ---- held-out documents: masks and labelling in progress ----------------

def test_extractions_touching_a_masked_example_are_not_scored():
    gold = evals.GoldDoc(source_url="u", promises=[evals.GoldPromise(0, 50)], exhaustive=True,
                         masked=[evals.GoldPromise(100, 150)])
    m, unjudged = evals.score_document([_promise(0, 50), _promise(140, 200)], gold, "x" * 300)
    assert (m.true_positives, m.false_positives, m.masked_ignored) == (1, 0, 1)
    assert not unjudged


def test_a_prompt_example_is_allowed_only_inside_a_mask(store):
    from pr_predictor.taxonomy import EXCLUSIONS

    example = EXCLUSIONS[0][1]
    text = "preamble " + example + " and more text afterwards"
    src = Source(company="Acme", url="u", text=text).finalise()
    store.put_source(src)
    start = text.index(example)
    unmasked = evals.GoldDoc(source_url="u", split="test")
    masked = evals.GoldDoc(source_url="u", split="test",
                           masked=[evals.GoldPromise(start, start + len(example))])
    assert evals.find_leakage(store, [unmasked], "prompt")
    assert evals.find_leakage(store, [masked], "prompt") == []


def test_a_document_still_being_labelled_is_not_scored(store, tmp_path):
    text = "we will do the thing by 2030. " * 5
    for url in ("done", "wip"):
        store.put_source(Source(company="Acme", url=url, text=text).finalise())
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()
    (gold_dir / "a.json").write_text(json.dumps({"source_url": "done", "split": "dev",
        "promises": [{"start_char": 0, "end_char": 28}]}))
    (gold_dir / "b.json").write_text(json.dumps({"source_url": "wip", "split": "test",
        "labelling_complete": False}))
    r = evals.run_eval(store, evals.GoldReplayClient(store, gold_dir=gold_dir),
                       gold_dir=gold_dir, record=False)
    assert r["documents"] == 1 and r["not_scored_labelling_incomplete"] == ["b.json"]


def test_label_phrases_survive_line_breaks_from_the_wrapped_view(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "build_gold", config.ROOT / "evals" / "build_gold.py")
    bg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bg)
    text = "intro we will reach net zero by 2050 outro"
    assert bg.locate(text, "we will reach\nnet zero by 2050", tmp_path / "t.txt") == (6, 36)
    with pytest.raises(SystemExit):
        bg.locate(text, "we will reach net-zero", tmp_path / "t.txt")


# ---- the label desk -------------------------------------------------------

def _label_server():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "label_server", config.ROOT / "evals" / "label_server.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_label_desk_accepts_a_valid_label_file():
    ls = _label_server()
    text = "intro we will reach net zero by 2050 and more. we will be your hosts today. outro"
    labels = {"labelling_complete": True, "labelled_by": "me", "masked": [],
              "promises": [["we will reach\nnet zero by 2050", "quantified_target", "firm", "n",
                            {"deadline_resolved": None, "slices": ["qa"]}]],
              "must_not_extract": [["we will be your hosts today", "logistics"]]}
    assert ls.validate(labels, text) == []


def test_label_desk_rejects_what_build_gold_would():
    ls = _label_server()
    text = "we will do it. we will do it again. secret example sentence here. end"
    labels = {"labelling_complete": True, "labelled_by": "",
              "masked": [["secret example sentence here", "prompt example"]],
              "promises": [["we will do it", "quantified_target", "firm", ""],
                           ["example sentence", "procedural_commitment", "sort of", ""],
                           ["do it again", "quantified_target", "firm", "", {"colour": "blue"}]],
              "must_not_extract": [["nowhere to be found", "x"]]}
    errors = " | ".join(ls.validate(labels, text))
    for expected in ("appears 2 times", "overlaps a masked", "unknown hedge",
                     "unknown field", "not in the transcript", "labelled by"):
        assert expected in errors, expected
