"""Tests for the mechanics that carry real risk.

Deliberately not a mock of the model. These cover the parts that are wrong
silently: offset arithmetic, span validation, duplicate reconciliation,
timestamp attachment, threading, cost accounting and the eval gate.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from pr_predictor import config, evals, outcomes, threading_
from pr_predictor.context import chunk_text
from pr_predictor.domain import Promise, PromiseThread, Segment, Source, Verdict
from pr_predictor.extract import reconcile
from pr_predictor.ingest import build_source
from pr_predictor.store import Store
from pr_predictor.taxonomy import HedgeLevel, PromiseType, VerdictStatus


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "test.db")
    yield s
    s.close()


def _promise(start, end, **kw):
    defaults = dict(
        source_id="src_1",
        company="Acme",
        promise_type=PromiseType.QUANTIFIED_TARGET,
        hedge_level=HedgeLevel.FIRM,
        normalized_claim="claim",
        prompt_version="v2",
        model="test",
        run_id="r1",
    )
    defaults.update(kw)
    return Promise(start_char=start, end_char=end, **defaults)


# ----------------------------------------------------------------------
# Chunking: offsets must survive
# ----------------------------------------------------------------------

def test_chunk_offsets_are_exact():
    text = " ".join("word{}".format(i) for i in range(5000))
    for c in chunk_text(text, chunk_words=400, overlap_words=50):
        assert text[c.offset : c.offset + len(c.text)] == c.text


def test_chunking_covers_the_whole_document():
    """The original pipeline's defect was silent truncation. Guard against it."""
    text = " ".join("word{}".format(i) for i in range(5000))
    chunks = chunk_text(text, chunk_words=400, overlap_words=50)
    assert chunks[0].offset == 0
    assert chunks[-1].end == len(text)
    covered = set()
    for c in chunks:
        covered.update(range(c.offset, c.end))
    assert len(covered) == len(text)


def test_chunks_overlap_so_boundary_promises_survive():
    text = " ".join("word{}".format(i) for i in range(2000))
    chunks = chunk_text(text, chunk_words=400, overlap_words=50)
    assert len(chunks) > 1
    assert chunks[1].offset < chunks[0].end, "chunks must overlap"


def test_overlap_must_be_smaller_than_chunk():
    with pytest.raises(ValueError):
        chunk_text("a b c", chunk_words=10, overlap_words=10)


# ----------------------------------------------------------------------
# Span validation: the point of storing offsets
# ----------------------------------------------------------------------

def test_verify_span_accepts_real_text():
    text = "we will become net zero by 2050"
    assert _promise(8, 31).verify_span(text)


@pytest.mark.parametrize(
    "start,end",
    [(0, 0), (5, 3), (-1, 10), (0, 10_000)],
)
def test_verify_span_rejects_impossible_ranges(start, end):
    assert not _promise(start, end).verify_span("short text")


def test_verify_span_rejects_whitespace_only():
    assert not _promise(5, 8).verify_span("abcde   fgh")


# ----------------------------------------------------------------------
# Reconciliation of chunk-overlap duplicates
# ----------------------------------------------------------------------

def test_reconcile_collapses_overlapping_duplicates():
    a = _promise(100, 200, confidence=0.9)
    b = _promise(105, 195, confidence=0.4)
    assert len(reconcile([a, b])) == 1


def test_reconcile_keeps_distinct_promises():
    a = _promise(100, 200)
    b = _promise(900, 1000)
    assert len(reconcile([a, b])) == 2


def test_reconcile_prefers_higher_confidence():
    low = _promise(100, 200, confidence=0.2, normalized_claim="low")
    high = _promise(102, 198, confidence=0.95, normalized_claim="high")
    assert reconcile([low, high])[0].normalized_claim == "high"


# ----------------------------------------------------------------------
# Timestamps: the provenance the original pipeline destroyed
# ----------------------------------------------------------------------

def test_build_source_indexes_cues_by_char_and_time():
    cues = [("hello there", 0.0, 2.0), ("we will invest", 2.0, 4.5)]
    src, segs = build_source("https://youtu.be/abc", cues, company="Acme")
    assert src.text == "hello there we will invest"
    assert src.text[segs[1].start_char : segs[1].end_char] == "we will invest"
    assert segs[1].t_start == 2.0


def test_segment_lookup_resolves_a_char_position_to_a_timestamp(store):
    cues = [("hello there", 0.0, 2.0), ("we will invest", 2.0, 4.5)]
    src, segs = build_source("https://youtu.be/abc", cues, company="Acme")
    store.put_source(src)
    store.put_segments(segs)
    row = store.segment_at(src.id, src.text.index("we will"))
    assert row is not None and row["t_start"] == 2.0


def test_video_url_deep_links_to_the_moment():
    p = _promise(0, 5, t_start=137.4)
    url = p.video_url("https://www.youtube.com/watch?v=abc123")
    assert url.endswith("&t=137s")


def test_video_url_is_unchanged_without_a_timestamp():
    p = _promise(0, 5)
    assert p.video_url("https://www.youtube.com/watch?v=abc") == (
        "https://www.youtube.com/watch?v=abc"
    )


# ----------------------------------------------------------------------
# Store: append-only, versioned, idempotent
# ----------------------------------------------------------------------

def test_store_keeps_both_prompt_versions(store):
    store.add_promises([_promise(0, 10, prompt_version="extract-v1")])
    store.add_promises([_promise(0, 10, prompt_version="extract-v2")])
    assert sorted(store.prompt_versions()) == ["extract-v1", "extract-v2"]
    assert len(store.list_promises(prompt_version="extract-v1")) == 1


def test_result_cache_round_trips(store):
    store.cache_put("k", '{"a": 1}')
    assert store.cache_get("k") == '{"a": 1}'
    assert store.cache_get("missing") is None


def test_memory_is_retrievable_by_kind(store):
    store.memory_put("Acme", "asr", "calabroth", "Callebaut")
    store.memory_put("Acme", "fact", "fy_end", "31 August")
    assert len(store.memory_list("Acme", kind="asr")) == 1
    assert len(store.memory_list("Acme")) == 2


# ----------------------------------------------------------------------
# Threading
# ----------------------------------------------------------------------

def test_similarity_matches_rewordings():
    assert threading_.similarity(
        "Lift 500,000 farmers out of poverty by 2025",
        "Lift half a million farmers out of poverty by 2025",
    ) > threading_.MATCH_THRESHOLD


def test_revised_deadline_stays_a_separate_thread():
    """A target moved from 2025 to 2030 must not be silently merged away.

    Losing the revision is losing the story.
    """
    p = _promise(0, 10, normalized_claim="100% sustainable ingredients", deadline=date(2030, 1, 1))
    t = PromiseThread(
        company="Acme",
        canonical_claim="100% sustainable ingredients",
        promise_type=PromiseType.QUANTIFIED_TARGET,
        deadline=date(2025, 1, 1),
    )
    assert threading_._match_score(p, t) < threading_.MATCH_THRESHOLD


def test_different_promise_types_never_merge():
    p = _promise(0, 10, promise_type=PromiseType.SCHEDULING_ANNOUNCEMENT)
    t = PromiseThread(
        company="Acme", canonical_claim="claim", promise_type=PromiseType.QUANTIFIED_TARGET
    )
    assert threading_._match_score(p, t) == 0.0


def test_threads_group_restatements_across_sources(store):
    for i, (pub, claim) in enumerate(
        [
            (date(2021, 1, 1), "Lift 500,000 farmers out of poverty by 2025"),
            (date(2023, 1, 1), "Lift half a million farmers out of poverty by 2025"),
        ]
    ):
        src = Source(company="Acme", url="u{}".format(i), published=pub, text="x" * 50)
        store.put_source(src.finalise())
        store.add_promises([_promise(0, 10, source_id=src.id, normalized_claim=claim)])

    stats = threading_.build_threads(store, "Acme", prompt_version="v2")
    assert stats["threads"] == 1
    assert stats["restated"] == 1


def test_silence_signal_counts_later_sources(store):
    for i, pub in enumerate([date(2021, 1, 1), date(2022, 1, 1), date(2023, 1, 1)]):
        s = Source(company="Acme", url="u{}".format(i), published=pub, text="x")
        store.put_source(s.finalise())
    t = PromiseThread(
        company="Acme",
        canonical_claim="c",
        promise_type=PromiseType.QUANTIFIED_TARGET,
        last_seen=date(2021, 1, 1),
    )
    signal = threading_.silence_signal(store, t)
    assert signal["sources_since"] == 2
    assert signal["silent"] is True


# ----------------------------------------------------------------------
# Cost accounting
# ----------------------------------------------------------------------

def test_cost_matches_published_sonnet_pricing():
    # 1M input + 1M output on Sonnet 5 = $2 + $10.
    assert config.estimate_cost("claude-sonnet-5", 1_000_000, 1_000_000) == pytest.approx(12.0)


def test_cache_reads_are_a_tenth_of_input_price():
    full = config.estimate_cost("claude-sonnet-5", input_tokens=1_000_000)
    cached = config.estimate_cost("claude-sonnet-5", cache_read_tokens=1_000_000)
    assert cached == pytest.approx(full * 0.1)


def test_cache_writes_carry_the_five_minute_premium():
    full = config.estimate_cost("claude-sonnet-5", input_tokens=1_000_000)
    write = config.estimate_cost("claude-sonnet-5", cache_write_tokens=1_000_000)
    assert write == pytest.approx(full * 1.25)


def test_unknown_model_costs_nothing_rather_than_crashing():
    assert config.estimate_cost("not-a-model", 1000, 1000) == 0.0


# ----------------------------------------------------------------------
# Evals
# ----------------------------------------------------------------------

def test_span_match_requires_real_overlap():
    assert evals.spans_match((100, 200), (110, 190))
    assert not evals.spans_match((100, 200), (300, 400))


def test_scoring_counts_a_forbidden_extraction(store):
    text = "x" * 500
    gold = evals.GoldDoc(
        source_url="u",
        promises=[evals.GoldPromise(0, 100)],
        must_not_extract=[evals.GoldPromise(200, 300)],
    )
    predicted = [_promise(0, 100), _promise(200, 300)]
    m, _ = evals.score_document(predicted, gold, text)
    assert m.true_positives == 1
    assert m.false_positives == 1
    assert m.forbidden_extracted == 1


def test_scoring_counts_missed_promises(store):
    gold = evals.GoldDoc(
        source_url="u", promises=[evals.GoldPromise(0, 50), evals.GoldPromise(300, 350)]
    )
    m, _ = evals.score_document([_promise(0, 50)], gold, "x" * 500)
    assert m.false_negatives == 1
    assert m.recall == 0.5


def test_gate_blocks_a_span_fidelity_regression():
    champion = {"overall": {"f1": 0.5, "span_fidelity": 1.0, "forbidden_extracted": 0}}
    candidate = {"overall": {"f1": 0.99, "span_fidelity": 0.8, "forbidden_extracted": 0}}
    passed, why = evals.gate(candidate, champion)
    assert not passed and "fidelity" in why.lower()


def test_gate_blocks_more_forbidden_extractions():
    champion = {"overall": {"f1": 0.5, "span_fidelity": 1.0, "forbidden_extracted": 0}}
    candidate = {"overall": {"f1": 0.9, "span_fidelity": 1.0, "forbidden_extracted": 2}}
    assert not evals.gate(candidate, champion)[0]


def test_gate_passes_a_real_improvement():
    champion = {"overall": {"f1": 0.5, "span_fidelity": 1.0, "forbidden_extracted": 0}}
    candidate = {"overall": {"f1": 0.7, "span_fidelity": 1.0, "forbidden_extracted": 0}}
    assert evals.gate(candidate, champion)[0]


def test_first_run_becomes_the_baseline():
    assert evals.gate({"overall": {"f1": 0.1}}, None)[0]


# ----------------------------------------------------------------------
# The shipped gold set must stay valid against the shipped corpus
# ----------------------------------------------------------------------

def test_gold_set_offsets_resolve_against_the_real_transcripts(store):
    """Catches a gold set that has gone stale -- which would silently make
    every later measurement meaningless."""
    from pr_predictor.ingest import ingest_legacy_transcripts

    if not config.TRANSCRIPTS_DIR.exists():
        pytest.skip("transcripts/ not present")
    ingest_legacy_transcripts(store, "Barry Callebaut")

    docs = evals.load_gold()
    if not docs:
        pytest.skip("no gold set built yet -- run evals/build_gold.py")
    assert evals.validate_gold(store, docs) == []


def test_gold_replay_reproduces_every_label(store):
    """End-to-end over the real corpus: extraction, offset translation across
    chunk boundaries, span validation and reconciliation."""
    from pr_predictor import context, extract
    from pr_predictor.ingest import ingest_legacy_transcripts

    if not config.TRANSCRIPTS_DIR.exists() or not evals.load_gold():
        pytest.skip("corpus or gold set not present")

    ingest_legacy_transcripts(store, "Barry Callebaut")
    context.seed_memory(store, "Barry Callebaut")
    client = evals.GoldReplayClient(store)
    expected = sum(len(d.promises) for d in evals.load_gold())

    stats = extract.extract_all(store, client, company="Barry Callebaut")
    assert stats["promises"] == expected
    assert stats["failed"] == 0

    for p in store.list_promises(company="Barry Callebaut"):
        src = store.get_source(p.source_id)
        assert p.verify_span(src.text), "offset translation produced a bad span"


# ----------------------------------------------------------------------
# Predictions
# ----------------------------------------------------------------------

def test_brier_score_rewards_a_confident_correct_call():
    from pr_predictor.domain import Prediction

    p = Prediction(thread_id="t", p_kept=0.9, horizon=date(2025, 1, 1))
    p.resolve(VerdictStatus.KEPT)
    assert p.brier_score == pytest.approx(0.01)


def test_brier_score_punishes_a_confident_wrong_call():
    from pr_predictor.domain import Prediction

    p = Prediction(thread_id="t", p_kept=0.9, horizon=date(2025, 1, 1))
    p.resolve(VerdictStatus.MISSED)
    assert p.brier_score == pytest.approx(0.81)


def test_hedged_promises_are_predicted_less_likely_than_firm_ones(store):
    src = Source(company="Acme", url="u", published=date(2021, 1, 1), text="x" * 100)
    store.put_source(src.finalise())

    scores = {}
    for hedge in (HedgeLevel.FIRM, HedgeLevel.ASPIRATIONAL):
        p = _promise(0, 10, source_id=src.id, hedge_level=hedge, deadline=date(2030, 1, 1))
        store.add_promises([p])
        t = PromiseThread(
            company="Acme",
            canonical_claim="c",
            promise_type=PromiseType.QUANTIFIED_TARGET,
            deadline=date(2030, 1, 1),
            promise_ids=[p.id],
            last_seen=date(2021, 1, 1),
        )
        store.put_thread(t)
        scores[hedge] = outcomes.predict_thread(store, t).p_kept

    assert scores[HedgeLevel.FIRM] > scores[HedgeLevel.ASPIRATIONAL]


def test_corrected_verdict_overrides_the_model(store):
    v = Verdict(
        thread_id="t",
        status=VerdictStatus.KEPT,
        as_of=date.today(),
        rationale="r",
        corrected_status=VerdictStatus.MISSED,
        review_action="corrected",
    )
    assert v.final_status() == VerdictStatus.MISSED


# ======================================================================
# Regression tests for the defects fixed in 0.3 (REPORT.md, section 6)
# ======================================================================

# ---- locating anchors instead of trusting model offsets ---------------

def test_locate_tolerates_case_punctuation_and_spacing():
    from pr_predictor.extract import locate

    text = "so we will become net zero by 2050 latest, and we will also set a target"
    assert locate(text, "We will become Net-Zero", "by 2050 latest.") == (3, 41)


def test_locate_rejects_changed_words():
    """A model that 'corrects' the transcript has not quoted it."""
    from pr_predictor.extract import locate

    assert locate("barry calabroth will invest", "Barry Callebaut will invest") is None


def test_locate_end_anchor_must_follow_the_start():
    from pr_predictor.extract import locate

    text = "target of 2030 is set. " + "x " * 20 + "we aim to reach net zero"
    assert locate(text, "we aim to reach", "target of 2030") is None


def test_locate_end_anchor_must_be_within_the_window():
    from pr_predictor.extract import locate

    text = "we will invest " + "filler " * 400 + "by 2030"
    assert locate(text, "we will invest", "by 2030", max_len=500) is None


def test_extraction_counts_unlocatable_quotes_as_uncitable(store):
    """Uncitable extractions are reported, never silently dropped."""
    from pr_predictor import extract
    from pr_predictor.llm import Result, Usage

    src = Source(company="Acme", url="u", text="we will invest 500 million by 2025").finalise()
    store.put_source(src)

    class Fake:
        model, dry_run = "fake", False

        def structured(self, **kw):
            item = dict(promise_type="quantified_target", hedge_level="firm",
                        normalized_claim="c")
            return Result(parsed=extract.ExtractionResult(promises=[
                dict(item, quote_start="we will invest", quote_end="by 2025"),
                dict(item, quote_start="we promise the moon", quote_end="tomorrow"),
            ]), usage=Usage())

    promises, _, uncitable = extract.extract_source(store, Fake(), src, stable_context="")
    assert len(promises) == 1 and uncitable == 1
    assert promises[0].verbatim(src.text) == "we will invest 500 million by 2025"


def test_uncitable_extractions_count_against_span_fidelity():
    gold = evals.GoldDoc(source_url="u", promises=[evals.GoldPromise(0, 10)])
    m, _ = evals.score_document([_promise(0, 10)], gold, "x" * 50, uncitable=1)
    assert m.span_fidelity == 0.5


# ---- eval contamination ------------------------------------------------

def test_prompt_versions_are_pinned():
    """Editing the taxonomy changes a prompt's text. Bump the version."""
    from pr_predictor import prompts

    for version, expected in prompts.PINNED.items():
        assert prompts.get(version).fingerprint == expected, (
            "{} changed content. Register a new version instead of editing it.".format(version)
        )
    assert set(prompts.PINNED) == {v for v in prompts.REGISTRY if v.startswith("extract-")}


def test_no_prompt_example_comes_from_a_gold_transcript(store):
    """The contamination that shipped in extract-v2. Never again."""
    from pr_predictor import prompts
    from pr_predictor.ingest import ingest_legacy_transcripts

    docs = evals.load_gold()
    if not config.TRANSCRIPTS_DIR.exists() or not docs:
        pytest.skip("corpus or gold set not present")
    ingest_legacy_transcripts(store, "Barry Callebaut")
    for version in prompts.REGISTRY:
        assert evals.find_leakage(store, docs, prompts.get(version).system) == []


def test_find_leakage_catches_a_contaminated_prompt(store):
    src = Source(company="Acme", url="u", text="we will lift half a million farmers out of poverty").finalise()
    store.put_source(src)
    doc = evals.GoldDoc(source_url="u", promises=[evals.GoldPromise(0, len(src.text))])
    leaks = evals.find_leakage(store, [doc], "Example: we will lift half a million farmers out of poverty")
    assert any("appears in the prompt" in l for l in leaks)


# ---- partial labelling -------------------------------------------------

def test_unlabelled_extractions_are_unjudged_not_wrong():
    gold = evals.GoldDoc(source_url="u", promises=[evals.GoldPromise(0, 50)])
    m, unjudged = evals.score_document([_promise(0, 50), _promise(300, 350)], gold, "x" * 500)
    assert (m.true_positives, m.false_positives, m.unjudged) == (1, 0, 1)
    assert m.precision == 1.0 and len(unjudged) == 1


def test_unlabelled_extractions_are_wrong_when_labelling_is_exhaustive():
    gold = evals.GoldDoc(source_url="u", promises=[evals.GoldPromise(0, 50)], exhaustive=True)
    m, _ = evals.score_document([_promise(0, 50), _promise(300, 350)], gold, "x" * 500)
    assert (m.false_positives, m.unjudged) == (1, 0)


def test_gate_refuses_to_promote_with_unjudged_extractions():
    result = {"overall": {"f1": 0.9, "span_fidelity": 1.0, "forbidden_extracted": 0, "unjudged": 3}}
    passed, why = evals.gate(result, None)
    assert not passed and "unjudged" in why


def test_gate_refuses_to_compare_across_gold_sets():
    base = {"f1": 0.5, "span_fidelity": 1.0, "forbidden_extracted": 0, "unjudged": 0}
    champion = {"overall": base, "gold_fingerprint": "old", "prompt_version": "extract-v1"}
    result = {"overall": dict(base, f1=0.9), "gold_fingerprint": "new"}
    passed, why = evals.gate(result, champion)
    assert not passed and "different gold set" in why


# ---- append-only records -----------------------------------------------

def test_records_are_insert_only():
    """Evidence and decisions are never overwritten. Derived state may be."""
    import re
    from pr_predictor import store as store_mod

    src = Path(store_mod.__file__).read_text()
    for table in store_mod.RECORD_TABLES:
        assert not re.search(r"INSERT OR REPLACE INTO {}\b".format(table), src), table
        assert not re.search(r"UPDATE {}\b".format(table), src), table


def test_duplicate_record_ids_are_rejected(store):
    import sqlite3

    p = _promise(0, 10)
    store.add_promises([p])
    with pytest.raises(sqlite3.IntegrityError):
        store.add_promises([p])


def test_reviews_append_and_never_rewrite_the_verdict(store):
    v = Verdict(thread_id="t", status=VerdictStatus.KEPT, as_of=date.today(), rationale="r")
    store.put_verdict(v)
    store.add_review(v.id, "accepted", reviewer="a")
    store.add_review(v.id, "corrected", reviewer="b", corrected_status=VerdictStatus.MISSED)

    latest = store.get_verdict(v.id)
    assert latest.status == VerdictStatus.KEPT, "the model's original call is preserved"
    assert latest.final_status() == VerdictStatus.MISSED
    assert [r["reviewer"] for r in store.list_reviews(v.id)] == ["a", "b"]


def test_superseded_source_is_hidden_but_still_citable(store):
    old = Source(company="Acme", url="u", text="old text").finalise()
    new = Source(company="Acme", url="u", text="new text").finalise()
    store.put_source(old)
    store.put_source(new)
    store.supersede_source(old.id, new.id)

    assert [s.id for s in store.list_sources("Acme")] == [new.id]
    assert store.source_by_url("u").id == new.id
    assert store.get_source(old.id) is not None, "old citations must still resolve"


# ---- idempotency --------------------------------------------------------

def test_a_promise_can_only_join_one_thread(store):
    import sqlite3

    store.add_member("thr_a", "prm_1")
    with pytest.raises(sqlite3.IntegrityError):
        store.add_member("thr_b", "prm_1")


def test_linking_twice_changes_nothing(store):
    """The bug: a second link run put every promise into its thread again."""
    src = Source(company="Acme", url="u", published=date(2021, 1, 1), text="x" * 50).finalise()
    store.put_source(src)
    store.add_promises([
        _promise(0, 10, source_id=src.id, normalized_claim="Lift 500,000 farmers out of poverty"),
        _promise(20, 30, source_id=src.id, normalized_claim="Net zero by 2050"),
    ])
    first = threading_.build_threads(store, "Acme", prompt_version="v2")
    second = threading_.build_threads(store, "Acme", prompt_version="v2")

    assert first["linked"] == 2 and second["linked"] == 0
    assert second["restated"] == first["restated"] == 0
    for t in store.list_threads("Acme"):
        assert len(t.promise_ids) == len(set(t.promise_ids))


def test_linking_never_mixes_extraction_versions(store):
    src = Source(company="Acme", url="u", published=date(2021, 1, 1), text="x" * 50).finalise()
    store.put_source(src)
    claim = "Lift 500,000 farmers out of poverty by 2025"
    store.add_promises([
        _promise(0, 10, source_id=src.id, normalized_claim=claim, prompt_version="extract-v1"),
        _promise(0, 10, source_id=src.id, normalized_claim=claim, prompt_version="extract-v3"),
    ])
    threading_.build_threads(store, "Acme", prompt_version="extract-v3")
    threads = store.list_threads("Acme")
    assert len(threads) == 1 and threads[0].restatement_count() == 0


def test_whole_pipeline_is_idempotent(tmp_path, monkeypatch):
    """Run every offline stage twice over the real corpus; nothing may change."""
    from pr_predictor.pipeline import Pipeline

    if not config.TRANSCRIPTS_DIR.exists() or not evals.load_gold():
        pytest.skip("corpus or gold set not present")

    s = Store(tmp_path / "idem.db")
    stages = ["seed", "ingest", "extract", "link", "verify", "predict"]

    def counts():
        c = s.conn
        return {t: c.execute("SELECT COUNT(*) FROM {}".format(t)).fetchone()[0]
                for t in ("sources", "promises", "thread_members", "threads",
                          "verdicts", "predictions", "extractions")}

    Pipeline(s, evals.GoldReplayClient(s), "Barry Callebaut").run(stages)
    once = counts()
    Pipeline(s, evals.GoldReplayClient(s), "Barry Callebaut").run(stages)
    assert counts() == once
    assert once["promises"] > 0
    s.close()


# ---- the link stage is metered like every other paid stage ------------

class _TieBreakClient:
    """Stands in for the model on thread tie-breaks only, with a known cost."""

    model = "claude-sonnet-5"
    dry_run = False

    def __init__(self, answer: bool, cost: float = 0.0015, fail: bool = False):
        self.answer, self.cost, self.fail, self.calls = answer, cost, fail, 0

    def structured(self, **kw):
        from pr_predictor.llm import LLMError, Result, Usage

        self.calls += 1
        if self.fail:
            raise LLMError("boom")
        return Result(
            parsed=threading_.ThreadMatch(same_commitment=self.answer),
            usage=Usage(input_tokens=400, output_tokens=20, cost_usd=self.cost),
        )


def _ambiguous_pair(store):
    """Two statements of the same commitment in different words (score 0.15-0.45)."""
    src_a = Source(company="Acme", url="a", published=date(2022, 11, 1), text="x" * 50).finalise()
    src_b = Source(company="Acme", url="b", published=date(2022, 11, 2), text="x" * 50).finalise()
    store.put_source(src_a)
    store.put_source(src_b)
    kw = dict(promise_type=PromiseType.SCHEDULING_ANNOUNCEMENT)
    store.add_promises([
        _promise(0, 10, source_id=src_a.id, normalized_claim="Give an update on new midterm guidance in Q1", **kw),
        _promise(0, 10, source_id=src_b.id, normalized_claim="Share new midterm guidance for three years after the first quarter", **kw),
    ])


def test_tie_break_cost_reaches_the_ledger(store):
    _ambiguous_pair(store)
    client = _TieBreakClient(answer=True)
    stats = threading_.build_threads(store, "Acme", prompt_version="v2", client=client)

    assert client.calls == 1 and stats["escalated_merges"] == 1
    runs = [r for r in store.list_runs() if r.stage == "link"]
    assert len(runs) == 1
    assert runs[0].cost_usd == pytest.approx(0.0015)
    assert runs[0].model == "claude-sonnet-5"
    assert runs[0].input_tokens == 400
    assert stats["cost_usd"] == pytest.approx(0.0015)


def test_memberships_carry_the_link_run_id(store):
    _ambiguous_pair(store)
    threading_.build_threads(store, "Acme", prompt_version="v2", client=_TieBreakClient(True))
    run_id = [r for r in store.list_runs() if r.stage == "link"][0].id
    ids = {r[0] for r in store.conn.execute("SELECT run_id FROM thread_members")}
    assert ids == {run_id}


def test_failed_tie_break_links_separately_and_is_noted(store):
    _ambiguous_pair(store)
    stats = threading_.build_threads(store, "Acme", prompt_version="v2", client=_TieBreakClient(True, fail=True))
    run = [r for r in store.list_runs() if r.stage == "link"][0]
    assert stats["tiebreak_errors"] == 1 and stats["threads"] == 2
    assert run.items_failed == 0 and run.items_ok == 2
    assert "tie-break" in run.notes


def test_link_with_nothing_to_do_writes_no_ledger_row(store):
    threading_.build_threads(store, "Acme", prompt_version="v2")
    assert [r for r in store.list_runs() if r.stage == "link"] == []


# ======================================================================
# verify: advisor decides escalated verdicts, per-model pricing, caching
# ======================================================================

from types import SimpleNamespace as _NS


def _it(kind, model, i, o):
    """One entry of usage.iterations, shaped like the SDK's iteration types."""
    return _NS(type=kind, model=model, input_tokens=i, output_tokens=o,
               cache_read_input_tokens=0, cache_creation_input_tokens=0)


def test_advisor_tokens_are_priced_at_the_advisor_rate():
    from pr_predictor.llm import usage_from_response

    resp = _NS(usage=_NS(
        input_tokens=3000, output_tokens=400,
        cache_read_input_tokens=0, cache_creation_input_tokens=0,
        iterations=[
            _it("message", "claude-sonnet-5", 1000, 100),
            _it("advisor_message", "claude-opus-4-8", 2000, 300),
        ],
    ))
    u = usage_from_response(resp, "claude-sonnet-5")
    expected = (config.estimate_cost("claude-sonnet-5", 1000, 100)
                + config.estimate_cost("claude-opus-4-8", 2000, 300))
    assert u.cost_usd == pytest.approx(expected)
    # The old code priced everything at the executor's rate -- visibly less.
    assert u.cost_usd > config.estimate_cost("claude-sonnet-5", 3000, 400)


def test_usage_without_iterations_uses_the_default_model():
    from pr_predictor.llm import usage_from_response

    resp = _NS(usage=_NS(input_tokens=1000, output_tokens=100,
                         cache_read_input_tokens=0, cache_creation_input_tokens=0))
    u = usage_from_response(resp, "claude-sonnet-5")
    assert u.cost_usd == pytest.approx(config.estimate_cost("claude-sonnet-5", 1000, 100))


def test_unpriced_model_is_flagged_not_silently_zero(store):
    from pr_predictor.llm import RunLedger, usage_from_response

    resp = _NS(usage=_NS(input_tokens=0, output_tokens=0,
                         cache_read_input_tokens=0, cache_creation_input_tokens=0,
                         iterations=[_it("advisor_message", "claude-future-9", 500, 50)]))
    u = usage_from_response(resp, "claude-sonnet-5")
    assert u.unpriced_models == ["claude-future-9"]
    with RunLedger(store, "verify") as run:
        run.add(u)
    assert "UNPRICED MODEL claude-future-9" in store.list_runs()[0].notes


class _VerifyClient:
    """Fakes the executor's first call and the advisor-assisted second call."""

    model = "claude-sonnet-5"
    dry_run = False

    def __init__(self, first, second=None, advisor_kind="advisor_result", raises=None):
        from pr_predictor.verify import Adjudication

        self._first = Adjudication(**first)
        self._second = Adjudication(**second) if second else None
        self._advisor_kind, self._raises = advisor_kind, raises
        self.parse_kwargs = None
        self.client = _NS(beta=_NS(messages=_NS(parse=self._parse)))

    def structured(self, **kw):
        from pr_predictor.llm import Result, Usage

        return Result(parsed=self._first, usage=Usage(cost_usd=0.001))

    def _parse(self, **kw):
        self.parse_kwargs = kw
        if self._raises:
            raise self._raises
        content = []
        if self._advisor_kind:
            payload = (_NS(type="advisor_result", text="The deadline has passed; treat as missed.")
                       if self._advisor_kind == "advisor_result"
                       else _NS(type="advisor_redacted_result", encrypted_content="xyz"))
            content.append(_NS(type="advisor_tool_result", content=payload))
        return _NS(
            content=content, stop_reason="end_turn", parsed_output=self._second,
            usage=_NS(input_tokens=0, output_tokens=0, cache_read_input_tokens=0,
                      cache_creation_input_tokens=0,
                      iterations=[_it("message", "claude-sonnet-5", 1000, 100),
                                  _it("advisor_message", "claude-opus-4-8", 2000, 300)]),
        )


def _thread(store):
    src = Source(company="Acme", url="u", published=date(2021, 1, 1), text="x" * 50).finalise()
    store.put_source(src)
    t = PromiseThread(company="Acme", canonical_claim="Net zero by 2050",
                      promise_type=PromiseType.QUANTIFIED_TARGET)
    store.put_thread(t)
    return t


_CONTESTED = dict(status="kept", rationale="probably met", confidence=0.5, contested=True)


def test_advisor_informed_answer_becomes_the_verdict(store):
    """The bug: the advisor's advice was appended as text and ignored."""
    from pr_predictor import verify

    client = _VerifyClient(_CONTESTED, second=dict(status="missed", rationale="deadline passed", confidence=0.8))
    v, usage = verify.adjudicate(store, client, _thread(store), [])

    assert v.status == VerdictStatus.MISSED, "the advisor-informed answer decides"
    assert v.initial_status == VerdictStatus.KEPT, "the executor's first call is kept for audit"
    assert v.escalated and v.advisor_model == config.ADVISOR_MODEL
    assert "treat as missed" in v.rationale
    assert client.parse_kwargs["betas"] == [config.ADVISOR_BETA]
    assert client.parse_kwargs["tools"][0]["model"] == config.ADVISOR_MODEL
    # Escalation cost includes the advisor at Opus rates, on top of the first call.
    assert usage.cost_usd == pytest.approx(
        0.001 + config.estimate_cost("claude-sonnet-5", 1000, 100)
        + config.estimate_cost("claude-opus-4-8", 2000, 300))


def test_uncontested_verdict_never_calls_the_advisor(store):
    from pr_predictor import verify

    client = _VerifyClient(dict(_CONTESTED, contested=False))
    v, _ = verify.adjudicate(store, client, _thread(store), [])
    assert client.parse_kwargs is None and not v.escalated and v.status == VerdictStatus.KEPT


@pytest.mark.parametrize("kwargs,reason", [
    (dict(raises=RuntimeError("400 advisor + output_format")), "escalation call failed"),
    (dict(advisor_kind=None, second=dict(status="missed", rationale="r")), "did not consult the advisor"),
    (dict(second=None), "no structured answer"),
])
def test_failed_escalation_keeps_the_initial_verdict(store, kwargs, reason):
    """Fail safe: nothing short of a validated, advisor-informed answer changes the outcome."""
    from pr_predictor import verify

    v, _ = verify.adjudicate(store, _VerifyClient(_CONTESTED, **kwargs), _thread(store), [])
    assert v.status == VerdictStatus.KEPT and not v.escalated
    assert reason in v.rationale


def test_encrypted_advice_is_recorded_as_unreadable(store):
    from pr_predictor import verify

    client = _VerifyClient(_CONTESTED, second=dict(status="missed", rationale="r"),
                           advisor_kind="advisor_redacted_result")
    v, _ = verify.adjudicate(store, client, _thread(store), [])
    assert v.status == VerdictStatus.MISSED
    assert "encrypted" in v.rationale


def test_evidence_loop_requests_prompt_caching(store):
    """The bug: up to 12 turns each re-paid the whole conversation at full price."""
    from pr_predictor import verify

    seen = {}

    def create(**kw):
        seen.update(kw)
        return _NS(content=[], stop_reason="end_turn",
                   usage=_NS(input_tokens=500, output_tokens=50, cache_read_input_tokens=400,
                             cache_creation_input_tokens=0))

    client = _NS(model="claude-sonnet-5", dry_run=False, client=_NS(messages=_NS(create=create)))
    _, usage, stop = verify.gather_evidence(store, client, _thread(store))
    assert seen["cache_control"] == {"type": "ephemeral"}
    assert stop == "satisfied" and usage.cache_read_tokens == 400


# ---- tools: search hits can be followed up with get_source_excerpt ----

def test_search_hit_position_chains_into_get_source_excerpt(store):
    """The bug: search returned no position, so the agent could not read around a hit."""
    from pr_predictor import tools

    text = ("intro " * 200) + "we will lift 500,000 farmers out of poverty by 2025. " + ("outro " * 200)
    src = Source(company="Acme", url="u", published=date(2023, 1, 1), text=text).finalise()
    store.put_source(src)

    hit = tools.search_corpus(store, "Acme", "farmers poverty")["results"][0]
    assert text[hit["match_char"]:].startswith("farmers")
    assert hit["excerpt"] == text[hit["start_char"]:hit["start_char"] + len(hit["excerpt"])]

    wider = tools.get_source_excerpt(store, hit["source_id"], start_char=hit["start_char"], length=1200)
    assert "we will lift 500,000 farmers" in wider["excerpt"]
    assert wider["start_char"] == hit["start_char"]


def test_negative_offset_does_not_read_from_the_end(store):
    from pr_predictor import tools

    src = Source(company="Acme", url="u", text="BEGINNING middle END").finalise()
    store.put_source(src)
    out = tools.get_source_excerpt(store, src.id, start_char=-3, length=9)
    assert out["excerpt"] == "BEGINNING" and out["start_char"] == 0


def test_tool_descriptions_tell_the_model_how_to_chain():
    from pr_predictor import tools

    d = {t["name"]: t["description"] for t in tools.tool_definitions()}
    assert "start_char" in d["search_corpus"] and "start_char" in d["get_source_excerpt"]
