"""The accountability profile and the thread audit trail behind the dashboard."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from pr_predictor import outcomes
from pr_predictor.domain import Evidence, Promise, PromiseThread, Source, Verdict
from pr_predictor.store import Store
from pr_predictor.taxonomy import HedgeLevel, PromiseType, VerdictStatus


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "test.db")
    yield s
    s.close()


def _seed(store):
    """Two threads, one adjudicated (and later corrected by an analyst)."""
    text = "we will open the plant by 2022. we aim to cut costs."
    src = Source(company="Acme", url="https://www.youtube.com/watch?v=x",
                 title="FY21", published=date(2021, 1, 1), text=text).finalise()
    store.put_source(src)
    lineage = dict(source_id=src.id, company="Acme", prompt_version="v", model="m", run_id="r")
    p1 = Promise(start_char=0, end_char=31, normalized_claim="Open the plant by 2022",
                 promise_type=PromiseType.SCHEDULING_ANNOUNCEMENT,
                 hedge_level=HedgeLevel.FIRM, t_start=65.0, **lineage)
    p2 = Promise(start_char=32, end_char=52, normalized_claim="Cut costs",
                 promise_type=PromiseType.DIRECTIONAL_GUIDANCE,
                 hedge_level=HedgeLevel.INTENDED, **lineage)
    store.add_promises([p1, p2])

    threads = []
    for p in (p1, p2):
        t = PromiseThread(company="Acme", canonical_claim=p.normalized_claim,
                          promise_type=p.promise_type, first_seen=src.published,
                          last_seen=src.published)
        store.put_thread(t)
        store.add_member(t.id, p.id)
        threads.append(t)

    cited = Evidence(thread_id=threads[0].id, source_id=src.id, excerpt="opened",
                     polarity="supports", found_by="search_corpus")
    other = Evidence(thread_id=threads[0].id, source_id=src.id, excerpt="unrelated",
                     polarity="ambiguous", found_by="search_corpus")
    store.add_evidence([cited, other])
    v = Verdict(thread_id=threads[0].id, status=VerdictStatus.MISSED, as_of=date(2023, 1, 1),
                rationale="See {}.".format(cited.id), evidence_ids=[cited.id, other.id],
                confidence=0.8)
    store.put_verdict(v)
    store.add_review(v.id, "corrected", reviewer="analyst",
                     corrected_status=VerdictStatus.KEPT, note="it opened")
    return threads, v, cited, other


def test_profile_counts_latest_corrected_verdict_and_unadjudicated(store):
    _seed(store)
    prof = outcomes.accountability_profile(store, "Acme")

    assert prof["coverage"]["threads"] == 2
    assert prof["coverage"]["adjudicated_threads"] == 1
    assert prof["coverage"]["unadjudicated_threads"] == 1
    # The analyst's correction wins over the model's call.
    assert prof["verdicts"]["kept"] == 1
    assert prof["verdicts"]["missed"] == 0
    assert prof["record"] == {"resolved": 1, "kept": 1, "kept_share": 1.0}
    assert prof["by_type"]["directional_guidance"] == {"threads": 1, "unadjudicated": 1}
    assert prof["hedge_mix"]["firm"] == 1 and prof["hedge_mix"]["intended"] == 1
    assert prof["reviewed"] == 1
    # Hard-coded for now, and must say so wherever it is shown.
    assert prof["rating"]["level"] == "pr_med_low"
    assert prof["rating"]["basis"] == "placeholder"


def test_rating_scale_runs_best_to_worst_and_every_level_has_an_image():
    from pr_predictor import rating

    levels = [s["level"] for s in rating.LEVELS]
    assert levels == ["pr_high", "pr_med_high", "pr_med_low", "pr_low"]
    assert rating.rate({})["level"] in levels
    img = Path(__file__).resolve().parents[1] / "src/pr_predictor/static/img/rating"
    assert all((img / (lvl + ".png")).is_file() for lvl in levels)


def test_thread_audit_carries_provenance_and_cited_evidence(store):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from pr_predictor import api

    threads, v, cited, other = _seed(store)

    def same_file():
        # Endpoints run in a worker thread; sqlite connections cannot cross threads.
        s = Store(store.path)
        try:
            yield s
        finally:
            s.close()

    api.app.dependency_overrides[api.get_store] = same_file
    try:
        client = TestClient(api.app)
        audit = client.get("/threads/{}/audit".format(threads[0].id)).json()
        profile = client.get("/companies/Acme/profile")
        missing = client.get("/companies/Nobody/profile")
        page = client.get("/dashboard")
        assets = [client.get("/dashboard/" + path).status_code for path in (
            "js/main.js", "js/components/rating.js", "css/rating.css", "img/rating/pr_low.png")]
    finally:
        api.app.dependency_overrides.clear()

    [promise] = audit["promises"]
    assert promise["provenance"]["verbatim"] == "we will open the plant by 2022."
    assert promise["provenance"]["video_url"].endswith("t=65s")

    [verdict] = audit["verdicts"]
    assert verdict["status"] == "kept"
    assert verdict["model_status"] == "missed"
    flags = {e["id"]: e["cited"] for e in verdict["evidence"]}
    assert flags == {cited.id: True, other.id: False}
    assert verdict["evidence"][0]["source"]["title"] == "FY21"

    assert profile.status_code == 200
    assert missing.status_code == 404
    assert page.status_code == 200 and "<title>" in page.text
    assert assets == [200, 200, 200, 200]


def test_eval_endpoints_list_runs_oldest_first_and_flag_the_champion(store, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from pr_predictor import api, evals

    base = {"model": "m", "gold_fingerprint": "g", "documents": 2, "repeat_f1": [0.5, 0.6],
            "manifest": {"harness": {"chunk_words": 1800, "chunk_overlap_words": 250}, "repeats": 2},
            "per_document": {"u": {"title": "FY21", "split": "dev", "f1": 0.5, "counts": {"true_positives": 1}}},
            "slices": {"all": {"recall": 0.5}},
            "unjudged_extractions": [{"text": "verbatim transcript words"}]}
    store.add_eval_run(dict(base, run_id="evl_old", prompt_version="extract-v1", overall={"f1": 0.4}))
    store.add_eval_run(dict(base, run_id="evl_new", prompt_version="extract-v3", overall={"f1": 0.7}))
    monkeypatch.setattr(evals, "load_champion", lambda gold_dir=None: {"run_id": "evl_new"})

    def same_file():
        s = Store(store.path)
        try:
            yield s
        finally:
            s.close()

    api.app.dependency_overrides[api.get_store] = same_file
    try:
        client = TestClient(api.app)
        listing = client.get("/evals").json()
        one = client.get("/evals/evl_new").json()
        missing = client.get("/evals/evl_nope")
        page = client.get("/dashboard").text
    finally:
        api.app.dependency_overrides.clear()

    assert [r["run_id"] for r in listing["runs"]] == ["evl_old", "evl_new"]
    assert [r["is_champion"] for r in listing["runs"]] == [False, True]
    assert listing["runs"][0]["recorded_at"] and listing["runs"][1]["unjudged"] == 1
    assert one["per_document"][0]["title"] == "FY21" and "counts" not in one["per_document"][0]
    assert one["manifest"]["repeats"] == 2
    assert "verbatim transcript words" not in str(listing) + str(one), "transcript text must not leak"
    assert missing.status_code == 404
    assert "Model evaluation" in page


def test_company_info_logos_exist_and_unknown_companies_have_none():
    from pr_predictor import company_info

    static = Path(__file__).resolve().parents[1] / "src/pr_predictor/static"
    for name, info in company_info.ABOUT.items():
        assert info["description"].strip(), name
        if info.get("logo"):
            assert (static / info["logo"]).is_file(), name
    assert company_info.about("Nobody") is None
