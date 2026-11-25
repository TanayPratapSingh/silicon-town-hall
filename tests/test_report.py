"""Analyst, pilot scorer and baselines. Offline: synthetic tags, synthetic residents, FakeLLM, fake retriever."""

import math

import pytest

from townhall.ingest import codebook as cb
from townhall.llm import FakeLLM
from townhall.report import analyst, baselines, score
from townhall.town import pilot

COMMENT_FIELDS = {"resident_id", "neighborhood", "text", "cited_pages", "stance", "simulated"}
CODEBOOK = [{"code": c, "topic": t, "definition": f"Example about {t}."}
            for c, t in [("4", "Community Grid"), ("6", "Transportation"), ("7-5", "Noise")]]


def person(rid, hood, race="White alone, not Hispanic or Latino", age=40, income=30_000):
    return {"resident_id": rid, "neighborhood": hood, "race_ethnicity": race, "age": age, "income": income,
            "tenure": "Rented", "disability": "Without a disability", "commute": "Bus", "profile": f"I am {rid}."}


def comment(rid, hood, codes, pages=(("FAKE_policy.pdf", 1),)):
    return {"resident_id": rid, "neighborhood": hood, "text": f"text by {rid}", "cited_pages": [list(p) for p in pages],
            "stance": "mixed", "simulated": True, "codes": codes}


def test_coverage_and_js_on_known_inputs():
    real, same = {"4": 2, "6": 2}, {"4": 5, "6": 5}
    assert score.coverage(real, {"4": 1}) == 0.5
    assert score.coverage(real, same) == 1.0
    assert score.coverage({}, same) is None
    assert score.js_distance(real, same) == pytest.approx(0.0)
    assert score.js_distance({"4": 1}, {"6": 1}) == pytest.approx(1.0)  # no topic in common
    # p = (1, 0), q = (1/2, 1/2): JSD = 1/2 (log2(4/3)) + 1/2 (1/2 log2(2/3) + 1/2 log2 2) = 0.3113
    expected = math.sqrt(0.5 * math.log2(4 / 3) + 0.25 * math.log2(2 / 3) + 0.25)
    assert score.js_distance({"4": 1}, {"4": 1, "6": 1}) == pytest.approx(expected)
    assert expected == pytest.approx(0.5579, abs=1e-4)
    assert score.js_distance({"4": 1}, {}) is None


def test_score_lists_concerns_only_real_residents_raised():
    out = score.score([["4", "6"], ["6"], ["7-5"], []], [["4"], [], ["4", "8"]], CODEBOOK)
    assert out["topic_coverage"] == pytest.approx(1 / 3, abs=1e-4)
    assert [c["code"] for c in out["concerns_only_real"]] == ["6", "7-5"]
    assert out["concerns_only_real"][0] == {"code": "6", "topic": "Transportation", "real_comments": 2}
    assert [c["code"] for c in out["topics_only_simulated"]] == ["8"]
    assert out["real_comments_with_no_topic"] == 1 and out["simulated_comments_with_no_topic"] == 1
    assert 0 < out["topic_mix_gap"] < 1


def test_bands():
    assert analyst._band(18, analyst.AGE_BANDS) == "18 to 29" and analyst._band(65, analyst.AGE_BANDS) == "65 and over"
    assert analyst._band(-500, analyst.INCOME_BANDS) == "under $25,000"
    assert analyst._band(100_000, analyst.INCOME_BANDS) == "$100,000 and over"


def test_who_to_reach_flags_missing_underrepresented_single_and_silent_groups():
    residents = [person("a", "Eastwood"), person("b", "Eastwood"), person("c", "Eastwood"), person("d", "Southside")]
    tagged = [comment("a", "Eastwood", ["4"]), comment("b", "Eastwood", ["6"]), comment("c", "Eastwood", []),
              comment("d", "Southside", [])]
    population = {"neighborhood": {"Eastwood": 0.5, "Southside": 0.2, "Near Westside": 0.3},
                  "race_ethnicity": {"White alone, not Hispanic or Latino": 0.6, "Hispanic or Latino": 0.4}}
    report = analyst.build_report(tagged, residents, CODEBOOK, population, "synthetic shares")
    reach = {(e["dimension"], e["group"]): e for e in report["who_to_reach"]}
    assert reach[("neighborhood", "Near Westside")]["simulated_residents"] == 0
    assert reach[("neighborhood", "Near Westside")]["expected_residents"] == 1.2
    assert "no simulated resident" in reach[("neighborhood", "Near Westside")]["reasons"][0]
    assert "no simulated resident" in reach[("race_ethnicity", "Hispanic or Latino")]["reasons"][0]
    southside = reach[("neighborhood", "Southside")]["reasons"]
    assert any("only one" in r for r in southside) and any("no codebook topic" in r for r in southside)
    assert ("neighborhood", "Eastwood") not in reach  # three residents, expected two, topics raised
    assert report["who_to_reach"][0]["group"] == "Hispanic or Latino"  # largest shortfall first: 1.6 expected, 0 seen
    assert report["by_neighborhood"]["Eastwood"] == {"residents": 3, "comments": 3, "no_topic_comments": 1,
                                                     "topics": {"4": 1, "6": 1}}
    assert report["by_group"]["age_band"]["30 to 44"]["residents"] == 4

    under = analyst.who_to_reach(analyst._stats(tagged, residents), 10, {"neighborhood": {"Southside": 0.5}})
    assert any("underrepresented: 1 simulated resident where its population share implies 5.0" in r
               for e in under if e["group"] == "Southside" for r in e["reasons"])


def test_who_to_reach_without_population_uses_only_simulated_evidence():
    residents = [person("a", "Eastwood")]
    report = analyst.build_report([comment("a", "Eastwood", ["4"])], residents, CODEBOOK)
    assert report["population_source"] == "population shares not available"
    assert all(e["expected_residents"] is None for e in report["who_to_reach"])
    assert {e["dimension"] for e in report["who_to_reach"]} == {*analyst.DIMENSIONS, "neighborhood"}


def test_every_sample_is_labeled_simulated_and_cites_pages():
    residents = [person("a", "Eastwood"), person("b", "Southside")]
    tagged = [comment("a", "Eastwood", ["4", "6"]), comment("b", "Southside", ["4"], pages=())]
    report = analyst.build_report(tagged, residents, CODEBOOK)
    assert report["label"].startswith("SIMULATED")
    assert [s["topic"] for s in report["samples"]] == ["4", "6"]  # b cites no page, so it is never a sample
    for s in report["samples"]:
        assert s["simulated"] is True and s["label"] == analyst.SIMULATED_LABEL and s["cited_pages"]
    with pytest.raises(ValueError):
        analyst.build_report([{**tagged[0], "simulated": False}], residents, CODEBOOK)


def test_analyst_tags_with_the_answer_key_tagger():
    llm = FakeLLM(reply='{"codes": ["7-5", "4", "not a code"]}')
    tagged = analyst.tag_comments([comment("a", "Eastwood", None)], CODEBOOK, llm)
    assert tagged[0]["codes"] == ["4", "7-5"]  # codebook order, unknown codes dropped by tagger.parse
    assert len(llm.calls) == 1 and llm.calls[0][0]["content"].startswith("You label public comments")


def test_baselines_write_well_formed_simulated_comments():
    llm = FakeLLM(responder=baselines.fake_responder)
    residents = [person("a", "Eastwood"), person("b", "Southside")]
    none = baselines.no_personas(pilot.POLICY, 3, pilot.fake_retriever, llm)
    skip = baselines.skip_meeting(pilot.POLICY, residents, pilot.fake_retriever, llm)
    assert len(none) == 3 and len(skip) == 2
    for c in none + skip:
        assert COMMENT_FIELDS <= set(c) and c["simulated"] is True and c["text"]
        assert c["cited_pages"] and all(p[0] == "FAKE_policy.pdf" for p in c["cited_pages"])
    assert [c["resident_id"] for c in skip] == ["a", "b"] and skip[1]["neighborhood"] == "Southside"
    assert not any(m[0]["content"] == pilot.meeting.COMMENT_SYSTEM for m in llm.calls)  # no meeting prompt used
    assert len(llm.calls) == 1 + len(residents)


def test_baselines_survive_bad_replies():
    llm = FakeLLM(reply="not json")
    assert baselines.no_personas(pilot.POLICY, 3, pilot.fake_retriever, llm) == []
    skip = baselines.skip_meeting(pilot.POLICY, [person("a", "Eastwood")], pilot.fake_retriever, llm)
    assert skip[0]["text"] == "not json" and skip[0]["cited_pages"] == [] and skip[0]["simulated"] is True


def test_fake_tagger_reply_uses_only_codebook_codes():
    llm = FakeLLM(responder=baselines.fake_responder)
    book = cb.load()
    tagged = analyst.tag_comments([comment(f"r{i}", "Eastwood", None) for i in range(20)], book, llm)
    valid = {t["code"] for t in book}
    assert all(set(c["codes"]) <= valid for c in tagged) and any(c["codes"] for c in tagged)


def test_scored_baselines_end_to_end():
    llm = FakeLLM(responder=baselines.fake_responder)
    residents = [person("a", "Eastwood"), person("b", "Southside")]
    sims = analyst.tag_comments(baselines.skip_meeting(pilot.POLICY, residents, pilot.fake_retriever, llm),
                                cb.load(), llm)
    out = score.score([["4"], ["6", "7-5"]], [c["codes"] for c in sims], cb.load())
    assert out["simulated_comments"] == 2 and out["real_comments"] == 2
    assert out["topic_coverage"] is None or 0 <= out["topic_coverage"] <= 1
