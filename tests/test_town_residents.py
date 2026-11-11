"""Residents: Census decoding and weighting, home tracts and neighborhoods, profiles and stereotype review.

Offline: tiny synthetic Census records and tracts, FakeLLM, no network.
"""

import json

import pandas as pd
import pytest

from townhall.llm import FakeLLM
from townhall.town import census, homes, personas


def person(**overrides):
    row = {"SERIALNO": "2024HU0000001", "SPORDER": "1", "AGEP": "40", "SEX": "2", "RAC1P": "2", "HISP": "01",
           "PINCP": "30000", "ADJINC": "1015250", "JWTRNS": "02", "DIS": "1", "PWGTP": "50", "TEN": "3"}
    return {**row, **overrides}


def test_decode_csv_style_codes():
    r = census.decode(person())
    assert r["sex"] == "Female"
    assert r["race_ethnicity"] == "Black or African American alone, not Hispanic or Latino"
    assert r["hispanic_origin"] == "Not Spanish/Hispanic/Latino"
    assert r["income"] == round(30000 * 1.015250)
    assert (r["commute"], r["tenure"], r["disability"], r["person_weight"]) == ("Bus", "Rented", "With a disability", 50)


def test_decode_api_style_codes_and_not_applicable():
    # the API writes 0 for N/A, drops leading zeros and gives ADJINC as a decimal
    r = census.decode(person(HISP="3", JWTRNS="0", TEN="0", ADJINC="1.049470"))
    assert (r["race_ethnicity"], r["hispanic_origin"]) == ("Hispanic or Latino", "Puerto Rican")
    assert r["commute"] == census.NO_COMMUTE
    assert r["tenure"] == census.NO_TENURE
    assert r["income"] == round(30000 * 1.049470)
    assert census.decode(person(JWTRNS=float("nan")))["commute"] == census.NO_COMMUTE  # blank in the CSV


def test_decode_rejects_unknown_code():
    with pytest.raises(KeyError):
        census.decode(person(RAC1P="42"))


def test_sample_is_weighted_adult_only_and_seeded():
    rows = ([person(SERIALNO=f"heavy{i}", PWGTP="1000") for i in range(10)]
            + [person(SERIALNO=f"light{i}", PWGTP="1") for i in range(90)]
            + [person(SERIALNO=f"child{i}", AGEP="17", PWGTP="100000") for i in range(10)])
    people = pd.DataFrame(rows, dtype=str)
    picked = census.sample_residents(n=10, seed=0, people=people)
    assert all(r["age"] >= 18 for r in picked)
    assert sum(r["serialno"].startswith("heavy") for r in picked) >= 9
    assert [r["serialno"] for r in picked] == [r["serialno"] for r in census.sample_residents(n=10, seed=0, people=people)]
    assert [r["resident_id"] for r in picked] == [f"r{i:03d}" for i in range(10)]
    shuffled = people.sample(frac=1, random_state=1)  # same records in another order, as the API sends them
    assert [r["serialno"] for r in census.sample_residents(n=10, seed=0, people=shuffled)] == [r["serialno"] for r in picked]


def tracts_table(**columns):
    base = {"tract": ["A", "B"], "tract_name": ["Census Tract 1", "Census Tract 2"], "neighborhood": ["North", "South"]}
    return pd.DataFrame({**base, **columns})


def test_homes_by_population():
    residents = [{"resident_id": f"r{i}"} for i in range(30)]
    homes.assign_homes(residents, tracts_table(population=[0, 100]))
    assert {(r["tract"], r["neighborhood"], r["home_draw"]) for r in residents} == {("B", "South", "population")}


def test_homes_conditioned_on_race_and_tenure():
    counts = {v: [0, 0] for v in homes.ACS_VARS}
    counts.update({"B01003_001E": [100, 100], "B03002_004E": [100, 0], "B03002_003E": [0, 100],
                   "B25008_003E": [100, 0], "B25008_002E": [0, 100]})
    tracts = tracts_table(population=[100, 100], **counts)
    black_renter = {"race_ethnicity": "Black or African American alone, not Hispanic or Latino", "tenure": "Rented"}
    white_owner = {"race_ethnicity": "White alone, not Hispanic or Latino", "tenure": "Owned free and clear"}
    in_dorm = {"race_ethnicity": "Asian alone, not Hispanic or Latino", "tenure": census.NO_TENURE}
    residents = [dict(black_renter) for _ in range(10)] + [dict(white_owner) for _ in range(10)] + [in_dorm]
    homes.assign_homes(residents, tracts)
    assert {r["tract"] for r in residents[:10]} == {"A"}
    assert {r["tract"] for r in residents[10:20]} == {"B"}
    assert residents[20]["home_draw"].startswith("population (no tract")  # nobody matches: falls back and says so


def test_neighborhood_point_in_polygon_with_hole():
    square = [[0, 0], [0, 10], [10, 10], [10, 0], [0, 0]]
    hole = [[4, 4], [4, 6], [6, 6], [6, 4], [4, 4]]
    features = [{"geometry": {"type": "Polygon", "coordinates": [square, hole]}, "properties": {"NAME": "Middle"}}]
    assert homes.neighborhood_of(1, 1, features) == "Middle"
    assert homes.neighborhood_of(5, 5, features) == homes.OUTSIDE
    assert homes.neighborhood_of(20, 20, features) == homes.OUTSIDE
    assert homes.tract_name("000501") == "Census Tract 5.01"
    assert homes.tract_name("003800") == "Census Tract 38"


def test_homes_unassigned_when_files_fail(monkeypatch):
    def offline(state, puma):
        raise OSError("no network")

    monkeypatch.setattr(homes, "load_tracts", offline)
    residents, info = homes.home_neighborhoods([{"resident_id": "r0"}, {"resident_id": "r1"}])
    assert {r["neighborhood"] for r in residents} == {homes.UNASSIGNED}
    assert info["source"].startswith("UNASSIGNED")


def test_profile_prompt_uses_facts_only():
    llm = FakeLLM(reply="I live in Westcott and take the bus.")
    resident = {**census.decode(person()), "neighborhood": "Westcott"}
    assert personas.build_profile(resident, llm) == "I live in Westcott and take the bus."
    sent = json.loads(llm.calls[0][1]["content"].split("Facts (JSON): ", 1)[1])
    assert set(sent) == set(personas.PROFILE_FIELDS) - {"income"} | {"annual_income_dollars"}
    assert sent["neighborhood"] == "Westcott"


def test_stereotype_keywords_and_identity_opener():
    review = personas.review_profile("As a Black woman, I live in a crime-ridden ghetto.")
    assert review["flagged"]
    assert any(f.startswith("ghetto") for f in review["keyword_flags"])
    assert any(f.startswith("crime ridden") for f in review["keyword_flags"])
    assert any("opens with" in f for f in review["keyword_flags"])
    clean = personas.review_profile("I am 40 and take the bus to work downtown. Rent takes a big part of my pay.")
    assert not clean["flagged"] and clean["llm_review"] is None


def test_stereotype_llm_review_hook():
    reviewer = FakeLLM(responder=lambda m: {"stereotype": True, "reasons": ["implies culture from race"]})
    review = personas.review_profile("I take the bus to work.", reviewer)
    assert review["flagged"] and review["keyword_flags"] == []
    assert review["llm_review"]["reasons"] == ["implies culture from race"]
    broken = personas.review_profile("I take the bus to work.", FakeLLM(reply="not json"))
    assert not broken["flagged"] and "error" in broken["llm_review"]
