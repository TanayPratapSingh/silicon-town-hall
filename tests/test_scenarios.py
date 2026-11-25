"""Final phase pieces: fact checks, districts, meeting groups, the dollar cap, Census facts in the prompts and
deleting uploads. Offline: synthetic residents, map squares and a fake OpenAI client."""

from types import SimpleNamespace

import pytest

from app import core
from townhall.report import fact_check as fc
from townhall.town import census, homes, meeting, pilot

OWNER = {"resident_id": "r1", "age": 40, "tenure": "Owned free and clear", "commute": census.NO_COMMUTE,
         "disability": "Without a disability"}


def square(name, lon, lat):
    ring = [[lon, lat], [lon + 0.01, lat], [lon + 0.01, lat + 0.01], [lon, lat + 0.01], [lon, lat]]
    return {"type": "Feature", "properties": {"NAME": name}, "geometry": {"type": "Polygon", "coordinates": [ring]}}


def test_fact_check_catches_claims_the_record_contradicts():
    assert fc.check("My daily commute on 81 takes an hour.", OWNER) == ["commute"]
    assert fc.check("As a renter, I worry my landlord will raise my rent.", OWNER) == ["tenure"]
    assert fc.check("I'm 62, and I have a disability.", OWNER) == ["age", "disability"]
    assert fc.check("As a 40-year-old homeowner I support the plan.", OWNER) == []
    for negated in ("I don't drive to work, but", "Since I do not commute to work regularly,",
                    "living here and not commuting to work, I"):  # restating the record is not a contradiction
        assert fc.check(negated, OWNER) == []
    out = fc.check_texts([{"resident_id": "r1", "text": "my commute"}, {"resident_id": "r1", "text": "fine"},
                          {"resident_id": "unknown", "text": "my commute"}], [OWNER])
    assert (out["texts_checked"], out["texts_with_a_contradiction"], out["by_kind"]) == (2, 1, {"commute": 1})


def test_districts_by_distance_and_direction_from_downtown():
    features = [square("Downtown", -76.15, 43.05), square("Near", -76.145, 43.052), square("Up", -76.15, 43.10),
                square("Right", -76.05, 43.05), square("Down", -76.15, 43.00), square("Left", -76.25, 43.05)]
    assert homes.districts(features) == {"Downtown": "Central", "Near": "Central", "Up": "North", "Right": "East",
                                         "Down": "South", "Left": "West"}


def test_meeting_groups(monkeypatch):
    rs = [{"resident_id": f"r{i}", "neighborhood": n} for i, n in enumerate(["A", "B", "A", "Z"])]
    sizes = lambda by: {k: len(v) for k, v in core.meeting_groups(rs, by).items()}
    assert sizes("citywide") == {core.CITYWIDE: 4} and sizes("neighborhood") == {"A": 2, "B": 1, "Z": 1}
    monkeypatch.setattr(homes, "districts", lambda: {"A": "North", "B": "North"})
    assert sizes("district") == {"North": 3, "Z": 1}  # off the map: its own meeting

    def no_map():
        raise FileNotFoundError("no neighborhood map")

    monkeypatch.setattr(homes, "districts", no_map)
    assert sizes("district") == {"A": 2, "B": 1, "Z": 1}


class FakeCompletions:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        usage = {"prompt_tokens": 1_000_000, "completion_tokens": 0, "total_tokens": 1_000_000}
        return SimpleNamespace(model="gpt-4.1-mini-2025-04-14", usage=SimpleNamespace(model_dump=lambda include: usage),
                               choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))])


def test_dollar_cap_stops_before_the_next_call(monkeypatch):
    fake = FakeCompletions()
    monkeypatch.setattr(pilot, "OpenAILLM", lambda: SimpleNamespace(
        client=SimpleNamespace(chat=SimpleNamespace(completions=fake))))
    llm = pilot.capped_openai(max_calls=10, max_dollars=0.70)
    for _ in range(2):  # $0.40 each at gpt-4.1-mini list price; the second starts at $0.40, under the cap
        llm.client.chat.completions.create(messages=[{"content": "x"}])
    with pytest.raises(RuntimeError, match="dollar cap"):
        llm.client.chat.completions.create(messages=[{"content": "x"}])
    assert fake.calls == 2 and llm.spent == pytest.approx(0.80)
    assert pilot.dollars("gpt-4.1-2025-04-14", 1e6, 1e6) == pytest.approx(10.0)
    assert pilot.dollars("some-new-model", 1e6, 0) == pytest.approx(2.0)  # unknown: priced as the dearest


def test_prompts_carry_the_census_facts():
    resident = {"resident_id": "r1", "profile": "I live here.", "age": 30, "sex": "Female", "race": "White alone",
                "hispanic_origin": "Not Spanish/Hispanic/Latino", "income": 0, "commute": census.NO_COMMUTE,
                "tenure": "Rented", "disability": "Without a disability", "neighborhood": "A"}
    line = meeting._facts(resident)
    assert line.startswith("Census facts about you") and '"age": 30' in line and "did not commute" in line
    assert meeting._facts({"resident_id": "r2", "profile": "p"}) == ""


def test_delete_upload_removes_only_uploads(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "UPLOADS", tmp_path / "uploads")
    home = core.UPLOADS / "upload_plan"
    (home / "chroma").mkdir(parents=True)
    (home / "plan.pdf").write_bytes(b"%PDF-1.7")
    assert [u["name"] for u in core.uploads()] == ["plan.pdf"]
    for bad in ("../x", "upload_missing", "upload_../../x"):
        with pytest.raises(ValueError):
            core.delete_upload(bad)
    core.delete_upload("upload_plan")
    assert not home.exists() and core.uploads() == []
