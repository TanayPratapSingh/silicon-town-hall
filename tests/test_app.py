"""Streamlit app: the password gate blocks, every page renders with FakeLLM, and delete works.

Offline: synthetic Census records and homes, a fake retriever over two fake pages, a fake memory
embedder, scenarios in tmp_path. A real model client fails the test if the app ever builds one.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from app import charts, core
from tests.test_town_meeting import fake_embed
from tests.test_town_residents import person
from townhall.llm import FakeLLM
from townhall.retrieval import Hit
from townhall.town import census, homes, pilot

SCRIPT = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
PASSWORD = "test-password"
BANNER = "Simulated residents. Not public comment."
PAGES = ["Community", "Policy", "Run meetings", "Report", "Saved scenarios"]


class FakeSearcher:
    def __init__(self, image):
        self.images = SimpleNamespace(count=lambda: 2)
        self.image = str(image)

    def search(self, query, k=10, mode="both", cited_pages=None, llm=None, **_):
        assert llm is None or isinstance(llm, FakeLLM)
        return [Hit("fake_doc", "FAKE_policy.pdf", page, f"FAKE PAGE {page} about noise and traffic.", self.image,
                    1.0 / page) for page in (1, 2)][:k]


def fake_homes(residents, state, puma, seed=0):
    for i, r in enumerate(residents):
        r.update(tract="T", tract_name="Census Tract 1", neighborhood=("North", "South")[i % 2], home_draw="test")
    return residents, {"source": "test homes", "method": "alternating"}


def refuse_real(*_, **__):
    raise AssertionError("the app built a real model client")


SQUARE = {"type": "Feature", "properties": {"NAME": "North"}, "geometry": {"type": "Polygon", "coordinates": [
    [[-76.2, 43.0], [-76.1, 43.0], [-76.1, 43.1], [-76.2, 43.1], [-76.2, 43.0]]]}}


@pytest.fixture
def app(tmp_path, monkeypatch):
    image = tmp_path / "page.png"
    Image.new("RGB", (20, 26), "white").save(image)
    people = pd.DataFrame([person(SERIALNO=f"s{i}", SPORDER="1", AGEP=str(20 + i), PWGTP="10",
                                  RAC1P=("1", "2")[i % 2]) for i in range(12)], dtype=str)
    monkeypatch.setattr(census, "load_people", lambda state, puma: (people, "synthetic test records"))
    monkeypatch.setattr(homes, "home_neighborhoods", fake_homes)
    monkeypatch.setattr(core, "get_retriever", lambda db_dir: FakeSearcher(image))
    monkeypatch.setattr(core, "page_count", lambda db_dir: 2)
    monkeypatch.setattr(core, "SCENARIOS", tmp_path / "scenarios")
    monkeypatch.setattr(core, "UPLOADS", tmp_path / "uploads")
    monkeypatch.setattr(core, "EMBED", fake_embed)
    monkeypatch.setattr(core, "population", lambda state, puma: (None, "test: no population shares"))
    monkeypatch.setattr(core, "neighborhood_shapes", lambda: [SQUARE])
    monkeypatch.setattr(core, "PILOT_SCORE_FILES", (tmp_path / "report_pilot_score.json",))
    monkeypatch.setattr(pilot, "capped_openai", refuse_real)
    monkeypatch.setattr(core.pilot, "capped_openai", refuse_real)
    at = AppTest.from_file(SCRIPT, default_timeout=60)
    at.secrets["APP_PASSWORD"] = PASSWORD
    return at


def login(at):
    at.run()
    at.text_input[0].input(PASSWORD).run()
    assert not at.exception
    return at


def goto(at, page):
    at.sidebar.radio(key="page").set_value(page).run()
    assert not at.exception, at.exception
    assert any(w.value == BANNER for w in at.warning)
    return at


def run_meetings(at, rounds=1):
    goto(at, "Community")
    at.number_input[0].set_value(4).run()
    at.button(key="sample").click().run()
    goto(at, "Run meetings")
    at.number_input[0].set_value(rounds).run()
    at.button(key="run").click().run()
    assert not at.exception, at.exception
    return core.list_scenarios()[0]


def test_gate_blocks_until_the_right_password(app):
    app.run()
    assert len(app.text_input) == 1 and not app.sidebar.radio and not app.dataframe
    assert not any(w.value == BANNER for w in app.warning)
    app.text_input[0].input("wrong").run()
    assert "Wrong password." in [e.value for e in app.error]
    assert not app.sidebar.radio and not app.button
    app.text_input[0].input(PASSWORD).run()
    assert app.sidebar.radio(key="page").value == "Community"


def test_gate_stays_locked_without_a_password(monkeypatch):
    monkeypatch.setattr(core.config, "APP_PASSWORD", "")
    at = AppTest.from_file(SCRIPT, default_timeout=60).run()
    assert not at.text_input and not at.sidebar.radio
    assert "APP_PASSWORD is not set" in at.error[0].value


def test_pages_render_with_the_free_model(app):
    login(app)
    assert app.sidebar.toggle(key="free").value is True
    for page in PAGES:
        goto(app, page)
    scenario = run_meetings(app)
    assert scenario["settings"]["meeting_text_by"] == core.FAKE_LABEL
    assert scenario["usage"]["meetings"]["real_calls"] == 0
    assert {r["neighborhood"] for r in scenario["residents"]} == {"North", "South"}
    comments = [c for m in scenario["meetings"] for c in m["comments"]]
    assert comments and all(c["simulated"] is True for c in comments)

    goto(app, "Community")
    assert (app.dataframe[0].value["label"] == "SIMULATED").all()
    goto(app, "Policy")
    app.text_input[0].input("noise").run()
    assert "FAKE_policy.pdf, page 1" in [m.value.strip("*") for m in app.markdown]
    goto(app, "Report")
    text = " ".join(m.value for m in app.markdown)
    assert "**SIMULATED** comment by" in text and "badge[topic: " in text
    assert any("FakeLLM picked the topics" in c.value for c in app.caption)
    heads = [s.value for s in app.subheader]
    assert heads[:5] == ["Pilot scores", "Who to reach in person", "Topics raised", "Topics by neighborhood",
                         "Topics by group"]
    assert scenario["report"]["reach_in_person"] and "report_error" not in scenario
    assert any(c.value.startswith("Cited: fake_doc, page") for c in app.caption)
    assert any(c.value == core.PILOT_PENDING for c in app.caption)
    core.PILOT_SCORE_FILES[0].write_text(json.dumps({"label": "Run type: test.", "command": "python -m test", "conditions": {
        "meeting": {"score": {"topic_coverage": 0.5, "topic_mix_gap": 0.25}}}}))
    app.run()
    assert any(c.value.startswith("Recorded pilot scores from runs/report_pilot_score.json") for c in app.caption)
    side = next(d.value for d in app.dataframe if "condition" in d.value.columns)
    assert side.to_dict("records") == [{"condition": "meeting", "real topics raised": "50.0%", "topic mix gap": 0.25}]
    goto(app, "Saved scenarios")
    assert app.dataframe[0].value["name"].tolist() == [scenario["name"]]


def test_charts_and_map_draw_the_report_numbers(app):
    login(app)
    scenario = run_meetings(app)
    report = scenario["report"]
    assert report["reach"] and report["matrix"]
    assert sum(m["comments"] for m in report["matrix"] if m["dimension"] == "neighborhood") == sum(
        r["simulated comments"] for r in report["tables"][0]["rows"])  # heatmap cells add up to the topic counts
    rows = core.representation(scenario["residents"], {"neighborhood": {"North": 0.75, "South": 0.25, "East": 0.1}})
    hood = {r["group"]: r for r in rows if r["dimension"] == "neighborhood"}
    assert (hood["North"]["simulated"], hood["North"]["expected"], hood["East"]["simulated"]) == (2, 3.0, 0)
    assert all(r["expected"] is None for r in rows if r["dimension"] == "age_band")
    for chart in (charts.topic_bars(report["tables"][0]["rows"]), charts.heatmap(report["matrix"]),
                  charts.representation([{"label": g, **{k: r[k] for k in ("simulated", "expected")}}
                                         for g, r in hood.items()])):
        assert chart.to_dict()["$schema"].startswith("https://vega.github.io/schema/vega-lite/")
    score = {"real_comments": 4, "simulated_comments": 2, "real_topic_counts": {"4": 2, "8": 1},
             "simulated_topic_counts": {"8": 2}}
    shares = charts.real_vs_simulated(score, {"4": "Purpose", "8": "Outreach"}).data.set_index(["topic", "source"]).share
    assert (shares["Purpose", charts.REAL], shares["Purpose", charts.SIM], shares["Outreach", charts.SIM]) == (0.5, 0, 1)
    assert charts.real_vs_simulated({"real_topic_counts": None}, {}) is None
    assert charts.gap_color(-3) == [227, 73, 72, 200] and charts.gap_color(0) == [240, 239, 236, 200]
    props = charts.neighborhood_map([SQUARE], list(hood.values())).layers[0].data["features"][0]["properties"]
    assert (props["simulated"], props["expected"], props["fill"]) == (2, "3.0", charts.gap_color(-1.0))
    goto(app, "Community")
    assert app.get("deck_gl_json_chart")  # the map renders
    goto(app, "Report")
    assert len(app.get("vega_lite_chart")) >= 3 and not app.exception


def test_delete_memories_and_scenario(app):
    login(app)
    scenario = run_meetings(app, rounds=2)
    sid = scenario["id"]
    assert len(scenario["meetings"]) == 4 and len(core.memories(sid)) == 4  # one row per resident and topic

    goto(app, "Saved scenarios")
    app.selectbox(key="memory_resident").set_value("r000").run()
    app.button(key="delete_resident").click().run()
    assert sorted(r["resident_id"] for r in core.memories(sid)) == ["r001", "r002", "r003"]
    assert app.button(key="delete_scenario").disabled
    app.checkbox(key="sure").check().run()
    app.button(key="delete_scenario").click().run()
    assert not (core.SCENARIOS / sid).exists()
    assert "Deleted scenario" in app.success[0].value and core.list_scenarios() == []


def test_upload_is_checked_and_indexed_apart(app, monkeypatch):
    calls = []
    monkeypatch.setattr(core, "index_document", lambda pdf, doc_id, db_dir, pages_dir: calls.append(
        (Path(pdf), doc_id, Path(db_dir))) or {"pages": 3, "pages_without_text": 1})
    with pytest.raises(ValueError):
        core.index_upload("notes.pdf", b"not a pdf")
    policy = core.index_upload("../Draft plan.pdf", b"%PDF-1.7 test")
    pdf, doc_id, db_dir = calls[0]
    assert doc_id == "upload_Draft_plan" and db_dir.is_relative_to(core.UPLOADS) and pdf.read_bytes().startswith(b"%PDF")
    assert (policy["kind"], policy["pages"], policy["db_dir"]) == ("upload", 3, str(db_dir))


def test_import_pilot_reuses_its_tags(app, tmp_path, monkeypatch):
    login(app)
    run = run_meetings(app)
    m = run["meetings"][0]
    town, report, runs = (tmp_path / d for d in ("town", "report", "runs"))
    for d in (town, report, runs):
        d.mkdir()
    write = lambda path, payload: path.write_text(json.dumps(payload))
    write(town / "residents_real.json", {"residents": run["residents"]})
    write(town / "meeting_pilot_real.json", {k: m[k] for k in ("agenda", "transcript", "summary")})
    write(town / "comments_real.json", {"comments": m["comments"]})
    write(report / "tagged_meeting.json", {"comments": [{**c, "codes": ["4"]} for c in m["comments"]]})
    write(runs / "town_pilot_real.json", {"date": "2026-11-11", "command": "python -m test", "real_calls": 0,
                                         "meeting": {"stance_changes": 0, "reply_parse_failures": 0},
                                         "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}})
    (town / "memory_real.sqlite").write_bytes((core.SCENARIOS / run["id"] / "memory.sqlite").read_bytes())
    monkeypatch.setattr(core.config, "PROCESSED", tmp_path)
    monkeypatch.setattr(core.config, "RUNS", runs)
    core.import_pilot()
    pilot_run = core.list_scenarios()[0]
    assert pilot_run["id"] == core.PILOT_ID and "pilot score run" in pilot_run["report"]["label"]
    assert pilot_run["report"]["tables"][0]["rows"][0]["code"] == "4" and core.memories(core.PILOT_ID)
