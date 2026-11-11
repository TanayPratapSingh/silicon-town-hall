"""App logic without Streamlit, so tests can call it and stub its edges (Census, retriever, memory embedder).

A scenario is one run of the meetings: data/processed/app/scenarios/<id>/scenario.json (settings,
residents, meetings, report, real call usage) plus memory.sqlite (long term memory for its rounds).
Every resident, turn, comment and memory in a scenario is simulated.
"""

import json
import re
import shutil
from datetime import datetime, timezone
from functools import cache
from pathlib import Path

import pandas as pd

from townhall import config
from townhall.ingest import codebook as cb
from townhall.llm import FakeLLM
from townhall.retrieval import Retriever, index, index_document
from townhall.town import census, homes, meeting, personas, pilot
from townhall.report import analyst, baselines
from townhall.town.memory import Memory

APP = config.PROCESSED / "app"
SCENARIOS = APP / "scenarios"
UPLOADS = APP / "uploads"
EMBED = None  # memory embedder; None loads all-MiniLM-L6-v2 on first use, tests pass a stand in

PRESETS = {"I-81 Syracuse (New York, PUMA 00701)": ("36", "00701")}
I81 = {"kind": "preset", "name": pilot.POLICY, "topic": pilot.TOPIC, "db_dir": str(index.DB_DIR)}
FAKE_LABEL = "Free test model: FakeLLM template text, not a model. Simulated."
PILOT_PENDING = "Pilot scores are not available yet: no scorer run is recorded in runs/."
PILOT_SCORE_FILES = (config.RUNS / "report_pilot_score.json", config.RUNS / "report_score_fake.json")


def make_llm(free, max_calls):
    """FakeLLM with canned replies (tagger and town pilot), or a real client that refuses call max_calls + 1."""
    if free or not config.OPENAI_API_KEY:
        return FakeLLM(responder=baselines.fake_responder)
    return pilot.capped_openai(max_calls)


def written_by(llm):
    return FAKE_LABEL if isinstance(llm, FakeLLM) else f"{config.RESIDENT_MODEL}, a real model. Simulated."


def usage(llm):
    """Real calls attempted and tokens used; zeros for FakeLLM."""
    rows = getattr(llm, "responses", [])
    return {"real_calls": getattr(llm, "attempts", 0),
            **{k: sum(r[k] for r in rows) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}}


def sample_community(state, puma, n, seed, llm):
    """n Census residents with home neighborhoods, a written profile and a stereotype review each."""
    people, source = census.load_people(state, puma)
    residents = census.sample_residents(state, puma, n=n, seed=seed, people=people, source=source)
    residents, home = homes.home_neighborhoods(residents, state, puma, seed=seed)
    label = written_by(llm)
    for r in residents:
        r["profile"] = personas.build_profile(r, llm)
        r["stereotype_review"] = personas.review_profile(r["profile"], llm, resident=r)
        r["profile_written_by"] = label
        r["simulated"] = True
    return residents, {"census_source": source, "homes": home}


def resident_table(residents):
    return pd.DataFrame([{"label": "SIMULATED", **{k: r.get(k) for k in (
        "resident_id", "neighborhood", "age", "sex", "race_ethnicity", "income", "commute", "tenure", "disability",
        "profile")}, "stereotype_flags": "; ".join(r["stereotype_review"]["keyword_flags"])
        or ("flagged by reviewer" if r["stereotype_review"]["flagged"] else "")} for r in residents])


@cache
def get_retriever(db_dir):
    return Retriever(db_dir=Path(db_dir))


def page_count(db_dir):
    return index.collections(Path(db_dir))[1].count()  # one page image row per page


def index_upload(filename, data):
    """Index an uploaded PDF into its own ChromaDB under UPLOADS, apart from the I-81 index."""
    if not data.startswith(b"%PDF"):
        raise ValueError(f"{filename} is not a PDF")
    stem = re.sub(r"[^\w.-]+", "_", Path(filename).stem)[:60]
    doc_id = f"upload_{stem}"  # the prefix keeps a name like ".." from escaping UPLOADS
    home = UPLOADS / doc_id
    home.mkdir(parents=True, exist_ok=True)
    pdf = home / f"{stem}.pdf"
    pdf.write_bytes(data)
    stats = index_document(pdf, doc_id, db_dir=home / "chroma", pages_dir=home / "pages")
    return {"kind": "upload", "name": filename, "topic": filename, "doc_id": doc_id, "db_dir": str(home / "chroma"),
            "pages": stats["pages"], "pages_without_text": stats["pages_without_text"]}


def cite(pages):
    return "; ".join(f"{doc}, page {page}" for doc, page in pages) or "no page cited"


def run_scenario(name, residents, policy, neighborhoods, rounds, llm, vision=False, change="", community=None,
                 on_progress=None):
    """One meeting per neighborhood per round, memory carried across rounds; saves and returns the scenario.
    A failure part way (such as the real call cap) is saved with what finished and the error."""
    sid = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "_" + (re.sub(r"[^\w-]+", "_", name)[:40] or "scenario")
    home = SCENARIOS / sid
    home.mkdir(parents=True)
    policy_text = policy["name"] + (f". Proposed change under test: {change}" if change.strip() else "")
    total, meetings, error = rounds * len(neighborhoods), [], None
    memory = Memory(home / "memory.sqlite", embed=EMBED)
    try:
        retrieve = meeting.retriever_from(get_retriever(policy["db_dir"]), k=5, llm=llm if vision else None)
        for rnd in range(1, rounds + 1):
            for hood in neighborhoods:
                if on_progress:
                    on_progress(len(meetings), total, rnd, hood)
                group = [r for r in residents if r["neighborhood"] == hood]
                state = meeting.run_meeting(policy_text, group, retrieve, llm, memory=memory, round=rnd,
                                            topic=policy["topic"])
                meetings.append({"round": rnd, "neighborhood": hood, **{k: state[k] for k in (
                    "agenda", "transcript", "summary", "comments", "stance_changes", "parse_failures")}})
    except Exception as e:  # noqa: BLE001 (saved and shown, not swallowed: real calls already spent stay saved)
        error = f"{type(e).__name__}: {e}"
    finally:
        memory.close()
    latest = {m["neighborhood"]: m["comments"] for m in meetings}  # each neighborhood's last finished round
    comments = [c for hood_comments in latest.values() for c in hood_comments]
    scenario = {
        "id": sid, "name": name, "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label": "Simulated residents. Not public comment.",
        "settings": {**(community or {}), "policy": policy["name"], "policy_kind": policy["kind"],
                     "policy_db": policy["db_dir"], "neighborhoods": list(neighborhoods), "rounds": rounds,
                     "meeting_text_by": written_by(llm), "vision_rerank": vision, "proposed_change": change},
        "status": "complete" if error is None else f"stopped early: {error}",
        "residents": residents, "meetings": meetings, "report": None, "usage": {"meetings": usage(llm)},
    }
    save = lambda: (home / "scenario.json").write_text(json.dumps(scenario, indent=1, ensure_ascii=False))
    save()  # before the report, so a report failure never loses the meetings
    try:
        scenario["report"] = build_report(comments, residents, llm, community)
    except Exception as e:  # noqa: BLE001 (saved and shown on the report page, such as the real call cap)
        scenario["report_error"] = f"{type(e).__name__}: {e}"
    scenario["usage"]["meetings_and_report"] = usage(llm)
    save()
    return scenario


@cache
def population(state, puma):
    """Census weighted group shares for the who to reach list, or (None, why) when the Census files and API fail."""
    try:
        return analyst.population_shares(state, puma)
    except Exception as e:  # noqa: BLE001 (the reason goes in the report label; the list then uses counts only)
        return None, f"not available ({type(e).__name__}: {e})"


def build_report(comments, residents, llm, community=None):
    """Tag the simulated comments with the answer key's tagger (one call each), then map analyst.build_report to
    what the app renders: label, tables [{title, rows}], sample_comments [{..., cited, label}], reach_in_person."""
    codebook = cb.load()
    shares, source = population(community["state"], community["puma"]) if community else (None, "no community")
    report = analyst.build_report(analyst.tag_comments(comments, codebook, llm), residents, codebook, shares, source)
    names = report["topic_names"]

    def cells(by):
        return [{"simulated residents": c["residents"], "comments": c["comments"],
                 "comments with no topic": c["no_topic_comments"],
                 "top topics": "; ".join(f"{names.get(k, k)} ({n})" for k, n in list(c["topics"].items())[:3])}
                for c in by.values()]

    def reach(e):
        expected = "" if e["expected_residents"] is None else f", {e['expected_residents']} expected"
        return (f"{e['dimension'].replace('_', ' ')}, {e['group']}: {e['simulated_residents']} simulated "
                f"residents{expected}. {'; '.join(e['reasons']).capitalize()}.")

    tagged_by = ("FakeLLM picked the topics from a hash of each comment, not by meaning" if isinstance(llm, FakeLLM)
                 else f"Topics tagged by {config.ANALYST_MODEL}, the tagger used on the real comments")
    return {
        "label": f"{report['label']}. {tagged_by}. Population shares: {report['population_source']}.",
        "tables": [
            {"title": "Topics raised", "rows": [{"code": k, "topic": names.get(k, k), "simulated comments": n}
                                               for k, n in report["topics"].items()]},
            {"title": "Topics by neighborhood", "rows": [{"neighborhood": h, **c} for h, c in zip(
                report["by_neighborhood"], cells(report["by_neighborhood"]))]},
            {"title": "Topics by group", "rows": [{"group type": d.replace("_", " "), "group": g, **c}
                                                  for d, by in report["by_group"].items()
                                                  for g, c in zip(by, cells(by))]}],
        "sample_comments": [{**s, "topic": names.get(s["topic"], s["topic"]), "cited": cite(s["cited_pages"]),
                             "label": "SIMULATED"} for s in report["samples"]],
        "reach_in_person": [reach(e) for e in report["who_to_reach"]],
    }


def pilot_scores():
    """The recorded I-81 pilot scores from runs/ (real run first), with their own label. Not this scenario's."""
    path = next((p for p in PILOT_SCORE_FILES if p.exists()), None)
    if path is None:
        return {"label": PILOT_PENDING, "scores": None}
    run = json.loads(path.read_text())
    scores = {name: {k: (c.get("score") or {}).get(k) for k in ("topic_coverage", "topic_mix_gap", "status")}
              for name, c in run.get("conditions", {}).items()}
    return {"label": f"Recorded pilot scores from runs/{path.name} ({run.get('command')}), not scores of this "
                     f"scenario. {run.get('label', '')}", "scores": scores}


def list_scenarios():
    """Saved scenarios, newest first."""
    paths = sorted(SCENARIOS.glob("*/scenario.json"), reverse=True)
    return [json.loads(p.read_text()) for p in paths]


def load_scenario(sid):
    return json.loads((SCENARIOS / sid / "scenario.json").read_text())


def memories(sid):
    """Every saved memory row in a scenario, without the embedding."""
    memory = Memory(SCENARIOS / sid / "memory.sqlite", embed=EMBED)
    try:
        # shortcut: Memory has no list method, so this reads its table directly; add one to Memory if this spreads
        rows = memory.db.execute("SELECT resident_id, topic, stance, reasons, comment, round FROM memories "
                                 "ORDER BY resident_id").fetchall()
    finally:
        memory.close()
    return [dict(zip(("resident_id", "topic", "stance", "reasons", "comment", "round"), row), label="SIMULATED")
            for row in rows]


def delete_memories(sid, resident_id=None):
    """Delete one resident's memories, or all of them in the scenario. Returns rows deleted."""
    memory = Memory(SCENARIOS / sid / "memory.sqlite", embed=EMBED)
    try:
        return memory.delete_all() if resident_id is None else memory.delete_resident(resident_id)
    finally:
        memory.close()


def delete_scenario(sid):
    """Delete a scenario's memories through Memory (secure delete), then its folder."""
    home = SCENARIOS / sid
    if home.resolve().parent != SCENARIOS.resolve():
        raise ValueError(f"not a scenario: {sid}")
    if (home / "memory.sqlite").exists():
        delete_memories(sid)
    shutil.rmtree(home)
