"""App logic without Streamlit, so tests can call it and stub its edges (Census, retriever, memory embedder).

A scenario is one run of the meetings: data/processed/app/scenarios/<id>/scenario.json (settings,
residents, meetings, report, real call usage) plus memory.sqlite (long term memory for its rounds).
Every resident, turn, comment and memory in a scenario is simulated.
"""

import json
import os
import re
import shutil
import zlib
from datetime import datetime, timezone
from functools import cache
from pathlib import Path

import numpy as np
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

PILOT_ID = "pilot_I-81_seed_0"  # sorts before the dated scenarios, so the pilot is listed first
DEMO = Path(__file__).resolve().parent / "demo"  # committed runs for the online demo: simulated text, public Census rows
DEMO_RUNS = ("control_seed0", "whatif_seed0", "scale_seed2")  # runs/scenario_<name>.json, exported with the pilot
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


def index_ready(db_dir):
    """False on the online demo: the 724 MB impact statement index stays on the team's computer."""
    return Path(db_dir).exists()


def hash_embed(texts):
    """Word hashing vectors for memory recall when sentence-transformers is not installed (the online demo).
    shortcut: matches shared words, not meaning; install sentence-transformers to get all-MiniLM-L6-v2."""
    out = np.zeros((len(texts), 256), dtype=np.float32)
    for i, text in enumerate(texts):
        for word in re.findall(r"[a-z]+", text.lower()):
            out[i, zlib.crc32(word.encode()) % 256] += 1
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


def memory_embedder():
    if EMBED is not None:
        return EMBED
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return hash_embed
    return None  # Memory loads all-MiniLM-L6-v2 itself


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


def uploads():
    """Indexed uploads on disk, newest first: [{doc_id, name, db_dir}]."""
    homes_ = sorted((p for p in UPLOADS.glob("upload_*") if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
    return [{"doc_id": h.name, "name": next((p.name for p in h.glob("*.pdf")), h.name), "db_dir": str(h / "chroma")}
            for h in homes_]


def delete_upload(doc_id):
    """Delete an uploaded PDF, its index and its page images (issue #22: a draft policy may be private)."""
    home = UPLOADS / doc_id
    if not doc_id.startswith("upload_") or home.resolve().parent != UPLOADS.resolve() or not home.is_dir():
        raise ValueError(f"not an upload: {doc_id}")
    get_retriever.cache_clear()  # drop an open handle on its index before the files go
    shutil.rmtree(home)


def cite(pages):
    return "; ".join(f"{doc}, page {page}" for doc, page in pages) or "no page cited"


GROUPINGS = {"neighborhood": "One meeting per neighborhood", "district": "One meeting per district (compass sectors)",
             "citywide": "One citywide meeting"}
CITYWIDE = "Citywide"


def meeting_groups(residents, group_by="neighborhood"):
    """{meeting name: residents in it}. District meetings fix neighborhood meetings of one or two people."""
    if group_by == "citywide":
        return {CITYWIDE: list(residents)}
    if group_by == "district":
        try:
            where = homes.districts()
        except OSError:  # no neighborhood map saved: every neighborhood is its own district
            where = {}
        key = lambda r: where.get(r["neighborhood"], r["neighborhood"])  # off the map: its own meeting
    else:
        key = lambda r: r["neighborhood"]
    out = {}
    for r in residents:
        out.setdefault(key(r), []).append(r)
    return dict(sorted(out.items()))


def run_scenario(name, residents, policy, neighborhoods, rounds, llm, vision=False, change="", community=None,
                 on_progress=None, group_by="neighborhood"):
    """One meeting per group (neighborhood, district or citywide) per round, memory carried across rounds; saves
    and returns the scenario. neighborhoods: the meeting names to run, from meeting_groups. A failure part way
    (such as the real call cap) is saved with what finished and the error."""
    groups = meeting_groups(residents, group_by)
    sid = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "_" + (re.sub(r"[^\w-]+", "_", name)[:40] or "scenario")
    home = SCENARIOS / sid
    home.mkdir(parents=True)
    policy_text = policy["name"] + (f". Proposed change under test: {change}" if change.strip() else "")
    total, meetings, error = rounds * len(neighborhoods), [], None
    memory = Memory(home / "memory.sqlite", embed=memory_embedder())
    ready = index_ready(policy["db_dir"])
    try:
        if ready:
            retrieve = meeting.retriever_from(get_retriever(policy["db_dir"]), k=5, llm=llm if vision else None)
        elif isinstance(llm, FakeLLM):
            retrieve = pilot.fake_retriever  # online demo: six FAKE pages, labeled in the settings below
        else:
            raise RuntimeError("the policy index is not on this server, so a real model run would cite nothing")
        for rnd in range(1, rounds + 1):
            for hood in neighborhoods:
                if on_progress:
                    on_progress(len(meetings), total, rnd, hood)
                state = meeting.run_meeting(policy_text, groups[hood], retrieve, llm, memory=memory, round=rnd,
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
                     "policy_db": policy["db_dir"], "neighborhoods": list(neighborhoods), "group_by": group_by,
                     "rounds": rounds,
                     "meeting_text_by": written_by(llm), "vision_rerank": vision, "proposed_change": change,
                     "retrieval": "impact statement index" if ready else "FAKE pages: the index is not on this server"},
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


def representation(residents, shares=None):
    """Simulated residents against the count the Census share implies, for every group on every dimension."""
    counts = {}
    for r in residents:
        for dim, group in analyst.groups(r).items():
            counts.setdefault(dim, {}).setdefault(group, 0)
            counts[dim][group] += 1
    dims = set(counts) | set(shares or {})
    return [{"dimension": dim, "group": g, "simulated": counts.get(dim, {}).get(g, 0),
             "expected": None if (shares or {}).get(dim, {}).get(g) is None else round(len(residents) * shares[dim][g], 2)}
            for dim in sorted(dims) for g in sorted(set(counts.get(dim, {})) | set((shares or {}).get(dim, {})), key=str)]


@cache
def neighborhood_shapes():
    """Syracuse neighborhood polygons that homes.py saved, or None; the app never downloads them itself."""
    path = census.CENSUS_RAW / "syracuse_neighborhoods.geojson"
    return json.loads(path.read_text())["features"] if path.exists() else None


def build_report(comments, residents, llm, community=None):
    """Tag the simulated comments with the answer key's tagger (one call each), then build the report."""
    tagged_by = ("FakeLLM picked the topics from a hash of each comment, not by meaning" if isinstance(llm, FakeLLM)
                 else f"Topics tagged by {config.ANALYST_MODEL}, the tagger used on the real comments")
    return report_from_tagged(analyst.tag_comments(comments, cb.load(), llm), residents, tagged_by, community)


def report_from_tagged(tagged, residents, tagged_by, community=None):
    """Map analyst.build_report to what the app renders:
    label, tables [{title, rows}], sample_comments [{..., cited, label}], reach_in_person [str]."""
    codebook = cb.load()
    shares, source = population(community["state"], community["puma"]) if community else (None, "no community")
    report = analyst.build_report(tagged, residents, codebook, shares, source)
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

    matrix = [{"dimension": dim, "group": g, "code": k, "topic": names.get(k, k), "comments": n}
              for dim, by in {"neighborhood": report["by_neighborhood"], **report["by_group"]}.items()
              for g, c in by.items() for k, n in c["topics"].items()]
    return {
        "label": f"{report['label']}. {tagged_by}. Population shares: {report['population_source']}.",
        "residents": report["residents"], "comments": report["comments"], "reach": report["who_to_reach"],
        "matrix": matrix,  # group by topic counts for the heatmaps; the tables below hold the same numbers
        "comment_codes": [{"resident_id": c["resident_id"], "codes": c["codes"]} for c in tagged],  # for scoring
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
    scores = {name: {k: (c.get("score") or {}).get(k) for k in (
        "topic_coverage", "topic_mix_gap", "status", "real_comments", "simulated_comments", "real_topic_counts",
        "simulated_topic_counts")} for name, c in run.get("conditions", {}).items()}
    return {"label": f"Recorded pilot scores from runs/{path.name}, not scores of this scenario. "
                     f"{run.get('label', '')}", "scores": scores}


def import_pilot():
    """Save the real 20 resident pilot as a scenario, reusing its tagged comments, so the app shows it with no new
    calls. Reads what python -m townhall.town.pilot --real-meeting and python -m townhall.report.score wrote."""
    town, run = config.PROCESSED / "town", json.loads((config.RUNS / "town_pilot_real.json").read_text())
    meeting_file = json.loads((town / "meeting_pilot_real.json").read_text())
    residents = json.loads((town / "residents_real.json").read_text())["residents"]
    comments = json.loads((town / "comments_real.json").read_text())["comments"]
    tagged = json.loads((config.PROCESSED / "report" / "tagged_meeting.json").read_text())["comments"]
    home = SCENARIOS / PILOT_ID
    home.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(town / "memory_real.sqlite", home / "memory.sqlite")
    community = {"community": next(iter(PRESETS)), "state": "36", "puma": "00701", "residents": len(residents),
                 "seed": 0}
    scenario = {
        "id": PILOT_ID, "name": "I-81 pilot: 20 residents, seed 0, one citywide meeting", "date": run["date"],
        "label": "Simulated residents. Not public comment.",
        "settings": {**community, "policy": I81["name"], "policy_kind": "preset", "policy_db": I81["db_dir"],
                     "neighborhoods": ["Citywide"], "rounds": 1, "vision_rerank": False, "proposed_change": "",
                     "meeting_text_by": f"{config.RESIDENT_MODEL}, a real model. Simulated.",
                     "source": f"{run['command']}, then python -m townhall.report.score"},
        "status": "complete", "residents": residents,
        "meetings": [{"round": 1, "neighborhood": "Citywide", "comments": comments,
                      **{k: meeting_file[k] for k in ("agenda", "transcript", "summary")},
                      "stance_changes": run["meeting"]["stance_changes"],
                      "parse_failures": run["meeting"]["reply_parse_failures"]}],
        "report": report_from_tagged(tagged, residents, f"Topics tagged by {config.ANALYST_MODEL} in the pilot "
                                     "score run (runs/report_pilot_score.json)", community),
        "usage": {"meetings": {"real_calls": run["real_calls"], **run["token_usage"]}},
    }
    (home / "scenario.json").write_text(json.dumps(scenario, indent=1, ensure_ascii=False))
    return scenario


def export_demo():
    """Copy the pilot and the DEMO_RUNS scenarios into app/demo/ for the online demo, without memory files and
    with local paths made relative. Returns the ids written."""
    DEMO.mkdir(exist_ok=True)
    ids = [PILOT_ID] + [json.loads((config.RUNS / f"scenario_{n}.json").read_text())["scenario_id"] for n in DEMO_RUNS
                        if (config.RUNS / f"scenario_{n}.json").exists()]
    root = str(config.ROOT) + "/"
    for sid in ids:
        text = (SCENARIOS / sid / "scenario.json").read_text().replace(root, "")
        assert "/Users/" not in text, f"{sid} still holds a local path"
        (DEMO / f"{sid}.json").write_text(text)
    return ids


def seed_demo():
    """With SILICON_DEMO=1 (the online demo), copy app/demo/ scenarios into SCENARIOS where missing."""
    if os.environ.get("SILICON_DEMO") != "1":
        return 0
    copied = 0
    for path in DEMO.glob("*.json"):
        home = SCENARIOS / path.stem
        if not (home / "scenario.json").exists():
            home.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, home / "scenario.json")
            copied += 1
    return copied


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


if __name__ == "__main__":  # python -m app.core imports the real pilot; --export-demo then writes app/demo/
    import sys

    print("saved scenario", import_pilot()["id"])
    if "--export-demo" in sys.argv:
        print("exported", export_demo())
