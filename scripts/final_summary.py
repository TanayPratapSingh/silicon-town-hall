"""Gather every scored real run into runs/final_summary.json, the one source for the final document and slides.

    python scripts/final_summary.py

Reads the pilot score files (seeds 0 and 1), the scenario run files (control, what if, scale) and, for the
pilots, the local transcripts and tags to add fact checks and group gaps. Counts and scores only, no text.
All of it is simulated; every pilot number is from 20 residents.
"""

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from townhall import config  # noqa: E402
from townhall.ingest import tagger  # noqa: E402
from townhall.report import fact_check, score  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from run_scenario import group_gaps  # noqa: E402

TOWN, REPORT = config.PROCESSED / "town", config.PROCESSED / "report"
SEEDS = {0: "", 1: "_seed1"}
PICK = ("topic_coverage", "topic_mix_gap", "simulated_comments", "simulated_comments_with_no_topic")


def brief(s):
    return {**{k: s.get(k) for k in PICK}, "missed": [x["topic"] for x in s.get("concerns_only_real", [])]}


def pilot(seed, suffix, real_counts):
    run = json.loads((config.RUNS / f"report_pilot_score{suffix}.json").read_text())
    out = {"source": f"runs/report_pilot_score{suffix}.json",
           "conditions": {n: brief(c["score"]) for n, c in run["conditions"].items()}}
    residents_path, meeting_path = TOWN / f"residents_real{suffix}.json", TOWN / f"meeting_pilot_real{suffix}.json"
    if residents_path.exists() and meeting_path.exists():
        residents = json.loads(residents_path.read_text())["residents"]
        turns = json.loads(meeting_path.read_text())["transcript"]
        comments = json.loads((TOWN / f"comments_real{suffix}.json").read_text())["comments"]
        out["fact_check"] = {"turns": fact_check.check_texts(turns, residents),
                             "comments": fact_check.check_texts(comments, residents)}
        tagged = REPORT / f"tagged_meeting{suffix}.json"
        if tagged.exists():
            codes = [{"resident_id": c["resident_id"], "codes": c["codes"]}
                     for c in json.loads(tagged.read_text())["comments"]]
            out["group_gaps"] = group_gaps(codes, residents, real_counts)
        out["stances"] = dict(Counter(c["stance"] for c in comments))
    return out


def shares(s):
    """Share of comments raising each topic code."""
    n = s["simulated_comments"] or 1
    return {code: round(k / n, 3) for code, k in s["simulated_topic_counts"].items()}


def main():
    real_counts = Counter(x for r in score.load_tags(tagger.TAGGED) for x in r["codes"])
    pilots = {seed: pilot(seed, suffix, real_counts) for seed, suffix in SEEDS.items()
              if (config.RUNS / f"report_pilot_score{suffix}.json").exists()}
    mean = lambda cond, key: round(sum(p["conditions"][cond][key] for p in pilots.values()) / len(pilots), 4)
    scenarios = {}
    for path in sorted(config.RUNS.glob("scenario_*.json")):
        run = json.loads(path.read_text())
        saved = config.PROCESSED / "app" / "scenarios" / run["scenario_id"] / "scenario.json"
        if saved.exists():  # fact checks are free, so they follow the current checker
            s = json.loads(saved.read_text())
            items = lambda key: [x for m in s["meetings"] for x in m[key]]
            fresh = {"turns": fact_check.check_texts(items("transcript"), s["residents"]),
                     "comments": fact_check.check_texts(items("comments"), s["residents"])}
            if fresh != run.get("fact_check"):
                run["fact_check"], run["fact_check_note"] = fresh, (
                    "recomputed by scripts/final_summary.py after the checker learned to skip negations such as "
                    "'I do not commute to work'")
                path.write_text(json.dumps(run, indent=2) + "\n")
        scenarios[path.stem.removeprefix("scenario_")] = run
    out = {"label": "Every resident, turn and comment is simulated. Pilot numbers are from 20 residents per seed.",
           "command": "python scripts/final_summary.py", "date": datetime.now(timezone.utc).date().isoformat(),
           "real_comments": sum(1 for _ in score.load_tags(tagger.TAGGED)),
           "pilots": pilots,
           "pilot_means": {cond: {k: mean(cond, k) for k in ("topic_coverage", "topic_mix_gap")}
                           for cond in ("meeting", "no_personas", "skip_meeting")},
           "scenarios": {name: {k: s.get(k) for k in ("name", "scenario_id", "residents", "seed", "group_by", "rounds",
                                                       "change", "status", "stances_by_round", "stance_changes_by_round",
                                                       "parse_failures", "fact_check", "real_calls", "estimated_dollars")}
                         | {"score": brief(s["score"]) if s.get("score") else None,
                            "topic_shares": shares(s["score"]) if s.get("score") else None,
                            "group_gaps": s.get("group_gaps")}
                         for name, s in scenarios.items()}}
    (config.RUNS / "final_summary.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"pilot_means": out["pilot_means"],
                      "scenarios": {n: (s["score"] or {}).get("topic_coverage") for n, s in out["scenarios"].items()}},
                     indent=2))


if __name__ == "__main__":
    main()
