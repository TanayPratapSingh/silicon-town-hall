"""Run a scenario with the real model under call and dollar caps, then score it against the real comments.

    python scripts/run_scenario.py --residents data/processed/town/residents_real.json \
        --name "I-81 control, seed 0" --group-by citywide --rounds 1 --cap 0.12 --out runs/scenario_control_seed0.json
    add --change "..." for a what if run

Goes through the app's own code (app.core.run_scenario), so the run appears on the app's Saved scenarios page.
Stops before the next call once the estimated list price reaches --cap dollars; what finished is saved.
Writes counts, scores and token usage to --out, never resident or comment text.
"""

import argparse
import json
import shlex
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import core  # noqa: E402
from townhall.ingest import codebook as cb  # noqa: E402
from townhall.ingest import tagger  # noqa: E402
from townhall.report import fact_check, score  # noqa: E402
from townhall.town import pilot  # noqa: E402

def summarize(scenario, real_codes, codebook):
    """Scores, groups, stances and fact checks of a saved scenario; counts only."""
    report, residents = scenario["report"], scenario["residents"]
    real_counts = Counter(x for cs in real_codes for x in cs)
    codes = report["comment_codes"] if report else []
    turns = [t for m in scenario["meetings"] for t in m["transcript"]]
    comments = [c for m in scenario["meetings"] for c in m["comments"]]
    rounds = sorted({m["round"] for m in scenario["meetings"]})
    return {
        "score": score.score(real_codes, [c["codes"] for c in codes], codebook) if codes else None,
        "group_gaps": score.group_gaps(codes, residents, real_counts) if codes else None,
        "stances_by_round": {r: dict(Counter(c["stance"] for m in scenario["meetings"] if m["round"] == r
                                             for c in m["comments"])) for r in rounds},
        "stance_changes_by_round": {r: sum(m["stance_changes"] for m in scenario["meetings"] if m["round"] == r)
                                    for r in rounds},
        "parse_failures": sum(m["parse_failures"] for m in scenario["meetings"]),
        "fact_check": {"turns": fact_check.check_texts(turns, residents),
                       "comments": fact_check.check_texts(comments, residents)},
    }


def main(argv=None):
    args = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    args.add_argument("--residents", required=True)
    args.add_argument("--name", required=True)
    args.add_argument("--group-by", choices=list(core.GROUPINGS), default="citywide")
    args.add_argument("--rounds", type=int, default=1)
    args.add_argument("--change", default="", help="a plan change residents hear as part of the policy")
    args.add_argument("--cap", type=float, required=True, help="dollar cap on the estimated list price")
    args.add_argument("--out", required=True)
    args = args.parse_args(argv)
    command = shlex.join(["python", "scripts/run_scenario.py", *(argv if argv is not None else sys.argv[1:])])
    data = json.loads(Path(args.residents).read_text())
    residents, seed = data["residents"], data.get("census", {}).get("seed")
    groups = core.meeting_groups(residents, args.group_by)
    max_calls = args.rounds * (len(groups) + 3 * len(residents)) + len(residents) + 5  # meetings, then one tag each
    llm = pilot.capped_openai(max_calls, max_dollars=args.cap)
    community = {"community": next(iter(core.PRESETS)), "state": "36", "puma": "00701", "residents": len(residents),
                 "seed": seed}
    scenario = core.run_scenario(args.name, residents, core.I81, list(groups), args.rounds, llm, change=args.change,
                                 community=community, group_by=args.group_by,
                                 on_progress=lambda done, total, rnd, hood: print(f"meeting {done + 1} of {total}: "
                                                                                 f"round {rnd}, {hood}", flush=True))
    codebook = cb.load()
    real_codes = [r["codes"] for r in score.load_tags(tagger.TAGGED)]
    run = {"label": f"Real model run ({pilot.config.RESIDENT_MODEL} residents, {pilot.config.ANALYST_MODEL} tagger). "
                    "Every resident, turn and comment is simulated.",
           "command": command, "date": datetime.now(timezone.utc).date().isoformat(), "name": args.name,
           "scenario_id": scenario["id"], "residents_file": args.residents, "residents": len(residents), "seed": seed,
           "group_by": args.group_by, "meetings_per_round": {g: len(rs) for g, rs in groups.items()},
           "rounds": args.rounds, "change": args.change, "status": scenario["status"],
           "report_error": scenario.get("report_error"), **summarize(scenario, real_codes, codebook),
           "real_calls": llm.attempts, "max_calls": max_calls, "cap_dollars": args.cap,
           "token_usage": {k: sum(c[k] for c in llm.responses) for k in ("prompt_tokens", "completion_tokens")},
           "models_returned": sorted({c["model"] for c in llm.responses}),
           "estimated_dollars": round(llm.spent, 4),
           "estimate_note": "list prices from townhall.town.pilot.PRICES, not the OpenAI bill"}
    Path(args.out).write_text(json.dumps(run, indent=2) + "\n")
    print(json.dumps({k: run[k] for k in ("status", "real_calls", "estimated_dollars")} |
                     {"coverage": (run["score"] or {}).get("topic_coverage"),
                      "gap": (run["score"] or {}).get("topic_mix_gap")}, indent=2))


if __name__ == "__main__":
    main()
