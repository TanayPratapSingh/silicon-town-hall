"""Pilot scorer: how closely simulated comments match the real ones, topic by topic.

    .venv/bin/python -m townhall.report.score --fake
    .venv/bin/python -m townhall.report.score --sim meeting=data/processed/town/comments.json \
        --sim no_personas=data/processed/report/baseline_no_personas.json \
        --sim skip_meeting=data/processed/report/baseline_skip_meeting.json \
        --residents data/processed/town/residents_real.json --run-type "20 resident pilot"

--fake tags everything with FakeLLM: the fake pilot meeting comments, both baselines run with FakeLLM over the
pilot's FAKE pages, and, standing in for the pending real tags, FakeLLM tags of the real comments. It proves
the plumbing and writes runs/report_score_fake.json. Without --fake, each simulated comment is tagged by
ANALYST_MODEL (one real call each, capped by --max-calls) and scored against the real tags file; if that file
does not exist yet, the real side is reported as pending. Every runs/ file holds counts and topic codes only.
"""

import argparse
import json
import shlex
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from townhall import config
from townhall.ingest import codebook as cb
from townhall.ingest import tagger
from townhall.llm import FakeLLM
from townhall.report import analyst, baselines
from townhall.town import pilot

OUT = config.PROCESSED / "report"
TOWN = config.PROCESSED / "town"
COVERAGE_NOTE = "Share of the topics in the real comments that at least one simulated comment raised."
JS_NOTE = ("Jensen Shannon distance (base 2) between the share of topic mentions in real and in simulated comments: "
           "0 means the same mix of topics, 1 means no topic in common.")


def coverage(real_counts, sim_counts):
    real = {c for c, n in real_counts.items() if n}
    return len(real & {c for c, n in sim_counts.items() if n}) / len(real) if real else None


def js_distance(real_counts, sim_counts):
    codes = sorted(set(real_counts) | set(sim_counts))
    p, q = (np.array([counts.get(c, 0) for c in codes], dtype=float) for counts in (real_counts, sim_counts))
    if not p.sum() or not q.sum():
        return None
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2

    def kl(a):
        return float(np.sum(a[a > 0] * np.log2(a[a > 0] / m[a > 0])))

    return float(np.sqrt(max(0.0, (kl(p) + kl(q)) / 2)))  # max: rounding can leave a tiny negative


def score(real_codes, sim_codes, codebook):
    """real_codes, sim_codes: one list of topic codes per comment."""
    names = {t["code"]: t["topic"] for t in codebook}
    real, sim = Counter(c for codes in real_codes for c in codes), Counter(c for codes in sim_codes for c in codes)
    cov, gap = coverage(real, sim), js_distance(real, sim)
    return {"real_comments": len(real_codes), "simulated_comments": len(sim_codes),
            "real_comments_with_no_topic": sum(not c for c in real_codes),
            "simulated_comments_with_no_topic": sum(not c for c in sim_codes),
            "topic_coverage": None if cov is None else round(cov, 4), "topic_coverage_note": COVERAGE_NOTE,
            "topic_mix_gap": None if gap is None else round(gap, 4), "topic_mix_gap_note": JS_NOTE,
            "concerns_only_real": [{"code": c, "topic": names.get(c), "real_comments": n}
                                   for c, n in real.most_common() if not sim[c]],
            "topics_only_simulated": [{"code": c, "topic": names.get(c), "simulated_comments": n}
                                      for c, n in sim.most_common() if not real[c]],
            "real_topic_counts": dict(real.most_common()), "simulated_topic_counts": dict(sim.most_common())}


MIN_GROUP_COMMENTS = 3  # a topic mix of one or two comments says nothing


def group_gaps(comment_codes, residents, real_counts):
    """Per dimension and group with MIN_GROUP_COMMENTS or more simulated comments: coverage and topic mix gap to the
    real comments. comment_codes: [{resident_id, codes}]. The real comments carry no demographics, so every group is
    compared with all of them."""
    by_id = {r["resident_id"]: analyst.groups(r) for r in residents}
    cells = {}
    for c in comment_codes:
        for dim, group in by_id.get(c["resident_id"], {}).items():
            cells.setdefault(dim, {}).setdefault(group, []).append(c["codes"])
    out = {}
    for dim, by_group in cells.items():
        for group, codes in by_group.items():
            if len(codes) >= MIN_GROUP_COMMENTS:
                sim = Counter(x for cs in codes for x in cs)
                cov, gap = coverage(real_counts, sim), js_distance(real_counts, sim)  # gap None: no topics
                out.setdefault(dim, {})[group] = {"comments": len(codes),
                                                  "topic_coverage": None if cov is None else round(cov, 4),
                                                  "topic_mix_gap": None if gap is None else round(gap, 4)}
    return out


def load_tags(path):
    """Records with a codes list, one per line, as townhall.ingest.tagger writes them."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_comments(path):
    data = json.loads(Path(path).read_text())
    return data["comments"] if isinstance(data, dict) else data


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def main(argv=None):
    args = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    args.add_argument("--fake", action="store_true", help="FakeLLM everywhere; writes runs/report_score_fake.json")
    args.add_argument("--sim", action="append", default=[], metavar="NAME=PATH", help="simulated comments file")
    args.add_argument("--residents", default=str(TOWN / "residents.json"), help="Census records of the residents")
    args.add_argument("--real", default=str(tagger.TAGGED), help="real tagged comments, one JSON object with codes per line")
    args.add_argument("--run-type", default="20 resident pilot", help="label for the numbers, such as '20 resident pilot'")
    args.add_argument("--out", default=str(config.RUNS / "report_pilot_score.json"))
    args.add_argument("--max-calls", type=int, default=100, help="cap on real tagger calls")
    args.add_argument("--no-population", action="store_true", help="skip the Census population shares")
    args.add_argument("--suffix", default="", help="added to the tagged and report file names, such as _seed1")
    args = args.parse_args(argv)
    command = shlex.join(["python", "-m", "townhall.report.score", *(argv if argv is not None else sys.argv[1:])])
    codebook = cb.load()
    residents = json.loads(Path(args.residents).read_text())["residents"]
    suffix = "_fake" if args.fake else args.suffix

    if args.fake:
        llm = FakeLLM(responder=baselines.fake_responder)
        sims = {"meeting": load_comments(TOWN / "comments.json"),
                "no_personas": baselines.no_personas(pilot.POLICY, len(residents), pilot.fake_retriever, llm),
                "skip_meeting": baselines.skip_meeting(pilot.POLICY, residents, pilot.fake_retriever, llm)}
        real_path = OUT / "real_tags_fake.jsonl"
        OUT.mkdir(parents=True, exist_ok=True)
        real_path.unlink(missing_ok=True)  # tag_all resumes from its file; start the fake tags fresh
        tagger.tag_all(tagger.load_comments(), codebook, llm, cap=float("inf"), out=real_path)
        run_type, out, real_side = "fake", config.RUNS / "report_score_fake.json", (
            "FAKE: FakeLLM tags of the real comments, picked by hash, not by meaning. The real score is pending "
            "until python -m townhall.ingest.tagger has tagged the comments.")
    else:
        llm = pilot.capped_openai(args.max_calls)
        sims = {name: load_comments(path) for name, path in (s.split("=", 1) for s in args.sim)}
        real_path, run_type, out = Path(args.real), args.run_type, Path(args.out)
        real_side = f"real tags from {baselines.rel(args.real)}" if real_path.exists() else (
            f"PENDING: {baselines.rel(args.real)} does not exist yet")

    population, pop_source = None, "skipped with --no-population"
    if not args.no_population:
        try:
            population, pop_source = analyst.population_shares()
        except (OSError, RuntimeError, ValueError, KeyError) as error:
            pop_source = f"population shares unavailable ({type(error).__name__}: {error})"
    real = load_tags(real_path) if real_path.exists() else None
    real_codes = None if real is None else [r["codes"] for r in real]
    known = {r["resident_id"] for r in residents}
    results = {}
    for name, comments in sims.items():
        tagged = analyst.tag_comments(comments, codebook, llm)
        _write(OUT / f"tagged_{name}{suffix}.json", {"label": analyst.SIMULATED_LABEL, "comments": tagged})
        entry = {"comments": len(tagged), "all_labeled_simulated": all(c.get("simulated") is True for c in tagged)}
        if all(c["resident_id"] in known for c in tagged):  # no_personas has no residents, so no group report
            speakers = {c["resident_id"] for c in tagged}  # the report covers the residents who wrote these comments
            report = analyst.build_report(tagged, [r for r in residents if r["resident_id"] in speakers], codebook,
                                          population, pop_source)
            _write(OUT / f"report_{name}{suffix}.json", report)
            entry["who_to_reach"] = report["who_to_reach"]
        entry["score"] = (score(real_codes, [c["codes"] for c in tagged], codebook) if real_codes is not None
                          else {"status": "PENDING: real tags not available",
                                "simulated_topic_counts": dict(Counter(x for c in tagged for x in c["codes"]))})
        results[name] = entry

    run = {"label": f"Run type: {run_type}. Every comment scored here is simulated. Real side: {real_side}",
           "run_type": run_type, "command": command, "date": datetime.now(timezone.utc).date().isoformat(),
           "tagger_model": "FakeLLM template, not a model" if args.fake else config.ANALYST_MODEL,
           "residents_in_file": len(residents), "population_source": pop_source, "real_tags_file": baselines.rel(real_path),
           "residents_file": baselines.rel(args.residents),
           "real_tags_available": real_codes is not None and not args.fake,
           "real_comments_by_source_group": None if real is None else dict(Counter(
               r.get("file", "?").split("/")[0] for r in real)),
           "conditions": results,
           "token_usage": "none: FakeLLM" if args.fake else baselines._usage(llm)}
    _write(out, run)
    print(json.dumps({k: v for k, v in run.items() if k != "conditions"} | {
        "scores": {n: {k: e["score"].get(k) for k in ("topic_coverage", "topic_mix_gap", "status")}
                   for n, e in results.items()}}, indent=2))


if __name__ == "__main__":
    main()
