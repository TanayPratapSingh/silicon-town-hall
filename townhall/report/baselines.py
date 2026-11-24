"""Two baselines that write simulated comments in the same shape as townhall.town.meeting comments.

    .venv/bin/python -m townhall.report.baselines --fake            # FakeLLM and the pilot's FAKE pages
    .venv/bin/python -m townhall.report.baselines --n 20 --residents data/processed/town/residents_real.json

no_personas: one LLM call, no residents, writes n comments about the policy from one retrieval.
skip_meeting: each resident writes its comment from its profile and its own retrieval, with no meeting.
Real mode searches the impact statement index (no vision stage, so no extra calls) with RESIDENT_MODEL,
capped at n + 1 real calls. Writes data/processed/report/baseline_*.json (gitignored) and
runs/report_baselines_fake.json or _real.json (counts and token usage only, no text).
"""

import argparse
import json
import shlex
import re
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path

from townhall import config
from townhall.ingest import tagger
from townhall.llm import FakeLLM
from townhall.town import meeting, pilot

OUT = config.PROCESSED / "report"
NO_PERSONAS_SYSTEM = (
    "You write simulated public comments on a draft policy. Write the requested number of different "
    "comments, each three to five sentences, grounded in the numbered policy excerpts, and cite the ones "
    'each comment used. Reply as JSON: {"comments": [{"text": "...", "stance": "support" or "oppose" or '
    '"mixed" or "undecided", "cited": [excerpt numbers]}]}.'
)
SKIP_SYSTEM = meeting.COMMENT_SYSTEM.replace("after a neighborhood meeting", "without attending any meeting")
assert SKIP_SYSTEM != meeting.COMMENT_SYSTEM, "meeting.COMMENT_SYSTEM changed; reword SKIP_SYSTEM"


def no_personas(policy, n, retriever, llm):
    """n comments from one call with no personas. Returns what the model wrote, which can be fewer than n."""
    hits = retriever(f"{policy}. Public concerns about the policy.", [])
    raw = llm.chat([{"role": "system", "content": NO_PERSONAS_SYSTEM},
                    {"role": "user", "content": f"Policy: {policy}\nNumber of comments: {n}\n"
                                                f"Policy excerpts:\n{meeting._excerpts(hits)}"}], json_mode=True)
    try:
        items = json.loads(raw).get("comments", [])
    except (json.JSONDecodeError, TypeError, AttributeError):
        items = []
    out = []
    for i, item in enumerate(items[:n] if isinstance(items, list) else []):
        text, stance, pages, _, _ = meeting._parse(json.dumps(item), hits, "undecided")
        out.append({"resident_id": f"llm{i:03d}", "neighborhood": None, "text": text, "cited_pages": pages,
                    "stance": stance, "simulated": True, "baseline": "no_personas"})
    return out


def skip_meeting(policy, residents, retriever, llm):
    """One comment per resident from its profile and retrieval alone: the meeting graph without the meeting."""
    out = []
    for r in residents:
        hits = retriever(meeting.build_query(r, "overall comment on the policy", ""), [])
        prompt = f"Your profile: {r['profile']}\nPolicy: {policy}\nPolicy excerpts:\n{meeting._excerpts(hits)}"
        raw = llm.chat([{"role": "system", "content": SKIP_SYSTEM}, {"role": "user", "content": prompt}], json_mode=True)
        text, stance, pages, _, _ = meeting._parse(raw, hits, "undecided")
        out.append({"resident_id": r["resident_id"], "neighborhood": r.get("neighborhood"), "text": text,
                    "cited_pages": pages, "stance": stance, "simulated": True, "baseline": "skip_meeting"})
    return out


def fake_responder(messages):
    """FakeLLM replies for the baselines and the tagger; other prompts go to the town pilot's fakes. Not a model."""
    system, user = messages[0]["content"], messages[-1]["content"]
    if system == NO_PERSONAS_SYSTEM:
        n = int(user.split("Number of comments: ", 1)[1].split("\n", 1)[0])
        return {"comments": [{"text": f"[FakeLLM, simulated, no persona] Comment {i} cites excerpt 1.",
                              "stance": "undecided", "cited": [1]} for i in range(n)]}
    if system == SKIP_SYSTEM:
        return {"text": "[FakeLLM, simulated, skipped the meeting] My comment cites excerpt 1.",
                "stance": "undecided", "reasons": ["FakeLLM placeholder reason"], "cited": [1]}
    if system == tagger.SYSTEM:  # 0 to 3 codes per comment picked from a hash of its text, unrelated to its meaning
        head, _, body = user.partition("\n\nComments:\n")
        codes = re.findall(r"^(\S+): ", head, re.M)
        tags = {}
        for cid, text in re.findall(r"^\[(\w+)\]\n(.*?)(?=\n\n\[\w+\]\n|\Z)", body, re.M | re.S):
            h = zlib.crc32(text.encode())
            tags[cid] = [codes[(h >> (8 * i)) % len(codes)] for i in range(h % 4)]
        return {"tags": tags}
    return pilot.fake_responder(messages)


def rel(path):
    """Path relative to the repo, so runs/ files carry no home directory."""
    path = Path(path).resolve()
    return str(path.relative_to(config.ROOT)) if path.is_relative_to(config.ROOT) else path.name


def _usage(llm):
    keys = ("prompt_tokens", "completion_tokens", "total_tokens")
    return {"calls": len(llm.responses), "models_returned": sorted({c["model"] for c in llm.responses}),
            **{k: sum(c[k] for c in llm.responses) for k in keys}}


def main(argv=None):
    args = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    args.add_argument("--fake", action="store_true", help="FakeLLM and the pilot's FAKE pages; no calls")
    args.add_argument("--n", type=int, default=20, help="no persona comments, and residents for skip_meeting")
    args.add_argument("--residents", default=str(config.PROCESSED / "town" / "residents.json"))
    args.add_argument("--suffix", default="", help="added to output file names, such as _seed1, so runs do not overwrite")
    args = args.parse_args(argv)
    command = shlex.join(["python", "-m", "townhall.report.baselines", *(argv if argv is not None else sys.argv[1:])])
    residents = json.loads(open(args.residents).read())["residents"][:args.n]
    if args.fake:
        llm, retriever, kind = FakeLLM(responder=fake_responder), pilot.fake_retriever, "fake"
        label = "FAKE: FakeLLM template and the pilot's 6 FAKE pages, not a model and not the policy. Simulated."
    else:
        from townhall.retrieval import Retriever

        llm, kind = pilot.capped_openai(max_calls=args.n + 1), "real"
        retriever = meeting.retriever_from(Retriever(), k=5, mode="both", llm=None)  # llm=None: no vision calls
        label = f"Simulated comments written by {config.RESIDENT_MODEL}, a real model, over the impact statement index."
    made = {"no_personas": no_personas(pilot.POLICY, args.n, retriever, llm),
            "skip_meeting": skip_meeting(pilot.POLICY, residents, retriever, llm)}
    OUT.mkdir(parents=True, exist_ok=True)
    for name, comments in made.items():
        path = OUT / f"baseline_{name}{'_fake' if args.fake else ''}{args.suffix}.json"
        path.write_text(json.dumps({"label": label, "comments": comments}, indent=2, ensure_ascii=False) + "\n")
    run = {"label": label, "run_type": kind, "command": command, "date": datetime.now(timezone.utc).date().isoformat(),
           "residents_file": rel(args.residents),
           **{name: {"comments": len(c), "with_cited_pages": sum(bool(x["cited_pages"]) for x in c),
                     "all_labeled_simulated": all(x["simulated"] is True for x in c)} for name, c in made.items()},
           "token_usage": "none: FakeLLM" if args.fake else _usage(llm)}
    (config.RUNS / f"report_baselines_{kind}{args.suffix}.json").write_text(json.dumps(run, indent=2) + "\n")
    print(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
