"""Fake pilot: 20 Census residents, one simulated meeting, one comment each, end to end.

    .venv/bin/python -m townhall.town.pilot [--n 20] [--seed 0] [--turns N]
    .venv/bin/python -m townhall.town.pilot --real-profiles   # n real profile calls, no meeting

The residents and their neighborhoods are real Census draws. In the fake pilot every piece of text
(profiles, agenda, turns, summary, comments, LLM reviews) comes from FakeLLM, a template that is not a
model, and the retriever searches six FAKE pages, not the impact statement. It tests the plumbing only.
Writes data/processed/town/residents.json, meeting_pilot.json, comments.json and memory.sqlite
(gitignored), and runs/town_pilot_fake.json with counts only: no Census microdata rows.

--real-profiles writes the same residents' profiles with RESIDENT_MODEL (one call each, capped at n),
reviews them with the keyword heuristics only, and writes data/processed/town/residents_real.json and
runs/town_profiles_real.json (counts and token usage only). No meeting runs.
"""

import argparse
import json
import re
import sys
import zlib
from collections import Counter
from datetime import datetime, timezone

from townhall import config
from townhall.llm import FakeLLM, OpenAILLM
from townhall.town import census, homes, meeting, personas
from townhall.town.memory import Memory

OUT = config.PROCESSED / "town"
RUN_FILE = config.RUNS / "town_pilot_fake.json"
REAL_RUN_FILE = config.RUNS / "town_profiles_real.json"
FAKE_LABEL = "FAKE: FakeLLM template and fake retriever, not a model and not the policy. Simulated."
POLICY = "NYSDOT I-81 Viaduct Project, Final Environmental Impact Statement, April 2022"
TOPIC = "I-81 Viaduct Project"

FAKE_PAGES = [
    {"file": "FAKE_policy.pdf", "page": i + 1, "text": f"FAKE PAGE for tests, not from the FEIS. Topic: {topic}."}
    for i, topic in enumerate(["traffic and commuting by car", "bus and transit service",
                               "walking and bicycle routes", "noise and air quality",
                               "construction disruption", "housing, rent and displacement"])
]


def fake_retriever(query, cited_pages, k=3):
    """Ranks the FAKE pages by word overlap with the query; already cited pages get a small boost."""
    words = set(re.findall(r"[a-z]+", query.lower()))
    cited = {(str(d), int(p)) for d, p in cited_pages}

    def score(page):
        overlap = len(words & set(re.findall(r"[a-z]+", page["text"].lower())))
        return overlap + 0.5 * ((page["file"], page["page"]) in cited)

    return sorted(FAKE_PAGES, key=score, reverse=True)[:k]


def _pick(text, options):
    return options[zlib.crc32(text.encode()) % len(options)]  # deterministic, and unrelated to who the resident is


def fake_responder(messages):
    """Canned FakeLLM replies, one shape per prompt. Not a model: nothing here reads the policy."""
    system, user = messages[0]["content"], messages[-1]["content"]
    if system == personas.PROFILE_SYSTEM:
        f = json.loads(user.split("Facts (JSON): ", 1)[1])
        return (f"[FakeLLM template, simulated] I am {f['age']} and live in {f['neighborhood']}. "
                f"Census record: {f['sex']}, {f['race_ethnicity']}, income about ${f['annual_income_dollars']:,} a year. "
                f"Commute: {f['commute']}. Housing: {f['tenure']}. {f['disability']}.")
    if system == personas.REVIEW_SYSTEM:
        return {"stereotype": False, "reasons": [], "note": "FakeLLM: no review was done"}
    if system == meeting.AGENDA_SYSTEM:
        return {"agenda": ["Traffic and commuting", "Noise and air quality", "Construction and housing"]}
    if system == meeting.SPEAK_SYSTEM:
        return {"text": "[FakeLLM, simulated] I respond to the last speaker and cite excerpt 1.",
                "stance": _pick(user, meeting.STANCES), "cited": [1]}
    if system == meeting.SUMMARY_SYSTEM:
        return "[FakeLLM, simulated] Summary placeholder after " + user.split("New turn by ", 1)[-1][:60]
    if system == meeting.COMMENT_SYSTEM:
        return {"text": "[FakeLLM, simulated] My one comment on the policy cites excerpts 1 and 2.",
                "stance": _pick(user, meeting.STANCES), "reasons": ["FakeLLM placeholder reason"], "cited": [1, 2]}
    raise ValueError("fake_responder got a prompt it does not know")


def capped_openai(max_calls):
    """OpenAILLM that refuses call number max_calls + 1 and records each response's model and token usage."""
    llm = OpenAILLM()
    create, llm.attempts, llm.responses = llm.client.chat.completions.create, 0, []

    def counted(**kwargs):
        if llm.attempts >= max_calls:
            raise RuntimeError(f"real call cap of {max_calls} reached")
        llm.attempts += 1
        response = create(**kwargs)
        llm.responses.append({"model": response.model, **response.usage.model_dump(include={
            "prompt_tokens", "completion_tokens", "total_tokens"})})
        return response

    llm.client.chat.completions.create = counted
    return llm


def real_profiles(residents, census_meta, home, command):
    llm = capped_openai(max_calls=len(residents))
    for r in residents:
        r["profile"] = personas.build_profile(r, llm)
        r["profile_written_by"] = f"{config.RESIDENT_MODEL}, a real model (simulated resident)"
        r["stereotype_review"] = personas.review_profile(r["profile"])  # keyword heuristics only: stays inside the cap
        r["simulated"] = True
    label = (f"REAL PROFILES written by {config.RESIDENT_MODEL} from real Census draws. The residents are simulated; "
             "the stereotype review here is keyword heuristics only, no LLM review was run.")
    _write(OUT / "residents_real.json", {"label": label, "census": census_meta, "home_source": home, "residents": residents})
    flags = [f for r in residents for f in r["stereotype_review"]["keyword_flags"]]
    usage = {k: sum(c[k] for c in llm.responses) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
    run = {"label": label, "command": command, "date": datetime.now(timezone.utc).date().isoformat(), "census": census_meta,
           "homes": {**home, "neighborhoods_drawn": len({r["neighborhood"] for r in residents})},
           "profiles": {"count": len(residents), "model_requested": config.RESIDENT_MODEL,
                        "models_returned": sorted({c["model"] for c in llm.responses}), "real_calls": llm.attempts,
                        "token_usage": usage,
                        "keyword_flagged_profiles": sum(bool(r["stereotype_review"]["keyword_flags"]) for r in residents),
                        "keyword_flag_counts": {f: flags.count(f) for f in sorted(set(flags))},
                        "llm_review": "not run, to stay within the cap of one real call per profile"}}
    _write(REAL_RUN_FILE, run)
    print(json.dumps(run, indent=2))


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def main(argv=None):
    args = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    args.add_argument("--n", type=int, default=20)
    args.add_argument("--seed", type=int, default=0)
    args.add_argument("--turns", type=int, default=None, help="meeting turns; default one per resident")
    args.add_argument("--real-profiles", action="store_true", help="write profiles with RESIDENT_MODEL, no meeting")
    args = args.parse_args(argv)
    command = " ".join(["python -m townhall.town.pilot", *(argv if argv is not None else sys.argv[1:])])

    people, census_source = census.load_people("36", "00701")
    pool = census.adults(people)
    residents = census.sample_residents("36", "00701", n=args.n, seed=args.seed, people=people, source=census_source)
    residents, home = homes.home_neighborhoods(residents, "36", "00701", seed=args.seed)
    census_meta = {"source": census_source, "state": "36", "puma": "00701", "person_records_in_puma": len(people),
                   "adult_records_in_puma": len(pool), "weighted_adults_in_puma": int(pool.PWGTP.astype(int).sum()),
                   "sampled": len(residents), "seed": args.seed,
                   **{f"sample_by_{key}": dict(Counter(r[key] for r in residents).most_common())
                      for key in ("race_ethnicity", "sex", "tenure", "commute", "disability")}}
    if args.real_profiles:
        return real_profiles(residents, census_meta, home, command)

    llm = FakeLLM(responder=fake_responder)
    for r in residents:
        r["profile"] = personas.build_profile(r, llm)
        r["stereotype_review"] = personas.review_profile(r["profile"], llm)
        r["simulated"] = True
    _write(OUT / "residents.json", {"label": FAKE_LABEL, "census": census_meta, "home_source": home,
                                    "residents": residents})

    memory = Memory(OUT / "memory.sqlite")
    state = meeting.run_meeting(POLICY, residents, fake_retriever, llm, max_turns=args.turns,
                                memory=memory, round=1, topic=TOPIC)
    _write(OUT / "meeting_pilot.json", {"label": FAKE_LABEL, "policy": POLICY, "agenda": state["agenda"],
                                        "transcript": state["transcript"], "summary": state["summary"]})
    _write(OUT / "comments.json", {"label": FAKE_LABEL, "comments": state["comments"]})

    comments = state["comments"]
    run = {
        "label": "FAKE PILOT. Residents and neighborhoods are real Census draws; all text is FakeLLM template "
                 "output and retrieval is over 6 FAKE pages. Counts test the plumbing only and say nothing "
                 "about what residents think. Simulated.",
        "command": command,
        "date": datetime.now(timezone.utc).date().isoformat(),
        "census": census_meta,
        "homes": {**home, "neighborhoods_drawn": len({r["neighborhood"] for r in residents}),
                  "residents_unassigned": sum(r["neighborhood"] == homes.UNASSIGNED for r in residents)},
        "profiles": {"written_by": "FakeLLM template", "count": len(residents),
                     "keyword_flagged": sum(bool(r["stereotype_review"]["keyword_flags"]) for r in residents),
                     "llm_review": "FakeLLM, no review was done"},
        "meeting": {"turns": len(state["transcript"]), "agenda_items": len(state["agenda"]),
                    "distinct_speakers": len({t["resident_id"] for t in state["transcript"]}),
                    "pages_cited_in_meeting": len(state["cited_pages"]), "reply_parse_failures": state["parse_failures"],
                    "llm_calls_total_including_profiles": len(llm.calls)},
        "comments": {"count": len(comments), "with_cited_pages": sum(bool(c["cited_pages"]) for c in comments),
                     "all_labeled_simulated": all(c["simulated"] is True for c in comments)},
        "memory": {"rows_after_pilot": memory.count(), "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"},
    }
    memory.close()
    _write(RUN_FILE, run)
    print(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
