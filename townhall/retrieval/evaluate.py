"""Score retrieval on runs/retrieval_testset.csv: recall@5, recall@10 and MRR@10 per pipeline setting.

`python -m townhall.retrieval.evaluate` runs the four free settings. `--vision` adds the vision
rerank row, which calls the real VISION_MODEL on the top 10 page images (capped, tokens counted).
Writes runs/retrieval_eval.json and runs/retrieval_eval_questions.csv.
"""

import argparse
import csv
import json
import re
import time

from townhall import config
from townhall.retrieval.search import Retriever

TESTSET = config.RUNS / "retrieval_testset.csv"
SETTINGS = {  # name: search keyword arguments
    "text_only": dict(mode="text", rerank=False, vision=False),
    "image_only": dict(mode="image", rerank=False, vision=False),
    "rrf": dict(mode="both", rerank=False, vision=False),
    "rrf_cross_encoder": dict(mode="both", rerank=True, vision=False),
}
STOP = set("""a an the and or of to in on for with by at from is are was were be been will would could can
do does did i my me we our it its this that these those there their they them what which who how when where
why if than then as into about up down out over under any all each per vs versus show near""".split())


def content_words(text):
    return {w.rstrip("s") for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP and len(w) > 2}


def counted_vision_llm(cap):
    """The real OpenAI client, with a hard cap on calls and token totals read from each API response."""
    from townhall.llm import OpenAILLM

    llm, usage = OpenAILLM(), {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cap": cap,
                        "unparsed_replies": 0}
    create = llm.client.chat.completions.create

    def create_counted(**kwargs):
        if usage["calls"] >= cap:
            raise RuntimeError(f"vision call cap of {cap} reached")
        usage["calls"] += 1
        response = create(**kwargs)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            usage[key] += getattr(response.usage, key)
        return response

    llm.client.chat.completions.create = create_counted
    return llm, usage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vision", action="store_true", help="add the paid vision rerank row")
    parser.add_argument("--cap", type=int, default=300, help="maximum vision calls")
    args = parser.parse_args()

    questions = list(csv.DictReader(TESTSET.open()))
    settings = dict(SETTINGS)
    usage = None
    if args.vision:
        if not config.OPENAI_API_KEY:
            raise SystemExit("--vision needs OPENAI_API_KEY in .env")
        llm, usage = counted_vision_llm(args.cap)
        settings["rrf_cross_encoder_vision"] = dict(mode="both", rerank=True, vision=True, llm=llm)

    retriever = Retriever()
    ranks = {name: [] for name in settings}
    near = {name: [] for name in settings}  # rank of the gold page or a page next to it in the same file
    seconds, stages = {}, {}
    for name, kwargs in settings.items():
        started = time.perf_counter()
        for q in questions:
            hits = retriever.search(q["question"], k=50, **kwargs)
            if usage and "llm" in kwargs:
                usage["unparsed_replies"] += sum(h.stage_scores["vision"] is None for h in hits[:10])
            gold = (q["gold_file"], int(q["gold_page"]))
            ranks[name].append(next((i for i, h in enumerate(hits, 1) if (h.file, h.page) == gold), None))
            near[name].append(next((i for i, h in enumerate(hits, 1) if h.file == gold[0] and abs(h.page - gold[1]) <= 1), None))
        seconds[name] = round((time.perf_counter() - started) / len(questions), 2)
        stages[name] = retriever.last_stages
        print(name, "done")

    def metrics(rows):
        n = len(rows)
        return {"questions": n,
                "recall@5": round(sum(r is not None and r <= 5 for r in rows) / n, 3),
                "recall@10": round(sum(r is not None and r <= 10 for r in rows) / n, 3),
                "mrr@10": round(sum(1 / r for r in rows if r is not None and r <= 10) / n, 3),
                "found_in_returned_list": sum(r is not None for r in rows)}

    kinds = sorted({q["kind"] for q in questions})
    overlaps = []
    for q in questions:
        words = content_words(q["question"])
        shared = sorted(words & content_words(q["source_anchor"]))
        q["anchor_words_in_question"] = " ".join(shared)
        q["anchor_overlap"] = round(len(shared) / len(words), 2)
        overlaps.append(q["anchor_overlap"])

    result = {
        "command": "python -m townhall.retrieval.evaluate" + (" --vision" if args.vision else ""),
        "testset": "runs/retrieval_testset.csv",
        "gold": "one page per question (file plus 1 based physical page); any other page counts as a miss",
        "secondary_metric": "recall@10 counting the page just before or after the gold page in the same file, because sections run across page breaks; strict numbers are the headline",
        "returned_per_setting": "top 50 without the cross encoder, top 20 with it (the cross encoder trims to 20)",
        "anchor_overlap": {"mean": round(sum(overlaps) / len(overlaps), 3), "max": max(overlaps),
                           "questions_with_any_overlap": sum(o > 0 for o in overlaps),
                           "meaning": "share of a question's content words that also appear in the heading or title it was written from"},
        "settings": {name: {"search_kwargs": {k: v for k, v in kwargs.items() if k != "llm"} | ({"llm": "OpenAI " + config.VISION_MODEL} if "llm" in kwargs else {}),
                            "stages": stages[name], "seconds_per_query": seconds[name],
                            **metrics(ranks[name]),
                "secondary_recall@10_gold_page_plus_or_minus_1": metrics(near[name])["recall@10"],
                            "by_kind": {kind: metrics([r for r, q in zip(ranks[name], questions) if q["kind"] == kind]) for kind in kinds}}
                     for name, kwargs in settings.items()},
    }
    if usage:
        result["vision_usage"] = usage | {"model": config.VISION_MODEL, "image_dpi": 100}
    (config.RUNS / "retrieval_eval.json").write_text(json.dumps(result, indent=1) + "\n")

    with (config.RUNS / "retrieval_eval_questions.csv").open("w", newline="") as f:
        cols = ["id", "kind", "gold_file", "gold_page", "anchor_overlap", "anchor_words_in_question"]
        writer = csv.writer(f)
        writer.writerow(cols + [f"rank_{name}" for name in settings])
        for i, q in enumerate(questions):
            writer.writerow([q[c] for c in cols] + [ranks[name][i] or "" for name in settings])
    print(json.dumps({name: {k: v for k, v in s.items() if k not in ("by_kind", "stages", "search_kwargs")}
                      for name, s in result["settings"].items()}, indent=1))
    if usage:
        print(json.dumps(result["vision_usage"]))


if __name__ == "__main__":
    main()
