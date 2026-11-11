"""Step 5: tag scrubbed comments with NYSDOT topic codes, and write the blind hand check sample.

    python -m townhall.ingest.tagger            # tag every comment (needs OPENAI_API_KEY; costs money)
    python -m townhall.ingest.tagger --limit 10  # tag the first 10 only
"""

import argparse
import csv
import json
import random
import sys
from collections import Counter

from townhall import config
from townhall.ingest import codebook as cb

DIR = config.PROCESSED / "ingest"
MAX_CHARS = 12000  # shortcut: long letters with exhibits are cut here; chunk them if their tags matter

SYSTEM = ("You label public comments on the I-81 Viaduct Project in Syracuse with NYSDOT's topic codes from "
          "Appendix A of the Record of Decision. Use only codes from the list. A comment can have several codes "
          "or none. Reply as JSON: {\"codes\": [\"4\", \"7-5\"]}.")


def prompt(comment_text, codebook):
    topics = "\n".join(f"{t['code']}: {t['topic']}. Example from NYSDOT: {t['definition']}" for t in codebook)
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Topic codes:\n{topics}\n\nComment:\n{comment_text[:MAX_CHARS]}"}]


def parse(reply, codebook):
    """Valid codes from the model's JSON reply, in codebook order; [] when the reply is not usable JSON."""
    try:
        data = json.loads(reply)
    except (json.JSONDecodeError, TypeError):
        return []
    codes = data.get("codes", []) if isinstance(data, dict) else data
    if not isinstance(codes, list):
        return []
    wanted = {str(c).strip() for c in codes}
    return [t["code"] for t in codebook if t["code"] in wanted]


def tag(comment_text, codebook, llm):
    return parse(llm.chat(prompt(comment_text, codebook), model=config.ANALYST_MODEL, temperature=0, json_mode=True),
                 codebook)


def load_comments(path=DIR / "comments.jsonl"):
    with open(path) as f:
        return [json.loads(line) for line in f]


def write_handcheck(comments, path=DIR / "handcheck.csv", n=100, seed=0):
    """Random sample for teammates to label blind: no model codes in the file, so they cannot anchor on them."""
    sample = random.Random(seed).sample(comments, min(n, len(comments)))
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "file", "pages", "kind", "text", "human_codes"])
        for c in sample:
            writer.writerow([c["id"], c["file"], f"{c['pages'][0]}-{c['pages'][1]}", c["kind"], c["text"], ""])
    return sample


def tag_all(comments, codebook, llm, out=DIR / "tags.jsonl"):
    counts = Counter()
    with open(out, "w") as f:
        for c in comments:
            codes = tag(c["text"], codebook, llm)
            counts.update(codes)
            f.write(json.dumps({"id": c["id"], "codes": codes}) + "\n")
    return counts


def main(argv=None):
    args = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    args.add_argument("--limit", type=int, help="tag only the first N comments")
    args = args.parse_args(argv)
    if not config.OPENAI_API_KEY:
        print("OPENAI_API_KEY is not set; skipping tagging. Add it to .env to tag comments.")
        return 0
    from townhall.llm import OpenAILLM

    comments = load_comments()[:args.limit]
    counts = tag_all(comments, cb.load(), OpenAILLM())
    stats = {"comments_tagged": len(comments), "model": config.ANALYST_MODEL, "topic_counts": dict(counts)}
    (config.RUNS / "ingest_tags.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
