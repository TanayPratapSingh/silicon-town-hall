"""Step 5: tag scrubbed comments with NYSDOT topic codes, and write the blind hand check sample.

    python -m townhall.ingest.tagger --groups feis_comments --cap 1.00   # needs OPENAI_API_KEY; costs money

Comments go to the model in batches, so the codebook is sent once per batch. Every batch's tags are appended to
data/processed/ingest/comments_tagged.jsonl as they arrive, comments already there are skipped (rerun to resume),
and the run stops before any call whose worst case cost would pass the dollar cap.
"""

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict

from townhall import config
from townhall.ingest import codebook as cb
from townhall.ingest.scrub import EMAIL, PHONE

DIR = config.PROCESSED / "ingest"
TAGGED = DIR / "comments_tagged.jsonl"
MAX_CHARS = 12000  # shortcut: long letters with exhibits are cut here; chunk them if their tags matter
BATCH_CHARS, BATCH_SIZE = 20000, 10
# Near duplicates are tagged once: word 5 gram overlap of 0.9 groups all 92 FEIS form emails, while 0.8 starts
# merging distinct letters. Tags differed across copies only because batch neighbors differed.
DEDUPE_JACCARD = 0.9
# gpt-4.1 list prices in dollars per token, from the lead; costs below are estimates from these, not the bill
PROMPT_PRICE, COMPLETION_PRICE = 2.00 / 1e6, 8.00 / 1e6

SYSTEM = ("You label public comments on the I-81 Viaduct Project in Syracuse with NYSDOT's topic codes from "
          "Appendix A of the Record of Decision. Use only codes from the list. A comment can have several codes "
          "or none. Reply as JSON with one entry per comment id: {\"tags\": {\"c00001\": [\"4\", \"7-5\"]}}.")


def prompt(batch, codebook):
    topics = "\n".join(f"{t['code']}: {t['topic']}. Example from NYSDOT: {t['definition']}" for t in codebook)
    comments = "\n\n".join(f"[{c['id']}]\n{c['text'][:MAX_CHARS]}" for c in batch)
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Topic codes:\n{topics}\n\nComments:\n{comments}"}]


def valid(codes, codebook):
    """Known codes in codebook order; [] for anything that is not a list."""
    if not isinstance(codes, list):
        return []
    wanted = {str(c).strip() for c in codes}
    return [t["code"] for t in codebook if t["code"] in wanted]


def parse(reply, ids, codebook):
    """{id: codes} for the ids the reply covers; ids it leaves out are missing and get retried on the next run."""
    try:
        data = json.loads(reply)
    except (json.JSONDecodeError, TypeError):
        return {}
    if isinstance(data, list) and len(ids) == 1:
        data = {"codes": data}
    if not isinstance(data, dict):
        return {}
    tags = data.get("tags", data)
    if len(ids) == 1 and "codes" in data:
        tags = {ids[0]: data["codes"]}
    elif len(ids) == 1 and isinstance(tags, dict) and len(tags) == 1:
        tags = {ids[0]: next(iter(tags.values()))}  # one comment sent: the model sometimes renames its id ("c1" for "c")
    return {i: valid(tags[i], codebook) for i in ids if isinstance(tags, dict) and i in tags}


def call(llm, messages, max_tokens):
    """(reply, prompt tokens, completion tokens). Usage comes from the OpenAI response; FakeLLM gets a length guess.
    shortcut: townhall/llm.py returns text only, so the OpenAI client is called here to read usage."""
    if hasattr(llm, "client"):
        response = llm.client.chat.completions.create(
            model=config.ANALYST_MODEL, messages=messages, temperature=0, max_tokens=max_tokens,
            response_format={"type": "json_object"})
        return response.choices[0].message.content, response.usage.prompt_tokens, response.usage.completion_tokens
    reply = llm.chat(messages, model=config.ANALYST_MODEL, temperature=0, json_mode=True)
    return reply, sum(len(m["content"]) for m in messages) // 4, len(reply) // 4


def cost(prompt_tokens, completion_tokens):
    return prompt_tokens * PROMPT_PRICE + completion_tokens * COMPLETION_PRICE


def tag(comment_text, codebook, llm):
    """Codes for one comment."""
    reply, _, _ = call(llm, prompt([{"id": "c", "text": comment_text}], codebook), max_tokens=100)
    return parse(reply, ["c"], codebook).get("c", [])


def shingles(text):
    words = re.findall(r"[a-z0-9]+", re.sub(r"\[(NAME|EMAIL|PHONE|ADDRESS)\]", " ", text.lower()))
    return {" ".join(words[i:i + 5]) for i in range(max(1, len(words) - 4))}


def representatives(comments, threshold=DEDUPE_JACCARD):
    """{comment id: id of the first comment in its near duplicate group}.
    shortcut: compares every pair, fine for a few thousand comments; use MinHash for a much bigger corpus."""
    sets = [shingles(c["text"]) for c in comments]
    parent = list(range(len(comments)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(sets):
        for j in range(i + 1, len(sets)):
            b = sets[j]
            if a == b or (min(len(a), len(b)) >= max(5, threshold * max(len(a), len(b)))
                          and len(a & b) >= threshold * len(a | b)):
                ri, rj = root(i), root(j)
                parent[max(ri, rj)] = min(ri, rj)
    return {c["id"]: comments[root(i)]["id"] for i, c in enumerate(comments)}


def batches(comments):
    batch, size = [], 0
    for c in comments:
        n = min(len(c["text"]), MAX_CHARS)
        if batch and (len(batch) == BATCH_SIZE or size + n > BATCH_CHARS):
            yield batch
            batch, size = [], 0
        batch.append(c)
        size += n
    if batch:
        yield batch


def load_comments(path=DIR / "comments.jsonl"):
    with open(path) as f:
        return [json.loads(line) for line in f]


def tag_all(comments, codebook, llm, cap, out=TAGGED, groups=None):
    """Tag the comments in `groups` (path prefixes such as 'feis_comments') not yet in `out`, under a dollar cap.
    Each near duplicate group is sent once and its codes are copied to every copy, marked with tags_from."""
    done = {c["id"] for c in load_comments(out)} if out.exists() else set()
    chosen = [c for c in comments if not groups or c["file"].split("/")[0] in groups]
    rep, copies = representatives(chosen), defaultdict(list)
    for c in chosen:
        copies[rep[c["id"]]].append(c)
    todo = [c for c in chosen if rep[c["id"]] == c["id"] and any(d["id"] not in done for d in copies[c["id"]])]
    stats = Counter(selected=len(chosen), distinct_texts=len(copies),
                    skipped_already_tagged=sum(c["id"] in done for c in chosen))
    spent = 0.0
    with open(out, "a") as f:
        for batch in batches(todo):
            if any(EMAIL.search(c["text"]) or PHONE.search(c["text"]) for c in batch):
                raise ValueError("refusing to send text with an email or phone number left in it")
            messages = prompt(batch, codebook)
            max_tokens = 40 * len(batch) + 50
            worst = cost(sum(len(m["content"]) for m in messages) // 3, max_tokens)  # 3 chars a token: high side
            if spent + worst > cap:
                stats["stopped_by_cap"] = 1
                break
            reply, p, q = call(llm, messages, max_tokens)
            spent += cost(p, q)
            stats.update(calls=1, prompt_tokens=p, completion_tokens=q)
            tags = parse(reply, [c["id"] for c in batch], codebook)
            stats.update(tagged=len(tags), missing_from_reply=len(batch) - len(tags),
                         copied_to_duplicates=sum(len(copies[i]) - 1 for i in tags))
            f.writelines(json.dumps({**d, "codes": tags[c["id"]], "model": config.ANALYST_MODEL,
                                     "tags_from": c["id"]}) + "\n"
                         for c in batch if c["id"] in tags for d in copies[c["id"]] if d["id"] not in done)
            f.flush()
    return stats, spent


def write_handcheck(comments, path=DIR / "handcheck.csv", n=100, seed=0):
    """Random sample for teammates to label blind: no model codes in the file, so they cannot anchor on them."""
    sample = random.Random(seed).sample(comments, min(n, len(comments)))
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "file", "pages", "kind", "text", "human_codes"])
        for c in sample:
            writer.writerow([c["id"], c["file"], f"{c['pages'][0]}-{c['pages'][1]}", c["kind"], c["text"], ""])
    return sample


def main(argv=None, llm=None):
    args = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    args.add_argument("--groups", nargs="*", help="source groups to tag, e.g. feis_comments (default: all)")
    args.add_argument("--cap", type=float, default=1.00, help="stop before estimated spend passes this many dollars")
    args = args.parse_args(argv)
    if llm is None:
        if not config.OPENAI_API_KEY:
            print("OPENAI_API_KEY is not set; skipping tagging. Add it to .env to tag comments.")
            return 0
        from townhall.llm import OpenAILLM

        llm = OpenAILLM()
    stats, spent = tag_all(load_comments(DIR / "comments.jsonl"), cb.load(), llm, args.cap, out=TAGGED,
                           groups=args.groups)
    groups = set(args.groups or [])
    name = "_".join(g.split("_")[0] for g in sorted(groups)) or "all"
    path = config.RUNS / f"ingest_tags_{name}.json"
    before = json.loads(path.read_text()) if path.exists() else {}  # a resumed run adds to the earlier totals
    tagged = [c for c in load_comments(TAGGED) if not groups or c["file"].split("/")[0] in groups]
    report = {"model": config.ANALYST_MODEL, "groups": sorted(groups) or "all", "cap_dollars": args.cap,
              "dedupe_jaccard": DEDUPE_JACCARD,
              **{k: stats[k] for k in ("selected", "distinct_texts", "skipped_already_tagged", "tagged",
                                       "copied_to_duplicates", "missing_from_reply", "calls", "prompt_tokens",
                                       "completion_tokens", "stopped_by_cap")},
              "estimated_dollars_this_run": round(spent, 4),
              **{f"{k}_all_runs": before.get(f"{k}_all_runs", before.get(k, 0)) + stats[k]
                 for k in ("calls", "prompt_tokens", "completion_tokens")},
              "estimated_dollars_all_runs": round(before.get("estimated_dollars_all_runs",
                                                             before.get("estimated_dollars_this_run", 0)) + spent, 4),
              "estimate_note": "from list prices of $2.00 per million prompt and $8.00 per million completion "
                               "tokens, not from the OpenAI bill",
              "comments_tagged_total": len(tagged),
              "comments_with_no_code": sum(not c["codes"] for c in tagged),
              "topic_counts": dict(Counter(code for c in tagged for code in c["codes"]))}
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
