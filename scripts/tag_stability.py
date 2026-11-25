"""How much the tagger's codes change between two tagging runs of the same comments.

    python scripts/tag_stability.py data/processed/ingest/comments_tagged_v1_batch_order.jsonl \
        data/processed/ingest/comments_tagged.jsonl runs/ingest_tag_stability_feis.json

Writes counts only, no comment text. Same model, prompt and temperature 0 in both runs; what differs is which
comments shared a batch, and in the second run near duplicates were sent once.
"""

import json
import sys
from collections import Counter
from pathlib import Path


def load(path):
    return {r["id"]: r for r in map(json.loads, Path(path).read_text().splitlines())}


def jaccard(a, b):
    return 1.0 if not a and not b else len(a & b) / len(a | b)


def compare(old, new):
    ids = sorted(set(old) & set(new))
    group = Counter(new[i].get("tags_from", i) for i in ids)
    single = [i for i in ids if group[new[i].get("tags_from", i)] == 1]
    same = lambda xs: sum(set(old[i]["codes"]) == set(new[i]["codes"]) for i in xs)
    biggest = group.most_common(1)[0][0]
    form = [i for i in ids if new[i].get("tags_from", i) == biggest]
    codes = sorted({c for i in ids for r in (old[i], new[i]) for c in r["codes"]}, key=lambda c: [int(x) for x in c.split("-")])
    return {
        "comments_compared": len(ids),
        "identical_code_sets": same(ids),
        "mean_jaccard_of_code_sets": round(sum(jaccard(set(old[i]["codes"]), set(new[i]["codes"])) for i in ids) / len(ids), 3),
        "comments_with_no_near_duplicate": len(single),
        "identical_code_sets_among_them": same(single),
        "biggest_near_duplicate_group": len(form),
        "distinct_code_sets_in_it_old": len({tuple(sorted(old[i]["codes"])) for i in form}),
        "distinct_code_sets_in_it_new": len({tuple(sorted(new[i]["codes"])) for i in form}),
        "topic_counts_old_vs_new": {c: [sum(c in old[i]["codes"] for i in ids), sum(c in new[i]["codes"] for i in ids)]
                                    for c in codes},
    }


if __name__ == "__main__":
    old_path, new_path, out = sys.argv[1:4]
    report = {"command": "python " + " ".join(sys.argv), "old": Path(old_path).name, "new": Path(new_path).name,
              "note": "same model (gpt-4.1), prompt and temperature 0; the runs differ in batch neighbors and in "
                      "near duplicates being sent once in the new run", **compare(load(old_path), load(new_path))}
    Path(out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
