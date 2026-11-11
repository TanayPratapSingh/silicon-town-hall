"""Pilot answer key, steps 1 to 5: pages, comments, scrubbing, codebook, hand check sample, optional tagging.

    python -m townhall.ingest.run                 # OCR included, about 2 minutes
    python -m townhall.ingest.run --reuse-pages   # skip extraction, reuse data/processed/ingest/pages.jsonl
    python -m townhall.ingest.run --tag           # also tag every comment with the OpenAI model (costs money)

Writes data/processed/ingest/{pages,comments}.jsonl and handcheck.csv (gitignored), townhall/ingest/codebook.json,
and runs/ingest_stats.json with aggregate counts only.
"""

import argparse
import json
from collections import Counter

from townhall import config
from townhall.ingest import codebook, pages, scrub, split, tagger

DIR = config.PROCESSED / "ingest"
# comment types in NYSDOT's lists; a few cells run names into the type ("<first> & <first> Email"), so they are
# counted as unparsed rather than copied into runs/
LIST_TYPES = {"Email", "Petition", "Letter", "Voicemail", "Postcard", "Online Form", "Written Form", "Oral"}


def build_comments(all_pages, listed, log=print):
    """Split and scrub; returns the scrubbed comments, replacement counts and a leak audit."""
    raw = split.split_all(all_pages)
    chapters = "\n".join(p["text"] for p in all_pages if p["file"].startswith("feis_chapters/"))
    lex = scrub.Lexicon([name for _, name, _ in listed if name] + [n for c in raw for n in c["names"]],
                        chapters, "\n".join(c["text"] for c in raw))
    log(f"lexicon: {len(lex.full)} full names, {len(lex.places)} place words")
    comments, replaced = [], Counter()
    for i, c in enumerate(raw):
        text, counts = scrub.scrub(c["text"], c["names"], lex)
        replaced += counts
        comments.append({"id": f"c{i:05d}", "file": c["file"], "pages": c["pages"], "zip": c["zip"],
                         "kind": c["kind"], "text": text, "quote_cut": c["quote_cut"]})
    audit = {"emails_left": sum(len(scrub.EMAIL.findall(c["text"])) for c in comments),
             "phones_left": sum(len(scrub.PHONE.findall(c["text"])) for c in comments),
             "known_full_names_left": sum(1 for c in comments for m in scrub.FULL.finditer(c["text"])
                                          if (m.group(1).lower(), m.group(2).lower()) in lex.full)}
    return comments, replaced, audit


def main(argv=None):
    args = argparse.ArgumentParser(description="Pilot answer key pipeline")
    args.add_argument("--reuse-pages", action="store_true", help="reuse pages.jsonl instead of extracting again")
    args.add_argument("--tag", action="store_true", help="tag all comments with the OpenAI model (costs money)")
    args = args.parse_args(argv)

    all_pages = pages.load() if args.reuse_pages else pages.extract()
    listed = scrub.commenter_lists(all_pages)
    comments, replaced, audit = build_comments(all_pages, listed)
    with open(DIR / "comments.jsonl", "w") as f:
        f.writelines(json.dumps(c) + "\n" for c in comments)
    topics, toc_entries, headings = codebook.write(all_pages)
    sample = tagger.write_handcheck(comments)

    stats = {
        "pages": len(all_pages),
        "pages_needing_ocr": sum(p["needed_ocr"] for p in all_pages),
        "pages_ocr_text_used": sum(p["ocr"] for p in all_pages),
        "pages_still_20_words_or_fewer": sum(p["words"] <= pages.MIN_WORDS for p in all_pages),
        "needing_ocr_by_group": dict(Counter(p["file"].split("/")[0] for p in all_pages if p["needed_ocr"])),
        "comments": len(comments),
        "comments_per_file": dict(Counter(c["file"] for c in comments)),
        "comments_per_kind": dict(Counter(c["kind"] for c in comments)),
        "comments_with_quote_cut": sum(c["quote_cut"] for c in comments),
        "comments_with_zip": sum(bool(c["zip"]) for c in comments),
        "replacements": dict(replaced),
        "audit_after_scrub": audit,
        # rows in NYSDOT's own lists of commenters by comment type, to compare the split against
        "nysdot_list_rows_by_type": {f.split("/")[-1]: dict(Counter(t if t in LIST_TYPES else "unparsed"
                                                             for g, _, t in listed if g == f))
                                     for f in sorted({g for g, _, _ in listed})},
        "codebook_topics": len(topics),
        "appendix_a_toc_entries": toc_entries,
        "appendix_a_headings_found": headings,
        "handcheck_sample": len(sample),
        "handcheck_kinds": dict(Counter(c["kind"] for c in sample)),
    }
    (config.RUNS / "ingest_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))
    if args.tag:
        tagger.main([])
    elif config.OPENAI_API_KEY:
        print("OPENAI_API_KEY is set but tagging was not requested; run with --tag or python -m townhall.ingest.tagger")
    else:
        print("OPENAI_API_KEY is not set; skipping tagging.")


if __name__ == "__main__":
    main()
