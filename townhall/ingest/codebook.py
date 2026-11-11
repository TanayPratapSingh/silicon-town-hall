"""Step 4: NYSDOT's topic codes from Appendix A (Responses to Comments on the FDR/FEIS, June 2022).

Appendix A gives no definitions, so each topic's line is NYSDOT's own wording: the first sentence of the
section that starts with "Commenters" (a group summary), else the first summarized comment in the section.
These read as examples more than definitions and should be reviewed before they steer a tagger.
Sections 1 (Introduction), 9 (letters grouped by organization) and 10 (list of commenters) are not topics,
and section 7 is represented by its subsections.
"""

import json
import re
from pathlib import Path

from townhall import config
from townhall.ingest.pages import ocr_page

FILE = "decision/appendix-response-comments.pdf"
OUT = Path(__file__).with_name("codebook.json")
TOC = re.compile(r"(\d{1,2}(?:-\d{1,2})?)\s+([A-Z][^.\n]*?)\s*\.{3,}\s*(\d+)")
RUNNING = re.compile(r"I-81 Viaduct Project Appendix A: Responses to Comments \d+ ")
LABEL = re.compile(r"\b[CR] \d{1,2}-\d{1,3}: ")


def flat(text):
    return re.sub(r"\s+", " ", text.replace("‐", "-")).strip() + " "


def first_line(section):
    section = LABEL.sub("", RUNNING.sub("", section)).strip()
    sentences = [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\"“])", section) if not s.startswith("See ")]
    if not sentences:
        return ""
    line = next((s for s in sentences if s.startswith("Commenters")), sentences[0])
    if len(line) > 200 and re.search(r"[:;]", line):
        line = re.split(r"[:;]", line)[0] + "."
    return line.strip()


def parse(pages):
    pages = [p for p in pages if p["file"] == FILE]
    toc = [(code, topic.strip()) for code, topic, _ in TOC.findall(pages[2]["text"])]
    body = [(p["page"], flat(p["text"])) for p in pages[3:]]
    joined = "".join(text for _, text in body)
    starts = [(len("".join(t for _, t in body[:i])), page) for i, (page, _) in enumerate(body)]
    heads = []
    for code, topic in toc:
        m = re.search(re.escape(f"{code} {topic.upper()} "), joined)
        if m:
            heads.append((m.start(), m.end(), code, topic))
    heads.sort()
    codebook = []
    for i, (start, end, code, topic) in enumerate(heads):
        children = any(c.startswith(code + "-") for _, _, c, _ in heads)
        if code in {"1", "10"} or code.split("-")[0] == "9" or children:
            continue
        section = joined[end:heads[i + 1][0] if i + 1 < len(heads) else len(joined)]
        page = max(p for offset, p in starts if offset <= start)
        line = first_line(section[:3000])
        if not line:  # pages 12 and 13 print section 5's comments without a text layer; read them by OCR
            ocr = flat(ocr_page((str(config.RAW / FILE), page - 1)))
            m = re.search(re.escape(topic.upper()), ocr)
            line = first_line(ocr[m.end():]) if m else ""
        codebook.append({"code": code, "topic": topic, "definition": line, "source_page": page})
    return codebook, len(toc), len(heads)


def write(pages, out=OUT):
    codebook, toc_entries, found = parse(pages)
    out.write_text(json.dumps(codebook, indent=2) + "\n")
    return codebook, toc_entries, found


def load(path=OUT):
    return json.loads(Path(path).read_text())
