"""Count pages, words and pages with no usable text layer for every PDF in data/raw/.

Run from the repo root after download.py: python scripts/pdf_stats.py
Writes runs/pdf_stats.csv, the source of every page and word count quoted in docs/.
"""
import csv
from pathlib import Path

from pypdf import PdfReader

# A page this short is a scan or a blank divider and needs OCR before tagging.
MIN_WORDS = 20


def main(raw=Path("data/raw"), out=Path("runs/pdf_stats.csv")):
    rows = []
    for pdf in sorted(raw.rglob("*.pdf")):
        pages = [page.extract_text() or "" for page in PdfReader(pdf).pages]
        rows.append({
            "file": pdf.relative_to(raw).as_posix(),
            "pages": len(pages),
            "words": sum(len(text.split()) for text in pages),
            "pages_needing_ocr": sum(len(text.split()) <= MIN_WORDS for text in pages),
        })
        print(rows[-1])
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
