"""Step 1: one record per PDF page, with OCR for pages that have no usable text layer."""

import json
import unicodedata
from concurrent.futures import ProcessPoolExecutor

import pymupdf

from townhall import config

# Same cutoff as scripts/pdf_stats.py: a page this short is a scan or a blank divider.
MIN_WORDS = 20
OUT = config.PROCESSED / "ingest" / "pages.jsonl"
LIGATURES = str.maketrans({"Ʃ": "tt", "Ɵ": "ti", "Ō": "ft"})
# Comment files and the decision documents first, the chapters residents read last.
PRIORITY = ["feis_comments", "deis_comments", "decision", "feis_chapters"]


def normalize(text):
    """Undo ligatures; some FEIS email fonts map 'tt', 'ti', 'ft' to these letters ('aƩached')."""
    return unicodedata.normalize("NFKC", text).translate(LIGATURES)


def ocr_page(args):
    path, index = args
    import pytesseract
    from PIL import Image

    with pymupdf.open(path) as doc:
        pix = doc[index].get_pixmap(dpi=300)
    image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return pytesseract.image_to_string(image)


def extract(raw=config.RAW, out=OUT, ocr=True, workers=8, log=print):
    pdfs = sorted(raw.rglob("*.pdf"), key=lambda p: (PRIORITY.index(p.parent.name), p.name))
    rows = []
    for pdf in pdfs:
        with pymupdf.open(pdf) as doc:
            texts = [normalize(page.get_text()) for page in doc]
        file = pdf.relative_to(raw).as_posix()
        todo = [i for i, t in enumerate(texts) if len(t.split()) <= MIN_WORDS]
        done = {}
        if ocr and todo:
            with ProcessPoolExecutor(workers) as pool:
                done = dict(zip(todo, map(normalize, pool.map(ocr_page, [(str(pdf), i) for i in todo]))))
        for i, text in enumerate(texts):
            used = i in done and len(done[i].split()) > len(text.split())  # OCR text replaces a thinner layer
            text = done[i] if used else text
            rows.append({"file": file, "page": i + 1, "text": text, "words": len(text.split()),
                         "ocr": used, "needed_ocr": i in todo})
        log(f"{file}: {len(texts)} pages, {len(todo)} short, {len(done)} OCR'd")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.writelines(json.dumps(row) + "\n" for row in rows)
    return rows


def load(path=OUT):
    with open(path) as f:
        return [json.loads(line) for line in f]
