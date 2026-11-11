"""Index a PDF as text chunks and page images in a persistent ChromaDB.

CLI: `python -m townhall.retrieval.index` indexes the 31 FEIS chapters and writes
runs/retrieval_index.json. Pass PDF paths to index other documents instead.
"""

import argparse
import json
import re
import time
from functools import cache
from pathlib import Path

import chromadb
import pymupdf
from chromadb.config import Settings
from PIL import Image

from townhall import config

HOME = config.PROCESSED / "retrieval"
DB_DIR = HOME / "chroma"
PAGES_DIR = HOME / "pages"
FEIS_DIR = config.RAW / "feis_chapters"

TEXT_MODEL = "BAAI/bge-small-en-v1.5"
IMAGE_MODEL = "clip-ViT-B-32"
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DPI = 100
CHUNK_WORDS, CHUNK_OVERLAP = 200, 40  # about 300 tokens, well under the 512 token limit of bge-small
BATCH_PAGES = 16


@cache
def load_text_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(TEXT_MODEL)


@cache
def load_image_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(IMAGE_MODEL)


@cache
def load_cross_encoder():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(CROSS_ENCODER)


def collections(db_dir=DB_DIR):
    """The two collections: text chunks (bge vectors) and page images (CLIP vectors)."""
    client = chromadb.PersistentClient(path=str(db_dir), settings=Settings(anonymized_telemetry=False))
    make = lambda name: client.get_or_create_collection(name, embedding_function=None, metadata={"hnsw:space": "cosine"})
    return make("text_chunks"), make("page_images")


def chunk_words(text, size=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, max(len(words) - overlap, 1), size - overlap)] if words else []


def index_document(pdf_path, doc_id, db_dir=DB_DIR, pages_dir=PAGES_DIR, text_model=None, image_model=None, dpi=DPI):
    """Index every page of one PDF; reindexing the same doc_id replaces it. Returns counts and timing."""
    started = time.perf_counter()
    pdf_path = Path(pdf_path)
    text_model, image_model = text_model or load_text_model(), image_model or load_image_model()
    texts, images = collections(db_dir)
    for col in (texts, images):
        col.delete(where={"doc_id": doc_id})
    out = Path(pages_dir) / re.sub(r"[^\w.-]+", "_", doc_id)  # doc_id may come from an uploaded file name
    out.mkdir(parents=True, exist_ok=True)
    stats = {"doc_id": doc_id, "file": pdf_path.name, "pages": 0, "chunks": 0, "images": 0, "pages_without_text": 0}

    with pymupdf.open(pdf_path) as pdf:
        for start in range(0, len(pdf), BATCH_PAGES):
            chunk_rows, page_rows, pils = [], [], []
            for page in pdf.pages(start, min(start + BATCH_PAGES, len(pdf))):
                number = page.number + 1  # 1 based physical page in the file, not the printed page number
                text = page.get_text()
                chunks = chunk_words(text)
                pix = page.get_pixmap(dpi=dpi)
                image_path = out / f"p{number:04d}.png"
                pix.save(image_path)
                pils.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
                meta = {"doc_id": doc_id, "file": pdf_path.name, "page": number}
                chunk_rows += [(f"{doc_id}:{number}:{i}", c, {**meta, "chunk": i}) for i, c in enumerate(chunks)]
                page_rows.append((f"{doc_id}:{number}", " ".join(text.split()),
                                  {**meta, "image_path": str(image_path.resolve()), "n_chunks": len(chunks)}))
                stats["pages_without_text"] += not chunks
            if chunk_rows:
                ids, docs, metas = zip(*chunk_rows)
                vectors = text_model.encode(list(docs), normalize_embeddings=True, batch_size=32)
                texts.upsert(ids=list(ids), embeddings=[v.tolist() for v in vectors], documents=list(docs), metadatas=list(metas))
            ids, docs, metas = zip(*page_rows)
            vectors = image_model.encode(pils, normalize_embeddings=True, batch_size=16)
            images.upsert(ids=list(ids), embeddings=[v.tolist() for v in vectors], documents=list(docs), metadatas=list(metas))
            stats["pages"] += len(page_rows)
            stats["images"] += len(page_rows)
            stats["chunks"] += len(chunk_rows)
    stats["seconds"] = round(time.perf_counter() - started, 1)
    return stats


def main():
    parser = argparse.ArgumentParser(description="Index PDFs for retrieval (default: the 31 FEIS chapters).")
    parser.add_argument("pdfs", nargs="*", type=Path, help="PDFs to index; doc_id is the file name without .pdf")
    args = parser.parse_args()
    pdfs = args.pdfs or sorted(FEIS_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"no PDFs found in {FEIS_DIR}; run scripts/download.py first")

    started = time.perf_counter()
    load_text_model(), load_image_model()
    load_seconds = round(time.perf_counter() - started, 1)
    docs = []
    for pdf in pdfs:
        docs.append(index_document(pdf, pdf.stem))
        print(json.dumps(docs[-1]))
    total = {key: sum(d[key] for d in docs) for key in ("pages", "chunks", "images", "pages_without_text")}
    texts, images = collections()
    record = {
        "command": "python -m townhall.retrieval.index" + "".join(f" '{p}'" for p in args.pdfs),
        "text_model": TEXT_MODEL, "image_model": IMAGE_MODEL, "dpi": DPI,
        "chunk_words": CHUNK_WORDS, "chunk_overlap": CHUNK_OVERLAP,
        "documents": len(docs), **total,
        "collection_counts": {"text_chunks": texts.count(), "page_images": images.count()},
        "model_load_seconds": load_seconds,
        "total_seconds": round(time.perf_counter() - started, 1),
        "per_document": docs,
    }
    print(json.dumps({k: v for k, v in record.items() if k != "per_document"}, indent=1))
    if not args.pdfs:  # only the FEIS run is the recorded artifact
        (config.RUNS / "retrieval_index.json").write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    main()
