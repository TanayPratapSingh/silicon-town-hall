"""Offline retrieval tests: a tiny synthetic PDF and stand in models, no downloads, no API key."""

import zlib
from pathlib import Path

import numpy as np
import pymupdf
import pytest

from townhall.llm import FakeLLM
from townhall.retrieval import Hit, Retriever, index_document, rrf

PAGES = ["Construction noise near Almond Street homes will be loud at night.",
         "Parking garages and bus routes downtown will change during the work.",
         None]  # page 3: a red block and no text layer, like a scanned map


def _unit(v):
    return v / (np.linalg.norm(v) or 1.0)


class WordHash:
    """Text stand in: bag of hashed words, so shared words mean similar vectors."""

    def encode(self, items, **_):
        out = []
        for text in items:
            v = np.zeros(64)
            for word in text.lower().split():
                v[zlib.crc32(word.strip(".,:").encode()) % 64] += 1
            out.append(_unit(v))
        return np.array(out)


class ColorClip:
    """Image stand in: images map to mean RGB, the words red/green/blue map to that channel."""

    def encode(self, items, **_):
        out = []
        for item in items:
            if isinstance(item, str):
                v = np.array([w in item.lower() for w in ("red", "green", "blue")], float) + 0.01
            else:
                v = np.asarray(item, float).reshape(-1, 3).mean(axis=0)
                v = v - v.min() + 0.01  # white pages come out flat, the red page points at red
            out.append(_unit(v))
        return np.array(out)


class Overlap:
    """Cross encoder stand in: count of shared words."""

    def predict(self, pairs, **_):
        return np.array([len(set(q.lower().split()) & set(d.lower().split())) for q, d in pairs], float)


class Looker:
    """A non fake LLM that rates page 2 highest, to exercise the vision stage."""

    def __init__(self):
        self.calls = []

    def look(self, prompt, image_path, model=None):
        self.calls.append(image_path)
        return "Score: 9" if image_path.endswith("p0002.png") else "2"


@pytest.fixture(scope="module")
def indexed(tmp_path_factory):
    root = tmp_path_factory.mktemp("retrieval")
    pdf = pymupdf.open()
    for text in PAGES:
        page = pdf.new_page()
        if text:
            page.insert_text((72, 72), text)
        else:
            page.draw_rect(pymupdf.Rect(0, 0, page.rect.width, page.rect.height), color=(1, 0, 0), fill=(1, 0, 0))
    pdf.save(root / "tiny.pdf")
    stats = index_document(root / "tiny.pdf", "tiny", db_dir=root / "chroma", pages_dir=root / "pages",
                           text_model=WordHash(), image_model=ColorClip())
    retriever = Retriever(db_dir=root / "chroma", text_model=WordHash(), image_model=ColorClip(), cross_encoder=Overlap())
    return root, stats, retriever


def test_rrf_math():
    fused = rrf([["a", "b", "c"], ["b", "c", "d"]], k=60)
    assert list(fused) == ["b", "c", "a", "d"]
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["d"] == pytest.approx(1 / 63)
    assert list(rrf([["x", "y"]])) == ["x", "y"]  # a single list keeps its order


def test_index_counts_every_page(indexed):
    _, stats, retriever = indexed
    assert stats["pages"] == stats["images"] == 3
    assert stats["chunks"] == 2 and stats["pages_without_text"] == 1
    assert retriever.texts.count() == 2 and retriever.images.count() == 3


def test_reindex_replaces(indexed):
    root, _, retriever = indexed
    again = index_document(root / "tiny.pdf", "tiny", db_dir=root / "chroma", pages_dir=root / "pages",
                           text_model=WordHash(), image_model=ColorClip())
    assert again["pages"] == 3 and retriever.images.count() == 3 and retriever.texts.count() == 2


def test_hit_fields(indexed):
    _, _, retriever = indexed
    hits = retriever.search("noise at night", k=3)
    assert hits and all(isinstance(h, Hit) for h in hits)
    top = hits[0]
    assert (top.doc_id, top.file, top.page) == ("tiny", "tiny.pdf", 1)
    assert "noise" in top.text and Path(top.image_path).exists()
    assert set(top.stage_scores) >= {"text", "image", "rrf", "cross_encoder", "vision", "boost"}
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_modes(indexed):
    _, _, retriever = indexed
    assert retriever.search("red", mode="image", rerank=False)[0].page == 3  # only image search finds the textless page
    text_only = retriever.search("parking bus", mode="text", rerank=False)
    assert text_only[0].page == 2 and all(h.stage_scores["image"] is None for h in text_only)
    assert {h.page for h in retriever.search("red parking", mode="both", rerank=False)} == {1, 2, 3}
    with pytest.raises(ValueError):
        retriever.search("x", mode="video")


def test_stage_switching(indexed):
    _, _, retriever = indexed
    off = retriever.search("parking downtown", rerank=False)
    assert all(h.stage_scores["cross_encoder"] is None for h in off) and retriever.last_stages["cross_encoder"] == "off"

    on = retriever.search("parking downtown")
    assert on[0].page == 2 and on[0].stage_scores["cross_encoder"] > 0
    assert on[-1].page == 3 and on[-1].stage_scores["cross_encoder"] is None  # textless page sorts last

    fake = FakeLLM(reply="10")
    retriever.search("noise", llm=fake)
    assert retriever.last_stages["vision"].startswith("skipped") and fake.calls == []

    looker = Looker()
    hits = retriever.search("noise", llm=looker)
    assert retriever.last_stages["vision"] == "ran" and len(looker.calls) == 3
    assert hits[0].page == 2 and hits[0].stage_scores["vision"] == 9.0
    retriever.search("noise", llm=looker, vision=False)
    assert retriever.last_stages["vision"] == "off" and len(looker.calls) == 3


def test_cited_boost(indexed):
    root, _, _ = indexed
    retriever = Retriever(db_dir=root / "chroma", text_model=WordHash(), image_model=ColorClip(),
                          cross_encoder=Overlap(), cited_boost=0.5)  # 3 candidates sit 1/3 apart
    plain = retriever.search("noise downtown parking", rerank=False)
    second = plain[1]
    boosted = retriever.search("noise downtown parking", rerank=False, cited_pages=[(second.doc_id, second.page)])
    assert boosted[0].page == second.page and boosted[0].stage_scores["boost"] == retriever.cited_boost
    assert boosted[0].score == pytest.approx(second.score + retriever.cited_boost)
