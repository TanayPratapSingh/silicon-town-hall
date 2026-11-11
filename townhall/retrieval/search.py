"""Search the index: text and image retrieval, RRF, cross encoder, vision, cited page boost."""

import re
from dataclasses import dataclass, field

from townhall.llm import FakeLLM
from townhall.retrieval import index

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge v1.5 query instruction
MODES = ("both", "text", "image")
FUSE_TOP, TEXT_CHUNKS, RRF_K = 50, 300, 60
RERANK_KEEP, VISION_TOP = 20, 10
STAGE_KEYS = ("text", "text_rank", "image", "image_rank", "rrf", "cross_encoder", "vision", "boost")
VISION_PROMPT = ("A resident asked: {query}\n"
                 "How useful is this document page for answering that? Reply with one number from 0 to 10.")


@dataclass
class Hit:
    doc_id: str
    file: str
    page: int            # 1 based physical page in the PDF
    text: str            # best matching chunk of the page ("" for a page with no text layer)
    image_path: str      # absolute path to the page PNG
    score: float         # final score: rank based base in (0, 1] plus any cited page boost
    stage_scores: dict = field(default_factory=dict)  # text, text_rank, image, image_rank, rrf, cross_encoder, vision, boost


def _best_first(score):
    """Sort key: higher scores first, missing scores last, ties keep their earlier order."""
    return (score is None, -(score or 0.0))


def rrf(ranked_lists, k=RRF_K):
    """Reciprocal rank fusion: score = sum over lists of 1 / (k + rank), rank starting at 1."""
    scores = {}
    for ranked in ranked_lists:
        for rank, key in enumerate(ranked, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    return dict(sorted(scores.items(), key=lambda item: -item[1]))


class Retriever:
    """Loads the three models once; pass stand ins (anything with .encode / .predict) for tests."""

    def __init__(self, db_dir=index.DB_DIR, text_model=None, image_model=None, cross_encoder=None, cited_boost=0.1):
        self.text_model = text_model or index.load_text_model()
        self.image_model = image_model or index.load_image_model()
        self.cross_encoder = cross_encoder or index.load_cross_encoder()
        self.texts, self.images = index.collections(db_dir)
        self.cited_boost = cited_boost
        self.last_stages = {}  # what ran in the last search, including skipped stages

    def search(self, query, k=10, mode="both", cited_pages=None, llm=None, rerank=True, vision=True):
        """Top k pages for a query. cited_pages: iterable of (doc_id, page) pairs already cited."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        stages, best_chunk, lists = {}, {}, []
        if mode in ("both", "text"):
            ranked = self._text_pages(query, stages, best_chunk)
            lists.append(ranked)
        if mode in ("both", "image"):
            lists.append(self._image_pages(query, stages))
        fused = list(rrf(lists).items())[:FUSE_TOP]
        pages = self._pages([key for key, _ in fused])
        for key, score in fused:
            stages[key] = {**dict.fromkeys(STAGE_KEYS), **stages[key], "rrf": score}
        cands = [key for key, _ in fused]
        run = {"retrieval": mode, "fusion": "rrf" if len(lists) > 1 else "single list", "candidates": len(cands)}

        if rerank and cands:
            self._rerank(query, cands, pages, stages, best_chunk)
            # shortcut: a page with no text layer cannot be scored by a text model, so it sorts last and
            # usually falls out of the 20; upgrade by OCR or by letting the vision stage see the full 50
            cands = sorted(cands, key=lambda c: _best_first(stages[c]["cross_encoder"]))[:RERANK_KEEP]
        run["cross_encoder"] = "ran" if rerank else "off"

        if not vision:
            run["vision"] = "off"
        elif llm is None or isinstance(llm, FakeLLM):
            run["vision"] = "skipped: no real LLM"
        else:
            top = cands[:VISION_TOP]
            for key in top:
                reply = llm.look(VISION_PROMPT.format(query=query), pages[key]["image_path"])
                found = re.search(r"\d+(?:\.\d+)?", str(reply))
                stages[key]["vision"] = min(float(found.group()), 10.0) if found else None
            top = sorted(top, key=lambda c: _best_first(stages[c]["vision"]))
            cands = top + cands[VISION_TOP:]
            run["vision"] = "ran"

        cited = {(str(d), int(p)) for d, p in cited_pages or ()}
        n = len(cands)
        hits = []
        for i, key in enumerate(cands):
            boost = self.cited_boost if key in cited else 0.0
            stages[key]["boost"] = boost
            page = pages[key]
            hits.append(Hit(doc_id=key[0], file=page["file"], page=key[1],
                            text=best_chunk.get(key, page["text"]), image_path=page["image_path"],
                            score=(n - i) / n + boost, stage_scores=stages[key]))
        run["cited_boost"] = "ran" if cited else "no cited pages"
        self.last_stages = run
        return sorted(hits, key=lambda h: -h.score)[:k]

    def _text_pages(self, query, stages, best_chunk):
        vector = self.text_model.encode([QUERY_PREFIX + query], normalize_embeddings=True)[0]
        found = self._query(self.texts, vector, TEXT_CHUNKS)
        ranked = []
        for meta, doc, sim in found:  # chunks come best first, so the first chunk per page is its best
            key = (meta["doc_id"], meta["page"])
            if key not in best_chunk:
                best_chunk[key] = doc
                ranked.append(key)
                stages.setdefault(key, {}).update(text=sim, text_rank=len(ranked))
        return ranked[:FUSE_TOP]

    def _image_pages(self, query, stages):
        vector = self.image_model.encode([query], normalize_embeddings=True)[0]
        ranked = []
        for meta, _, sim in self._query(self.images, vector, FUSE_TOP):
            key = (meta["doc_id"], meta["page"])
            ranked.append(key)
            stages.setdefault(key, {}).update(image=sim, image_rank=len(ranked))
        return ranked

    @staticmethod
    def _query(col, vector, n):
        n = min(n, col.count())
        if not n:
            return []
        r = col.query(query_embeddings=[vector.tolist()], n_results=n)
        return [(m, d, 1.0 - dist) for m, d, dist in zip(r["metadatas"][0], r["documents"][0], r["distances"][0])]

    def _pages(self, keys):
        """Page text, file, image path and chunk count for each (doc_id, page) key."""
        if not keys:
            return {}
        got = self.images.get(ids=[f"{d}:{p}" for d, p in keys])
        rows = {(m["doc_id"], m["page"]): {**m, "text": doc} for m, doc in zip(got["metadatas"], got["documents"])}
        return {key: rows[key] for key in keys}

    def _rerank(self, query, cands, pages, stages, best_chunk):
        """Score every chunk of every candidate page; a page's score is its best chunk's score."""
        ids = [f"{d}:{p}:{i}" for d, p in cands for i in range(pages[(d, p)]["n_chunks"])]
        if not ids:
            return
        got = self.texts.get(ids=ids)
        pairs = list(zip(got["metadatas"], got["documents"]))
        scores = self.cross_encoder.predict([(query, doc) for _, doc in pairs])
        for (meta, doc), score in zip(pairs, scores):
            key, score = (meta["doc_id"], meta["page"]), float(score)
            if stages[key]["cross_encoder"] is None or score > stages[key]["cross_encoder"]:
                stages[key]["cross_encoder"] = score
                best_chunk[key] = doc
