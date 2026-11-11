"""Document indexing and retrieval (issues #4 and #5).

Index any PDF (every page as text chunks plus one page image) into a persistent
ChromaDB at data/processed/retrieval/chroma/:

    from townhall.retrieval import index_document
    index_document("plan.pdf", doc_id="plan")      # reindexing a doc_id replaces it

Search it. This signature is stable; residents and meetings call it:

    from townhall.retrieval import Retriever
    retriever = Retriever()                         # loads the three models once; reuse it
    hits = retriever.search(query, k=10, mode="both", cited_pages=None, llm=None)

    query        str, the resident's question
    k            how many pages to return
    mode         "both" (text and image search fused by RRF), "text" or "image"
    cited_pages  iterable of (doc_id, page) pairs already cited; they get a small additive boost
    llm          a townhall.llm client; a real one scores the top 10 page images with llm.look,
                 None or FakeLLM skips that stage (recorded in retriever.last_stages)
    rerank, vision  optional switches (default True) for ablations

Each hit is a `Hit(doc_id, file, page, text, image_path, score, stage_scores)`.
`page` is the 1 based physical page in the PDF, `text` the best matching chunk,
`stage_scores` holds text, text_rank, image, image_rank, rrf, cross_encoder,
vision and boost (None where a stage did not run for that page).

Pipeline (docs/proposal.md, "Under the hood"): text top 50 pages and image top 50
pages merged by reciprocal rank fusion (k=60) to the top 50; a text cross encoder
trims 50 to 20; a vision model reorders the top 10; cited pages get the boost.
"""

from townhall.retrieval.index import index_document
from townhall.retrieval.search import Hit, Retriever, rrf

__all__ = ["Hit", "Retriever", "index_document", "rrf"]
