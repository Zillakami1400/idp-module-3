"""
search/router.py
================
Module 5: Search API Endpoints

Provides:
  - GET /api/v1/search?q=<query>&top_k=5   → semantic text search
  - GET /api/v1/search/{doc_id}/similar?top_k=5 → find similar docs
"""

import logging
from fastapi import APIRouter, Query, Path, HTTPException

from search.vector_store import VectorStore

logger = logging.getLogger("search.router")

router = APIRouter(prefix="/api/v1", tags=["Semantic Search"])


@router.get("/search", summary="Semantic natural-language document search")
def search_documents(
    q: str = Query(..., description="Natural language query text", min_length=1),
    top_k: int = Query(5, description="Number of results to return", ge=1, le=50),
):
    """
    Search all indexed documents using a natural language query.

    The query is converted into a semantic embedding and compared against
    all stored document vectors using cosine similarity.

    Example queries:
      - "Show me Q3 expense reports over 5000"
      - "Contracts with penalty clauses above 5%"
      - "invoice customer Roland"

    Returns ranked results with similarity scores, document type, and tags.
    """
    store = VectorStore()

    if store.total_documents == 0:
        raise HTTPException(
            status_code=404,
            detail="No documents indexed yet. Upload some documents first.",
        )

    results = store.search(query_text=q, top_k=top_k)
    return results


@router.get(
    "/search/{doc_id}/similar",
    summary="Find documents similar to a given document",
)
def find_similar_documents(
    doc_id: str = Path(..., description="Document ID to find similar docs for"),
    top_k: int = Query(5, description="Number of similar documents to return", ge=1, le=50),
):
    """
    Find documents most semantically similar to a reference document.

    Uses the existing embedding for the given doc_id to search
    the FAISS index for other similar documents.
    """
    store = VectorStore()

    if store.total_documents == 0:
        raise HTTPException(
            status_code=404,
            detail="No documents indexed yet. Upload some documents first.",
        )

    results = store.search_by_doc_id(doc_id=doc_id, top_k=top_k)

    if not results.get("results") and doc_id not in store.id_to_pos:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{doc_id}' not found in the index.",
        )

    return results


@router.post("/search/rebuild", summary="Rebuild the search index from all embeddings")
def rebuild_index():
    """
    Rebuild the FAISS search index from all stored embedding files.

    Use this if you have manually added/removed embeddings, or if
    the index seems out of sync.
    """
    store = VectorStore()
    count = store.rebuild_index()
    return {
        "status": "rebuilt",
        "total_documents": count,
    }
