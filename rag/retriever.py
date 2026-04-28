"""
rag/retriever.py
================
Module 6 — RAG Retriever

Bridges the existing FAISS singleton (search/vector_store.py :: VectorStore)
into a fully LangChain-compatible retriever.

Design decisions
----------------
* **No new FAISS index** — we read the same on-disk index that Modules 4 & 5
  already maintain.  The existing VectorStore singleton owns the FAISS state;
  this module only wraps it.
* **MMR search** (Maximal Marginal Relevance) is performed manually so we do
  not need langchain-community's `FAISS` class, which would require its own
  separate index lifecycle.
* **Singleton embedding model** — reuses embeddings.embedder._get_model() so
  the 384-dim sentence-transformer loads at most once per process.
* **doc_filter** — optional metadata filter on `document_type` or `tags` field
  stored in storage/structured_data/<doc_id>.json.
* **Graceful import handling** — if langchain is missing, raises a clear
  ImportError at call-time, not at import-time, so the rest of the system
  keeps working.

Public API
----------
    build_faiss_retriever(doc_filter: str | None = None, k: int = 5)
        → langchain_core.retrievers.BaseRetriever
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

logger = logging.getLogger("rag.retriever")

# ---------------------------------------------------------------------------
# Storage config (mirrors search/vector_store.py)
# ---------------------------------------------------------------------------
EMBEDDINGS_DIR = Path("storage/embeddings")
STRUCTURED_DATA_DIR = Path("storage/structured_data")

# ---------------------------------------------------------------------------
# Optional imports — fail loudly only when the public function is called
# ---------------------------------------------------------------------------
try:
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    _LC_AVAILABLE = True
except ImportError:
    _LC_AVAILABLE = False

try:
    import faiss as _faiss_lib  # type: ignore
    _FAISS_AVAILABLE = True
except ImportError:
    _faiss_lib = None  # type: ignore
    _FAISS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _load_doc_metadata(doc_id: str) -> dict:
    """
    Load the structured-data JSON for a document.

    Returns a dict with at least:
        document_type (str), tags (list[str]), cleaned_text (str)
    """
    json_path = STRUCTURED_DATA_DIR / f"{doc_id}.json"
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return data
        except Exception as exc:
            logger.warning("Could not read metadata for %s: %s", doc_id, exc)
    return {"document_type": "unknown", "tags": [], "cleaned_text": ""}


def _passes_filter(metadata: dict, doc_filter: str | None) -> bool:
    """
    Return True if the document matches the optional filter string.

    Matching logic (case-insensitive substring):
      - Checks document_type
      - Checks any tag in the tags list
    If doc_filter is None, always returns True.
    """
    if doc_filter is None:
        return True
    needle = doc_filter.lower()
    if needle in metadata.get("document_type", "").lower():
        return True
    for tag in metadata.get("tags", []):
        if needle in tag.lower():
            return True
    return False


def _get_query_vector(query_text: str) -> Optional[np.ndarray]:
    """
    Encode a query string into a unit-normalized embedding vector.

    Reuses the already-loaded sentence-transformer singleton from
    embeddings.embedder so the model is never loaded twice.
    """
    try:
        from embeddings.embedder import _get_model  # noqa: PLC0415
        model = _get_model()
    except Exception as exc:
        logger.error("Cannot load embedding model: %s", exc)
        return None

    vec = model.encode([query_text], show_progress_bar=False, convert_to_numpy=True)
    vec = vec.flatten().astype("float32")
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def _mmr(
    query_vec: np.ndarray,
    candidate_vecs: np.ndarray,
    candidate_ids: list[str],
    k: int,
    lambda_param: float = 0.5,
) -> list[str]:
    """
    Maximal Marginal Relevance selection.

    Selects k items from `candidate_ids` that balance:
      - relevance  to query_vec  (similarity score)
      - diversity  among already-selected items

    Args:
        query_vec:      1-D float32 unit vector  (dim,)
        candidate_vecs: 2-D float32 matrix       (n_candidates, dim)
        candidate_ids:  doc_ids parallel to rows of candidate_vecs
        k:              number of items to select
        lambda_param:   weight between relevance (1.0) and diversity (0.0)

    Returns:
        Ordered list of selected doc_ids (length ≤ k).
    """
    relevance_scores = (candidate_vecs @ query_vec).tolist()  # shape (n,)
    selected_indices: list[int] = []
    remaining = list(range(len(candidate_ids)))

    while remaining and len(selected_indices) < k:
        if not selected_indices:
            # First pick: pure relevance
            best = max(remaining, key=lambda i: relevance_scores[i])
        else:
            selected_vecs = candidate_vecs[selected_indices]  # (s, dim)

            def mmr_score(i: int) -> float:
                rel = relevance_scores[i]
                # max cosine similarity to any already-selected vector
                redundancy = float(np.max(candidate_vecs[i] @ selected_vecs.T))
                return lambda_param * rel - (1 - lambda_param) * redundancy

            best = max(remaining, key=mmr_score)

        selected_indices.append(best)
        remaining.remove(best)

    return [candidate_ids[i] for i in selected_indices]


# ---------------------------------------------------------------------------
# LangChain retriever implementation
# ---------------------------------------------------------------------------

class IDPFaissRetriever(BaseRetriever):  # type: ignore[misc]
    """
    LangChain-compatible retriever backed by the IDP System's FAISS index.

    Attributes:
        k:          Maximum number of documents to return.
        doc_filter: Optional string to filter by document_type or tag.
        _fetch_k:   Candidate pool size for MMR (default: 4 * k).
    """

    # Pydantic fields — LangChain BaseRetriever is a pydantic v1 BaseModel
    k: int = 5
    doc_filter: Optional[str] = None
    fetch_k_multiplier: int = 4

    class Config:
        arbitrary_types_allowed = True

    # ------------------------------------------------------------------
    # Internal helpers (not Pydantic fields — created in __init__)
    # ------------------------------------------------------------------

    def _get_store(self):  # type: ignore[return]
        """Return the VectorStore singleton (lazy import to avoid circular deps)."""
        from search.vector_store import VectorStore  # noqa: PLC0415
        return VectorStore()

    # ------------------------------------------------------------------
    # Core retrieval
    # ------------------------------------------------------------------

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: "CallbackManagerForRetrieverRun",
    ) -> List[Document]:
        """
        Main retrieval method called by LangChain chains.

        Steps:
          1. Encode query → unit vector
          2. Fetch fetch_k candidates from FAISS (pure ANN, fast)
          3. Apply doc_filter on structured metadata
          4. Re-rank filtered candidates with MMR
          5. Return top-k as LangChain Document objects
        """
        if not _FAISS_AVAILABLE:
            logger.error("faiss-cpu is not installed — retrieval unavailable.")
            return []

        store = self._get_store()
        if store.index is None or store.index.ntotal == 0:
            logger.warning("FAISS index is empty — returning no results.")
            return []

        # 1. Encode query
        query_vec = _get_query_vector(query)
        if query_vec is None:
            return []

        # 2. Fetch a larger candidate pool so filtering + MMR have choices
        fetch_k = min(self.k * self.fetch_k_multiplier, store.index.ntotal)
        query_2d = query_vec.reshape(1, -1)
        _scores, indices = store.index.search(query_2d, fetch_k)
        candidate_indices = [
            int(idx) for idx in indices[0] if 0 <= idx < len(store.doc_ids)
        ]

        if not candidate_indices:
            return []

        # 3. Load candidate doc_ids, metadata, and vectors
        candidate_ids: list[str] = []
        candidate_vecs_list: list[np.ndarray] = []

        for idx in candidate_indices:
            doc_id = store.doc_ids[idx]
            meta = _load_doc_metadata(doc_id)

            if not _passes_filter(meta, self.doc_filter):
                continue

            npy_path = EMBEDDINGS_DIR / f"{doc_id}.npy"
            if not npy_path.exists():
                logger.debug("No .npy for %s — skipping.", doc_id)
                continue

            try:
                vec = np.load(npy_path).astype("float32").flatten()
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                candidate_ids.append(doc_id)
                candidate_vecs_list.append(vec)
            except Exception as exc:
                logger.warning("Could not load vector for %s: %s", doc_id, exc)

        if not candidate_ids:
            logger.info("No candidates passed filter=%r.", self.doc_filter)
            return []

        candidate_vecs = np.vstack(candidate_vecs_list)  # (n, dim)

        # 4. MMR re-ranking
        selected_ids = _mmr(
            query_vec=query_vec,
            candidate_vecs=candidate_vecs,
            candidate_ids=candidate_ids,
            k=self.k,
        )

        # 5. Build LangChain Document objects
        documents: list[Document] = []
        for doc_id in selected_ids:
            meta = _load_doc_metadata(doc_id)
            page_content = meta.get("cleaned_text") or ""
            lc_doc = Document(
                page_content=page_content,
                metadata={
                    "doc_id": doc_id,
                    "document_type": meta.get("document_type", "unknown"),
                    "tags": meta.get("tags", []),
                    "source": str(STRUCTURED_DATA_DIR / f"{doc_id}.json"),
                },
            )
            documents.append(lc_doc)

        logger.info(
            "IDPFaissRetriever: query=%r | filter=%r | returned %d/%d docs (MMR)",
            query[:60], self.doc_filter, len(documents), self.k,
        )
        return documents

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: "CallbackManagerForRetrieverRun",
    ) -> List[Document]:
        """Async wrapper — runs the sync path (FAISS is CPU-bound, no benefit from true async)."""
        return self._get_relevant_documents(query, run_manager=run_manager)


# ---------------------------------------------------------------------------
# Public factory function
# ---------------------------------------------------------------------------

def build_faiss_retriever(
    doc_filter: str | None = None,
    k: int = 5,
) -> "IDPFaissRetriever":
    """
    Build and return a LangChain-compatible FAISS retriever for the IDP system.

    Wraps the existing ``VectorStore`` singleton (search/vector_store.py) —
    **no new index is created**; we reuse the one maintained by Modules 4 & 5.

    Args:
        doc_filter: Optional string.  If provided, only documents whose
                    ``document_type`` or any ``tag`` contains this string
                    (case-insensitive substring match) are returned.
                    Examples: ``"invoice"``, ``"Finance"``, ``"Risk:High"``.
        k:          Maximum number of documents to return per query.
                    MMR is applied over a candidate pool of ``4 * k`` to
                    improve result diversity.

    Returns:
        An ``IDPFaissRetriever`` instance (subclass of
        ``langchain_core.retrievers.BaseRetriever``).

    Raises:
        ImportError: If ``langchain-core`` is not installed.

    Example::

        retriever = build_faiss_retriever(doc_filter="invoice", k=3)
        docs = retriever.invoke("Q3 invoice over 5000")
        for d in docs:
            print(d.metadata["doc_id"], d.page_content[:80])
    """
    if not _LC_AVAILABLE:
        raise ImportError(
            "langchain-core is required for build_faiss_retriever(). "
            "Install with:  pip install langchain langchain-community"
        )

    logger.info(
        "build_faiss_retriever called — doc_filter=%r, k=%d", doc_filter, k
    )
    return IDPFaissRetriever(k=k, doc_filter=doc_filter)
