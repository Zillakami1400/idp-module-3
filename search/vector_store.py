"""
search/vector_store.py
=======================
Module 5: FAISS Vector Search

Responsibilities:
  - Maintain a FAISS IndexFlatIP index for cosine similarity search
    (inner product on L2-normalized vectors = cosine similarity)
  - Map doc_id ↔ FAISS index position
  - Add new document vectors on upload
  - Search by natural-language query text
  - Search by doc_id (find similar documents)
  - Rebuild index from stored .npy files
  - Persist / load index to/from storage/search/

Graceful degradation:
  - If faiss is NOT installed → warn + return empty results
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("search.vector_store")

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
SEARCH_DIR = Path("storage/search")
SEARCH_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDINGS_DIR = Path("storage/embeddings")
STRUCTURED_DATA_DIR = Path("storage/structured_data")

INDEX_FILE = SEARCH_DIR / "faiss_index.bin"
MAP_FILE = SEARCH_DIR / "index_map.json"

# ---------------------------------------------------------------------------
# Optional FAISS import
# ---------------------------------------------------------------------------
try:
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
except ImportError:
    faiss = None  # type: ignore
    _FAISS_AVAILABLE = False
    logger.warning(
        "faiss-cpu is not installed. Vector search will be unavailable. "
        "Install with:  pip install faiss-cpu"
    )

# ---------------------------------------------------------------------------
# Optional sentence-transformers for query encoding
# ---------------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _ST_AVAILABLE = True
except ImportError:
    SentenceTransformer = None  # type: ignore
    _ST_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VECTOR_DIM = 384  # must match embedder output


class VectorStore:
    """
    FAISS-based vector store for semantic document search.

    Uses IndexFlatIP (inner product) on L2-normalized vectors,
    which is equivalent to cosine similarity.
    """

    _instance = None

    def __new__(cls):
        """Singleton — only one VectorStore instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.index = None
        self.doc_ids: list[str] = []       # position → doc_id
        self.id_to_pos: dict[str, int] = {}  # doc_id  → position
        self._model = None

        if _FAISS_AVAILABLE:
            self._load_or_create_index()
        else:
            logger.warning("VectorStore created but FAISS is not available.")

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _load_or_create_index(self):
        """Load existing index from disk, or create a new empty one."""
        if INDEX_FILE.exists() and MAP_FILE.exists():
            try:
                self.index = faiss.read_index(str(INDEX_FILE))
                with open(MAP_FILE, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                self.doc_ids = data.get("doc_ids", [])
                self.id_to_pos = {did: i for i, did in enumerate(self.doc_ids)}
                logger.info(
                    "Loaded FAISS index: %d vectors | dim=%d",
                    self.index.ntotal, VECTOR_DIM,
                )
                return
            except Exception as exc:
                logger.warning("Failed to load existing index: %s — creating new.", exc)

        # Create fresh
        self.index = faiss.IndexFlatIP(VECTOR_DIM)
        self.doc_ids = []
        self.id_to_pos = {}
        logger.info("Created new FAISS index (dim=%d).", VECTOR_DIM)

    def save(self):
        """Persist index and mapping to disk."""
        if not _FAISS_AVAILABLE or self.index is None:
            return

        faiss.write_index(self.index, str(INDEX_FILE))
        with open(MAP_FILE, "w", encoding="utf-8") as fh:
            json.dump({"doc_ids": self.doc_ids}, fh, indent=2)
        logger.info("FAISS index saved → %s (%d vectors)", INDEX_FILE, self.index.ntotal)

    # ------------------------------------------------------------------
    # Document operations
    # ------------------------------------------------------------------

    def add_document(self, doc_id: str, vector: np.ndarray) -> bool:
        """
        Add a document vector to the index.

        Args:
            doc_id: Unique document identifier.
            vector: 1-D numpy array of shape (384,), L2-normalized.

        Returns:
            True if added successfully, False otherwise.
        """
        if not _FAISS_AVAILABLE or self.index is None:
            logger.warning("Cannot add document — FAISS unavailable.")
            return False

        # Skip if already indexed
        if doc_id in self.id_to_pos:
            logger.debug("doc_id=%s already in index — skipping.", doc_id)
            return True

        # Ensure correct shape
        vec = vector.reshape(1, -1).astype("float32")

        self.index.add(vec)
        pos = len(self.doc_ids)
        self.doc_ids.append(doc_id)
        self.id_to_pos[doc_id] = pos

        # Auto-save
        self.save()

        logger.info("Added doc_id=%s to index (pos=%d, total=%d)", doc_id, pos, self.index.ntotal)
        return True

    def rebuild_index(self) -> int:
        """
        Rebuild the entire index from all .npy files in storage/embeddings/.

        Returns:
            Number of vectors indexed.
        """
        if not _FAISS_AVAILABLE:
            logger.warning("Cannot rebuild — FAISS unavailable.")
            return 0

        # Fresh index
        self.index = faiss.IndexFlatIP(VECTOR_DIM)
        self.doc_ids = []
        self.id_to_pos = {}

        npy_files = sorted(EMBEDDINGS_DIR.glob("*.npy"))
        for npy_path in npy_files:
            doc_id = npy_path.stem  # e.g. "DOC_A8692491"
            try:
                vector = np.load(npy_path)
                vec = vector.reshape(1, -1).astype("float32")
                self.index.add(vec)
                pos = len(self.doc_ids)
                self.doc_ids.append(doc_id)
                self.id_to_pos[doc_id] = pos
            except Exception as exc:
                logger.error("Failed to load %s: %s", npy_path, exc)

        self.save()
        logger.info("Index rebuilt: %d vectors from %s", self.index.ntotal, EMBEDDINGS_DIR)
        return self.index.ntotal

    # ------------------------------------------------------------------
    # Query encoding
    # ------------------------------------------------------------------

    def _get_model(self):
        """Lazy load the embedding model for query encoding."""
        if self._model is None:
            if not _ST_AVAILABLE:
                return None
            from embeddings.embedder import _get_model
            self._model = _get_model()
        return self._model

    def _encode_query(self, query_text: str) -> Optional[np.ndarray]:
        """Encode a query string into a normalized vector."""
        model = self._get_model()
        if model is None:
            logger.error("Cannot encode query — sentence-transformers unavailable.")
            return None

        vec = model.encode([query_text], show_progress_bar=False, convert_to_numpy=True)
        vec = vec.flatten()
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    # ------------------------------------------------------------------
    # Search operations
    # ------------------------------------------------------------------

    def _get_doc_metadata(self, doc_id: str) -> dict:
        """Load structured data JSON for a doc_id (for tags, type, entities)."""
        json_path = STRUCTURED_DATA_DIR / f"{doc_id}.json"
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                return {
                    "document_type": data.get("document_type", "unknown"),
                    "tags": data.get("tags", []),
                    "entities": data.get("entities", {}),
                }
            except Exception:
                pass
        return {"document_type": "unknown", "tags": [], "entities": {}}

    def search(self, query_text: str, top_k: int = 5) -> dict:
        """
        Semantic search: find documents most similar to a natural language query.

        Args:
            query_text: Natural language query string.
            top_k:      Number of top results to return.

        Returns:
            dict with keys:
              - query      (str)
              - total_docs (int)
              - results    (list of {doc_id, score, rank, document_type, tags})
        """
        empty = {"query": query_text, "total_docs": 0, "results": []}

        if not _FAISS_AVAILABLE or self.index is None or self.index.ntotal == 0:
            logger.warning("Search unavailable — index empty or FAISS not installed.")
            return empty

        # Encode query
        query_vec = self._encode_query(query_text)
        if query_vec is None:
            return empty

        # Search
        k = min(top_k, self.index.ntotal)
        query_vec_2d = query_vec.reshape(1, -1).astype("float32")
        scores, indices = self.index.search(query_vec_2d, k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0 or idx >= len(self.doc_ids):
                continue
            doc_id = self.doc_ids[idx]
            meta = self._get_doc_metadata(doc_id)
            results.append({
                "rank": rank,
                "doc_id": doc_id,
                "score": round(float(score), 4),
                "document_type": meta["document_type"],
                "tags": meta["tags"],
            })

        logger.info(
            "Search '%s' → %d results (top score=%.4f)",
            query_text[:50], len(results),
            results[0]["score"] if results else 0,
        )

        return {
            "query": query_text,
            "total_docs": self.index.ntotal,
            "results": results,
        }

    def search_by_doc_id(self, doc_id: str, top_k: int = 5) -> dict:
        """
        Find documents similar to an existing document.

        Args:
            doc_id: The reference document's ID.
            top_k:  Number of similar docs to return.

        Returns:
            dict with keys:
              - reference_doc_id (str)
              - total_docs       (int)
              - results          (list of {doc_id, score, rank, document_type, tags})
        """
        empty = {"reference_doc_id": doc_id, "total_docs": 0, "results": []}

        if not _FAISS_AVAILABLE or self.index is None:
            return empty

        # Load the document's vector
        npy_path = EMBEDDINGS_DIR / f"{doc_id}.npy"
        if not npy_path.exists():
            logger.warning("No embedding found for doc_id=%s", doc_id)
            return empty

        doc_vec = np.load(npy_path).reshape(1, -1).astype("float32")

        # Search (top_k + 1 because the doc itself will be in results)
        k = min(top_k + 1, self.index.ntotal)
        scores, indices = self.index.search(doc_vec, k)

        results = []
        for rank_counter, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0 or idx >= len(self.doc_ids):
                continue
            found_id = self.doc_ids[idx]
            # Skip the document itself
            if found_id == doc_id:
                continue
            meta = self._get_doc_metadata(found_id)
            results.append({
                "rank": len(results) + 1,
                "doc_id": found_id,
                "score": round(float(score), 4),
                "document_type": meta["document_type"],
                "tags": meta["tags"],
            })
            if len(results) >= top_k:
                break

        logger.info(
            "Similar docs to %s → %d results", doc_id, len(results),
        )

        return {
            "reference_doc_id": doc_id,
            "total_docs": self.index.ntotal,
            "results": results,
        }

    @property
    def total_documents(self) -> int:
        """Number of documents currently indexed."""
        if self.index is None:
            return 0
        return self.index.ntotal
