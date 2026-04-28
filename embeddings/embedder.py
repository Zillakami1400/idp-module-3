"""
embeddings/embedder.py
=======================
Module 4: Semantic Embeddings

Responsibilities:
  - Load the sentence-transformers model (all-MiniLM-L6-v2) as a lazy singleton
  - Chunk long documents into overlapping segments
  - Embed each chunk, mean-pool into a single document vector
  - Normalize to unit length (for cosine similarity via inner product)
  - Persist vector (.npy) and metadata (.json) to storage/embeddings/
  - Return a structured result dict

Graceful degradation:
  - If sentence-transformers is NOT installed → warn + return empty result
"""

import json
import time
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("embeddings.embedder")

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
EMBEDDINGS_DIR = Path("storage/embeddings")
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------
MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_DIM = 384            # output dimension for MiniLM-L6-v2
CHUNK_SIZE = 500            # characters per chunk
CHUNK_OVERLAP = 50          # overlap between consecutive chunks

# ---------------------------------------------------------------------------
# Optional sentence-transformers import
# ---------------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _ST_AVAILABLE = True
except ImportError:
    SentenceTransformer = None  # type: ignore
    _ST_AVAILABLE = False
    logger.warning(
        "sentence-transformers is not installed. Embedding generation will be "
        "skipped. Install with:  pip install sentence-transformers"
    )

# ---------------------------------------------------------------------------
# Lazy model singleton
# ---------------------------------------------------------------------------
_model_instance = None


def _get_model():
    """Load the embedding model once (lazy singleton)."""
    global _model_instance
    if _model_instance is None:
        logger.info("Loading embedding model '%s' (first call — may take a few seconds)…", MODEL_NAME)
        _model_instance = SentenceTransformer(MODEL_NAME)
        logger.info("Model loaded successfully. Vector dimension = %d", VECTOR_DIM)
    return _model_instance


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks.

    Args:
        text:       The input text string.
        chunk_size: Maximum character length per chunk.
        overlap:    Number of overlapping characters between chunks.

    Returns:
        List of text chunks.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    logger.debug("Text chunked: %d chars → %d chunks", len(text), len(chunks))
    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_embedding(
    doc_id: str,
    cleaned_text: Optional[str] = None,
) -> dict:
    """
    Generate a semantic embedding vector for a document.

    Pipeline:
      1. Chunk the cleaned text into overlapping segments
      2. Encode each chunk with all-MiniLM-L6-v2
      3. Mean-pool all chunk vectors → single document vector
      4. L2-normalize to unit length (cosine similarity via inner product)
      5. Save .npy vector file + metadata JSON
      6. Return result dict

    Args:
        doc_id:       Unique document identifier.
        cleaned_text: Pre-cleaned document text. If None, tries to load from
                      storage/ocr_text/{doc_id}.txt

    Returns:
        dict with keys:
          - doc_id       (str)
          - model_name   (str)
          - vector_dim   (int)
          - chunk_count  (int)
          - output_path  (str)   path to saved .npy file
          - meta_path    (str)   path to saved metadata JSON
          - status       (str)   "success" | "empty" | "skipped"
          - duration_s   (float)
    """
    empty_result = {
        "doc_id": doc_id,
        "model_name": MODEL_NAME,
        "vector_dim": VECTOR_DIM,
        "chunk_count": 0,
        "output_path": None,
        "meta_path": None,
        "status": "skipped",
        "duration_s": 0.0,
    }

    # Guard: sentence-transformers not installed
    if not _ST_AVAILABLE:
        logger.warning(
            "Skipping embedding for doc_id=%s — sentence-transformers not installed.",
            doc_id,
        )
        return empty_result

    start_time = time.perf_counter()

    # Load text if not provided
    if cleaned_text is None:
        text_file = Path("storage/ocr_text") / f"{doc_id}.txt"
        if not text_file.exists():
            logger.error("No text found for doc_id=%s (expected %s)", doc_id, text_file)
            empty_result["status"] = "error"
            return empty_result
        cleaned_text = text_file.read_text(encoding="utf-8")

    if not cleaned_text.strip():
        logger.warning("Empty text for doc_id=%s — returning empty embedding.", doc_id)
        empty_result["status"] = "empty"
        empty_result["duration_s"] = round(time.perf_counter() - start_time, 3)
        return empty_result

    logger.info("Generating embedding for doc_id=%s (%d chars)…", doc_id, len(cleaned_text))

    try:
        # 1. Chunk
        chunks = _chunk_text(cleaned_text)

        # 2. Encode
        model = _get_model()
        chunk_vectors = model.encode(chunks, show_progress_bar=False, convert_to_numpy=True)

        # 3. Mean-pool
        if len(chunk_vectors.shape) == 1:
            # Single chunk — already a 1-D vector
            doc_vector = chunk_vectors
        else:
            doc_vector = np.mean(chunk_vectors, axis=0)

        # 4. L2-normalize
        norm = np.linalg.norm(doc_vector)
        if norm > 0:
            doc_vector = doc_vector / norm

        # 5. Save
        npy_path = EMBEDDINGS_DIR / f"{doc_id}.npy"
        np.save(npy_path, doc_vector)
        logger.info("Embedding saved → %s", npy_path)

        meta = {
            "doc_id": doc_id,
            "model_name": MODEL_NAME,
            "vector_dim": int(doc_vector.shape[0]),
            "chunk_count": len(chunks),
            "text_length": len(cleaned_text),
        }
        meta_path = EMBEDDINGS_DIR / f"{doc_id}_meta.json"
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=4)

        duration = round(time.perf_counter() - start_time, 3)

        result = {
            "doc_id": doc_id,
            "model_name": MODEL_NAME,
            "vector_dim": int(doc_vector.shape[0]),
            "chunk_count": len(chunks),
            "output_path": str(npy_path),
            "meta_path": str(meta_path),
            "status": "success",
            "duration_s": duration,
        }

        logger.info(
            "Embedding complete — dim=%d | chunks=%d | %.3fs",
            doc_vector.shape[0], len(chunks), duration,
        )
        return result

    except Exception as exc:
        duration = round(time.perf_counter() - start_time, 3)
        logger.exception("Embedding failed for doc_id=%s: %s", doc_id, exc)
        return {
            "doc_id": doc_id,
            "model_name": MODEL_NAME,
            "vector_dim": VECTOR_DIM,
            "chunk_count": 0,
            "output_path": None,
            "meta_path": None,
            "status": "error",
            "error": str(exc),
            "duration_s": duration,
        }
