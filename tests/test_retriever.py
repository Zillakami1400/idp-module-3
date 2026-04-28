"""
tests/test_retriever.py
========================
Unit tests for rag/retriever.py

Coverage:
  1. build_faiss_retriever() returns an IDPFaissRetriever instance.
  2. The retriever returns exactly k results when the FAISS index contains
     more than k documents.
  3. The retriever returns fewer than k results when the index has fewer
     than k documents (e.g. only 2 docs, k=5).
  4. doc_filter correctly filters results.
  5. _mmr() selects the expected number of items.
  6. _passes_filter() logic for document_type and tag matching.

All FAISS / embedding I/O is monkey-patched so the tests run offline,
without a GPU, and without touching real disk storage.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path when running from any directory
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helpers to build a fake VectorStore and fake structured_data on disk
# ---------------------------------------------------------------------------

def _make_fake_store(doc_ids: list[str], dim: int = 384):
    """
    Return a mock VectorStore whose .index and .doc_ids reflect doc_ids.

    The underlying faiss index stores random unit vectors so search()
    returns valid indices.
    """
    import faiss  # real faiss must be installed for the integration path

    index = faiss.IndexFlatIP(dim)
    rng = np.random.default_rng(seed=42)
    vecs = rng.standard_normal((len(doc_ids), dim)).astype("float32")
    # L2-normalize each row
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / norms
    index.add(vecs)

    store = MagicMock()
    store.index = index
    store.doc_ids = list(doc_ids)
    store.id_to_pos = {did: i for i, did in enumerate(doc_ids)}
    return store, vecs


def _write_fake_structured_data(tmp_path: Path, doc_ids: list[str], doc_type: str = "invoice"):
    """
    Write minimal structured_data JSON files for each doc_id into tmp_path.
    Returns the tmp_path so callers can point STRUCTURED_DATA_DIR at it.
    """
    for doc_id in doc_ids:
        data = {
            "doc_id": doc_id,
            "document_type": doc_type,
            "tags": ["Finance", "Q3"],
            "cleaned_text": f"Sample content for {doc_id}.",
        }
        (tmp_path / f"{doc_id}.json").write_text(json.dumps(data), encoding="utf-8")
    return tmp_path


def _write_fake_embeddings(tmp_path: Path, doc_ids: list[str], vecs: np.ndarray):
    """Write .npy embedding files for each doc_id into tmp_path."""
    for i, doc_id in enumerate(doc_ids):
        np.save(tmp_path / f"{doc_id}.npy", vecs[i])
    return tmp_path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def fake_dirs(tmp_path):
    """
    Create temporary structured_data/ and embeddings/ directories and return
    their paths as a dict.
    """
    struct_dir = tmp_path / "structured_data"
    embed_dir = tmp_path / "embeddings"
    struct_dir.mkdir()
    embed_dir.mkdir()
    return {"structured": struct_dir, "embeddings": embed_dir}


# ---------------------------------------------------------------------------
# 1. build_faiss_retriever returns the correct type
# ---------------------------------------------------------------------------

def test_build_faiss_retriever_returns_retriever():
    """build_faiss_retriever() should return an IDPFaissRetriever instance."""
    from rag.retriever import build_faiss_retriever, IDPFaissRetriever

    retriever = build_faiss_retriever(k=3)
    assert isinstance(retriever, IDPFaissRetriever)
    assert retriever.k == 3
    assert retriever.doc_filter is None


def test_build_faiss_retriever_with_filter():
    from rag.retriever import build_faiss_retriever

    retriever = build_faiss_retriever(doc_filter="invoice", k=7)
    assert retriever.doc_filter == "invoice"
    assert retriever.k == 7


# ---------------------------------------------------------------------------
# 2. Retriever returns exactly k results when index has more than k docs
# ---------------------------------------------------------------------------

def test_retriever_returns_exactly_k_results(fake_dirs):
    """
    Core requirement: when the FAISS index has N > k documents,
    the retriever must return exactly k results.
    """
    import faiss  # noqa: F401  — guard: test requires faiss

    k = 3
    n_docs = 10  # deliberately more than k
    doc_ids = [f"DOC_{i:04d}" for i in range(n_docs)]

    fake_store, vecs = _make_fake_store(doc_ids)
    _write_fake_structured_data(fake_dirs["structured"], doc_ids)
    _write_fake_embeddings(fake_dirs["embeddings"], doc_ids, vecs)

    from rag.retriever import build_faiss_retriever, IDPFaissRetriever

    retriever = build_faiss_retriever(k=k)

    # Patch the paths and the store inside the retriever
    with (
        patch("rag.retriever.STRUCTURED_DATA_DIR", fake_dirs["structured"]),
        patch("rag.retriever.EMBEDDINGS_DIR", fake_dirs["embeddings"]),
        patch.object(IDPFaissRetriever, "_get_store", return_value=fake_store),
        patch("rag.retriever._get_query_vector", return_value=vecs[0]),
    ):
        docs = retriever.invoke("show me invoices")  # .invoke() is the LangChain standard

    assert len(docs) == k, (
        f"Expected exactly {k} documents but got {len(docs)}. "
        "MMR or filtering may be returning too many/few results."
    )


# ---------------------------------------------------------------------------
# 3. Retriever returns ≤ k results when index has fewer than k documents
# ---------------------------------------------------------------------------

def test_retriever_returns_fewer_than_k_when_index_small(fake_dirs):
    """
    When the index has fewer documents than k, the retriever should return
    all available documents (not crash, not pad).
    """
    k = 5
    n_docs = 2  # less than k
    doc_ids = [f"DOC_{i:04d}" for i in range(n_docs)]

    fake_store, vecs = _make_fake_store(doc_ids)
    _write_fake_structured_data(fake_dirs["structured"], doc_ids)
    _write_fake_embeddings(fake_dirs["embeddings"], doc_ids, vecs)

    from rag.retriever import build_faiss_retriever, IDPFaissRetriever

    retriever = build_faiss_retriever(k=k)

    with (
        patch("rag.retriever.STRUCTURED_DATA_DIR", fake_dirs["structured"]),
        patch("rag.retriever.EMBEDDINGS_DIR", fake_dirs["embeddings"]),
        patch.object(IDPFaissRetriever, "_get_store", return_value=fake_store),
        patch("rag.retriever._get_query_vector", return_value=vecs[0]),
    ):
        docs = retriever.invoke("find contracts")

    assert len(docs) == n_docs
    assert len(docs) < k


# ---------------------------------------------------------------------------
# 4. doc_filter correctly narrows results
# ---------------------------------------------------------------------------

def test_retriever_doc_filter_excludes_non_matching(fake_dirs):
    """
    Documents whose document_type and tags do not match doc_filter should be
    excluded from results entirely.
    """
    k = 5
    invoice_ids = [f"INV_{i:04d}" for i in range(4)]
    contract_ids = [f"CON_{i:04d}" for i in range(4)]
    all_ids = invoice_ids + contract_ids

    fake_store, vecs = _make_fake_store(all_ids)

    # Write different types
    struct_dir = fake_dirs["structured"]
    embed_dir = fake_dirs["embeddings"]

    for i, doc_id in enumerate(invoice_ids):
        data = {"doc_id": doc_id, "document_type": "invoice",
                "tags": ["Finance"], "cleaned_text": f"Invoice {doc_id}"}
        (struct_dir / f"{doc_id}.json").write_text(json.dumps(data))
        np.save(embed_dir / f"{doc_id}.npy", vecs[i])

    for i, doc_id in enumerate(contract_ids):
        data = {"doc_id": doc_id, "document_type": "contract",
                "tags": ["Legal"], "cleaned_text": f"Contract {doc_id}"}
        (struct_dir / f"{doc_id}.json").write_text(json.dumps(data))
        np.save(embed_dir / f"{doc_id}.npy", vecs[len(invoice_ids) + i])

    from rag.retriever import build_faiss_retriever, IDPFaissRetriever

    retriever = build_faiss_retriever(doc_filter="invoice", k=k)

    with (
        patch("rag.retriever.STRUCTURED_DATA_DIR", struct_dir),
        patch("rag.retriever.EMBEDDINGS_DIR", embed_dir),
        patch.object(IDPFaissRetriever, "_get_store", return_value=fake_store),
        patch("rag.retriever._get_query_vector", return_value=vecs[0]),
    ):
        docs = retriever.invoke("billing details")

    returned_types = {d.metadata["document_type"] for d in docs}
    assert "contract" not in returned_types, "doc_filter='invoice' should exclude contracts"
    assert all(d.metadata["document_type"] == "invoice" for d in docs)


# ---------------------------------------------------------------------------
# 5. _mmr() selects exactly k distinct items
# ---------------------------------------------------------------------------

def test_mmr_selects_k_distinct_items():
    """_mmr() must return exactly k doc_ids with no duplicates."""
    from rag.retriever import _mmr

    rng = np.random.default_rng(0)
    dim = 384
    n = 20
    k = 5
    query_vec = rng.standard_normal(dim).astype("float32")
    query_vec /= np.linalg.norm(query_vec)

    candidate_vecs = rng.standard_normal((n, dim)).astype("float32")
    norms = np.linalg.norm(candidate_vecs, axis=1, keepdims=True)
    candidate_vecs /= norms

    candidate_ids = [f"DOC_{i}" for i in range(n)]

    selected = _mmr(query_vec, candidate_vecs, candidate_ids, k=k)

    assert len(selected) == k
    assert len(set(selected)) == k, "MMR returned duplicate doc_ids"


def test_mmr_returns_all_when_fewer_than_k():
    """When n < k, _mmr() should return all n items."""
    from rag.retriever import _mmr

    rng = np.random.default_rng(1)
    dim = 384
    n = 3
    k = 8
    query_vec = rng.standard_normal(dim).astype("float32")
    query_vec /= np.linalg.norm(query_vec)
    candidate_vecs = rng.standard_normal((n, dim)).astype("float32")
    candidate_vecs /= np.linalg.norm(candidate_vecs, axis=1, keepdims=True)
    candidate_ids = [f"DOC_{i}" for i in range(n)]

    selected = _mmr(query_vec, candidate_vecs, candidate_ids, k=k)
    assert len(selected) == n


# ---------------------------------------------------------------------------
# 6. _passes_filter logic
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("doc_type,tags,needle,expected", [
    ("invoice", ["Finance"],        "invoice",  True),
    ("invoice", ["Finance"],        "INVOICE",  True),   # case-insensitive
    ("contract", ["Legal"],         "invoice",  False),
    ("report",  ["Risk:High"],      "risk",     True),   # tag match
    ("unknown", [],                 None,       True),   # no filter → always True
    ("generic", ["Q3", "Finance"],  "q3",       True),
])
def test_passes_filter(doc_type, tags, needle, expected):
    from rag.retriever import _passes_filter

    meta = {"document_type": doc_type, "tags": tags}
    assert _passes_filter(meta, needle) == expected


# ---------------------------------------------------------------------------
# 7. Document metadata is populated correctly
# ---------------------------------------------------------------------------

def test_returned_documents_have_correct_metadata(fake_dirs):
    """Each returned LangChain Document should carry doc_id, type, tags, source."""
    k = 2
    doc_ids = [f"DOC_{i:04d}" for i in range(6)]
    fake_store, vecs = _make_fake_store(doc_ids)
    _write_fake_structured_data(fake_dirs["structured"], doc_ids, doc_type="invoice")
    _write_fake_embeddings(fake_dirs["embeddings"], doc_ids, vecs)

    from rag.retriever import build_faiss_retriever, IDPFaissRetriever

    retriever = build_faiss_retriever(k=k)

    with (
        patch("rag.retriever.STRUCTURED_DATA_DIR", fake_dirs["structured"]),
        patch("rag.retriever.EMBEDDINGS_DIR", fake_dirs["embeddings"]),
        patch.object(IDPFaissRetriever, "_get_store", return_value=fake_store),
        patch("rag.retriever._get_query_vector", return_value=vecs[0]),
    ):
        docs = retriever.invoke("test query")

    for doc in docs:
        assert "doc_id" in doc.metadata
        assert "document_type" in doc.metadata
        assert "tags" in doc.metadata
        assert "source" in doc.metadata
        assert doc.page_content != ""  # cleaned_text was written


# ---------------------------------------------------------------------------
# 8. Empty index returns no results (no crash)
# ---------------------------------------------------------------------------

def test_empty_index_returns_empty_list():
    """An empty FAISS index must return [] without raising."""
    fake_store = MagicMock()
    fake_store.index = None  # signals empty

    from rag.retriever import build_faiss_retriever, IDPFaissRetriever

    retriever = build_faiss_retriever(k=5)

    with patch.object(IDPFaissRetriever, "_get_store", return_value=fake_store):
        docs = retriever.invoke("anything")

    assert docs == []
