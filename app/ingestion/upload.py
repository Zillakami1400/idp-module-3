"""
app/ingestion/upload.py
=======================
Module 1 + Module 2 + Module 3 Integration

POST /upload
  1. Validate file type
  2. Generate unique doc_id
  3. Save file to storage/documents/
  4. Save metadata to database/metadata.json
  5. Trigger OCR pipeline automatically
  6. Trigger information extraction pipeline
  7. Return enriched response with OCR + extraction status
"""

import os
import shutil
import uuid
import logging
from datetime import datetime
from pathlib import Path
import json
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.ingestion.metadata import save_metadata
from ocr.processor import process_document
from intelligence.pipeline import run_pipeline, DocumentIntelligence

# Vector search — optional
try:
    from search.vector_store import VectorStore
    _SEARCH_AVAILABLE = True
except ImportError:
    _SEARCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("ingestion.upload")

# ---------------------------------------------------------------------------
# Router & constants
# ---------------------------------------------------------------------------
router = APIRouter(prefix="/api/v1", tags=["Document Ingestion"])

UPLOAD_FOLDER = "storage/documents"
ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _get_extension(filename: str) -> str:
    """Extract and return the lowercase file extension (without the dot)."""
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def _save_ocr_text(doc_id: str, text: str) -> str:
    """
    Persist OCR text to `storage/ocr_text/{doc_id}.txt`.
    Never raises; best-effort.
    """
    try:
        out_dir = Path("storage/ocr_text")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{doc_id}.txt"
        out_path.write_text(text or "", encoding="utf-8")
        return str(out_path)
    except Exception:
        logger.exception("Failed to save OCR text for doc_id=%s", doc_id)
        return str(Path("storage/ocr_text") / f"{doc_id}.txt")


def _save_structured_data(doc_id: str, intelligence: DocumentIntelligence) -> str:
    """
    Persist intelligence JSON to `storage/structured_data/{doc_id}.json`.
    Never raises; best-effort.
    """
    try:
        out_dir = Path("storage/structured_data")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{doc_id}.json"

        payload = {
            "doc_id": doc_id,
            "doc_type": intelligence.doc_type,
            "confidence": float(intelligence.confidence),
            "profile": intelligence.profile.to_dict(),
            "entities": intelligence.entities,
            "tables": intelligence.normalised_tables,
            "summary": intelligence.summary_text,
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "chunk_type": c.chunk_type,
                    "page": c.page_number,
                    "text": (c.text or "")[:500],
                }
                for c in (intelligence.chunks or [])
            ],
        }

        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=4, ensure_ascii=False)
        return str(out_path)
    except Exception:
        logger.exception("Failed to save structured intelligence for doc_id=%s", doc_id)
        return str(Path("storage/structured_data") / f"{doc_id}.json")


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/upload", summary="Upload a document, trigger OCR & extraction")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF, PNG, JPG, or JPEG document.

    Pipeline (automatic):
      1. Validate file extension
      2. Generate a unique doc_id
      3. Save the file to storage/documents/
      4. Persist upload metadata
      5. Run the OCR processing pipeline
      6. Return enriched response

    Returns:
        JSON with upload metadata + OCR summary.
    """
    # ------------------------------------------------------------------
    # 1. Validate file type
    # ------------------------------------------------------------------
    ext = _get_extension(file.filename)
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '.{ext}'. "
                f"Allowed types: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
            ),
        )

    # ------------------------------------------------------------------
    # 2. Generate unique document ID
    # ------------------------------------------------------------------
    doc_id = f"DOC_{uuid.uuid4().hex[:8].upper()}"
    upload_time = datetime.utcnow().isoformat()

    # ------------------------------------------------------------------
    # 3. Save the uploaded file
    # ------------------------------------------------------------------
    # Format:  storage/documents/DOC_XXXXXXXX_originalname.ext
    safe_filename = f"{doc_id}_{file.filename}"
    file_path = os.path.join(UPLOAD_FOLDER, safe_filename)

    logger.info("Saving uploaded file → %s", file_path)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except OSError as exc:
        logger.error("Failed to save file: %s", exc)
        raise HTTPException(status_code=500, detail=f"Could not save file: {exc}")
    finally:
        await file.close()

    # ------------------------------------------------------------------
    # 4. Persist upload metadata (pre-OCR)
    # ------------------------------------------------------------------
    metadata = {
        "doc_id": doc_id,
        "original_filename": file.filename,
        "stored_filename": safe_filename,
        "file_path": file_path,
        "file_type": ext,
        "upload_time": upload_time,
        "status": "uploaded",      # will be updated after OCR
        "ocr_status": "pending",
    }
    save_metadata(metadata)

    # ------------------------------------------------------------------
    # 5. Run OCR pipeline (in threadpool to avoid blocking event loop)
    # ------------------------------------------------------------------
    try:
        logger.info("Triggering OCR for doc_id=%s …", doc_id)
        ocr_result = await run_in_threadpool(process_document, str(file_path), doc_id)
    except Exception as exc:
        logger.error("OCR pipeline raised an error: %s", exc)
        metadata["status"] = "ocr_failed"
        metadata["ocr_status"] = "error"
        save_metadata(metadata)
        raise HTTPException(status_code=500, detail="OCR processing failed.")

    # ------------------------------------------------------------------
    # 6. Run intelligence pipeline (in threadpool to avoid blocking event loop)
    # ------------------------------------------------------------------
    try:
        logger.info("Triggering intelligence pipeline for doc_id=%s …", doc_id)
        intelligence = await run_in_threadpool(
            run_pipeline,
            doc_id,
            str(file_path),
            ocr_result.full_text,
            ocr_result.all_raw_tables,
            ocr_result.pages,
        )
    except Exception as exc:
        logger.error("Intelligence pipeline raised an error: %s", exc)
        metadata["status"] = "extraction_failed"
        save_metadata(metadata)
        raise HTTPException(status_code=500, detail="Extraction processing failed.")

    # ------------------------------------------------------------------
    # 7. Persist OCR + structured data
    # ------------------------------------------------------------------
    ocr_text_path = _save_ocr_text(doc_id, ocr_result.full_text)
    structured_path = _save_structured_data(doc_id, intelligence)

    # ------------------------------------------------------------------
    # 8. Update metadata with pipeline outcomes
    # ------------------------------------------------------------------
    metadata["status"] = "processed"
    metadata["ocr_status"] = "success" if (ocr_result.full_text or "").strip() else "empty"
    metadata["ocr_output_path"] = ocr_text_path
    metadata["ocr_char_count"] = len(ocr_result.full_text or "")
    metadata["ocr_word_count"] = len((ocr_result.full_text or "").split())
    metadata["ocr_page_count"] = len(ocr_result.pages or [])
    metadata["extraction_status"] = "success"
    metadata["document_type"] = intelligence.doc_type
    metadata["doc_type"] = intelligence.doc_type
    metadata["doc_type_confidence"] = round(float(intelligence.confidence or 0.0), 3)
    metadata["chunk_count"] = len(intelligence.chunks or [])
    metadata["table_count"] = len(intelligence.normalised_tables or [])
    metadata["extraction_output_path"] = structured_path
    save_metadata(metadata)

    # ------------------------------------------------------------------
    # 9. Embedding (existing call; now uses chunks text)
    # ------------------------------------------------------------------
    embedding_result = {"status": "skipped"}
    try:
        from embeddings.embedder import generate_embedding

        chunks_text = "\n\n".join((c.text or "") for c in (intelligence.chunks or []) if (c.text or "").strip())
        embedding_result = generate_embedding(doc_id=doc_id, cleaned_text=chunks_text)
    except Exception as exc:
        logger.warning("Embedding generation failed for doc_id=%s: %s", doc_id, exc)

    # Keep the existing FAISS block unchanged by providing `extraction_result`
    extraction_result = {"embedding": embedding_result}

    # ------------------------------------------------------------------
    # 9. Auto-index in FAISS (if embedding + search available)
    # ------------------------------------------------------------------
    if extraction_result and _SEARCH_AVAILABLE:
        embedding_info = extraction_result.get("embedding", {})
        if embedding_info.get("status") == "success":
            try:
                import numpy as np
                npy_path = embedding_info.get("output_path")
                if npy_path:
                    vector = np.load(npy_path)
                    store = VectorStore()
                    store.add_document(doc_id, vector)
                    logger.info("Auto-indexed doc_id=%s in FAISS.", doc_id)
            except Exception as exc:
                logger.warning("Failed to auto-index doc_id=%s: %s", doc_id, exc)

    # ------------------------------------------------------------------
    # 9. Build API response
    # ------------------------------------------------------------------
    return {
        "doc_id": doc_id,
        "doc_type": intelligence.doc_type,
        "confidence": round(float(intelligence.confidence or 0.0), 3),
        "page_count": len(ocr_result.pages or []),
        "table_count": len(intelligence.normalised_tables or []),
        "chunk_count": len(intelligence.chunks or []),
        "typed_fields": (intelligence.entities or {}).get("typed_fields", {}),
        "summary": intelligence.summary_text,
    }