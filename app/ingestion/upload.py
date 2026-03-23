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

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.ingestion.metadata import save_metadata
from ocr.processor import process_document, OCRProcessingError
from extraction.extractor import extract_information, ExtractionError

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
    # 5. Run OCR pipeline
    # ------------------------------------------------------------------
    logger.info("Triggering OCR for doc_id=%s …", doc_id)
    try:
        ocr_result = process_document(doc_id=doc_id, file_path=file_path)
        ocr_status = ocr_result["status"]  # "success" | "empty" | "error"
    except OCRProcessingError as exc:
        logger.error("OCR pipeline raised an error: %s", exc)
        ocr_result = {
            "status": "error",
            "error": str(exc),
            "char_count": 0,
            "word_count": 0,
            "page_count": 0,
            "output_path": None,
            "duration_s": 0.0,
        }
        ocr_status = "error"

    # ------------------------------------------------------------------
    # 6. Update metadata with OCR outcome
    # ------------------------------------------------------------------
    metadata["status"] = "processed" if ocr_status == "success" else "ocr_failed"
    metadata["ocr_status"] = ocr_status
    metadata["ocr_output_path"] = ocr_result.get("output_path")
    metadata["ocr_char_count"] = ocr_result.get("char_count", 0)
    metadata["ocr_word_count"] = ocr_result.get("word_count", 0)
    metadata["ocr_page_count"] = ocr_result.get("page_count", 0)
    metadata["ocr_duration_s"] = ocr_result.get("duration_s", 0.0)

    save_metadata(metadata)

    # ------------------------------------------------------------------
    # 7. Run extraction pipeline (only if OCR succeeded)
    # ------------------------------------------------------------------
    extraction_result = None
    if ocr_status == "success":
        logger.info("Triggering extraction for doc_id=%s …", doc_id)
        try:
            extraction_result = extract_information(
                doc_id=doc_id,
                ocr_text=ocr_result.get("ocr_text", ""),
                file_path=file_path,
            )
        except ExtractionError as exc:
            logger.error("Extraction pipeline raised an error: %s", exc)
            extraction_result = {
                "status": "error",
                "error": str(exc),
                "document_type": "generic_document",
                "entities": {},
                "tables": {"table_count": 0, "tables": []},
            }

    # ------------------------------------------------------------------
    # 8. Update metadata with extraction outcome
    # ------------------------------------------------------------------
    if extraction_result:
        metadata["extraction_status"] = extraction_result.get("status", "error")
        metadata["document_type"] = extraction_result.get("document_type")
        metadata["extraction_output_path"] = extraction_result.get("output_path")
        save_metadata(metadata)

    # ------------------------------------------------------------------
    # 9. Build API response
    # ------------------------------------------------------------------
    response = {
        "doc_id": doc_id,
        "original_filename": file.filename,
        "file_path": file_path,
        "file_type": ext,
        "upload_time": upload_time,
        "ocr": {
            "status": ocr_status,
            "output_path": ocr_result.get("output_path"),
            "page_count": ocr_result.get("page_count", 0),
            "char_count": ocr_result.get("char_count", 0),
            "word_count": ocr_result.get("word_count", 0),
            "duration_s": ocr_result.get("duration_s", 0.0),
        },
    }

    # Include OCR error info
    if ocr_status == "error":
        response["ocr"]["error"] = ocr_result.get("error", "Unknown OCR error")

    # Include extraction results
    if extraction_result:
        tables_info = extraction_result.get("tables", {"table_count": 0, "tables": []})
        response["extraction"] = {
            "status": extraction_result.get("status"),
            "document_type": extraction_result.get("document_type"),
            "entities": extraction_result.get("entities", {}),
            "output_path": extraction_result.get("output_path"),
            "duration_s": extraction_result.get("duration_s", 0.0),
            "tables": {
                "table_count": tables_info.get("table_count", 0),
                "flavor_used": tables_info.get("flavor_used"),
                "json_path": tables_info.get("json_path"),
            },
        }
        if extraction_result.get("status") == "error":
            response["extraction"]["error"] = extraction_result.get("error")

    return response