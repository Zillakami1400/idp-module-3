"""
ocr/processor.py
================
Module 2: OCR Processing Pipeline

Responsibilities:
  - Accept a file path (PDF, PNG, JPG, JPEG)
  - Pre-process images for better OCR accuracy (grayscale, denoise, threshold)
  - Run Tesseract OCR on every page / image
  - Combine all extracted text into a single string
  - Persist the text to  storage/ocr_text/{doc_id}.txt
  - Return a structured result dict ready for downstream modules
    (classification, extraction, embeddings, etc.)

Downstream consumers:
  - Module 3 : Document Classification  → result["text"]
  - Module 4 : Information Extraction   → result["text"]
  - Module 5 : Embeddings & Search      → result["text"]
"""

import os
import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pytesseract
from PIL import Image
from pdf2image import convert_from_path

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ocr.processor")

# ---------------------------------------------------------------------------
# Tesseract executable path (Windows default install location)
# ---------------------------------------------------------------------------
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# ---------------------------------------------------------------------------
# Storage paths
# ---------------------------------------------------------------------------
OCR_OUTPUT_DIR = Path("storage/ocr_text")
OCR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Supported file types
# ---------------------------------------------------------------------------
PDF_EXTENSION = ".pdf"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------
class OCRProcessingError(Exception):
    """Raised when the OCR pipeline encounters an unrecoverable error."""


# ---------------------------------------------------------------------------
# Image pre-processing helpers
# ---------------------------------------------------------------------------

def _pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """Convert a Pillow Image to an OpenCV BGR ndarray."""
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def _preprocess_image(pil_image: Image.Image) -> Image.Image:
    """
    Pre-process a Pillow image to improve OCR accuracy:

    Steps:
      1. Convert to grayscale
      2. Apply Gaussian blur to reduce noise
      3. Apply adaptive thresholding (binarisation)
      4. Convert back to Pillow for pytesseract

    Returns:
        A pre-processed Pillow Image ready for OCR.
    """
    # Step 1 – Grayscale
    cv_img = _pil_to_cv2(pil_image)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    # Step 2 – Gaussian blur (mild, to reduce fine noise without blurring text)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Step 3 – Adaptive threshold → clean black/white image
    thresh = cv2.adaptiveThreshold(
        blurred,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=31,
        C=15,
    )

    # Step 4 – Back to Pillow
    return Image.fromarray(thresh)


# ---------------------------------------------------------------------------
# Core OCR helpers
# ---------------------------------------------------------------------------

def _ocr_single_image(pil_image: Image.Image, page_num: Optional[int] = None) -> str:
    """
    Run Tesseract on a single Pillow Image after pre-processing.

    Args:
        pil_image: The image to process.
        page_num:  Optional page number for logging (PDF pages).

    Returns:
        Extracted text string (may be empty if nothing detected).
    """
    label = f"page {page_num}" if page_num is not None else "image"
    logger.debug("Pre-processing %s …", label)

    processed = _preprocess_image(pil_image)

    logger.debug("Running OCR on %s …", label)
    # lang="eng" – extend to multi-language by passing e.g. lang="eng+fra"
    text = pytesseract.image_to_string(processed, lang="eng")

    logger.debug("OCR complete for %s — %d characters extracted.", label, len(text))
    return text


def _extract_text_from_pdf(file_path: str) -> str:
    """
    Convert each PDF page to an image and run OCR.

    Args:
        file_path: Absolute or relative path to the PDF file.

    Returns:
        Concatenated text for all pages, separated by page markers.

    Raises:
        OCRProcessingError: If pdf2image fails to convert the file.
    """
    logger.info("Converting PDF to images: %s", file_path)
    try:
        # dpi=300 gives a good quality/speed trade-off for OCR
        pages = convert_from_path(file_path, dpi=300)
    except Exception as exc:
        raise OCRProcessingError(
            f"Failed to convert PDF '{file_path}' to images: {exc}"
        ) from exc

    logger.info("  → %d page(s) found.", len(pages))

    all_text_parts = []
    for page_num, page_img in enumerate(pages, start=1):
        logger.info("  Processing page %d / %d …", page_num, len(pages))
        page_text = _ocr_single_image(page_img, page_num=page_num)
        # Tag each page so downstream modules can split on demand
        all_text_parts.append(f"--- Page {page_num} ---\n{page_text}")

    return "\n\n".join(all_text_parts)


def _extract_text_from_image(file_path: str) -> str:
    """
    Open an image file and run OCR directly.

    Args:
        file_path: Absolute or relative path to the image file.

    Returns:
        Extracted text string.

    Raises:
        OCRProcessingError: If the image cannot be opened.
    """
    logger.info("Opening image file: %s", file_path)
    try:
        pil_image = Image.open(file_path)
        # Ensure image data is fully loaded (avoids deferred-loading issues)
        pil_image.load()
    except Exception as exc:
        raise OCRProcessingError(
            f"Failed to open image '{file_path}': {exc}"
        ) from exc

    return _ocr_single_image(pil_image)


# ---------------------------------------------------------------------------
# Output persistence
# ---------------------------------------------------------------------------

def _save_text(doc_id: str, text: str) -> str:
    """
    Save extracted OCR text to  storage/ocr_text/{doc_id}.txt

    Args:
        doc_id: Unique document identifier.
        text:   The extracted text to persist.

    Returns:
        The absolute path of the saved file.
    """
    output_path = OCR_OUTPUT_DIR / f"{doc_id}.txt"
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    logger.info("OCR text saved → %s", output_path)
    return str(output_path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_document(doc_id: str, file_path: str) -> dict:
    """
    Main entry point for the OCR processing pipeline.

    Supports:
      - PDF  : converted page-by-page via pdf2image
      - PNG / JPG / JPEG : processed directly via Pillow + OpenCV

    Pipeline:
      1. Detect file type
      2. Pre-process image(s) (grayscale → denoise → threshold)
      3. Run Tesseract OCR
      4. Combine all extracted text
      5. Persist to storage/ocr_text/{doc_id}.txt
      6. Return structured result dict

    Args:
        doc_id:    Unique document identifier (e.g. "DOC_abc123ef")
        file_path: Path to the uploaded document.

    Returns:
        dict with the following keys:
          - doc_id      (str)  : same as input
          - file_path   (str)  : same as input
          - ocr_text    (str)  : full extracted text
          - output_path (str)  : path where text was saved
          - page_count  (int)  : number of pages / images processed
          - char_count  (int)  : total characters extracted
          - word_count  (int)  : approximate word count
          - status      (str)  : "success" | "empty" | "error"
          - error       (str?)  : present only when status == "error"
          - duration_s  (float): processing time in seconds

    Raises:
        OCRProcessingError: Re-raised after logging for unexpected errors.
    """
    logger.info("=" * 60)
    logger.info("Starting OCR pipeline for doc_id=%s", doc_id)
    logger.info("  File: %s", file_path)

    start_time = time.perf_counter()

    # Validate file existence
    if not os.path.exists(file_path):
        msg = f"File not found: {file_path}"
        logger.error(msg)
        return {
            "doc_id": doc_id,
            "file_path": file_path,
            "ocr_text": "",
            "output_path": None,
            "page_count": 0,
            "char_count": 0,
            "word_count": 0,
            "status": "error",
            "error": msg,
            "duration_s": 0.0,
        }

    ext = Path(file_path).suffix.lower()

    try:
        # ------------------------------------------------------------------
        # Route by file type
        # ------------------------------------------------------------------
        if ext == PDF_EXTENSION:
            extracted_text = _extract_text_from_pdf(file_path)
            # Count pages from the markers we inserted
            page_count = extracted_text.count("--- Page ")

        elif ext in IMAGE_EXTENSIONS:
            extracted_text = _extract_text_from_image(file_path)
            page_count = 1

        else:
            raise OCRProcessingError(
                f"Unsupported file type '{ext}'. "
                f"Supported: PDF, PNG, JPG, JPEG."
            )

        # ------------------------------------------------------------------
        # Persist
        # ------------------------------------------------------------------
        output_path = _save_text(doc_id, extracted_text)

        # ------------------------------------------------------------------
        # Build result
        # ------------------------------------------------------------------
        duration = round(time.perf_counter() - start_time, 3)
        char_count = len(extracted_text)
        word_count = len(extracted_text.split())
        status = "success" if char_count > 0 else "empty"

        result = {
            "doc_id": doc_id,
            "file_path": file_path,
            "ocr_text": extracted_text,
            "output_path": output_path,
            "page_count": page_count,
            "char_count": char_count,
            "word_count": word_count,
            "status": status,
            "duration_s": duration,
        }

        logger.info(
            "OCR complete — status=%s | pages=%d | chars=%d | words=%d | %.3fs",
            status, page_count, char_count, word_count, duration,
        )
        logger.info("=" * 60)
        return result

    except OCRProcessingError as exc:
        duration = round(time.perf_counter() - start_time, 3)
        logger.error("OCR failed for doc_id=%s: %s", doc_id, exc)
        return {
            "doc_id": doc_id,
            "file_path": file_path,
            "ocr_text": "",
            "output_path": None,
            "page_count": 0,
            "char_count": 0,
            "word_count": 0,
            "status": "error",
            "error": str(exc),
            "duration_s": duration,
        }

    except Exception as exc:
        # Catch-all — log and re-raise so the API layer can return HTTP 500
        duration = round(time.perf_counter() - start_time, 3)
        logger.exception("Unexpected error in OCR pipeline for doc_id=%s", doc_id)
        raise OCRProcessingError(
            f"Unexpected OCR error for doc_id={doc_id}: {exc}"
        ) from exc