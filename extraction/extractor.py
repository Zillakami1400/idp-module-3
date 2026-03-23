"""
extraction/extractor.py
=======================
Module 3: Intelligent Information Extraction

Responsibilities:
  - Load raw OCR text from storage/ocr_text/{doc_id}.txt
  - Clean OCR noise (junk symbols, whitespace, currency normalization)
  - Detect document type via keyword rules
  - Extract structured entities via regex patterns
  - Save structured JSON to storage/structured_data/{doc_id}.json
  - Return a result dict for pipeline integration

Downstream consumers:
  - Module 4 : Semantic Embeddings   → result["entities"]
  - Module 5 : Vector Search         → result["document_type"], result["entities"]
"""

import os
import re
import json
import time
import logging
from pathlib import Path
from typing import Optional

# Table extraction (camelot) — imported lazily so it degrades gracefully
try:
    from extraction.table_extractor import extract_tables as _extract_tables
    _TABLE_EXTRACTION_AVAILABLE = True
except ImportError:
    _TABLE_EXTRACTION_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("extraction.extractor")

# ---------------------------------------------------------------------------
# Storage paths
# ---------------------------------------------------------------------------
OCR_TEXT_DIR = Path("storage/ocr_text")
STRUCTURED_DATA_DIR = Path("storage/structured_data")
STRUCTURED_DATA_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------
class ExtractionError(Exception):
    """Raised when the extraction pipeline encounters an unrecoverable error."""


# ===========================================================================
#  STEP 1 — TEXT CLEANING
# ===========================================================================

# Common OCR mis-reads that we can fix deterministically
_OCR_TYPO_MAP = {
    "UPL ID":  "UPI ID",
    "UPl ID":  "UPI ID",
    "UP| ID":  "UPI ID",
    "Payim":   "Paytm",
    "Paytin":  "Paytm",
    "Paytmn":  "Paytm",
    "PhonePa": "PhonePe",
}

# Regex for characters that are almost never meaningful in document text
_JUNK_CHARS_RE = re.compile(r"[©®™¢°¬«»¤§¶]+")

# Lines that are clearly ad / UI noise from payment‑app screenshots
_NOISE_KEYWORDS = [
    "scratch", "save on", "education", "no cost emi",
    "pay again", "share)", "check balance",
    "cratchcar", "demy", "ubustiwontatsc",
]


def clean_text(raw_text: str) -> str:
    """
    Clean raw OCR text by removing noise and normalizing content.

    Pipeline:
      1. Fix known OCR typos (UPL→UPI, Payim→Paytm, …)
      2. Normalize currency symbols (~ → ₹, Rs.→ ₹, INR → ₹)
      3. Remove junk unicode characters (©, ®, ¢, °, …)
      4. Strip advertisement / UI noise lines
      5. Collapse excessive whitespace
      6. Strip leading/trailing whitespace per line

    Args:
        raw_text: The raw OCR text string.

    Returns:
        Cleaned text string ready for entity extraction.
    """
    text = raw_text

    # 1. Fix known OCR typos
    for typo, fix in _OCR_TYPO_MAP.items():
        text = text.replace(typo, fix)

    # 2. Normalize currency symbols  →  ₹
    #    Handles:  ~300  |  Rs. 300  |  Rs 300  |  INR 300  |  ₹300
    text = re.sub(r"~\s*(\d)", r"₹\1", text)           # ~300 → ₹300
    text = re.sub(r"Rs\.?\s*", "₹", text, flags=re.I)  # Rs. 300 → ₹300
    text = re.sub(r"INR\s*", "₹", text, flags=re.I)    # INR 300 → ₹300

    # 3. Remove junk unicode characters
    text = _JUNK_CHARS_RE.sub("", text)

    # 4. Strip advertisement / UI noise lines
    cleaned_lines = []
    for line in text.splitlines():
        line_lower = line.strip().lower()
        # Skip empty lines and lines that match noise keywords
        if not line_lower:
            continue
        if any(kw in line_lower for kw in _NOISE_KEYWORDS):
            continue
        # Skip lines that are mostly non-alphanumeric (OCR garbage)
        alpha_ratio = sum(c.isalnum() or c.isspace() for c in line) / max(len(line), 1)
        if alpha_ratio < 0.4 and len(line.strip()) > 2:
            continue
        cleaned_lines.append(line.strip())

    # 5. Collapse excessive whitespace
    text = "\n".join(cleaned_lines)
    text = re.sub(r"[ \t]+", " ", text)         # multiple spaces → one
    text = re.sub(r"\n{3,}", "\n\n", text)      # 3+ newlines → 2

    logger.debug("Text cleaned: %d chars → %d chars", len(raw_text), len(text))
    return text.strip()


# ===========================================================================
#  STEP 2 — DOCUMENT TYPE DETECTION
# ===========================================================================

# Each document type has a set of indicator keywords.
# We count hits and pick the type with the highest score.
_DOC_TYPE_RULES = {
    "payment_receipt": [
        "upi", "ref", "paid", "payment", "paytm", "phonepe", "gpay",
        "google pay", "received", "completed", "transaction",
        "upi id", "ref. no", "reference",
    ],
    "invoice": [
        "invoice", "bill to", "total", "tax", "gst", "gstin",
        "subtotal", "due date", "invoice number", "inv no",
        "qty", "quantity", "unit price", "discount",
    ],
    "bank_statement": [
        "account", "balance", "statement", "ifsc", "branch",
        "credit", "debit", "opening balance", "closing balance",
        "transaction history", "passbook",
    ],
}


def detect_document_type(cleaned_text: str) -> str:
    """
    Classify document type using keyword frequency scoring.

    Strategy:
      - Convert text to lowercase
      - Count how many indicator keywords appear for each type
      - The type with the highest score wins
      - Falls back to 'generic_document' if no type scores > 0

    Args:
        cleaned_text: Pre-cleaned OCR text.

    Returns:
        One of: 'payment_receipt', 'invoice', 'bank_statement', 'generic_document'
    """
    text_lower = cleaned_text.lower()
    scores = {}

    for doc_type, keywords in _DOC_TYPE_RULES.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        scores[doc_type] = score

    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]

    if best_score == 0:
        logger.info("No keywords matched — defaulting to generic_document.")
        return "generic_document"

    logger.info(
        "Document type detected: %s (score=%d) | all scores: %s",
        best_type, best_score, scores,
    )
    return best_type


# ===========================================================================
#  STEP 3 — ENTITY EXTRACTION (Regex-based)
# ===========================================================================

def _extract_name(text: str) -> Optional[str]:
    """
    Extract a person's name — typically the first line with 2+ capitalized words.

    Heuristic: find a line where most words start with an uppercase letter
    and there are at least 2 words. Skip lines with known non-name patterns.
    """
    skip_patterns = ["upi", "ref", "invoice", "total", "page", "---"]
    for line in text.splitlines():
        line = line.strip()
        if not line or len(line) < 4:
            continue
        # Skip lines containing known non-name keywords
        if any(pat in line.lower() for pat in skip_patterns):
            continue
        words = line.split()
        if len(words) < 2 or len(words) > 5:
            continue
        # Check if most words start with uppercase (name-like)
        capitalized = sum(1 for w in words if w[0].isupper())
        if capitalized >= len(words) * 0.6:
            # Exclude lines that are mostly digits
            digit_ratio = sum(c.isdigit() for c in line) / max(len(line), 1)
            if digit_ratio < 0.3:
                return line

    return None


def _extract_upi_id(text: str) -> Optional[str]:
    """
    Extract UPI ID — pattern: word@word (e.g., mumtajbegamm65-1@oksbi).
    """
    match = re.search(r"[\w.\-]+@[\w]+", text)
    return match.group(0) if match else None


def _extract_amount(text: str) -> Optional[str]:
    """
    Extract monetary amount.

    Handles patterns:
      - ₹300  |  ₹1,500  |  ₹1,500.00
      - Standalone large numbers near currency context
    """
    # Try currency symbol first: ₹300, ₹1,500.00
    match = re.search(r"₹\s*([\d,]+(?:\.\d{1,2})?)", text)
    if match:
        return match.group(1).replace(",", "")

    # Fallback: look for number near payment-related words
    match = re.search(
        r"(?:amount|total|paid|received)\s*[:\-]?\s*([\d,]+(?:\.\d{1,2})?)",
        text, re.I
    )
    if match:
        return match.group(1).replace(",", "")

    return None


def _extract_date(text: str) -> Optional[str]:
    """
    Extract date/time from text.

    Supports:
      - 21 Feb, 01:00 AM
      - 21/02/2026
      - 21-02-2026
      - Feb 21, 2026
      - 2026-02-21
    """
    patterns = [
        # "21 Feb, 01:00 AM" or "21 Feb 01:00 AM"
        r"\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s,]*\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?",
        # "21 Feb 2026" or "Feb 21, 2026"
        r"\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s,]*\d{4}",
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}[\s,]+\d{4}",
        # "21/02/2026" or "21-02-2026"
        r"\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}",
        # ISO "2026-02-21"
        r"\d{4}-\d{2}-\d{2}",
        # Just "21 Feb" (no year)
        r"\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            return match.group(0).strip().rstrip(",")

    return None


def _extract_reference_number(text: str) -> Optional[str]:
    """
    Extract a transaction reference / receipt number.

    Looks for: "Ref" or "Reference" followed by a long numeric string (10+ digits).
    Also tries standalone 10+ digit numbers if Ref keyword is absent.
    """
    # Pattern 1: "Ref. No: 605275164369" or "Reference: 12345..."
    match = re.search(
        r"(?:Ref\.?\s*(?:No\.?|Number)?|Reference)\s*[:\-]?\s*(\d{6,})",
        text, re.I
    )
    if match:
        return match.group(1)

    # Pattern 2: "Transaction ID: 12345..."
    match = re.search(
        r"(?:Transaction|Txn)\s*(?:ID|No\.?|Number)?\s*[:\-]?\s*(\d{6,})",
        text, re.I
    )
    if match:
        return match.group(1)

    # Pattern 3: Standalone 10+ digit number (likely a reference)
    match = re.search(r"\b(\d{10,})\b", text)
    if match:
        return match.group(1)

    return None


def extract_entities(cleaned_text: str, document_type: str) -> dict:
    """
    Run all entity extractors and return a unified entities dict.

    Args:
        cleaned_text:  Pre-cleaned OCR text.
        document_type: Detected document type (for future type-specific logic).

    Returns:
        Dictionary of extracted entities (values may be None if not found).
    """
    entities = {
        "name": _extract_name(cleaned_text),
        "upi_id": _extract_upi_id(cleaned_text),
        "amount": _extract_amount(cleaned_text),
        "date_time": _extract_date(cleaned_text),
        "reference_number": _extract_reference_number(cleaned_text),
    }

    # Log what we found
    found = {k: v for k, v in entities.items() if v is not None}
    missed = [k for k, v in entities.items() if v is None]
    logger.info("Entities extracted: %s", found)
    if missed:
        logger.info("Entities not found: %s", missed)

    return entities


# ===========================================================================
#  STEP 4 — PERSISTENCE
# ===========================================================================

def _save_structured_data(doc_id: str, data: dict) -> str:
    """
    Save extraction results as JSON to storage/structured_data/{doc_id}.json

    Args:
        doc_id: Unique document identifier.
        data:   The structured extraction result dict.

    Returns:
        Path to the saved JSON file.
    """
    output_path = STRUCTURED_DATA_DIR / f"{doc_id}.json"
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=4, ensure_ascii=False)
    logger.info("Structured data saved → %s", output_path)
    return str(output_path)


# ===========================================================================
#  PUBLIC API
# ===========================================================================

def extract_information(
    doc_id: str,
    ocr_text: Optional[str] = None,
    file_path: Optional[str] = None,
) -> dict:
    """
    Main entry point for the information extraction pipeline.

    Pipeline:
      1. Load OCR text (from file or passed directly)
      2. Clean the text
      3. Detect document type
      4. Extract entities
      5. Extract tables (PDF only, via camelot — skipped gracefully if unavailable)
      6. Save structured JSON
      7. Return result dict

    Args:
        doc_id:    Unique document identifier (e.g. "DOC_371377E3")
        ocr_text:  Optional — if provided, use this text instead of loading from file.
                   Useful when called from the upload pipeline where OCR text is
                   already in memory.
        file_path: Optional — original file path. Required to enable table extraction
                   for PDF files. Images are skipped automatically.

    Returns:
        dict with keys:
          - doc_id          (str)
          - document_type   (str)
          - entities        (dict)
          - cleaned_text    (str)
          - tables          (dict)  table_count + per-table metadata
          - output_path     (str)
          - status          (str)  "success" | "empty" | "error"
          - error           (str?) present only when status == "error"
          - duration_s      (float)
    """
    logger.info("=" * 60)
    logger.info("Starting extraction pipeline for doc_id=%s", doc_id)

    start_time = time.perf_counter()

    try:
        # ------------------------------------------------------------------
        # 1. Load OCR text
        # ------------------------------------------------------------------
        if ocr_text is None:
            ocr_file = OCR_TEXT_DIR / f"{doc_id}.txt"
            if not ocr_file.exists():
                raise ExtractionError(
                    f"OCR text file not found: {ocr_file}"
                )
            ocr_text = ocr_file.read_text(encoding="utf-8")
            logger.info("Loaded OCR text from %s (%d chars)", ocr_file, len(ocr_text))

        if not ocr_text.strip():
            duration = round(time.perf_counter() - start_time, 3)
            logger.warning("OCR text is empty for doc_id=%s", doc_id)
            return {
                "doc_id": doc_id,
                "document_type": "generic_document",
                "entities": {},
                "cleaned_text": "",
                "output_path": None,
                "status": "empty",
                "duration_s": duration,
            }

        # ------------------------------------------------------------------
        # 2. Clean text
        # ------------------------------------------------------------------
        cleaned = clean_text(ocr_text)
        logger.info("Text cleaned: %d → %d chars", len(ocr_text), len(cleaned))

        # ------------------------------------------------------------------
        # 3. Detect document type
        # ------------------------------------------------------------------
        doc_type = detect_document_type(cleaned)

        # ------------------------------------------------------------------
        # 4. Extract entities
        # ------------------------------------------------------------------
        entities = extract_entities(cleaned, doc_type)

        # ------------------------------------------------------------------
        # 5. Extract tables (PDF only)
        # ------------------------------------------------------------------
        tables_result: dict = {"table_count": 0, "tables": []}
        if file_path and str(file_path).lower().endswith(".pdf"):
            if _TABLE_EXTRACTION_AVAILABLE:
                logger.info("Extracting tables from PDF: %s", file_path)
                tables_result = _extract_tables(
                    pdf_path=file_path,
                    doc_id=doc_id,
                )
            else:
                logger.warning(
                    "Table extraction skipped for doc_id=%s — "
                    "extraction.table_extractor could not be imported.",
                    doc_id,
                )
        else:
            logger.debug(
                "Table extraction skipped for doc_id=%s — "
                "file is not a PDF or file_path not provided.",
                doc_id,
            )

        # ------------------------------------------------------------------
        # 6. Build result
        # ------------------------------------------------------------------
        result = {
            "doc_id": doc_id,
            "document_type": doc_type,
            "entities": entities,
            "cleaned_text": cleaned,
            "tables": tables_result,
        }

        # ------------------------------------------------------------------
        # 7. Save to JSON
        # ------------------------------------------------------------------
        output_path = _save_structured_data(doc_id, result)

        duration = round(time.perf_counter() - start_time, 3)

        result["output_path"] = output_path
        result["status"] = "success"
        result["duration_s"] = duration

        logger.info(
            "Extraction complete — type=%s | entities=%d found | %.3fs",
            doc_type,
            sum(1 for v in entities.values() if v is not None),
            duration,
        )
        logger.info("=" * 60)
        return result

    except ExtractionError as exc:
        duration = round(time.perf_counter() - start_time, 3)
        logger.error("Extraction failed for doc_id=%s: %s", doc_id, exc)
        return {
            "doc_id": doc_id,
            "document_type": "generic_document",
            "entities": {},
            "cleaned_text": "",
            "output_path": None,
            "status": "error",
            "error": str(exc),
            "duration_s": duration,
        }

    except Exception as exc:
        duration = round(time.perf_counter() - start_time, 3)
        logger.exception("Unexpected error in extraction for doc_id=%s", doc_id)
        raise ExtractionError(
            f"Unexpected extraction error for doc_id={doc_id}: {exc}"
        ) from exc
