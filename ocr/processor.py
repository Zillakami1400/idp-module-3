"""
`ocr/processor.py`

Digital-first OCR pipeline using Docling.

Strategy for all documents (PDFs & Images):
  - Uses `DocumentConverter` from `docling`
  - Automatic detection of digital text, tables, layout
  - Automatic fallback to Tesseract for OCR when necessary

Session constraints honored:
  - No cloud calls (local-only).
  - Never raise to caller; catch exceptions and return partial results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional
import logging
import os
import time
import threading

from docling.document_converter import DocumentConverter, PdfFormatOption, WordFormatOption, ImageFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions

logger = logging.getLogger("ocr.processor")

OCR_TEXT_DIR = Path("storage/ocr_text")
OCR_TEXT_DIR.mkdir(parents=True, exist_ok=True)

ExtractionMethod = Literal["docling"]

# Configure Docling Converter pipeline to use local Tesseract
# Setting the path if necessary (if not on systemic PATH)
import os
if "TESSDATA_PREFIX" not in os.environ:
    os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"

# HuggingFace hub requires this on windows
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

def _get_converter() -> DocumentConverter:
    """Initialize DocumentConverter using Docling's RapidOcr."""
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True

    # Use RapidOCR which is installed natively with docling
    try:
        pipeline_options.ocr_options = RapidOcrOptions()
    except Exception:
        pass

    return DocumentConverter(
        allowed_formats=[InputFormat.PDF, InputFormat.IMAGE, InputFormat.DOCX],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline_options)
        }
    )
CONVERTER = _get_converter()
_CONVERTER_LOCK = threading.Lock()


@dataclass
class PageResult:
    """
    Result for a single page (PDF) or the single page of an image file.

    Attributes:
        page_number: 1-based page index.
        text: Extracted text (may be empty).
        raw_tables: List of raw pdfplumber-style table rows, cleaned:
            each table is a list of rows, each row is a list of cell strings.
        extraction_method: "docling"
    """
    page_number: int
    text: str
    raw_tables: list[list[list[str]]]
    extraction_method: str = "docling"


@dataclass
class OCRResult:
    """
    Full document OCR result.

    Attributes:
        doc_id: Document identifier.
        file_path: Original file path.
        pages: Per-page extraction results.
        full_text: Joined text across pages (populated by `finalise()`).
        all_raw_tables: Flattened tables across pages (populated by `finalise()`).
    """
    doc_id: str
    file_path: str
    pages: list[PageResult] = field(default_factory=list)
    full_text: str = ""
    all_raw_tables: list[list[list[str]]] = field(default_factory=list)

    def finalise(self) -> None:
        """
        Populate `full_text` and `all_raw_tables` from `pages`.
        """
        self.full_text = "\n\n".join((p.text or "").strip() for p in self.pages).strip()
        self.all_raw_tables = [t for p in self.pages for t in (p.raw_tables or [])]

    def __getitem__(self, key: str) -> Any:
        mapping = {
            "doc_id": self.doc_id,
            "file_path": self.file_path,
            "ocr_text": self.full_text,
            "output_path": str(OCR_TEXT_DIR / f"{self.doc_id}.txt"),
            "page_count": len(self.pages),
            "char_count": len(self.full_text or ""),
            "word_count": len((self.full_text or "").split()),
            "status": "success" if (self.full_text or "").strip() else "empty",
        }
        if key in mapping:
            return mapping[key]
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


_SUPPORTED_EXTS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}


def _save_text_to_storage(doc_id: str, text: str) -> Optional[str]:
    """
    Save text to `storage/ocr_text/{doc_id}.txt`.
    """
    try:
        out_path = OCR_TEXT_DIR / f"{doc_id}.txt"
        out_path.write_text(text or "", encoding="utf-8")
        return str(out_path)
    except Exception:
        logger.exception("Failed saving OCR text. doc_id=%s", doc_id)
        return None


def process_document(file_path: str, doc_id: str) -> OCRResult:
    """
    Entry point: process a document and return an `OCRResult`.

    - Detects file type from extension
    - Uses Docling pipeline
    - Never raises — logs errors and returns partial results
    """
    start = time.perf_counter()
    result = OCRResult(doc_id=doc_id, file_path=file_path)

    try:
        ext = Path(file_path).suffix.lower()
        if ext not in _SUPPORTED_EXTS:
            logger.warning("Unsupported file type: %s", file_path)
            return result

        if not os.path.exists(file_path):
            logger.error("File not found: %s", file_path)
            return result

        logger.info("Processing document with Docling. doc_id=%s path=%s", doc_id, file_path)
        
        with _CONVERTER_LOCK:
            conv_res = CONVERTER.convert(file_path)
            doc = conv_res.document

        if not doc:
            raise RuntimeError(f"Docling conversion failed for {file_path}. Document object is empty or None.")

        # 1. Use Docling's native markdown export for high-quality text representation
        try:
            full_markdown = doc.export_to_markdown()
        except Exception:
            logger.exception("Docling failed to export markdown.")
            full_markdown = ""

        # Group tables by page
        page_tables: dict[int, list[list[list[str]]]] = {}

        # 2. Iterate over all tables
        if hasattr(doc, "tables"):
            for table in doc.tables:
                try:
                    # Figure out page
                    page_no = 1
                    if hasattr(table, "prov") and table.prov:
                        page_no = table.prov[0].page_no

                    # Convert table to DataFrame then to nested lists
                    df = table.export_to_dataframe()
                    # Include header as first row, then data values
                    raw_table: list[list[str]] = [df.columns.tolist()] + df.astype(str).values.tolist()
                    
                    page_tables.setdefault(page_no, []).append(raw_table)
                except Exception:
                    logger.exception("Failed to export docling table on page %d", page_no)

        # 3. Create a single PageResult since markdown collapses pages beautifully, OR
        # split by page if strictly required. In our case, `finalise()` concatenates all page texts anyway.
        pages_extracted = [
            PageResult(
                page_number=1,
                text=full_markdown,
                raw_tables=page_tables.get(1, []),
                extraction_method="docling"
            )
        ]
        
        # Append any tables found on other pages without repeating full text
        for p_num in sorted(page_tables.keys()):
            if p_num != 1:
                pages_extracted.append(
                    PageResult(
                        page_number=p_num,
                        text="",  # Full text is already in page 1
                        raw_tables=page_tables.get(p_num, []),
                        extraction_method="docling"
                    )
                )

        result.pages = pages_extracted

    except Exception:
        logger.exception("process_document failed; returning partial result. doc_id=%s", doc_id)

    try:
        result.finalise()
    except Exception:
        logger.exception("finalise() failed; continuing. doc_id=%s", doc_id)

    _save_text_to_storage(doc_id, result.full_text)

    elapsed = time.perf_counter() - start
    logger.info(
        "OCR (Docling) done. doc_id=%s status=%s pages=%d chars=%d %.3fs",
        doc_id,
        result.get("status"),
        result.get("page_count", 0),
        result.get("char_count", 0),
        elapsed,
    )
    return result