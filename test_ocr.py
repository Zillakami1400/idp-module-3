"""
test_ocr.py
===========
Quick smoke test for the OCR processing pipeline.

Usage (from project root):
    python test_ocr.py <path_to_document>

Example:
    python test_ocr.py dataset/sample_invoice.pdf
    python test_ocr.py dataset/sample_receipt.png
"""

import sys
import json
import logging

# Configure logging so OCR processing logs are visible during the test
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

from ocr.processor import process_document


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_ocr.py <path_to_document>")
        print("Example: python test_ocr.py dataset/sample_invoice.pdf")
        sys.exit(1)

    file_path = sys.argv[1]
    doc_id = "TEST_DOC_001"

    print("\n" + "=" * 60)
    print(f"  Testing OCR Pipeline")
    print(f"  File   : {file_path}")
    print(f"  Doc ID : {doc_id}")
    print("=" * 60 + "\n")

    result = process_document(doc_id=doc_id, file_path=file_path)

    # Print structured result (excluding the full text to keep output readable)
    summary = {k: v for k, v in result.items() if k != "ocr_text"}
    print("\n--- OCR Result Summary ---")
    print(json.dumps(summary, indent=2))

    print("\n--- Extracted Text Preview (first 500 chars) ---")
    preview = result.get("ocr_text", "")[:500]
    print(preview if preview else "(empty)")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
