"""
test_extraction.py
==================
Smoke test for Module 3: Information Extraction pipeline.

Usage (from project root):
    python test_extraction.py <doc_id>

Example:
    python test_extraction.py DOC_371377E3

This will:
  1. Load OCR text from storage/ocr_text/{doc_id}.txt
  2. Clean the text
  3. Detect document type
  4. Extract entities
  5. Save JSON to storage/structured_data/{doc_id}.json
  6. Print a summary
"""

import sys
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

from extraction.extractor import extract_information, clean_text


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_extraction.py <doc_id>")
        print("Example: python test_extraction.py DOC_371377E3")
        sys.exit(1)

    doc_id = sys.argv[1]

    print("\n" + "=" * 60)
    print(f"  Testing Extraction Pipeline")
    print(f"  Doc ID : {doc_id}")
    print("=" * 60)

    # Run the full extraction pipeline
    result = extract_information(doc_id=doc_id)

    # Print cleaned text
    print("\n--- Cleaned Text ---")
    print(result.get("cleaned_text", "(empty)"))

    # Print structured result (excluding cleaned_text for readability)
    summary = {k: v for k, v in result.items() if k != "cleaned_text"}
    print("\n--- Extraction Result ---")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n" + "=" * 60)

    if result.get("status") == "success":
        print(f"\n✓ JSON saved to: {result.get('output_path')}")
    else:
        print(f"\n✗ Extraction status: {result.get('status')}")
        if result.get("error"):
            print(f"  Error: {result['error']}")


if __name__ == "__main__":
    main()
