"""
ocr/__init__.py
===============
Exposes the public API for the OCR module.
"""

from ocr.processor import process_document

__all__ = ["process_document"]
