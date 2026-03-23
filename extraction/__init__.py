"""
extraction/__init__.py
======================
Exposes the public API for the extraction module.
"""

from extraction.extractor import extract_information, ExtractionError
from extraction.table_extractor import extract_tables

__all__ = ["extract_information", "ExtractionError", "extract_tables"]
