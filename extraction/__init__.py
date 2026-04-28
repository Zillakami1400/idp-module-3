"""
extraction/__init__.py
======================
Exposes the public API for the extraction module.
"""

from extraction.extractor import extract_information, ExtractionError, extract_entities, entities_to_text
from extraction.table_extractor import normalise_tables, fuzzy_match_column

__all__ = [
    "extract_information",
    "ExtractionError",
    "extract_entities",
    "entities_to_text",
    "normalise_tables",
    "fuzzy_match_column",
]
