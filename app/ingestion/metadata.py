"""
app/ingestion/metadata.py
=========================
Lightweight JSON-based metadata store.

Supports:
  - save_metadata(data)    → upsert by doc_id
  - load_all_metadata()    → list of all records
  - get_metadata(doc_id)   → single record or None
"""

import json
import os
import logging
from typing import Optional

logger = logging.getLogger("ingestion.metadata")

DB_FILE = "database/metadata.json"

# Ensure the database directory exists
os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)


def _read_db() -> list:
    """Read and return all records from the JSON store."""
    if not os.path.exists(DB_FILE):
        return []
    with open(DB_FILE, "r", encoding="utf-8") as fh:
        try:
            return json.load(fh)
        except json.JSONDecodeError:
            logger.warning("metadata.json is corrupt — resetting to empty list.")
            return []


def _write_db(records: list) -> None:
    """Persist all records to the JSON store."""
    with open(DB_FILE, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=4, ensure_ascii=False)


def save_metadata(data: dict) -> None:
    """
    Upsert a metadata record keyed by 'doc_id'.

    If a record with the same doc_id already exists it is replaced;
    otherwise the new record is appended.  This allows the upload route
    to call save_metadata() twice — once pre-OCR and once post-OCR —
    without creating duplicate entries.

    Args:
        data: Dictionary that must contain a 'doc_id' key.
    """
    doc_id = data.get("doc_id")
    records = _read_db()

    # Upsert: replace existing record with matching doc_id
    updated = False
    for i, record in enumerate(records):
        if record.get("doc_id") == doc_id:
            records[i] = data
            updated = True
            break

    if not updated:
        records.append(data)

    _write_db(records)
    logger.debug("Metadata %s for doc_id=%s.", "updated" if updated else "created", doc_id)


def load_all_metadata() -> list:
    """Return a list of all stored metadata records."""
    return _read_db()


def get_metadata(doc_id: str) -> Optional[dict]:
    """
    Fetch the metadata record for a specific doc_id.

    Args:
        doc_id: The document identifier to look up.

    Returns:
        The metadata dict, or None if not found.
    """
    for record in _read_db():
        if record.get("doc_id") == doc_id:
            return record
    return None