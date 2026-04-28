"""
Schema-free, multi-layer entity extraction.

This module is designed to work on arbitrary documents without relying on
trained models or fixed schemas. Extraction runs in four independent layers;
each layer has its own try/except so failures never block other layers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import json
import logging
import os
import re
import time

logger = logging.getLogger("extraction.extractor")


OCR_TEXT_DIR = Path("storage/ocr_text")
STRUCTURED_DATA_DIR = Path("storage/structured_data")
STRUCTURED_DATA_DIR.mkdir(parents=True, exist_ok=True)


class ExtractionError(Exception):
    """
    Backwards-compatibility exception type for existing callers.

    Note:
        Per session rules, this module avoids raising to callers; this type is
        kept so `app/ingestion/upload.py` can continue importing it unchanged.
    """


# -------------------------
# LAYER 1 — Universal KV
# -------------------------
_KV_RE = re.compile(r"(?P<key>[A-Z][A-Za-z0-9 /\-_]{1,50}?)\s*[:\-–]\s*(?P<value>[^\n]{1,200})")


# -------------------------
# LAYER 2 — Typed patterns
# -------------------------
_DATES_RE = re.compile(
    r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{2,4})\b",
    re.IGNORECASE,
)
_AMOUNTS_RE = re.compile(
    r"(?:[\$€£₹¥₩])\s?\d[\d,]*(?:\.\d{1,2})?|\d[\d,]*(?:\.\d{1,2})?\s?(?:USD|EUR|GBP|INR|JPY|AUD|CAD)\b",
    re.IGNORECASE,
)
_PHONE_RE = re.compile(r"\(?\+?[\d\s\-\(\)]{7,20}\d")
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_ORDER_ID_RE = re.compile(r"(?:order\s*(?:id|#|no\.?|number)\s*[:\-]?\s*)([A-Z0-9\-]{3,30})", re.IGNORECASE)
_INVOICE_ID_RE = re.compile(
    r"(?:invoice\s*(?:id|#|no\.?|number)\s*[:\-]?\s*)([A-Z0-9\-]{3,30})",
    re.IGNORECASE,
)
_TOTALS_RE = re.compile(
    r"(?:total\s*price|totalprice|grand\s*total|amount\s*due|net\s*total|subtotal|balance\s*due)[\s:]*([\$€£₹¥]?\s?\d[\d,]*(?:\.\d{1,2})?)",
    re.IGNORECASE,
)
_POSTAL_CODE_RE = re.compile(r"\b(?:[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}|\d{5}(?:-\d{4})?|[A-Z]\d[A-Z]\s?\d[A-Z]\d|\d{4,6})\b")


_KNOWN_TYPES = {"INVOICE", "PURCHASE_ORDER", "RECEIPT"}

_TARGET_FIELDS: dict[str, dict[str, list[str]]] = {
    "INVOICE": {
        "invoice_id": ["invoice id", "invoice no", "invoice number", "invoice #"],
        "order_id": ["order id", "order no", "order #", "order number"],
        "customer_id": ["customer id", "customer no", "client id"],
        "date": ["order date", "invoice date", "date"],
        "due_date": ["due date", "payment due"],
        "total_amount": ["total price", "totalprice", "grand total", "amount due", "total"],
        "customer_name": ["contact name", "customer name", "billed to", "bill to"],
    },
    "PURCHASE_ORDER": {
        "order_id": ["order id", "po number", "po #", "order no"],
        "date": ["order date", "date"],
        "customer_name": ["customer name", "buyer", "ship to"],
        "vendor_name": ["vendor", "supplier", "seller"],
    },
    "RECEIPT": {
        "receipt_id": ["receipt no", "transaction id", "ref"],
        "date": ["date", "transaction date"],
        "total_amount": ["total", "amount", "grand total"],
    },
}


def _dedupe_keep_order(items: list[str], cap: int) -> list[str]:
    try:
        seen: set[str] = set()
        out: list[str] = []
        for it in items:
            s = (it or "").strip()
            if not s or s in seen:
                continue
            seen.add(s)
            out.append(s)
            if len(out) >= cap:
                break
        return out
    except Exception:
        return (items or [])[:cap]


def _match_alias_value(text: str, alias: str) -> Optional[str]:
    """
    Pattern per alias:
      r"(?:ALIAS)\s*[:\-–]\s*([^\n]{1,100})" IGNORECASE
    """
    try:
        pat = re.compile(rf"(?:{re.escape(alias)})\s*[:\-–]\s*([^\n]{{1,100}})", re.IGNORECASE)
        m = pat.search(text or "")
        if not m:
            return None
        return (m.group(1) or "").strip()
    except Exception:
        return None


def extract_entities(
    text: str,
    doc_type: str = "UNKNOWN",
    normalised_tables: list | None = None,
) -> dict[str, Any]:
    """
    Four-layer schema-free entity extraction.

    Each layer runs independently in its own try/except block; a failure in one
    layer never stops the others.

    Args:
        text: Document text.
        doc_type: Optional doc type hint.
        normalised_tables: Optional output of `extraction.table_extractor.normalise_tables`.

    Returns:
        A dict with keys always present:
          doc_type, key_value_pairs, dates, amounts, phone_numbers, email_addresses,
          postal_codes, order_ids, invoice_ids, totals, table_entities,
          typed_fields
    """
    entities: dict[str, Any] = {
        "doc_type": doc_type or "UNKNOWN",
        "key_value_pairs": {},
        "dates": [],
        "amounts": [],
        "phone_numbers": [],
        "email_addresses": [],
        "postal_codes": [],
        "order_ids": [],
        "invoice_ids": [],
        "totals": [],
        "table_entities": [],
        "typed_fields": {},
    }

    safe_text = ""
    try:
        safe_text = text or ""
    except Exception:
        safe_text = ""

    # -------------------------
    # LAYER 1 — Universal KV extraction
    # -------------------------
    try:
        kv: dict[str, str] = {}
        for m in _KV_RE.finditer(safe_text):
            key = (m.group("key") or "").strip().title()
            value = (m.group("value") or "").strip()
            if len(key) <= 2 or not value:
                continue
            if key in kv:
                idx = 2
                while f"{key} {idx}" in kv:
                    idx += 1
                key = f"{key} {idx}"
            kv[key] = value
        entities["key_value_pairs"] = kv
    except Exception:
        logger.exception("Layer 1 KV extraction failed.")

    # -------------------------
    # LAYER 2 — Typed pattern extraction
    # -------------------------
    try:
        dates = [m.group(1) for m in _DATES_RE.finditer(safe_text)]
        amounts = [m.group(0) for m in _AMOUNTS_RE.finditer(safe_text)]
        phones = [m.group(0) for m in _PHONE_RE.finditer(safe_text)]
        emails = [m.group(0) for m in _EMAIL_RE.finditer(safe_text)]
        order_ids = [m.group(1) for m in _ORDER_ID_RE.finditer(safe_text)]
        invoice_ids = [m.group(1) for m in _INVOICE_ID_RE.finditer(safe_text)]
        totals = [m.group(1) for m in _TOTALS_RE.finditer(safe_text)]
        postal_codes = [m.group(0) for m in _POSTAL_CODE_RE.finditer(safe_text)]

        entities["dates"] = _dedupe_keep_order(dates, 10)
        entities["amounts"] = _dedupe_keep_order(amounts, 10)
        entities["phone_numbers"] = _dedupe_keep_order(phones, 5)
        entities["email_addresses"] = _dedupe_keep_order(emails, 5)
        entities["order_ids"] = _dedupe_keep_order(order_ids, 10)
        entities["invoice_ids"] = _dedupe_keep_order(invoice_ids, 10)
        entities["totals"] = _dedupe_keep_order(totals, 10)
        entities["postal_codes"] = _dedupe_keep_order(postal_codes, 10)
    except Exception:
        logger.exception("Layer 2 typed extraction failed.")

    # -------------------------
    # LAYER 3 — Table-derived entities
    # -------------------------
    try:
        if normalised_tables:
            table_entities: list[str] = []
            merged_kv: dict[str, str] = dict(entities.get("key_value_pairs") or {})
            totals_list: list[str] = list(entities.get("totals") or [])

            for table in normalised_tables:
                try:
                    if not isinstance(table, dict):
                        continue
                    headers = table.get("headers") or []
                    rows = table.get("rows") or []

                    # a) Row strings
                    row_cap = 50
                    for r in rows[:row_cap]:
                        if isinstance(r, dict):
                            parts = [f"{k}: {str(v).strip()}" for k, v in r.items()]
                            row_str = " | ".join(parts).strip()
                            if row_str:
                                table_entities.append(row_str)

                    # b) KV merge (non-overwriting)
                    if table.get("table_type") == "key_value":
                        kv_pairs = table.get("kv_pairs") or {}
                        if isinstance(kv_pairs, dict):
                            for k_raw, v_raw in kv_pairs.items():
                                k = (str(k_raw) if k_raw is not None else "").strip().title()
                                v = (str(v_raw) if v_raw is not None else "").strip()
                                if not k or not v:
                                    continue
                                if k not in merged_kv:
                                    merged_kv[k] = v

                    # c) Totals from footer dicts
                    if "totals" in table and isinstance(table.get("totals"), list):
                        for t in table.get("totals") or []:
                            if isinstance(t, dict):
                                val = (t.get("value") or "").strip()
                                if val:
                                    totals_list.append(val)
                except Exception:
                    logger.exception("Layer 3 failed processing a table; continuing.")
                    continue

            entities["table_entities"] = _dedupe_keep_order(table_entities, 50 * max(len(normalised_tables), 1))
            entities["key_value_pairs"] = merged_kv
            entities["totals"] = _dedupe_keep_order(totals_list, 10)
    except Exception:
        logger.exception("Layer 3 table-derived extraction failed.")

    # -------------------------
    # LAYER 4 — Doc-type targeted fields (known types only)
    # -------------------------
    try:
        dt = (entities.get("doc_type") or "UNKNOWN").upper()
        if dt in _KNOWN_TYPES:
            typed: dict[str, str] = {}
            spec = _TARGET_FIELDS.get(dt, {})
            for field_name, aliases in spec.items():
                for alias in aliases:
                    value = _match_alias_value(safe_text, alias)
                    if value:
                        typed[field_name] = value
                        break
            entities["typed_fields"] = typed
    except Exception:
        logger.exception("Layer 4 typed_fields extraction failed.")

    return entities


def entities_to_text(entities: dict) -> str:
    """
    Convert entities dict into a compact, embedding-friendly text block.

    Output format:
      "[doc_id / doc_type]\n"
      "Key Fields:\n  field: value\n..."
      "Total Amount: x\n"
      "Dates: x\n"
      "Document Fields:\n  Key: Value\n..."
      "Table Data:\n  row string\n..."

    Notes:
      - Uses `doc_type` from entities dict.
      - If typed_fields empty, skips Key Fields section.
      - If totals present, always includes each as "Total Amount: value".
      - Caps table_entities at 20 lines.
    """
    try:
        doc_id = str(entities.get("doc_id") or "").strip()
        doc_type = str(entities.get("doc_type") or "UNKNOWN").strip()

        lines: list[str] = [f"[{doc_id} / {doc_type}]".rstrip()]

        typed_fields = entities.get("typed_fields") or {}
        if isinstance(typed_fields, dict) and typed_fields:
            lines.append("Key Fields:")
            for k, v in typed_fields.items():
                lines.append(f"  {k}: {str(v).strip()}")

        totals = entities.get("totals") or []
        if isinstance(totals, list):
            for t in totals:
                s = str(t).strip()
                if s:
                    lines.append(f"Total Amount: {s}")

        dates = entities.get("dates") or []
        if isinstance(dates, list) and dates:
            lines.append("Dates: " + ", ".join(str(d).strip() for d in dates if str(d).strip()))

        kv = entities.get("key_value_pairs") or {}
        if isinstance(kv, dict) and kv:
            lines.append("Document Fields:")
            for k, v in kv.items():
                ks = str(k).strip()
                vs = str(v).strip()
                if ks and vs:
                    lines.append(f"  {ks}: {vs}")

        table_entities = entities.get("table_entities") or []
        if isinstance(table_entities, list) and table_entities:
            lines.append("Table Data:")
            for row_str in table_entities[:20]:
                s = str(row_str).strip()
                if s:
                    lines.append(f"  {s}")

        return "\n".join(lines).strip()
    except Exception:
        logger.exception("entities_to_text failed; returning empty string.")
        return ""


def _save_structured_data(doc_id: str, data: dict) -> Optional[str]:
    """Persist structured extraction output to `storage/structured_data/{doc_id}.json`."""
    try:
        out_path = STRUCTURED_DATA_DIR / f"{doc_id}.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=4, ensure_ascii=False)
        return str(out_path)
    except Exception:
        logger.exception("Failed saving structured data for doc_id=%s", doc_id)
        return None


def extract_information(doc_id: str, ocr_text: Optional[str] = None, file_path: Optional[str] = None) -> dict:
    """
    Backwards-compatible pipeline entry point used by `upload.py`.

    This function keeps the response shape expected by the existing API, while
    delegating entity extraction to the schema-free `extract_entities()` above.

    Never raises; returns partial results on errors.
    """
    start = time.perf_counter()
    result: dict[str, Any] = {
        "doc_id": doc_id,
        "document_type": "UNKNOWN",
        "entities": {},
        "cleaned_text": "",
        "tables": {"table_count": 0, "tables": []},
        "tags": [],
        "embedding": {"status": "skipped"},
        "output_path": None,
        "status": "error",
        "duration_s": 0.0,
    }

    try:
        text_val = ocr_text
        if text_val is None:
            ocr_file = OCR_TEXT_DIR / f"{doc_id}.txt"
            if ocr_file.exists():
                text_val = ocr_file.read_text(encoding="utf-8")
            else:
                text_val = ""

        cleaned_text = (text_val or "").strip()
        result["cleaned_text"] = cleaned_text

        if not cleaned_text:
            result["status"] = "empty"
            result["duration_s"] = round(time.perf_counter() - start, 3)
            return result

        # Normalise tables if available (from OCR digital extraction)
        normalised_tables: Optional[list] = None
        try:
            from extraction.table_extractor import normalise_tables  # local import

            raw_tables: list = []
            # Prefer raw tables passed via OCRResult if caller included them in ocr_text payload
            # (No guaranteed channel here; leave empty unless upstream wires it.)
            if raw_tables:
                normalised_tables = normalise_tables(raw_tables, doc_type="UNKNOWN")
        except Exception:
            normalised_tables = None

        entities = extract_entities(cleaned_text, doc_type="UNKNOWN", normalised_tables=normalised_tables)
        entities["doc_id"] = doc_id
        result["entities"] = entities
        result["document_type"] = str(entities.get("doc_type") or "UNKNOWN")

        out_path = _save_structured_data(doc_id, result)
        result["output_path"] = out_path
        result["status"] = "success"
        result["duration_s"] = round(time.perf_counter() - start, 3)
        return result
    except Exception as exc:
        logger.exception("extract_information failed for doc_id=%s", doc_id)
        result["error"] = str(exc)
        result["duration_s"] = round(time.perf_counter() - start, 3)
        return result
