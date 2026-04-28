"""
Schema-free table normalisation.

Input is raw pdfplumber output:
  - list of tables
  - each table is a list of rows
  - each row is a list of str|None cells

This module does not assume any column names in advance and aims to work on any
table structure. All functions are exception-safe.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any, Optional
import logging
import re

logger = logging.getLogger("extraction.table_extractor")


_KV_LEFT_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9 _\-/\.]{1,40}$")
_HEADER_CELL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9 _\-/\.%#]{0,40}$")
_NUMERIC_ONLY_RE = re.compile(r"^\s*[\d.,]+\s*$")

_FOOTER_LABELS = [
    "totalprice",
    "total price",
    "grand total",
    "subtotal",
    "amount due",
    "net total",
    "balance due",
]


def fuzzy_match_column(query: str, candidates: list[str], threshold: float = 0.6) -> str | None:
    """
    Fuzzy match a query against candidate column names.

    Uses SequenceMatcher ratio. Substring match gets a +0.9 score boost (capped at 1.0).
    Returns the best match above threshold, else None.
    """
    try:
        q = (query or "").strip().lower()
        if not q:
            return None
        best: Optional[str] = None
        best_score = 0.0
        for cand in candidates or []:
            c = (cand or "").strip()
            if not c:
                continue
            c_lower = c.lower()
            score = SequenceMatcher(None, q, c_lower).ratio()
            if q in c_lower or c_lower in q:
                score = min(1.0, score + 0.9)
            if score > best_score:
                best_score = score
                best = c
        return best if best is not None and best_score >= float(threshold) else None
    except Exception:
        logger.exception("fuzzy_match_column failed.")
        return None


def _clean_cell(v: Any) -> str:
    """Strip None and normalise whitespace."""
    try:
        if v is None:
            return ""
        return re.sub(r"\s+", " ", str(v)).strip()
    except Exception:
        return ""


def _is_numeric_cell(cell: str) -> bool:
    try:
        return bool(_NUMERIC_ONLY_RE.match(cell or ""))
    except Exception:
        return False


def _table_dimensions(table: list[list[Any]]) -> int:
    try:
        return max((len(r) for r in table if isinstance(r, list)), default=0)
    except Exception:
        return 0


def _pad_or_truncate(row: list[str], n: int) -> list[str]:
    try:
        if n <= 0:
            return []
        if len(row) >= n:
            return row[:n]
        return row + ([""] * (n - len(row)))
    except Exception:
        return (row or [])[: max(n, 0)]


def _make_unique_headers(headers: list[str]) -> list[str]:
    """Strip trailing ':' and whitespace; dedupe by appending _2, _3, ..."""
    try:
        base = [re.sub(r":\s*$", "", (h or "")).strip() for h in headers]
        seen: dict[str, int] = {}
        out: list[str] = []
        for h in base:
            name = h or ""
            if not name:
                name = "Column"
            if name not in seen:
                seen[name] = 1
                out.append(name)
            else:
                seen[name] += 1
                out.append(f"{name}_{seen[name]}")
        return out
    except Exception:
        logger.exception("_make_unique_headers failed.")
        return headers


def _detect_kv_table(table: list[list[Any]]) -> bool:
    """
    CASE 1 — KV table detection.

    Condition:
      - exactly 2 columns AND
      - >= 70% of non-empty left-column cells end with ':' OR match key regex.
    """
    try:
        if not table:
            return False
        if _table_dimensions(table) != 2:
            return False
        left_cells: list[str] = []
        for row in table:
            if not isinstance(row, list):
                continue
            left = _clean_cell(row[0]) if len(row) > 0 else ""
            if left.strip():
                left_cells.append(left.strip())
        if not left_cells:
            return False
        qualifying = 0
        for left in left_cells:
            if left.endswith(":") or _KV_LEFT_KEY_RE.match(left):
                qualifying += 1
        return (qualifying / max(len(left_cells), 1)) >= 0.7
    except Exception:
        logger.exception("_detect_kv_table failed.")
        return False


def _kv_table_to_output(table: list[list[Any]]) -> dict[str, Any]:
    kv: dict[str, str] = {}
    lines: list[str] = []
    try:
        for row in table or []:
            if not isinstance(row, list):
                continue
            key_raw = _clean_cell(row[0]) if len(row) > 0 else ""
            val_raw = _clean_cell(row[1]) if len(row) > 1 else ""
            key = re.sub(r":\s*$", "", key_raw).strip()
            value = val_raw.strip()
            if not key:
                continue
            kv[key] = value
            lines.append(f"{key}: {value}".rstrip())
    except Exception:
        logger.exception("_kv_table_to_output failed.")

    return {
        "table_type": "key_value",
        "headers": list(kv.keys()),
        "rows": [dict(kv)],
        "kv_pairs": dict(kv),
        "raw_text": "\n".join(lines).strip(),
    }


def _infer_header_row(table: list[list[str]]) -> tuple[list[str], int]:
    """
    CASE 3 — Header row inference.

    Only check first 4 rows. First qualifying row is header.
    If no header found: synthesise Column_1..Column_N
    """
    try:
        max_cols = _table_dimensions(table)  # type: ignore[arg-type]
        scan_rows = table[:4] if table else []
        for idx, row in enumerate(scan_rows):
            non_empty = [c for c in row if c.strip()]
            if not non_empty:
                continue
            if all(c.strip().endswith(":") for c in non_empty):
                hdr = [re.sub(r":\s*$", "", c).strip() for c in row]
                return _make_unique_headers(hdr), idx
            matches = 0
            total = 0
            for c in row:
                c2 = re.sub(r":\s*$", "", c).strip()
                if not c2:
                    continue
                total += 1
                if _is_numeric_cell(c2):
                    continue
                if _HEADER_CELL_RE.match(c2):
                    matches += 1
            if total > 0 and (matches / total) >= 0.6:
                hdr = [re.sub(r":\s*$", "", c).strip() for c in row]
                return _make_unique_headers(hdr), idx
        hdr = [f"Column_{i}" for i in range(1, max_cols + 1)]
        return _make_unique_headers(hdr), -1
    except Exception:
        logger.exception("_infer_header_row failed.")
        return [], -1


def _is_footer_row(row: list[str]) -> bool:
    """
    CASE 2 — Footer/total row detection.
    """
    try:
        non_empty = [c.strip() for c in row if c.strip()]
        if not non_empty:
            return False
        labels = set(_FOOTER_LABELS)
        if len(non_empty) == 2 and non_empty[0].lower() in labels:
            return True
        return any(c.lower() in labels for c in non_empty)
    except Exception:
        return False


def _extract_footer_totals(rows: list[list[str]]) -> tuple[list[dict[str, str]], list[list[str]]]:
    totals: list[dict[str, str]] = []
    remaining: list[list[str]] = []
    try:
        labels = set(_FOOTER_LABELS)

        def first_label_cell(r: list[str]) -> Optional[str]:
            for c in r:
                c2 = (c or "").strip()
                if c2.lower() in labels:
                    return c2
            return None

        for row in rows:
            if _is_footer_row(row):
                label = first_label_cell(row) or ""
                value = ""
                try:
                    if label:
                        found_label = False
                        for c in row:
                            c2 = (c or "").strip()
                            if not c2:
                                continue
                            if not found_label and c2.lower() == label.lower():
                                found_label = True
                                continue
                            if found_label:
                                value = c2
                                break
                    if not value:
                        non_empty = [c.strip() for c in row if c.strip()]
                        if len(non_empty) >= 2:
                            value = non_empty[1]
                except Exception:
                    value = value or ""
                if label:
                    totals.append({"label": label, "value": value})
            else:
                remaining.append(row)
        return totals, remaining
    except Exception:
        logger.exception("_extract_footer_totals failed.")
        return totals, rows


def _table_to_text_kv(kv_pairs: dict[str, str]) -> str:
    try:
        return "\n".join(f"{k}: {v}".rstrip() for k, v in (kv_pairs or {}).items()).strip()
    except Exception:
        return ""


def _table_to_text_data(headers: list[str], rows: list[dict[str, str]], totals: list[dict[str, str]]) -> str:
    try:
        lines: list[str] = []
        for r in rows or []:
            parts = []
            for h in headers or []:
                parts.append(f"{h}: {str(r.get(h, '')).strip()}")
            lines.append(" | ".join(parts).strip())
        for t in totals or []:
            label = (t.get("label") or "").strip()
            value = (t.get("value") or "").strip()
            if label:
                lines.append(f"{label}: {value}".rstrip())
        return "\n".join(lines).strip()
    except Exception:
        return ""


def normalise_tables(raw_tables: list, doc_type: str = "UNKNOWN") -> list[dict[str, Any]]:
    """
    Normalise raw pdfplumber tables into schema-free dicts.

    Returns [] on total failure.
    """
    _ = doc_type  # schema-free; reserved for future behaviour
    outputs: list[dict[str, Any]] = []
    try:
        for table in raw_tables or []:
            try:
                if not isinstance(table, list) or not table:
                    continue

                # CASE 1 — KV
                if _detect_kv_table(table):
                    kv_out = _kv_table_to_output(table)
                    kv_out["raw_text"] = _table_to_text_kv(kv_out.get("kv_pairs", {}))
                    outputs.append(kv_out)
                    continue

                cleaned_table: list[list[str]] = []
                for row_any in table:
                    if not isinstance(row_any, list):
                        continue
                    cleaned_table.append([_clean_cell(c) for c in row_any])
                if not cleaned_table:
                    continue

                # CASE 3 — header inference
                headers, header_idx = _infer_header_row(cleaned_table)
                if not headers:
                    col_count = _table_dimensions(cleaned_table)  # type: ignore[arg-type]
                    headers = [f"Column_{i}" for i in range(1, col_count + 1)]
                headers = _make_unique_headers(headers)
                col_count = len(headers)

                data_rows_raw = cleaned_table[header_idx + 1 :] if header_idx >= 0 else cleaned_table

                padded_rows: list[list[str]] = []
                for r in data_rows_raw:
                    r2 = _pad_or_truncate(list(r), col_count)
                    if any(c.strip() for c in r2):
                        padded_rows.append(r2)

                # CASE 2 — footer totals
                totals, remaining_rows = _extract_footer_totals(padded_rows)

                rows_dicts: list[dict[str, str]] = []
                for r in remaining_rows:
                    if not any((c or "").strip() for c in r):
                        continue
                    rows_dicts.append({headers[i]: (r[i] if i < len(r) else "") for i in range(col_count)})

                out: dict[str, Any] = {
                    "table_type": "data",
                    "headers": headers,
                    "rows": rows_dicts,
                    "raw_text": _table_to_text_data(headers, rows_dicts, totals),
                }
                if totals:
                    out["totals"] = totals
                outputs.append(out)
            except Exception:
                logger.exception("Failed normalising a table; skipping.")
                continue
        return outputs
    except Exception:
        logger.exception("normalise_tables failed; returning [].")
        return []
