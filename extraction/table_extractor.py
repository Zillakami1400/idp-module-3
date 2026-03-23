"""
extraction/table_extractor.py
==============================
Camelot-based PDF Table Extraction

Responsibilities:
  - Extract tables from a PDF using camelot (lattice mode first, stream fallback)
  - Clean each table: drop fully-empty rows and columns
  - Save a summary JSON:  storage/structured_data/{doc_id}_tables.json
  - Save individual CSVs: storage/structured_data/{doc_id}_table_{n}.csv
  - Return a result dict  { table_count, tables: [{page, rows, cols, accuracy, data}] }

Graceful degradation:
  - If camelot is NOT installed  → log a warning, return empty result (no crash)
  - If the PDF is scanned/image-only → log a warning, return empty result
  - Any other unexpected error    → log the error,  return empty result

Downstream consumers:
  - extraction/extractor.py  → extract_information() includes the result under "tables"
"""

import csv
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("extraction.table_extractor")

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
STRUCTURED_DATA_DIR = Path("storage/structured_data")
STRUCTURED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Optional camelot import — fail gracefully if not installed
# ---------------------------------------------------------------------------
try:
    import camelot  # type: ignore
    _CAMELOT_AVAILABLE = True
except ImportError:
    camelot = None  # type: ignore
    _CAMELOT_AVAILABLE = False
    logger.warning(
        "camelot-py is not installed. Table extraction will be skipped. "
        "Install it with:  pip install camelot-py[cv]"
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clean_dataframe(df: Any) -> list[list[str]]:
    """
    Drop rows and columns that are entirely empty strings, then return
    the remaining data as a plain list-of-lists (all strings).

    Args:
        df: A pandas DataFrame (as returned by camelot table.df).

    Returns:
        Cleaned 2-D list of strings.
    """
    # Replace NaN/None with empty string first
    df = df.fillna("")
    # Drop columns where every cell is an empty string
    df = df.loc[:, (df != "").any(axis=0)]
    # Drop rows where every cell is an empty string
    df = df[(df != "").any(axis=1)]
    # Reset index so row numbers are clean
    df = df.reset_index(drop=True)
    return df.values.tolist()


def _read_tables_with_camelot(pdf_path: str, flavor: str):
    """
    Wrapper around camelot.read_pdf — returns a TableList or raises.

    Separating this out makes the fallback logic in extract_tables() cleaner.
    """
    return camelot.read_pdf(
        pdf_path,
        flavor=flavor,
        pages="all",
        suppress_stdout=True,
    )


def _save_tables_json(doc_id: str, result: dict) -> str:
    """Persist the table summary dict to storage/structured_data/{doc_id}_tables.json."""
    output_path = STRUCTURED_DATA_DIR / f"{doc_id}_tables.json"
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=4, ensure_ascii=False)
    logger.info("Table JSON saved → %s", output_path)
    return str(output_path)


def _save_table_csv(doc_id: str, table_index: int, data: list[list[str]]) -> str:
    """
    Persist a single table's data to storage/structured_data/{doc_id}_table_{n}.csv.

    Args:
        doc_id:      Document identifier.
        table_index: 1-based table number.
        data:        2-D list of strings (rows × cols).

    Returns:
        Path to the saved CSV file.
    """
    csv_path = STRUCTURED_DATA_DIR / f"{doc_id}_table_{table_index}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerows(data)
    logger.info("Table CSV saved  → %s", csv_path)
    return str(csv_path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_tables(pdf_path: str, doc_id: str) -> dict:
    """
    Extract all tables from a PDF file using camelot-py.

    Strategy:
      1. Try camelot in *lattice* mode (best for bordered/grid tables).
      2. If 0 tables found, retry in *stream* mode (good for whitespace-delimited tables).
      3. For each table:
           a. Convert to pandas DataFrame
           b. Drop fully-empty rows / columns
           c. Save as CSV
      4. Save a summary JSON with all table metadata.
      5. Return the result dict.

    Graceful degradation:
      - camelot not installed   → warn + return empty dict
      - scanned / image-only PDF → warn + return empty dict (camelot cannot parse images)
      - any other error         → log error + return empty dict

    Args:
        pdf_path: Absolute or relative path to the PDF file.
        doc_id:   Unique document identifier used for output filenames.

    Returns:
        dict with keys:
          - table_count  (int)  : number of tables found
          - tables       (list) : list of table info dicts, each containing:
              - index    (int)  : 1-based table index
              - page     (int)  : page number where the table was found
              - rows     (int)  : number of data rows after cleaning
              - cols     (int)  : number of data columns after cleaning
              - accuracy (float): camelot's parse accuracy score (0–100)
              - csv_path (str)  : path to saved CSV file
              - data     (list) : 2-D list of string cell values
          - json_path    (str?) : path to the saved summary JSON (if tables found)
    """
    empty_result: dict = {"table_count": 0, "tables": []}

    # ------------------------------------------------------------------
    # Guard: camelot not installed
    # ------------------------------------------------------------------
    if not _CAMELOT_AVAILABLE:
        logger.warning(
            "Skipping table extraction for doc_id=%s — camelot-py is not installed.",
            doc_id,
        )
        return empty_result

    logger.info("Starting table extraction for doc_id=%s | path=%s", doc_id, pdf_path)

    # ------------------------------------------------------------------
    # Step 1: Try lattice mode, fall back to stream
    # ------------------------------------------------------------------
    table_list = None
    flavor_used = None

    for flavor in ("lattice", "stream"):
        try:
            logger.info("  Trying camelot flavor='%s' …", flavor)
            result = _read_tables_with_camelot(pdf_path, flavor)
            if len(result) > 0:
                table_list = result
                flavor_used = flavor
                logger.info(
                    "  Found %d table(s) with flavor='%s'.", len(result), flavor
                )
                break
            else:
                logger.info("  No tables found with flavor='%s', trying next.", flavor)
        except Exception as exc:
            # Covers NotImplementedError for scanned PDFs, PDFSyntaxError, etc.
            exc_type = type(exc).__name__
            logger.warning(
                "  camelot (%s) raised %s: %s — trying next flavor or aborting.",
                flavor, exc_type, exc,
            )
            # If it looks like a scanned/image-only PDF, stop immediately
            if "pdf" in str(exc).lower() and "text" in str(exc).lower():
                logger.warning(
                    "PDF '%s' appears to be scanned/image-only — "
                    "camelot cannot extract tables from it. Returning empty.",
                    pdf_path,
                )
                return empty_result

    # No tables at all (both flavors returned 0 or errored out silently)
    if not table_list:
        logger.info(
            "No tables extracted from '%s' (doc_id=%s) — "
            "PDF may be scanned or contain no structured tables.",
            pdf_path, doc_id,
        )
        return empty_result

    # ------------------------------------------------------------------
    # Step 2: Process each table
    # ------------------------------------------------------------------
    tables_meta = []

    for idx, cam_table in enumerate(table_list, start=1):
        try:
            # camelot table attributes
            page_num = int(cam_table.page)
            accuracy = round(float(cam_table.accuracy), 2)

            # Clean the DataFrame
            cleaned_data = _clean_dataframe(cam_table.df)

            rows = len(cleaned_data)
            cols = len(cleaned_data[0]) if cleaned_data else 0

            if rows == 0 or cols == 0:
                logger.info(
                    "  Table %d on page %d is empty after cleaning — skipping.",
                    idx, page_num,
                )
                continue

            # Save CSV
            csv_path = _save_table_csv(doc_id, idx, cleaned_data)

            table_info = {
                "index":    idx,
                "page":     page_num,
                "rows":     rows,
                "cols":     cols,
                "accuracy": accuracy,
                "csv_path": csv_path,
                "data":     cleaned_data,
            }
            tables_meta.append(table_info)

            logger.info(
                "  Table %d: page=%d  rows=%d  cols=%d  accuracy=%.1f%%",
                idx, page_num, rows, cols, accuracy,
            )

        except Exception as exc:
            logger.error(
                "  Error processing table %d (doc_id=%s): %s", idx, doc_id, exc
            )
            # Continue with remaining tables

    # ------------------------------------------------------------------
    # Step 3: Save summary JSON
    # ------------------------------------------------------------------
    final_result: dict = {
        "table_count": len(tables_meta),
        "flavor_used": flavor_used,
        "tables": tables_meta,
    }

    if tables_meta:
        json_path = _save_tables_json(doc_id, final_result)
        final_result["json_path"] = json_path
    else:
        logger.info("No non-empty tables found — skipping JSON save.")

    logger.info(
        "Table extraction complete for doc_id=%s — %d table(s) extracted.",
        doc_id, len(tables_meta),
    )
    return final_result
