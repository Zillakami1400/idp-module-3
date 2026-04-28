"""
Unified intelligence pipeline orchestrator.

Runs, in order:
  1) classification
  2) table normalisation
  3) entity extraction
  3b) typed-field enrichment from table content
  4) chunk construction
  5) summary construction

The goal is robust, schema-free document understanding for semantic search and RAG.
All functions are exception-safe and return partial results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
import logging
import re

from intelligence.classifier import classify, DocumentProfile
from extraction.table_extractor import normalise_tables, fuzzy_match_column
from extraction.extractor import extract_entities, entities_to_text

logger = logging.getLogger("intelligence.pipeline")


@dataclass
class DocumentChunk:
    """
    A retrieval-ready chunk of document information.

    Attributes:
        chunk_id: Unique ID using format "{doc_id}_chunk_{n:04d}".
        doc_id: Document identifier.
        text: Chunk text used for embeddings/retrieval.
        chunk_type: Type of chunk (document_header, page_text, table, entity_summary).
        page_number: Source page number when available.
        metadata: Additional chunk metadata.
    """

    chunk_id: str
    doc_id: str
    text: str
    chunk_type: str
    page_number: Optional[int]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentIntelligence:
    """
    Full intelligence output for one document.

    Attributes:
        doc_id: Document identifier.
        file_path: Source file path.
        doc_type: Inferred or fallback document type.
        confidence: Doc type confidence score.
        profile: Full classification profile.
        entities: Extracted entities dict.
        normalised_tables: Normalised table structures.
        chunks: Final retrieval chunks.
        full_text: Raw full text provided to pipeline.
        summary_text: Compact summary for fast retrieval.
    """

    doc_id: str
    file_path: str
    doc_type: str
    confidence: float
    profile: DocumentProfile
    entities: dict[str, Any]
    normalised_tables: list[dict[str, Any]]
    chunks: list[DocumentChunk]
    full_text: str
    summary_text: str


def _default_profile() -> DocumentProfile:
    """Return a safe fallback profile."""
    try:
        return DocumentProfile(
            doc_type="UNKNOWN",
            confidence=0.0,
            has_tables=False,
            layout_hints={},
            key_value_pairs={},
            detected_dates=[],
            detected_amounts=[],
        )
    except Exception:
        return DocumentProfile("UNKNOWN", 0.0, False, {}, {}, [], [])


def _new_chunk(doc_id: str, index: int, text: str, chunk_type: str, page_number: Optional[int], metadata: dict[str, Any]) -> DocumentChunk:
    """Create one DocumentChunk with canonical chunk_id format."""
    try:
        return DocumentChunk(
            chunk_id=f"{doc_id}_chunk_{index:04d}",
            doc_id=doc_id,
            text=text,
            chunk_type=chunk_type,
            page_number=page_number,
            metadata=metadata or {},
        )
    except Exception:
        return DocumentChunk(
            chunk_id=f"{doc_id}_chunk_{index:04d}",
            doc_id=doc_id,
            text=text or "",
            chunk_type=chunk_type or "unknown",
            page_number=page_number,
            metadata={},
        )


def _extract_pages_text(pages: Any, full_text: str) -> list[tuple[int, str]]:
    """
    Build a list of `(page_number, page_text)` pairs.

    If `pages` is unavailable or invalid, returns one pseudo-page from full_text.
    """
    pairs: list[tuple[int, str]] = []
    try:
        if isinstance(pages, list) and pages:
            for idx, p in enumerate(pages, start=1):
                try:
                    page_num = idx
                    page_text = ""
                    if hasattr(p, "page_number"):
                        page_num = int(getattr(p, "page_number"))
                    if hasattr(p, "text"):
                        page_text = str(getattr(p, "text") or "")
                    elif isinstance(p, dict):
                        page_num = int(p.get("page_number", idx))
                        page_text = str(p.get("text", "") or "")
                    if page_text.strip():
                        pairs.append((page_num, page_text))
                except Exception:
                    continue
        if not pairs and (full_text or "").strip():
            pairs.append((1, full_text))
        return pairs
    except Exception:
        logger.exception("_extract_pages_text failed; returning fallback single page.")
        if (full_text or "").strip():
            return [(1, full_text)]
        return []


def _find_breakpoint(window: str, min_pos: int) -> int:
    """
    Find a boundary in second half of window.

    Priority:
      1) last ". "
      2) last ".\n"
      3) last "\n\n"
      4) end of window
    """
    try:
        candidates = [window.rfind(". "), window.rfind(".\n"), window.rfind("\n\n")]
        candidates = [c for c in candidates if c >= min_pos]
        if not candidates:
            return len(window)
        best = max(candidates)
        # Include boundary length where applicable
        if window[best:best + 2] in {". ", ".\n", "\n\n"}:
            return min(len(window), best + 2)
        return best
    except Exception:
        return len(window)


def _split_page_text(text: str) -> list[str]:
    """
    Split page text into chunks.

    Rules:
      - If text length <= 800: one chunk.
      - If > 800: sliding window size=400 overlap=80, with sentence/paragraph boundary break.
    """
    try:
        clean = (text or "").strip()
        if not clean:
            return []
        if len(clean) <= 800:
            return [clean]

        out: list[str] = []
        size = 400
        overlap = 80
        start = 0
        n = len(clean)

        while start < n:
            end = min(n, start + size)
            window = clean[start:end]
            min_pos = len(window) // 2
            cut = _find_breakpoint(window, min_pos=min_pos)
            piece = window[:cut].strip()
            if piece:
                out.append(piece)
            if end >= n:
                break
            advance = max(cut - overlap, 1)
            start = start + advance

        return out
    except Exception:
        logger.exception("_split_page_text failed; returning unsplit text.")
        return [text.strip()] if (text or "").strip() else []


def _first_non_empty(value: Any) -> str:
    """Return cleaned non-empty string or empty string."""
    try:
        return str(value or "").strip()
    except Exception:
        return ""


def _build_document_header_text(
    doc_id: str,
    doc_type: str,
    entities: dict[str, Any],
    normalised_tables: list[dict[str, Any]],
) -> str:
    """
    Build the single document_header chunk text.

    Includes fields only when available; no empty lines are emitted.
    """
    lines: list[str] = []
    try:
        lines.append(f"Document: {doc_id}")
        lines.append(f"Type: {doc_type}")

        typed = entities.get("typed_fields") or {}
        kv = entities.get("key_value_pairs") or {}

        def pick(*keys: str) -> str:
            for k in keys:
                v = _first_non_empty(typed.get(k))
                if v:
                    return v
            for k in keys:
                # key_value_pairs may use title-case keys
                v = _first_non_empty(kv.get(k) or kv.get(k.replace("_", " ").title()))
                if v:
                    return v
            return ""

        order_id = pick("order_id")
        customer_name = pick("customer_name")
        customer_id = pick("customer_id")
        country = pick("country")
        city = pick("city")
        date = pick("date")
        total_amount = pick("total_amount")

        if order_id:
            lines.append(f"Order ID: {order_id}")
        if customer_name and customer_id:
            lines.append(f"Customer: {customer_name} ({customer_id})")
        elif customer_name:
            lines.append(f"Customer: {customer_name}")
        elif customer_id:
            lines.append(f"Customer: ({customer_id})")
        if country:
            lines.append(f"Country: {country}")
        if city:
            lines.append(f"City: {city}")
        if date:
            lines.append(f"Date: {date}")
        if total_amount:
            lines.append(f"Total: {total_amount}")

        # Product summary from data tables via fuzzy matching
        products: list[str] = []
        for tbl in normalised_tables or []:
            try:
                if (tbl.get("table_type") or "") != "data":
                    continue
                headers = [str(h) for h in (tbl.get("headers") or [])]
                if not headers:
                    continue
                name_col = (
                    fuzzy_match_column("product name", headers)
                    or fuzzy_match_column("product", headers)
                    or fuzzy_match_column("description", headers)
                )
                qty_col = (
                    fuzzy_match_column("quantity", headers)
                    or fuzzy_match_column("qty", headers)
                )
                if not name_col:
                    continue
                for row in (tbl.get("rows") or []):
                    if not isinstance(row, dict):
                        continue
                    pname = _first_non_empty(row.get(name_col))
                    if not pname:
                        continue
                    pqty = _first_non_empty(row.get(qty_col)) if qty_col else ""
                    if pqty:
                        products.append(f"{pname} x{pqty}")
                    else:
                        products.append(pname)
                    if len(products) >= 10:
                        break
                if len(products) >= 10:
                    break
            except Exception:
                continue
                
        if not products:
            schema = entities.get("normalized_schema") or {}
            for item in schema.get("line_items") or []:
                name = item.get("product")
                if not name:
                    continue
                qty = item.get("quantity")
                if qty:
                    products.append(f"{name} x{qty}")
                else:
                    products.append(name)
            
        if products:
            lines.append("Products: " + ", ".join(products[:10]))

        return "\n".join(lines).strip()
    except Exception:
        logger.exception("_build_document_header_text failed; returning minimal header.")
        return f"Document: {doc_id}\nType: {doc_type}".strip()


def _enrich_typed_fields(entities: dict[str, Any], normalised_tables: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Fill missing typed_fields from kv pairs and table rows.

    Target fields:
      order_id, customer_name, date, customer_id, total_amount, country, city, contact_name.
    Also pulls total_amount from table footer rows when still missing.
    """
    try:
        out = dict(entities or {})
        typed = dict(out.get("typed_fields") or {})
        kv = dict(out.get("key_value_pairs") or {})

        targets = [
            "order_id",
            "customer_name",
            "date",
            "customer_id",
            "total_amount",
            "country",
            "city",
            "contact_name",
        ]
        aliases: dict[str, list[str]] = {
            "order_id": ["order id", "order no", "order number", "po number", "po #"],
            "customer_name": ["customer name", "billed to", "bill to", "buyer", "ship to"],
            "date": ["date", "order date", "invoice date", "transaction date"],
            "customer_id": ["customer id", "customer no", "client id"],
            "total_amount": ["total", "total price", "totalprice", "amount due", "grand total", "net total", "subtotal", "balance due"],
            "country": ["country"],
            "city": ["city"],
            "contact_name": ["contact name"],
        }

        # 1) Fill from KV pairs
        for field_name in targets:
            try:
                if _first_non_empty(typed.get(field_name)):
                    continue
                for a in aliases.get(field_name, []):
                    title_key = a.strip().title()
                    val = _first_non_empty(kv.get(title_key))
                    if val:
                        typed[field_name] = val
                        break
            except Exception:
                continue

        # 2) Fill from data table rows using fuzzy header matching
        for tbl in normalised_tables or []:
            try:
                if (tbl.get("table_type") or "") != "data":
                    continue
                headers = [str(h) for h in (tbl.get("headers") or [])]
                if not headers:
                    continue
                for field_name in targets:
                    if _first_non_empty(typed.get(field_name)):
                        continue
                    best_col = None
                    for a in aliases.get(field_name, []):
                        best_col = fuzzy_match_column(a, headers)
                        if best_col:
                            break
                    if not best_col:
                        continue
                    for row in (tbl.get("rows") or []):
                        if not isinstance(row, dict):
                            continue
                        val = _first_non_empty(row.get(best_col))
                        if val:
                            typed[field_name] = val
                            break
            except Exception:
                continue

        # 3) Pull total_amount from table footers if still missing
        if not _first_non_empty(typed.get("total_amount")):
            for tbl in normalised_tables or []:
                try:
                    totals = tbl.get("totals") or []
                    for t in totals:
                        if isinstance(t, dict):
                            val = _first_non_empty(t.get("value"))
                            if val:
                                typed["total_amount"] = val
                                break
                    if _first_non_empty(typed.get("total_amount")):
                        break
                except Exception:
                    continue

        out["typed_fields"] = typed
        return out
    except Exception:
        logger.exception("_enrich_typed_fields failed; returning original entities.")
        return entities or {}


def _build_chunks(
    doc_id: str,
    doc_type: str,
    full_text: str,
    pages: Any,
    entities: dict[str, Any],
    normalised_tables: list[dict[str, Any]],
) -> list[DocumentChunk]:
    """
    Build chunks in required order:
      1) document_header (one)
      2) page text chunks
      3) table chunks
      4) entity_summary (one)
    """
    chunks: list[DocumentChunk] = []
    try:
        idx = 1

        # 1) document_header
        header_text = _build_document_header_text(doc_id, doc_type, entities, normalised_tables)
        chunks.append(
            _new_chunk(
                doc_id=doc_id,
                index=idx,
                text=header_text,
                chunk_type="document_header",
                page_number=None,
                metadata={"doc_type": doc_type},
            )
        )
        idx += 1

        # 2) text chunks per page
        page_pairs = _extract_pages_text(pages, full_text)
        for page_number, page_text in page_pairs:
            try:
                segments = _split_page_text(page_text)
                for seg in segments:
                    seg_clean = (seg or "").strip()
                    if not seg_clean:
                        continue
                    txt = f"[{doc_id} / {doc_type}]\n{seg_clean}"
                    chunks.append(
                        _new_chunk(
                            doc_id=doc_id,
                            index=idx,
                            text=txt,
                            chunk_type="page_text",
                            page_number=page_number,
                            metadata={"doc_type": doc_type},
                        )
                    )
                    idx += 1
            except Exception:
                continue

        # 3) table chunks
        for tbl in normalised_tables or []:
            try:
                raw_text = str(tbl.get("raw_text") or "").strip()
                if not raw_text:
                    continue
                txt = f"[{doc_id} / {doc_type} / table]\n{raw_text}"
                chunks.append(
                    _new_chunk(
                        doc_id=doc_id,
                        index=idx,
                        text=txt,
                        chunk_type="table",
                        page_number=None,
                        metadata={
                            "headers": tbl.get("headers", []),
                            "table_type": tbl.get("table_type"),
                        },
                    )
                )
                idx += 1
            except Exception:
                continue

        # 4) entity_summary chunk
        summary_body = entities_to_text(entities or {})
        summary_txt = f"[{doc_id} / {doc_type}]\n{summary_body}".strip()
        chunks.append(
            _new_chunk(
                doc_id=doc_id,
                index=idx,
                text=summary_txt,
                chunk_type="entity_summary",
                page_number=None,
                metadata={"doc_type": doc_type},
            )
        )

        return chunks
    except Exception:
        logger.exception("_build_chunks failed; returning partial chunks.")
        return chunks


def _build_summary(
    doc_id: str,
    doc_type: str,
    entities: dict[str, Any],
    normalised_tables: list[dict[str, Any]],
) -> str:
    """
    Build summary text for fast retrieval.

    Format:
      Document: {doc_id}
      Document Type: {doc_type}
      {typed_field}: {value}
      Dates: ...
      Total: ...
      Table ({n} rows): ...
      {top 8 kv pairs}
    """
    lines: list[str] = []
    try:
        lines.append(f"Document: {doc_id}")
        lines.append(f"Document Type: {doc_type}")

        typed = entities.get("typed_fields") or {}
        if isinstance(typed, dict):
            for k, v in typed.items():
                ks = str(k).strip()
                vs = str(v).strip()
                if ks and vs:
                    lines.append(f"{ks}: {vs}")

        dates = entities.get("dates") or []
        if isinstance(dates, list) and dates:
            date_vals = [str(d).strip() for d in dates if str(d).strip()]
            if date_vals:
                lines.append("Dates: " + ", ".join(date_vals))

        totals = entities.get("totals") or []
        if isinstance(totals, list):
            first_total = next((str(t).strip() for t in totals if str(t).strip()), "")
            if first_total:
                lines.append(f"Total: {first_total}")

        for tbl in normalised_tables or []:
            try:
                headers = [str(h).strip() for h in (tbl.get("headers") or []) if str(h).strip()]
                rows_count = len(tbl.get("rows") or [])
                if headers:
                    lines.append(f"Table ({rows_count} rows): {', '.join(headers)}")
            except Exception:
                continue
                
        schema = entities.get("normalized_schema") or {}
        line_items = schema.get("line_items") or []
        if line_items:
            lines.append(f"Line Items ({len(line_items)}):")
            for item in line_items[:10]:
                name = item.get("product") or "Item"
                qs = f" x{item['quantity']}" if item.get("quantity") else ""
                lines.append(f"  - {name}{qs} (Total: {item.get('line_total', 'N/A')})")

        kv = entities.get("key_value_pairs") or {}
        if isinstance(kv, dict) and kv:
            added = 0
            for k, v in kv.items():
                ks = str(k).strip()
                vs = str(v).strip()
                if not ks or not vs:
                    continue
                lines.append(f"{ks}: {vs}")
                added += 1
                if added >= 8:
                    break

        return "\n".join(lines).strip()
    except Exception:
        logger.exception("_build_summary failed.")
        return f"Document: {doc_id}\nDocument Type: {doc_type}".strip()


def run_pipeline(
    doc_id: str,
    file_path: str,
    full_text: str,
    raw_tables: Any,
    pages: Any = None,
) -> DocumentIntelligence:
    """
    Unified orchestration entry point.

    Steps:
      1. classify(full_text, raw_tables)
      2. normalise_tables(raw_tables, doc_type)
      3. extract_entities(full_text, doc_type, normalised_tables)
      3b. _enrich_typed_fields(entities, normalised_tables)
      4. _build_chunks(...)
      5. _build_summary(...)

    Never raises. Each main step has its own try/except.
    """
    profile = _default_profile()
    doc_type = "UNKNOWN"
    confidence = 0.0
    entities: dict[str, Any] = {
        "doc_type": doc_type,
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
    tables: list[dict[str, Any]] = []
    chunks: list[DocumentChunk] = []
    summary_text = ""
    safe_full_text = ""

    try:
        safe_full_text = str(full_text or "")
    except Exception:
        safe_full_text = ""

    # STEP 1: classify
    try:
        profile = classify(safe_full_text, raw_tables)
        doc_type = str(profile.doc_type or "UNKNOWN")
        confidence = float(profile.confidence or 0.0)
    except Exception:
        logger.exception("Step 1 classify failed; continuing with UNKNOWN profile.")

    # STEP 2: normalise tables
    try:
        tables = normalise_tables(raw_tables or [], doc_type=doc_type)
    except Exception:
        logger.exception("Step 2 table normalisation failed; continuing with empty tables.")
        tables = []

    # STEP 3: extract entities
    try:
        entities = extract_entities(safe_full_text, doc_type=doc_type, normalised_tables=tables)
        entities["doc_type"] = doc_type
    except Exception:
        logger.exception("Step 3 entity extraction failed; continuing with empty entities.")

    # STEP 3b: enrich typed fields
    try:
        entities = _enrich_typed_fields(entities, tables)
        entities["doc_type"] = doc_type
    except Exception:
        logger.exception("Step 3b typed enrichment failed; continuing.")

    # STEP 3c: mode detection, block-based parsing, and normalization
    try:
        import re as _re

        # ── helpers ──────────────────────────────────────────────────────────
        def _float(s: str) -> Optional[float]:
            try:
                return float(_re.sub(r"[^\d\.]", "", s or ""))
            except (ValueError, TypeError):
                return None

        def _re_pick(pattern: str, text: str) -> Optional[str]:
            m = _re.search(pattern, text or "", _re.IGNORECASE)
            return m.group(1).strip() if m else None

        # ── block-based line_item parser ──────────────────────────────────────
        _FIELD_SPLIT = _re.compile(
            r"(?i)(Product:|Quantity:|Unit\s*Price:|Total:)"
        )
        _HAS_PRODUCT = _re.compile(r"(?i)\bProduct\s*:")

        def _parse_line_items(text: str) -> list:
            items: list = []
            current: dict = {}
            for raw_line in text.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                parts = _FIELD_SPLIT.split(line)
                if len(parts) == 1:
                    # continuation of product name on its own line
                    if current and "product" in current and "quantity" not in current:
                        current["product"] = (current["product"] + " " + line).strip()
                    continue
                i = 1
                while i < len(parts) - 1:
                    kw = parts[i].strip().lower().rstrip(":")
                    kw = _re.sub(r"\s+", "_", kw)   # "unit price" → "unit_price"
                    val = (parts[i + 1] if i + 1 < len(parts) else "").strip()
                    if kw == "product":
                        if current and "product" in current:
                            items.append(current)
                            current = {}
                        current["product"] = val
                    elif kw == "quantity":
                        v = _float(val)
                        if v is not None:
                            current["quantity"] = v
                    elif kw == "unit_price":
                        v = _float(val)
                        if v is not None:
                            current["unit_price"] = v
                    elif kw == "total":
                        v = _float(val)
                        if v is not None:
                            current["line_total"] = v
                    i += 2
            if current and "product" in current:
                items.append(current)
            return items

        # ── mode detection ────────────────────────────────────────────────────
        has_product_blocks = bool(_HAS_PRODUCT.search(safe_full_text))
        if tables:
            document_mode = "TABLE_MODE"
        elif has_product_blocks:
            document_mode = "FORM_MODE"
        else:
            document_mode = "UNKNOWN"

        entities["document_mode"] = document_mode

        # ── line items ────────────────────────────────────────────────────────
        line_items: list = []
        if document_mode == "FORM_MODE":
            line_items = _parse_line_items(safe_full_text)

        # ── key field extraction ──────────────────────────────────────────────
        typed = dict(entities.get("typed_fields") or {})
        kv = entities.get("key_value_pairs") or {}

        def _pick(*keys: str) -> Optional[str]:
            for k in keys:
                v = (typed.get(k) or "").strip()
                if v:
                    return v
            for k in keys:
                v = (kv.get(k.replace("_", " ").title()) or "").strip()
                if v:
                    return v
            return None

        # derive nested sub-fields from compound KV blobs
        cust_details = kv.get("Customer Details") or ""
        order_details = kv.get("Order Details") or ""

        customer_id = (
            _pick("customer_id")
            or _re_pick(r"Customer\s+ID\s*[:\-]\s*(\S+)", cust_details)
            or _re_pick(r"Customer\s+ID\s*[:\-]\s*(\S+)", safe_full_text)
        )
        customer_name = (
            _pick("customer_name")
            or _re_pick(r"Customer\s+Name\s*[:\-]\s*([^\n]+)", cust_details)
            or _re_pick(r"Customer\s+Name\s*[:\-]\s*([^\n]+)", safe_full_text)
        )
        order_date = (
            _pick("date")
            or _re_pick(r"Order\s+Date\s*[:\-]\s*(\S+)", order_details)
            or _re_pick(r"Order\s+Date\s*[:\-]\s*(\S+)", safe_full_text)
        )
        order_id = _pick("order_id")

        # ── total validation ──────────────────────────────────────────────────
        raw_total_str = _re.sub(r"[^\d\.]", "", str(_pick("total_amount") or "0"))
        try:
            ext_total = float(raw_total_str) if raw_total_str else 0.0
        except ValueError:
            ext_total = 0.0

        if line_items:
            calc_total = sum(item.get("line_total", 0.0) for item in line_items)
            final_total = calc_total if abs(calc_total - ext_total) > 0.01 else ext_total
        else:
            final_total = ext_total

        # ── confidence scoring ────────────────────────────────────────────────
        conf = 0.0
        if line_items:
            conf += 0.4
            if all("line_total" in it for it in line_items):
                conf += 0.2
            if line_items and abs(sum(it.get("line_total", 0) for it in line_items) - ext_total) <= 0.01:
                conf += 0.2
        if order_id:
            conf += 0.1
        if customer_name:
            conf += 0.1
        confidence = min(max(conf, 0.0), 1.0)

        entities["normalized_schema"] = {
            "order_id": order_id,
            "customer_name": customer_name,
            "customer_id": customer_id,
            "date": order_date,
            "line_items": line_items,
            "total_amount": final_total,
        }
    except Exception:
        logger.exception("Step 3c normalization failed; continuing.")

    # STEP 4: build chunks
    try:
        chunks = _build_chunks(
            doc_id=doc_id,
            doc_type=doc_type,
            full_text=safe_full_text,
            pages=pages,
            entities=entities,
            normalised_tables=tables,
        )
    except Exception:
        logger.exception("Step 4 chunking failed; continuing with empty chunks.")
        chunks = []

    # STEP 5: build summary
    try:
        summary_text = _build_summary(doc_id, doc_type, entities, tables)
    except Exception:
        logger.exception("Step 5 summary build failed; using minimal summary.")
        summary_text = f"Document: {doc_id}\nDocument Type: {doc_type}"

    try:
        return DocumentIntelligence(
            doc_id=doc_id,
            file_path=file_path,
            doc_type=doc_type,
            confidence=max(0.0, min(confidence, 1.0)),
            profile=profile,
            entities=entities,
            normalised_tables=tables,
            chunks=chunks,
            full_text=safe_full_text,
            summary_text=summary_text,
        )
    except Exception:
        logger.exception("run_pipeline failed constructing output object; returning fallback.")
        return DocumentIntelligence(
            doc_id=doc_id,
            file_path=file_path,
            doc_type="UNKNOWN",
            confidence=0.0,
            profile=_default_profile(),
            entities={
                "doc_type": "UNKNOWN",
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
            },
            normalised_tables=[],
            chunks=[],
            full_text=safe_full_text,
            summary_text=f"Document: {doc_id}\nDocument Type: UNKNOWN",
        )
