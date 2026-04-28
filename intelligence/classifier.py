"""
Zero-shot, rules-based document classifier.

This module intentionally avoids any trained models or hardcoded schemas beyond
lightweight keyword scoring and generic layout signals. It is designed to work
on arbitrary documents, including types never seen before.

Key guarantees:
  - Runs locally (no network/cloud calls).
  - Never raises to caller: every function catches its own exceptions and
    returns partial results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import logging
import re

logger = logging.getLogger("intelligence.classifier")


DocType = str


@dataclass
class DocumentProfile:
    """
    High-level profile of a document inferred from plain text (and optional tables).

    Attributes:
        doc_type: Inferred document type label (e.g. "INVOICE", "PURCHASE_ORDER", "UNKNOWN").
        confidence: Confidence score in [0.0, 1.0].
        has_tables: Whether the input included any non-empty tables.
        layout_hints: Generic, type-agnostic signals about document layout.
        key_value_pairs: Extracted "Key: Value" pairs from text.
        detected_dates: Up to 10 detected date strings.
        detected_amounts: Up to 10 detected currency amounts.
    """

    doc_type: DocType
    confidence: float
    has_tables: bool
    layout_hints: dict[str, Any]
    key_value_pairs: dict[str, str]
    detected_dates: list[str]
    detected_amounts: list[str]

    def is_known(self) -> bool:
        """Return True if doc_type is not UNKNOWN and confidence is at least 0.4."""
        try:
            return self.doc_type != "UNKNOWN" and float(self.confidence) >= 0.4
        except Exception:
            return False

    def to_dict(self) -> dict[str, Any]:
        """Convert this profile into a JSON-serializable dict."""
        try:
            return {
                "doc_type": self.doc_type,
                "confidence": float(self.confidence),
                "has_tables": bool(self.has_tables),
                "layout_hints": dict(self.layout_hints or {}),
                "key_value_pairs": dict(self.key_value_pairs or {}),
                "detected_dates": list(self.detected_dates or []),
                "detected_amounts": list(self.detected_amounts or []),
            }
        except Exception:
            logger.exception("DocumentProfile.to_dict failed; returning minimal dict.")
            return {
                "doc_type": self.doc_type,
                "confidence": self.confidence,
                "has_tables": self.has_tables,
            }


_TIER1_RULES: dict[DocType, dict[str, list[str]]] = {
    "INVOICE": {
        "required": ["invoice"],
        "optional": [
            "customer id",
            "order id",
            "total",
            "unit price",
            "totalprice",
            "product name",
            "subtotal",
            "due date",
            "amount due",
            "bill to",
        ],
    },
    "PURCHASE_ORDER": {
        "required": ["purchase order"],
        "optional": [
            "order id",
            "order date",
            "unit price",
            "quantity",
            "product",
            "vendor",
            "ship to",
            "po number",
            "delivery",
        ],
    },
    "RECEIPT": {
        "required": ["receipt"],
        "optional": [
            "total",
            "tax",
            "cash",
            "transaction",
            "cashier",
            "change",
            "thank you",
            "store",
        ],
    },
    "CONTRACT": {
        "required": ["agreement", "contract"],
        "optional": [
            "party",
            "whereas",
            "obligations",
            "termination",
            "governing law",
            "effective date",
            "signed",
            "clause",
            "hereinafter",
        ],
    },
    "RESUME": {
        "required": ["resume", "curriculum vitae"],
        "optional": [
            "experience",
            "education",
            "skills",
            "employment",
            "objective",
            "references",
            "linkedin",
        ],
    },
    "EMAIL": {
        "required": ["from:", "to:", "subject:"],
        "optional": ["cc:", "bcc:", "sent:", "dear", "regards", "sincerely", "forwarded"],
    },
    "BANK_STATEMENT": {
        "required": ["statement", "account number"],
        "optional": [
            "balance",
            "debit",
            "credit",
            "opening balance",
            "closing balance",
            "iban",
            "routing",
            "sort code",
        ],
    },
    "REPORT": {
        "required": ["report", "summary", "analysis", "findings"],
        "optional": [
            "executive summary",
            "conclusion",
            "methodology",
            "recommendation",
            "table of contents",
            "appendix",
        ],
    },
    "FORM": {
        "required": ["form", "please fill", "signature"],
        "optional": ["date of birth", "full name", "applicant", "submit", "checkbox", "tick"],
    },
    "MEDICAL_RECORD": {
        "required": ["patient", "diagnosis"],
        "optional": [
            "prescription",
            "medication",
            "doctor",
            "hospital",
            "symptoms",
            "treatment",
            "blood pressure",
            "icd",
        ],
    },
}


_CURRENCY_RE = re.compile(r"[$€£₹¥₩]")
_DATE_ISO_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_DATE_DMY_RE = re.compile(r"\b\d{2}/\d{2}/\d{4}\b")
_NUMBERED_LIST_RE = re.compile(r"^\s*\d+[\.\)]\s+\w", re.MULTILINE)
_BULLET_LIST_RE = re.compile(r"^\s*[•\-\*]\s+\w", re.MULTILINE)
_ALL_CAPS_HEADERS_RE = re.compile(r"^[A-Z][A-Z\s]{5,}$", re.MULTILINE)
_AMOUNT_RE = re.compile(r"(?P<cur>[$€£₹¥₩])\s*(?P<num>\d[\d,]*(?:\.\d{1,2})?)")
_KV_RE = re.compile(
    r"(?P<key>[A-Z][A-Za-z0-9 /\-_]{1,50}?)\s*[:\-–]\s*(?P<value>[^\n]{1,200})"
)


def _safe_lower(text: str) -> str:
    """Lowercase text safely."""
    try:
        return (text or "").lower()
    except Exception:
        return ""


def _compute_layout_hints(text: str) -> dict[str, Any]:
    """
    Compute generic layout signals from raw text.

    Always catches exceptions and returns partial hints.
    """
    hints: dict[str, Any] = {}
    try:
        raw = text or ""
        lines = raw.splitlines()
        non_empty_lines = [ln for ln in lines if ln.strip()]
        avg_line_length = (
            sum(len(ln.strip()) for ln in non_empty_lines) / max(len(non_empty_lines), 1)
        )

        has_currency = bool(_CURRENCY_RE.search(raw))
        has_dates = bool(_DATE_ISO_RE.search(raw) or _DATE_DMY_RE.search(raw))

        hints = {
            "total_lines": len(lines),
            "non_empty_lines": len(non_empty_lines),
            "avg_line_length": float(avg_line_length),
            "has_currency": has_currency,
            "has_dates": has_dates,
            "has_numbered_list": bool(_NUMBERED_LIST_RE.search(raw)),
            "has_bullet_list": bool(_BULLET_LIST_RE.search(raw)),
            "has_all_caps_headers": bool(_ALL_CAPS_HEADERS_RE.search(raw)),
            "colon_ratio": float(raw.count(":") / max(len(non_empty_lines), 1)),
        }
        return hints
    except Exception:
        logger.exception("Layout hint computation failed.")
        return hints


def _has_tables(raw_tables: Any) -> bool:
    """
    Decide whether `raw_tables` contains any non-empty table content.

    This is intentionally permissive with input shape because upstream may pass:
      - None
      - list of tables
      - list of rows
      - any nested list-like structure
    """
    try:
        if not raw_tables:
            return False
        if isinstance(raw_tables, (list, tuple)):
            if len(raw_tables) == 0:
                return False
            # Look for any non-empty string/number somewhere inside.
            stack = list(raw_tables)
            while stack:
                item = stack.pop()
                if item is None:
                    continue
                if isinstance(item, (list, tuple)):
                    stack.extend(list(item))
                    continue
                s = str(item).strip()
                if s:
                    return True
            return False
        return True
    except Exception:
        logger.exception("Failed determining has_tables; defaulting to False.")
        return False


def _extract_dates(text: str, limit: int = 10) -> list[str]:
    """Extract up to `limit` dates from text (ISO + DMY)."""
    try:
        raw = text or ""
        matches: list[str] = []
        for m in _DATE_ISO_RE.finditer(raw):
            matches.append(m.group(0))
            if len(matches) >= limit:
                return matches
        for m in _DATE_DMY_RE.finditer(raw):
            matches.append(m.group(0))
            if len(matches) >= limit:
                return matches
        # De-dupe while preserving order
        seen: set[str] = set()
        uniq: list[str] = []
        for d in matches:
            if d not in seen:
                seen.add(d)
                uniq.append(d)
        return uniq[:limit]
    except Exception:
        logger.exception("Date extraction failed; returning partial.")
        return []


def _extract_amounts(text: str, limit: int = 10) -> list[str]:
    """Extract up to `limit` currency amounts from text."""
    try:
        raw = text or ""
        amounts: list[str] = []
        for m in _AMOUNT_RE.finditer(raw):
            cur = (m.group("cur") or "").strip()
            num = (m.group("num") or "").strip()
            if cur and num:
                amounts.append(f"{cur}{num}")
            if len(amounts) >= limit:
                break
        # De-dupe while preserving order
        seen: set[str] = set()
        uniq: list[str] = []
        for a in amounts:
            if a not in seen:
                seen.add(a)
                uniq.append(a)
        return uniq[:limit]
    except Exception:
        logger.exception("Amount extraction failed; returning partial.")
        return []


def _extract_key_value_pairs(text: str, limit: int = 200) -> dict[str, str]:
    """
    Extract key/value pairs of the form 'Key: Value' / 'Key - Value' / 'Key – Value'.

    Keys are normalized with `.strip().title()`.
    """
    kv: dict[str, str] = {}
    try:
        raw = text or ""
        for m in _KV_RE.finditer(raw):
            key = (m.group("key") or "").strip().title()
            value = (m.group("value") or "").strip()
            if not key or not value:
                continue
            if key in kv:
                idx = 2
                while f"{key} {idx}" in kv:
                    idx += 1
                key = f"{key} {idx}"
            kv[key] = value
            if len(kv) >= limit:
                break
        return kv
    except Exception:
        logger.exception("Key/value extraction failed; returning partial.")
        return kv


def _tier1_score(text_lower: str) -> tuple[DocType, float, dict[DocType, float]]:
    """
    Tier 1 keyword scoring:
      - required_keywords: each hit = 2 pts; needs at least 1 hit to qualify
      - optional_keywords: each hit = 1 pt
      - confidence = score / max_possible capped at 1.0

    Returns:
        best_doc_type, base_confidence, per_type_confidences
    """
    per_type: dict[DocType, float] = {}
    best_type: DocType = "UNKNOWN"
    best_conf: float = 0.0

    try:
        for doc_type, rules in _TIER1_RULES.items():
            required = rules.get("required", [])
            optional = rules.get("optional", [])

            req_hits = sum(1 for kw in required if kw in text_lower)
            if req_hits < 1:
                per_type[doc_type] = 0.0
                continue

            opt_hits = sum(1 for kw in optional if kw in text_lower)
            score = (req_hits * 2) + (opt_hits * 1)
            max_possible = (len(required) * 2) + (len(optional) * 1)
            conf = 0.0 if max_possible <= 0 else min(float(score) / float(max_possible), 1.0)

            per_type[doc_type] = conf
            if conf > best_conf:
                best_conf = conf
                best_type = doc_type

        return best_type, best_conf, per_type
    except Exception:
        logger.exception("Tier 1 scoring failed; returning UNKNOWN.")
        return "UNKNOWN", 0.0, per_type


def _apply_confidence_boosts(
    doc_type: DocType, confidence: float, layout_hints: dict[str, Any], has_tables: bool
) -> float:
    """
    Apply Tier 2 confidence boosts after Tier 1.
    """
    try:
        conf = float(confidence)
        has_currency = bool(layout_hints.get("has_currency"))
        avg_line_length = float(layout_hints.get("avg_line_length") or 0.0)
        colon_ratio = float(layout_hints.get("colon_ratio") or 0.0)

        if doc_type in {"INVOICE", "PURCHASE_ORDER", "RECEIPT"}:
            if has_currency:
                conf += 0.15
            if has_tables:
                conf += 0.10

        if doc_type == "CONTRACT" and avg_line_length > 60:
            conf += 0.10

        if doc_type == "EMAIL" and colon_ratio > 1.5:
            conf += 0.10

        return min(conf, 1.0)
    except Exception:
        logger.exception("Failed applying confidence boosts; returning original confidence.")
        try:
            return min(float(confidence), 1.0)
        except Exception:
            return 0.0


def _heuristic_fallback(
    confidence: float, has_tables: bool, layout_hints: dict[str, Any]
) -> DocType:
    """
    Heuristic fallback when confidence < 0.4 after both tiers.
    """
    try:
        conf = float(confidence)
        if conf >= 0.4:
            return "UNKNOWN"

        has_currency = bool(layout_hints.get("has_currency"))
        has_dates = bool(layout_hints.get("has_dates"))
        colon_ratio = float(layout_hints.get("colon_ratio") or 0.0)
        avg_line_length = float(layout_hints.get("avg_line_length") or 0.0)
        has_numbered_list = bool(layout_hints.get("has_numbered_list"))
        has_bullet_list = bool(layout_hints.get("has_bullet_list"))

        if has_currency and has_tables:
            return "FINANCIAL_DOCUMENT"
        if colon_ratio > 1.5 and has_dates:
            return "STRUCTURED_FORM"
        if avg_line_length > 70:
            return "NARRATIVE_DOCUMENT"
        if has_numbered_list or has_bullet_list:
            return "LIST_DOCUMENT"
        return "UNKNOWN"
    except Exception:
        logger.exception("Heuristic fallback failed; returning UNKNOWN.")
        return "UNKNOWN"


def classify(text: str, raw_tables: Optional[Any] = None) -> DocumentProfile:
    """
    Classify a document from text and optional raw tables.

    Args:
        text: The document text.
        raw_tables: Optional raw/extracted tables (any nested shape).

    Returns:
        DocumentProfile. Never raises; returns partial results on errors.
    """
    try:
        safe_text = text or ""
    except Exception:
        safe_text = ""

    try:
        has_tables = _has_tables(raw_tables)
        layout_hints = _compute_layout_hints(safe_text)
        detected_dates = _extract_dates(safe_text, limit=10)
        detected_amounts = _extract_amounts(safe_text, limit=10)
        key_value_pairs = _extract_key_value_pairs(safe_text)

        text_lower = _safe_lower(safe_text)
        best_type, base_conf, _per_type = _tier1_score(text_lower)

        # Tier 2 boosts (always computed signals, boosts only apply post Tier 1)
        boosted_conf = _apply_confidence_boosts(
            doc_type=best_type,
            confidence=base_conf,
            layout_hints=layout_hints,
            has_tables=has_tables,
        )

        final_type = best_type
        final_conf = boosted_conf

        # Heuristic fallback if still not "known"
        if final_conf < 0.4 or final_type == "UNKNOWN":
            fallback_type = _heuristic_fallback(final_conf, has_tables, layout_hints)
            if fallback_type != "UNKNOWN":
                final_type = fallback_type
                # keep confidence < 0.4 to reflect heuristic nature
                final_conf = min(final_conf, 0.39)
            else:
                final_type = "UNKNOWN" if final_conf < 0.4 else final_type

        return DocumentProfile(
            doc_type=final_type,
            confidence=min(max(float(final_conf), 0.0), 1.0),
            has_tables=has_tables,
            layout_hints=layout_hints,
            key_value_pairs=key_value_pairs,
            detected_dates=detected_dates,
            detected_amounts=detected_amounts,
        )
    except Exception:
        logger.exception("classify() failed; returning UNKNOWN profile.")
        return DocumentProfile(
            doc_type="UNKNOWN",
            confidence=0.0,
            has_tables=False,
            layout_hints={},
            key_value_pairs={},
            detected_dates=[],
            detected_amounts=[],
        )

