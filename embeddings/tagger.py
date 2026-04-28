"""
embeddings/tagger.py
====================
Module 4b: Auto-Tagging

Automatically generate contextual tags for documents based on:
  - Domain classification (Finance, Legal, Medical, HR, etc.)
  - Vendor/company extraction
  - Risk level assessment
  - Amount bucketing
  - Document type
  - Default status tag

Returns a list of string tags like:
  ["Finance", "Vendor:AcmeCorp", "Risk:High", "Amount:5K-50K",
   "DocType:invoice", "Status:Under Review"]
"""

import re
import logging
from typing import Optional

logger = logging.getLogger("embeddings.tagger")


# ---------------------------------------------------------------------------
# Domain classification keywords
# ---------------------------------------------------------------------------
_DOMAIN_RULES = {
    "Finance": [
        "invoice", "payment", "receipt", "tax", "gst", "amount", "balance",
        "credit", "debit", "bank", "account", "transaction", "upi", "total",
        "revenue", "expense", "profit", "loss", "budget", "salary", "payroll",
        "refund", "billing", "ledger", "audit",
    ],
    "Legal": [
        "contract", "agreement", "clause", "penalty", "liability",
        "terms and conditions", "indemnity", "arbitration", "jurisdiction",
        "warranty", "confidential", "non-disclosure", "nda", "compliance",
        "regulation", "statute", "court", "legal", "attorney", "law",
    ],
    "Medical": [
        "patient", "diagnosis", "prescription", "hospital", "doctor",
        "medical", "health", "clinical", "treatment", "pharmacy",
        "lab report", "blood", "test result", "symptom", "disease",
    ],
    "HR": [
        "employee", "resume", "cv", "hire", "onboarding", "leave",
        "attendance", "appraisal", "performance review", "designation",
        "department", "human resource", "training", "offer letter",
    ],
    "Operations": [
        "shipment", "delivery", "logistics", "warehouse", "inventory",
        "supply chain", "purchase order", "vendor", "procurement",
        "shipping", "tracking", "dispatch",
    ],
}

# ---------------------------------------------------------------------------
# Risk keywords
# ---------------------------------------------------------------------------
_HIGH_RISK_KEYWORDS = [
    "penalty", "liability", "termination", "breach", "dispute",
    "overdue", "default", "fraud", "violation", "critical",
    "urgent", "lawsuit", "damages", "forfeiture",
]

_MEDIUM_RISK_KEYWORDS = [
    "warning", "caution", "review", "amendment", "exception",
    "delay", "escalation", "discrepancy", "pending",
]

# ---------------------------------------------------------------------------
# Vendor / company extraction patterns
# ---------------------------------------------------------------------------
_VENDOR_PATTERNS = [
    # "Vendor: AcmeCorp" or "Vendor Name: AcmeCorp"
    r"(?:vendor|supplier|company|firm|seller|merchant)[\s]*(?:name)?[\s]*[:\-]\s*(.+)",
    # "Bill From: AcmeCorp"
    r"(?:bill\s*from|from|sold\s*by|issued\s*by)[\s]*[:\-]\s*(.+)",
]


# ---------------------------------------------------------------------------
# Amount bucketing
# ---------------------------------------------------------------------------

def _bucket_amount(amount_str: Optional[str]) -> Optional[str]:
    """
    Convert an amount string to a human-readable bucket tag.

    Buckets:
      - Under1K   (< 1,000)
      - 1K-5K     (1,000 – 4,999)
      - 5K-50K    (5,000 – 49,999)
      - Over50K   (≥ 50,000)
    """
    if not amount_str:
        return None

    try:
        amount = float(amount_str.replace(",", ""))
    except (ValueError, TypeError):
        return None

    if amount < 1_000:
        return "Amount:Under1K"
    elif amount < 5_000:
        return "Amount:1K-5K"
    elif amount < 50_000:
        return "Amount:5K-50K"
    else:
        return "Amount:Over50K"


# ---------------------------------------------------------------------------
# Vendor extraction
# ---------------------------------------------------------------------------

def _extract_vendor(text: str) -> Optional[str]:
    """Try to extract a vendor/company name from text using patterns."""
    text_lower = text.lower()
    for pattern in _VENDOR_PATTERNS:
        match = re.search(pattern, text_lower)
        if match:
            vendor = match.group(1).strip()
            # Clean up: take first line, capitalize words, limit length
            vendor = vendor.split("\n")[0].strip()
            if len(vendor) > 50:
                vendor = vendor[:50]
            # Title case
            vendor = vendor.title()
            # Remove trailing punctuation
            vendor = vendor.rstrip(".,;:")
            if vendor and len(vendor) > 1:
                return vendor
    return None


# ---------------------------------------------------------------------------
# Risk assessment
# ---------------------------------------------------------------------------

def _assess_risk(text: str, amount_str: Optional[str] = None) -> str:
    """
    Assess document risk level based on keywords and amount.

    Returns: "Risk:High", "Risk:Medium", or "Risk:Low"
    """
    text_lower = text.lower()

    # High risk keywords
    high_hits = sum(1 for kw in _HIGH_RISK_KEYWORDS if kw in text_lower)
    if high_hits >= 2:
        return "Risk:High"

    # High amount (> 50K)
    if amount_str:
        try:
            amount = float(amount_str.replace(",", ""))
            if amount >= 50_000:
                return "Risk:High"
        except (ValueError, TypeError):
            pass

    # Medium risk keywords
    medium_hits = sum(1 for kw in _MEDIUM_RISK_KEYWORDS if kw in text_lower)
    if high_hits >= 1 or medium_hits >= 2:
        return "Risk:Medium"

    return "Risk:Low"


# ---------------------------------------------------------------------------
# Domain classification
# ---------------------------------------------------------------------------

def _classify_domain(text: str) -> list[str]:
    """
    Classify document into one or more domains using keyword scoring.
    Returns domain tags for any category scoring above threshold.
    """
    text_lower = text.lower()
    domain_tags = []

    scores = {}
    for domain, keywords in _DOMAIN_RULES.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        scores[domain] = score

    # Include domains with score >= 3 (strong signal)
    for domain, score in sorted(scores.items(), key=lambda x: -x[1]):
        if score >= 3:
            domain_tags.append(domain)

    # Always include at least the top domain if any score > 0
    if not domain_tags:
        best = max(scores, key=scores.get)
        if scores[best] > 0:
            domain_tags.append(best)

    return domain_tags


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_tags(
    cleaned_text: str,
    document_type: str = "generic_document",
    entities: Optional[dict] = None,
) -> list[str]:
    """
    Auto-generate contextual tags for a document.

    Tags include:
      - Domain tags (Finance, Legal, Medical, HR, Operations)
      - Vendor:{name} if extractable
      - Risk:{High|Medium|Low}
      - Amount:{bucket} if amount is available
      - DocType:{type} from document classification
      - Status:Under Review (default for new documents)

    Args:
        cleaned_text:  Pre-cleaned document text.
        document_type: Detected document type from extraction.
        entities:      Extracted entities dict (name, amount, etc.)

    Returns:
        List of tag strings, e.g.:
        ["Finance", "Vendor:AcmeCorp", "Risk:High", "Amount:5K-50K",
         "DocType:invoice", "Status:Under Review"]
    """
    if entities is None:
        entities = {}

    tags = []

    # 1. Domain tags
    domain_tags = _classify_domain(cleaned_text)
    tags.extend(domain_tags)

    # 2. Document type tag
    if document_type and document_type != "generic_document":
        tags.append(f"DocType:{document_type}")

    # 3. Vendor tag
    vendor = _extract_vendor(cleaned_text)
    if vendor:
        tags.append(f"Vendor:{vendor}")

    # 4. Risk assessment
    amount_str = entities.get("amount")
    risk_tag = _assess_risk(cleaned_text, amount_str)
    tags.append(risk_tag)

    # 5. Amount bucket
    amount_tag = _bucket_amount(amount_str)
    if amount_tag:
        tags.append(amount_tag)

    # 6. Default status
    tags.append("Status:Under Review")

    logger.info("Generated %d tags: %s", len(tags), tags)
    return tags
