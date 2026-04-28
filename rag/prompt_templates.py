"""
rag/prompt_templates.py
=======================
Module 6 — Prompt Templates for RAG Chain

Design:
  - BASE_SYSTEM_TEMPLATE: core system prompt with {context}, {chat_history}, {question}
  - RISK_ADDENDUM: composable addition appended to the base system prompt
    when retrieved documents contain contracts, legal content, or Risk:High tags.
  - build_prompt(): returns a ChatPromptTemplate with or without the addendum.
  - should_include_risk_addendum(): auto-detects from retrieved Document metadata.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.documents import Document

from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger("rag.prompt_templates")

# ---------------------------------------------------------------------------
# Base system prompt
# ---------------------------------------------------------------------------
BASE_SYSTEM_TEMPLATE = """\
You are an intelligent document assistant for the IDP (Intelligent Document \
Processing) system. Your role is to answer questions about uploaded and \
processed documents using ONLY the provided context.

INSTRUCTIONS:
- Answer based strictly on the retrieved context below. Never fabricate information.
- If the context is insufficient, state clearly what is missing.
- Cite document IDs (e.g. DOC_A8692491) when referencing specific information.
- Be concise but thorough.
- When presenting numerical data or tables, preserve the original formatting.

RETRIEVED CONTEXT:
{context}

CONVERSATION HISTORY:
{chat_history}"""

# ---------------------------------------------------------------------------
# Composable risk addendum — appended to base, never used alone
# ---------------------------------------------------------------------------
RISK_ADDENDUM = """

ADDITIONAL RISK-ANALYSIS INSTRUCTIONS (risk/contract content detected):
- Identify and highlight penalty clauses, liability terms, and breach conditions.
- Flag financial exposure amounts, deadlines, and termination triggers.
- Assess overall risk level (High / Medium / Low) with brief justification.
- Note compliance concerns or regulatory references.
- If indemnity or forfeiture clauses exist, summarize their conditions explicitly."""


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

_RISK_DOC_TYPES = frozenset({"contract", "legal", "agreement", "nda"})
_RISK_TAG_PREFIXES = ("Risk:High", "Risk:Medium")


def should_include_risk_addendum(docs: list[Document]) -> bool:
    """
    Return True if any retrieved document is risk-related.

    Checks:
      - document_type ∈ {contract, legal, agreement, nda}
      - any tag starting with Risk:High or Risk:Medium
    """
    for doc in docs:
        meta = doc.metadata or {}
        if meta.get("document_type", "").lower() in _RISK_DOC_TYPES:
            return True
        for tag in meta.get("tags", []):
            if tag.startswith(_RISK_TAG_PREFIXES):
                return True
    return False


def build_prompt(include_risk_addendum: bool = False) -> ChatPromptTemplate:
    """
    Build and return a ChatPromptTemplate.

    When *include_risk_addendum* is True the RISK_ADDENDUM is **composed**
    into the system message — it extends the base prompt, it does NOT
    replace it.

    Placeholders filled at call-time: {context}, {chat_history}, {question}.
    """
    system_content = BASE_SYSTEM_TEMPLATE
    if include_risk_addendum:
        system_content += RISK_ADDENDUM
        logger.debug("Risk addendum composed into prompt.")

    return ChatPromptTemplate.from_messages([
        ("system", system_content),
        ("human", "{question}"),
    ])
