"""
rag/__init__.py
===============
RAG (Retrieval-Augmented Generation) module for the IDP System.

Public API:
  - build_faiss_retriever(doc_filter, k)  → LangChain BaseRetriever
  - RAGChainManager()                     → singleton chain manager
  - build_prompt(include_risk_addendum)   → ChatPromptTemplate
"""

from rag.retriever import build_faiss_retriever
from rag.chain import RAGChainManager
from rag.prompt_templates import build_prompt

__all__ = ["build_faiss_retriever", "RAGChainManager", "build_prompt"]
