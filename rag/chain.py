"""
rag/chain.py
============
Module 6 — RAG Chain Manager (Singleton)

Thread-safety guarantees:
  - One ``asyncio.Lock`` per session_id prevents concurrent mutations to the
    same conversation's memory.
  - A global ``asyncio.Lock`` serialises creation of new per-session locks.
  - ``ConversationBufferWindowMemory`` is scoped per session_id — never shared.

Ollama integration:
  - Uses ChatOllama (langchain-community or langchain-ollama).
  - Connection errors are caught and re-raised as ``OllamaConnectionError``
    so the router can return HTTP 503.

Environment variables:
  - OLLAMA_MODEL        (default: llama3.1)
  - OLLAMA_BASE_URL     (default: http://localhost:11434)
  - OLLAMA_TEMPERATURE  (default: 0.3)
  - RAG_MEMORY_WINDOW   (default: 10)
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, AsyncGenerator

import numpy as np

logger = logging.getLogger("rag.chain")

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
try:
    from langchain_ollama import ChatOllama
except ImportError:
    try:
        from langchain_community.chat_models import ChatOllama  # type: ignore[assignment]
    except ImportError:
        ChatOllama = None  # type: ignore[assignment,misc]

try:
    from langchain.memory import ConversationBufferWindowMemory
    from langchain_core.messages import HumanMessage, AIMessage
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------
class OllamaConnectionError(Exception):
    """Raised when the Ollama server is unreachable."""


# ---------------------------------------------------------------------------
# Singleton chain manager
# ---------------------------------------------------------------------------
class RAGChainManager:
    """
    Singleton that owns:
      - per-session ``ConversationBufferWindowMemory``
      - per-session ``asyncio.Lock``
      - a lazily-initialised ``ChatOllama`` LLM
    """

    _instance: RAGChainManager | None = None

    def __new__(cls) -> RAGChainManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        self._sessions: dict[str, ConversationBufferWindowMemory] = {}
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()
        self._llm = None

        logger.info("RAGChainManager singleton created.")

    # ------------------------------------------------------------------
    # Lock management
    # ------------------------------------------------------------------
    async def _get_session_lock(self, session_id: str) -> asyncio.Lock:
        async with self._global_lock:
            if session_id not in self._session_locks:
                self._session_locks[session_id] = asyncio.Lock()
            return self._session_locks[session_id]

    # ------------------------------------------------------------------
    # Memory management (call under session lock only)
    # ------------------------------------------------------------------
    def _get_or_create_memory(self, sid: str) -> ConversationBufferWindowMemory:
        if sid not in self._sessions:
            window = int(os.getenv("RAG_MEMORY_WINDOW", "10"))
            self._sessions[sid] = ConversationBufferWindowMemory(
                k=window,
                memory_key="chat_history",
                return_messages=False,
                human_prefix="User",
                ai_prefix="Assistant",
            )
            logger.debug("Created new memory for session=%s (window=%d)", sid, window)
        return self._sessions[sid]

    # ------------------------------------------------------------------
    # LLM management
    # ------------------------------------------------------------------
    def _get_llm(self):
        if ChatOllama is None:
            raise ImportError(
                "Neither langchain-ollama nor langchain-community ChatOllama "
                "is installed.  pip install langchain-ollama"
            )
        if self._llm is None:
            self._llm = ChatOllama(
                model=os.getenv("OLLAMA_MODEL", "llama3.1"),
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.3")),
            )
            logger.info("ChatOllama initialised (model=%s).", self._llm.model)
        return self._llm

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _retrieve_docs(question: str, doc_filter: str | None = None):
        from rag.retriever import build_faiss_retriever
        retriever = build_faiss_retriever(doc_filter=doc_filter)
        return retriever.invoke(question)

    @staticmethod
    def _build_sources(docs, question: str) -> list[dict]:
        from rag.retriever import _get_query_vector

        query_vec = _get_query_vector(question)
        sources: list[dict] = []
        for doc in docs:
            doc_id = doc.metadata.get("doc_id", "unknown")
            score = 0.0
            if query_vec is not None:
                npy = Path("storage/embeddings") / f"{doc_id}.npy"
                if npy.exists():
                    dv = np.load(npy).flatten().astype("float32")
                    n = np.linalg.norm(dv)
                    if n > 0:
                        dv /= n
                    score = float(np.dot(query_vec, dv))
            sources.append({
                "doc_name": doc_id,
                "page": 1,
                "similarity_score": round(score, 4),
                "document_type": doc.metadata.get("document_type", "unknown"),
                "tags": doc.metadata.get("tags", []),
                "content_preview": doc.page_content[:200],
            })
        return sources

    @staticmethod
    def _format_context(docs) -> str:
        if not docs:
            return "No relevant documents found."
        parts = []
        for d in docs:
            did = d.metadata.get("doc_id", "?")
            dtype = d.metadata.get("document_type", "unknown")
            parts.append(f"[{did}] ({dtype})\n{d.page_content}")
        return "\n\n---\n\n".join(parts)

    def _build_messages(self, question: str, memory, docs):
        from rag.prompt_templates import build_prompt, should_include_risk_addendum

        context = self._format_context(docs)
        include_risk = should_include_risk_addendum(docs)
        prompt = build_prompt(include_risk_addendum=include_risk)
        history = memory.load_memory_variables({}).get("chat_history", "")
        return prompt.format_messages(
            context=context,
            chat_history=history or "No previous conversation.",
            question=question,
        )

    @staticmethod
    def _wrap_ollama_error(exc: Exception) -> OllamaConnectionError:
        msg = str(exc).lower()
        keywords = ("connect", "refused", "unreachable", "timeout", "connection")
        if any(kw in msg for kw in keywords):
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            return OllamaConnectionError(
                f"Cannot connect to Ollama at {base_url}. "
                "Ensure Ollama is running with: ollama serve"
            )
        return OllamaConnectionError(f"Ollama error: {exc}")

    # ------------------------------------------------------------------
    # Public API — chat (non-streaming)
    # ------------------------------------------------------------------
    async def chat(
        self,
        question: str,
        session_id: str,
        doc_filter: str | None = None,
    ) -> dict[str, Any]:
        lock = await self._get_session_lock(session_id)
        async with lock:
            memory = self._get_or_create_memory(session_id)
            docs = self._retrieve_docs(question, doc_filter)
            sources = self._build_sources(docs, question)
            messages = self._build_messages(question, memory, docs)

            try:
                llm = self._get_llm()
                response = await llm.ainvoke(messages)
                answer = response.content
            except Exception as exc:
                raise self._wrap_ollama_error(exc) from exc

            memory.save_context({"input": question}, {"output": answer})

            return {
                "session_id": session_id,
                "question": question,
                "answer": answer,
                "source_chunks": sources,
            }

    # ------------------------------------------------------------------
    # Public API — streaming (async generator)
    # ------------------------------------------------------------------
    async def stream_chat(
        self,
        question: str,
        session_id: str,
        doc_filter: str | None = None,
    ) -> AsyncGenerator[dict, None]:
        """
        Async generator yielding:
          {"type": "token", "content": "..."}   — per token
          {"type": "done",  "sources": [...]}   — final frame
        Session lock is held for the entire duration.
        """
        lock = await self._get_session_lock(session_id)
        async with lock:
            memory = self._get_or_create_memory(session_id)
            docs = self._retrieve_docs(question, doc_filter)
            sources = self._build_sources(docs, question)
            messages = self._build_messages(question, memory, docs)

            full_response = ""
            try:
                llm = self._get_llm()
                async for chunk in llm.astream(messages):
                    token = chunk.content
                    if token:
                        full_response += token
                        yield {"type": "token", "content": token}
            except Exception as exc:
                yield {"type": "error", "message": str(self._wrap_ollama_error(exc))}
                return

            memory.save_context({"input": question}, {"output": full_response})
            yield {"type": "done", "sources": sources, "answer": full_response}

    # ------------------------------------------------------------------
    # Public API — history & session management
    # ------------------------------------------------------------------
    async def get_history(self, session_id: str) -> list[dict]:
        if session_id not in self._sessions:
            return []
        lock = await self._get_session_lock(session_id)
        async with lock:
            memory = self._sessions.get(session_id)
            if memory is None:
                return []
            msgs = memory.chat_memory.messages
            result = []
            for msg in msgs:
                if isinstance(msg, HumanMessage):
                    role = "user"
                elif isinstance(msg, AIMessage):
                    role = "assistant"
                else:
                    role = "system"
                result.append({"role": role, "content": msg.content})
            return result

    async def clear_session(self, session_id: str) -> bool:
        lock = await self._get_session_lock(session_id)
        async with lock:
            if session_id in self._sessions:
                self._sessions[session_id].clear()
                del self._sessions[session_id]
                logger.info("Cleared session %s", session_id)
                return True
            return False
