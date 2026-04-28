"""
rag/router.py
=============
Module 6 — RAG API Endpoints

Endpoints:
  POST   /rag/chat              → answer + source_chunks[]
  GET    /rag/chat/{session_id}  → conversation history
  DELETE /rag/chat/{session_id}  → clear session memory
  WS     /rag/stream             → stream tokens, final done frame with sources

Ollama connection errors → HTTP 503 / WebSocket error frame.
"""

from __future__ import annotations

import json
import logging
import uuid

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from rag.chain import RAGChainManager, OllamaConnectionError

logger = logging.getLogger("rag.router")

router = APIRouter(prefix="/rag", tags=["RAG Chat"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question")
    session_id: str | None = Field(
        None, description="Session ID; auto-generated if omitted"
    )
    doc_filter: str | None = Field(
        None, description="Filter by document_type or tag (substring match)"
    )


class SourceChunk(BaseModel):
    doc_name: str
    page: int
    similarity_score: float
    document_type: str = "unknown"
    tags: list[str] = []
    content_preview: str = ""


class ChatResponse(BaseModel):
    session_id: str
    question: str
    answer: str
    source_chunks: list[SourceChunk]


class HistoryMessage(BaseModel):
    role: str
    content: str


class HistoryResponse(BaseModel):
    session_id: str
    messages: list[HistoryMessage]


# ---------------------------------------------------------------------------
# POST /rag/chat — non-streaming chat
# ---------------------------------------------------------------------------

@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Ask a question over your documents (RAG)",
)
async def rag_chat(body: ChatRequest):
    """
    Send a natural-language question.  The system retrieves the most
    relevant document chunks via FAISS + MMR, composes a prompt
    (with optional risk addendum), and returns an LLM-generated answer
    together with the source chunks used.

    Conversation memory is maintained per ``session_id``.
    """
    session_id = body.session_id or uuid.uuid4().hex[:12]

    try:
        manager = RAGChainManager()
        result = await manager.chat(
            question=body.question,
            session_id=session_id,
            doc_filter=body.doc_filter,
        )
    except OllamaConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return result


# ---------------------------------------------------------------------------
# GET /rag/chat/{session_id} — conversation history
# ---------------------------------------------------------------------------

@router.get(
    "/chat/{session_id}",
    response_model=HistoryResponse,
    summary="Retrieve conversation history for a session",
)
async def get_chat_history(session_id: str):
    """Return the full conversation history (user + assistant turns)."""
    manager = RAGChainManager()
    messages = await manager.get_history(session_id)

    if not messages:
        raise HTTPException(
            status_code=404,
            detail=f"No conversation history for session '{session_id}'.",
        )

    return {"session_id": session_id, "messages": messages}


# ---------------------------------------------------------------------------
# DELETE /rag/chat/{session_id} — clear memory
# ---------------------------------------------------------------------------

@router.delete(
    "/chat/{session_id}",
    summary="Clear conversation memory for a session",
)
async def clear_chat_session(session_id: str):
    """Delete the ConversationBufferWindowMemory for a session."""
    manager = RAGChainManager()
    cleared = await manager.clear_session(session_id)

    if not cleared:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found or already cleared.",
        )

    return {"status": "cleared", "session_id": session_id}


# ---------------------------------------------------------------------------
# WS /rag/stream — WebSocket token streaming
# ---------------------------------------------------------------------------

@router.websocket("/stream")
async def stream_chat(websocket: WebSocket):
    """
    WebSocket endpoint for streaming RAG responses.

    Client sends a JSON frame::

        {"question": "...", "session_id": "...", "doc_filter": "..."}

    Server streams back::

        {"type": "token",  "content": "partial text"}   (repeated)
        {"type": "done",   "sources": [...]}             (final frame)

    On Ollama connection failure the server sends::

        {"type": "error",  "message": "..."}
    """
    await websocket.accept()
    logger.info("WebSocket connection accepted.")

    try:
        while True:
            # Wait for the client to send a question
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json(
                    {"type": "error", "message": "Invalid JSON payload."}
                )
                continue

            question = data.get("question", "").strip()
            if not question:
                await websocket.send_json(
                    {"type": "error", "message": "Missing 'question' field."}
                )
                continue

            session_id = data.get("session_id") or uuid.uuid4().hex[:12]
            doc_filter = data.get("doc_filter")

            manager = RAGChainManager()

            try:
                async for frame in manager.stream_chat(
                    question=question,
                    session_id=session_id,
                    doc_filter=doc_filter,
                ):
                    await websocket.send_json(frame)
            except OllamaConnectionError as exc:
                await websocket.send_json(
                    {"type": "error", "message": str(exc)}
                )
            except Exception as exc:
                logger.exception("Streaming error: %s", exc)
                await websocket.send_json(
                    {"type": "error", "message": f"Internal error: {exc}"}
                )

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as exc:
        logger.exception("WebSocket unexpected error: %s", exc)
