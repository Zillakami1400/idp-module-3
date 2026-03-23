"""
app/main.py
===========
IDP System — FastAPI application entry point.
"""

from fastapi import FastAPI

from app.ingestion.upload import router as upload_router

# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Intelligent Document Processing (IDP) System",
    description=(
        "A modular AI-powered document pipeline:\n"
        "  • Module 1 — Document Ingestion\n"
        "  • Module 2 — OCR Processing\n"
        "  • Module 3 — Information Extraction\n"
        "  • Module 4 — Semantic Embeddings     (coming soon)\n"
        "  • Module 5 — Vector Search            (coming soon)\n"
    ),
    version="0.3.0",
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(upload_router)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/", tags=["Health"])
def health_check():
    """Quick health-check endpoint."""
    return {
        "status": "running",
        "system": "IDP System",
        "version": "0.3.0",
        "modules": {
            "module_1_ingestion": "active",
            "module_2_ocr": "active",
            "module_3_extraction": "active",
            "module_4_embeddings": "pending",
            "module_5_vector_search": "pending",
        },
    }