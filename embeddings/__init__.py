"""
embeddings/__init__.py
=======================
Exposes the public API for the embeddings module.
"""

from embeddings.embedder import generate_embedding
from embeddings.tagger import generate_tags

__all__ = ["generate_embedding", "generate_tags"]
