"""Retrieval service — turns a question into scored, grounded textbook chunks.

Combines an EmbeddingProvider and a VectorStore behind one retrieve() call.
Takes both as constructor arguments (dependency injection) rather than
calling app.models.get_embedding_provider()/app.knowledge.get_vector_store()
itself, so callers control provider selection and tests can inject fakes —
the same convention as the rest of this module.

Note: RetrievedChunk here is intentionally NOT re-exported from
knowledge/__init__.py's top level — that name is already taken by
golden_qa.RetrievedChunk (a distinct, lighter type used by the golden-QA
sanity check). Import it from app.knowledge.retrieval directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from app.models import EmbeddingProvider, EmbeddingResponse

from .base import VectorStore

__all__ = [
    "RetrievalResult",
    "RetrievalService",
    "RetrievedChunk",
    "get_retrieval_score_threshold",
    "get_retrieval_top_k",
]

_DEFAULT_RETRIEVAL_TOP_K = 5
_DEFAULT_RETRIEVAL_SCORE_THRESHOLD = 0.55


def get_retrieval_top_k() -> int:
    """Default number of chunks to retrieve, from the RETRIEVAL_TOP_K env var."""
    return int(os.getenv("RETRIEVAL_TOP_K", str(_DEFAULT_RETRIEVAL_TOP_K)))


def get_retrieval_score_threshold() -> float:
    """Minimum cosine similarity for a chunk to count as a real match, from
    the RETRIEVAL_SCORE_THRESHOLD env var — the grounding guardrail's
    "nothing good enough was found" cutoff, applied by orchestration.
    """
    return float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", str(_DEFAULT_RETRIEVAL_SCORE_THRESHOLD)))


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    page_number: int | None
    score: float
    document_id: int | None


@dataclass(frozen=True)
class RetrievalResult:
    chunks: list[RetrievedChunk]
    best_score: float
    query_embedding_usage: EmbeddingResponse


class RetrievalService:
    """Embeds a query, searches a subject's collection, and returns scored chunks."""

    def __init__(self, embedding_provider: EmbeddingProvider, vector_store: VectorStore) -> None:
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store

    async def retrieve(self, query: str, collection: str, top_k: int = _DEFAULT_RETRIEVAL_TOP_K) -> RetrievalResult:
        embedding_response = self._embedding_provider.embed([query])
        (query_embedding,) = embedding_response.embeddings

        hits = await self._vector_store.search(collection, query_embedding, top_k=top_k)

        chunks = sorted(
            (
                RetrievedChunk(
                    text=hit.text,
                    page_number=hit.page_number,
                    score=1.0 - hit.distance,
                    document_id=hit.document_id,
                )
                for hit in hits
            ),
            key=lambda chunk: chunk.score,
            reverse=True,
        )[:top_k]

        best_score = chunks[0].score if chunks else 0.0

        return RetrievalResult(chunks=chunks, best_score=best_score, query_embedding_usage=embedding_response)
