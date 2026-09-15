from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    text: str
    embedding: list[float]
    document_id: int | None = None
    page_number: int | None = None


@dataclass(frozen=True)
class SearchResult:
    text: str
    document_id: int | None
    page_number: int | None
    distance: float


class VectorStore(ABC):
    """Common interface every vector-store adapter (sqlite-vec, ...) implements.

    Ingestion and knowledge's own sanity-check code depend on this interface,
    never on a concrete adapter — see technical-design.md's dependency-inversion
    convention (the same pattern as app.models.ModelProvider).
    """

    @abstractmethod
    async def upsert_chunks(self, collection: str, chunks: list[Chunk]) -> None:
        """Embed-and-store chunks into a subject's collection (creating it if needed)."""
        raise NotImplementedError

    @abstractmethod
    async def search(self, collection: str, query_embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        """Return the top_k nearest chunks (by similarity) in a subject's collection."""
        raise NotImplementedError

    @abstractmethod
    async def delete_document_chunks(self, collection: str, document_id: int) -> None:
        """Remove all chunks belonging to one document from a collection.

        Used before re-ingesting a document with corrected pipeline code, so
        the old (possibly buggy) chunks don't linger alongside the new ones.
        """
        raise NotImplementedError
