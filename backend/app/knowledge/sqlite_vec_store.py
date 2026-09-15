from __future__ import annotations

import asyncio

from . import store
from .base import Chunk, SearchResult, VectorStore


class SqliteVecStore(VectorStore):
    """VectorStore adapter backed by sqlite-vec — a lightweight, local,
    file-based vector store, no server-based vector DB needed at this scale.
    Wraps the underlying sync sqlite3 calls in asyncio.to_thread() so callers
    can await them without blocking the event loop.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    async def upsert_chunks(self, collection: str, chunks: list[Chunk]) -> None:
        if not chunks:
            return

        def _run() -> None:
            conn = store.connect(self._db_path)
            try:
                store.ensure_collection(conn, collection, dimensions=len(chunks[0].embedding))
                rows = [(c.document_id, c.page_number, c.text) for c in chunks]
                embeddings = [c.embedding for c in chunks]
                store.insert_chunks(conn, collection, rows, embeddings)
            finally:
                conn.close()

        await asyncio.to_thread(_run)

    async def search(self, collection: str, query_embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        def _run() -> list[SearchResult]:
            conn = store.connect(self._db_path)
            try:
                rows = store.search(conn, collection, query_embedding, top_k)
                return [
                    SearchResult(text=text, page_number=page_number, document_id=document_id, distance=distance)
                    for text, page_number, document_id, distance in rows
                ]
            finally:
                conn.close()

        return await asyncio.to_thread(_run)

    async def delete_document_chunks(self, collection: str, document_id: int) -> None:
        def _run() -> None:
            conn = store.connect(self._db_path)
            try:
                store.delete_chunks_for_document(conn, collection, document_id)
            finally:
                conn.close()

        await asyncio.to_thread(_run)
