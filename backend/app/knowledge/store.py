from __future__ import annotations

import re
import sqlite3

import sqlite_vec  # type: ignore[import-untyped]


def connect(db_path: str) -> sqlite3.Connection:
    """Open (and prepare) the vector store's SQLite file.

    Synchronous by design: sqlite-vec is a runtime-loadable extension for
    the stdlib sqlite3 module, not aiosqlite, and this is used only from
    background threads (see knowledge/__init__.py's asyncio.to_thread calls).
    """
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            collection TEXT NOT NULL,
            document_id INTEGER,
            page_number INTEGER,
            text TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def collection_table(collection: str) -> str:
    """A safe vec0 virtual-table name for a collection (defensive sanitizing —
    callers pass a subject-derived token, e.g. "subject_3", not raw user text).
    """
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", collection)
    return f"vec_{safe}"


def ensure_collection(conn: sqlite3.Connection, collection: str, dimensions: int) -> None:
    table = collection_table(collection)
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING vec0("
        f"embedding float[{dimensions}] distance_metric=cosine)"
    )
    conn.commit()


def insert_chunks(
    conn: sqlite3.Connection,
    collection: str,
    rows: list[tuple[int | None, int | None, str]],
    embeddings: list[list[float]],
) -> None:
    """rows: (document_id, page_number, text) tuples, same order as embeddings."""
    table = collection_table(collection)
    cursor = conn.cursor()
    for (document_id, page_number, text), embedding in zip(rows, embeddings, strict=True):
        cursor.execute(
            "INSERT INTO chunks (collection, document_id, page_number, text) VALUES (?, ?, ?, ?)",
            (collection, document_id, page_number, text),
        )
        chunk_id = cursor.lastrowid
        cursor.execute(
            f"INSERT INTO {table} (rowid, embedding) VALUES (?, ?)",
            (chunk_id, sqlite_vec.serialize_float32(embedding)),
        )
    conn.commit()


def delete_chunks_for_document(conn: sqlite3.Connection, collection: str, document_id: int) -> None:
    """Removes every chunk (both metadata row and vector entry) belonging to
    one document from a collection — used to clear out a document's old
    chunks before re-ingesting it with corrected pipeline code.
    """
    table = collection_table(collection)
    cursor = conn.cursor()
    ids = [
        row[0]
        for row in cursor.execute(
            "SELECT id FROM chunks WHERE collection = ? AND document_id = ?",
            (collection, document_id),
        ).fetchall()
    ]
    if ids:
        placeholders = ",".join("?" for _ in ids)
        cursor.execute(f"DELETE FROM {table} WHERE rowid IN ({placeholders})", ids)
        cursor.execute(f"DELETE FROM chunks WHERE id IN ({placeholders})", ids)
    conn.commit()


def search(
    conn: sqlite3.Connection,
    collection: str,
    query_embedding: list[float],
    top_k: int,
) -> list[tuple[str, int | None, int | None, float]]:
    """Returns (text, page_number, document_id, distance) tuples, nearest first."""
    table = collection_table(collection)
    rows = conn.execute(
        f"""
        SELECT c.text, c.page_number, c.document_id, v.distance
        FROM {table} AS v
        JOIN chunks AS c ON c.id = v.rowid
        WHERE v.embedding MATCH ? AND k = ?
        ORDER BY v.distance
        """,
        (sqlite_vec.serialize_float32(query_embedding), top_k),
    ).fetchall()
    return [(text, page_number, document_id, distance) for text, page_number, document_id, distance in rows]
