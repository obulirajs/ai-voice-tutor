"""Knowledge module — vector store access and retrieval.

Public surface: the `VectorStore` interface, its `Chunk`/`SearchResult`
types, `get_vector_store()` (reads VECTOR_STORE from the environment and
returns the configured adapter), `RetrievalService` (embeds a question and
returns scored, grounded chunks — see `retrieval.py` for its config
getters), the automatic post-ingestion `check_ingestion_consistency()` and
`check_subject_language_mismatch()`, and the standalone, read-only
`run_golden_qa_check()`. Callers (ingestion, orchestration, api) must depend
on these, never import `SqliteVecStore` directly — the same
dependency-inversion convention as app.models' ModelProvider/get_provider().

Note: `retrieval.RetrievedChunk` is deliberately not re-exported here — the
name `RetrievedChunk` below is golden_qa's (a distinct, lighter type).
Import the retrieval one from `app.knowledge.retrieval` directly.
"""

from __future__ import annotations

import os

from .base import Chunk, SearchResult, VectorStore
from .consistency import (
    ConsistencyCheckResult,
    ConsistencyFailure,
    check_ingestion_consistency,
)
from .golden_qa import (
    GoldenQASummary,
    GoldenQuestion,
    GoldenQuestionResult,
    RetrievedChunk,
    TypeSummary,
    parse_expected_page,
    run_golden_qa_check,
    summarize_golden_qa_results,
)
from .language_check import check_subject_language_mismatch, detect_chunk_language
from .retrieval import (
    RetrievalResult,
    RetrievalService,
    get_retrieval_score_threshold,
    get_retrieval_top_k,
)
from .sqlite_vec_store import SqliteVecStore

__all__ = [
    "Chunk",
    "ConsistencyCheckResult",
    "ConsistencyFailure",
    "GoldenQASummary",
    "GoldenQuestion",
    "GoldenQuestionResult",
    "RetrievalResult",
    "RetrievalService",
    "RetrievedChunk",
    "SearchResult",
    "TypeSummary",
    "VectorStore",
    "check_ingestion_consistency",
    "check_subject_language_mismatch",
    "detect_chunk_language",
    "get_retrieval_score_threshold",
    "get_retrieval_top_k",
    "get_vector_store",
    "parse_expected_page",
    "run_golden_qa_check",
    "summarize_golden_qa_results",
]

_DEFAULT_VECTOR_STORE_PATH = "./vector_store.db"


def get_vector_store() -> VectorStore:
    """Build the VectorStore adapter configured via the VECTOR_STORE env var."""
    backend = os.getenv("VECTOR_STORE", "sqlite_vec").lower()

    if backend == "sqlite_vec":
        db_path = os.getenv("VECTOR_STORE_PATH", _DEFAULT_VECTOR_STORE_PATH)
        return SqliteVecStore(db_path=db_path)

    raise ValueError(f"Unknown VECTOR_STORE: {backend!r} (expected 'sqlite_vec')")
