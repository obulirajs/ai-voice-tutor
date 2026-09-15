"""Ingestion module — upload handling, OCR/vision, and structure-aware chunking.

Public surface: ingest_pdf() (the full pipeline), IngestResult, and the
upload-validation helpers (looks_like_pdf, compute_content_hash,
open_validated_pdf, InvalidPdfError) callers should run before invoking
ingest_pdf. Depends on app.models' and app.knowledge's interfaces only,
never a concrete adapter — see technical-design.md's dependency-inversion
convention. Retrieval-quality checking (both the automatic post-ingestion
consistency check and the standalone golden-question endpoint) lives in
app.knowledge, not here — see IngestResult.consistency_check and
app.knowledge.golden_qa.
"""

from __future__ import annotations

from .pipeline import IngestResult, collection_for_subject, ingest_pdf
from .validation import (
    InvalidPdfError,
    compute_content_hash,
    looks_like_pdf,
    open_validated_pdf,
)

__all__ = [
    "IngestResult",
    "InvalidPdfError",
    "collection_for_subject",
    "compute_content_hash",
    "ingest_pdf",
    "looks_like_pdf",
    "open_validated_pdf",
]
