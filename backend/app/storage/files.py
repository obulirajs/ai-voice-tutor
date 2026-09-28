"""On-disk storage for uploaded PDFs and their cached page-image renders.

architecture.md's Storage module owns "filesystem for the actual
documents/images" alongside the SQLite metadata DB (db.py/repository.py) --
this is that half. Layout under get_data_dir() (DATA_DIR env var, default
STORAGE_DEFAULTS.data_dir):

    data/documents/<document_id>.pdf              the stored source PDF
    data/documents/cache/<document_id>/<page>.png  cached page-view renders

Callers (app.api.documents) never build these paths themselves -- that
would leak the on-disk layout into the API layer, and any change here
(e.g. sharding by subject) would ripple outward.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from app.config import STORAGE_DEFAULTS

__all__ = [
    "delete_document_files",
    "ensure_documents_dir",
    "get_cache_dir",
    "get_data_dir",
    "get_document_pdf_path",
    "get_documents_dir",
    "get_page_cache_path",
    "save_document_pdf",
]


def get_data_dir() -> Path:
    return Path(os.getenv("DATA_DIR", STORAGE_DEFAULTS.data_dir))


def get_documents_dir() -> Path:
    return get_data_dir() / "documents"


def get_cache_dir() -> Path:
    return get_documents_dir() / "cache"


def ensure_documents_dir() -> None:
    """Creates the documents dir and its cache subdir if missing. Called
    from app.main's startup lifespan, same as storage.init_db()."""
    get_documents_dir().mkdir(parents=True, exist_ok=True)
    get_cache_dir().mkdir(parents=True, exist_ok=True)


def get_document_pdf_path(document_id: int) -> Path:
    return get_documents_dir() / f"{document_id}.pdf"


def get_page_cache_path(document_id: int, page_number: int) -> Path:
    return get_cache_dir() / str(document_id) / f"{page_number}.png"


def save_document_pdf(document_id: int, pdf_bytes: bytes) -> None:
    path = get_document_pdf_path(document_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pdf_bytes)


def delete_document_files(document_id: int) -> None:
    """Removes the stored PDF and any cached page-image renders for a
    document. Safe to call even if neither exists (e.g. the PDF was never
    stored for a pre-page-view document) -- delete is best-effort cleanup,
    not a guarantee something was there.
    """
    get_document_pdf_path(document_id).unlink(missing_ok=True)

    cache_dir = get_cache_dir() / str(document_id)
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
