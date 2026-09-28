"""Storage module — SQLite metadata via SQLAlchemy + aiosqlite, plus
on-disk storage for uploaded PDFs and their cached page-image renders.

Public surface: the ORM row types, the repository functions, get_db() (a
FastAPI dependency yielding a session), init_db() (creates tables, called
from app.main's startup lifespan), and the files.py functions
(ensure_documents_dir(), save_document_pdf(), delete_document_files(),
get_document_pdf_path(), get_page_cache_path()) for the on-disk half. Callers
use these, never the raw engine, sqlalchemy internals, or on-disk paths
built by hand.
"""

from __future__ import annotations

from .db import get_db, init_db
from .files import (
    delete_document_files,
    ensure_documents_dir,
    get_document_pdf_path,
    get_page_cache_path,
    save_document_pdf,
)
from .models import ChatSession, Document, Message, Subject, UsageEvent
from .repository import (
    Period,
    RetrievalHealth,
    SessionPreview,
    SessionSummary,
    UsageSummary,
    append_message,
    create_document,
    create_session,
    delete_document,
    get_document,
    get_document_by_hash,
    get_history,
    get_or_create_subject,
    get_retrieval_health,
    get_session,
    get_session_list,
    get_session_usage,
    get_subject,
    get_subject_by_name,
    get_usage_summary,
    list_documents,
    list_sessions_for_subject,
    list_subjects,
    log_usage_event,
)

__all__ = [
    "ChatSession",
    "Document",
    "Message",
    "Period",
    "RetrievalHealth",
    "SessionPreview",
    "SessionSummary",
    "Subject",
    "UsageEvent",
    "UsageSummary",
    "append_message",
    "create_document",
    "create_session",
    "delete_document",
    "delete_document_files",
    "ensure_documents_dir",
    "get_db",
    "get_document",
    "get_document_by_hash",
    "get_document_pdf_path",
    "get_history",
    "get_or_create_subject",
    "get_page_cache_path",
    "get_retrieval_health",
    "get_session",
    "get_session_list",
    "get_session_usage",
    "get_subject",
    "get_subject_by_name",
    "get_usage_summary",
    "init_db",
    "list_documents",
    "list_sessions_for_subject",
    "list_subjects",
    "log_usage_event",
    "save_document_pdf",
]
