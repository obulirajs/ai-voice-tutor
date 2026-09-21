"""Storage module — SQLite metadata via SQLAlchemy + aiosqlite.

Public surface: the ORM row types, the repository functions, get_db() (a
FastAPI dependency yielding a session) and init_db() (creates tables,
called from app.main's startup lifespan). Callers use these, never the
raw engine or sqlalchemy internals directly.
"""

from __future__ import annotations

from .db import get_db, init_db
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
    "get_db",
    "get_document",
    "get_document_by_hash",
    "get_history",
    "get_or_create_subject",
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
]
