from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .models import ChatSession, Document, Message, Subject, UsageEvent


async def get_subject_by_name(db: AsyncSession, name: str) -> Subject | None:
    result = await db.execute(select(Subject).where(Subject.name == name))
    return result.scalar_one_or_none()


async def get_or_create_subject(db: AsyncSession, name: str) -> Subject:
    subject = await get_subject_by_name(db, name)
    if subject is not None:
        return subject

    subject = Subject(name=name)
    db.add(subject)
    await db.flush()
    return subject


async def create_session(db: AsyncSession, subject_id: int) -> ChatSession:
    session = ChatSession(subject_id=subject_id)
    db.add(session)
    await db.flush()
    return session


async def get_session(db: AsyncSession, session_id: str) -> ChatSession | None:
    result = await db.execute(select(ChatSession).where(ChatSession.id == session_id))
    return result.scalar_one_or_none()


async def get_history(db: AsyncSession, session_id: str) -> list[Message]:
    result = await db.execute(select(Message).where(Message.session_id == session_id).order_by(Message.id))
    return list(result.scalars().all())


async def append_message(db: AsyncSession, session_id: str, role: str, content: str) -> Message:
    message = Message(session_id=session_id, role=role, content=content)
    db.add(message)
    await db.flush()
    return message


async def log_usage_event(
    db: AsyncSession,
    *,
    session_id: str | None,
    subject_id: int | None,
    provider: str,
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
    cost_usd: float,
    latency_ms: float | None = None,
) -> UsageEvent:
    event = UsageEvent(
        session_id=session_id,
        subject_id=subject_id,
        provider=provider,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
    )
    db.add(event)
    await db.flush()
    return event


async def create_document(
    db: AsyncSession,
    *,
    subject_id: int,
    filename: str,
    collection: str,
    page_count: int,
    scanned_page_count: int = 0,
    chunk_count: int = 0,
    content_hash: str | None = None,
) -> Document:
    document = Document(
        subject_id=subject_id,
        filename=filename,
        collection=collection,
        page_count=page_count,
        scanned_page_count=scanned_page_count,
        chunk_count=chunk_count,
        content_hash=content_hash,
    )
    db.add(document)
    await db.flush()
    return document


async def get_document_by_hash(db: AsyncSession, subject_id: int, content_hash: str) -> Document | None:
    result = await db.execute(
        select(Document).where(Document.subject_id == subject_id, Document.content_hash == content_hash)
    )
    return result.scalar_one_or_none()


async def list_documents(db: AsyncSession, subject_id: int) -> list[Document]:
    result = await db.execute(select(Document).where(Document.subject_id == subject_id).order_by(Document.id))
    return list(result.scalars().all())


async def get_document(db: AsyncSession, document_id: int) -> Document | None:
    result = await db.execute(select(Document).where(Document.id == document_id))
    return result.scalar_one_or_none()


async def delete_document(db: AsyncSession, document_id: int) -> None:
    document = await get_document(db, document_id)
    if document is not None:
        await db.delete(document)
        await db.flush()
