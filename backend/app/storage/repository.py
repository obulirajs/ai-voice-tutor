from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal

from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from .models import ChatSession, Document, Message, Subject, UsageEvent

Period = Literal["day", "week", "month", "all"]

_PERIOD_LOOKBACK = {
    "day": timedelta(days=1),
    "week": timedelta(days=7),
    "month": timedelta(days=30),
}


def _period_cutoff(period: Period) -> datetime | None:
    """The earliest created_at a row must have to count for `period`, or
    None for "all" (no filter)."""
    lookback = _PERIOD_LOOKBACK.get(period)
    return None if lookback is None else datetime.now(timezone.utc) - lookback


async def get_subject_by_name(db: AsyncSession, name: str) -> Subject | None:
    result = await db.execute(select(Subject).where(Subject.name == name))
    return result.scalar_one_or_none()


async def list_subjects(db: AsyncSession) -> list[tuple[Subject, int]]:
    """All subjects with their document count, for the /api/subjects list view."""
    result = await db.execute(
        select(Subject, func.count(Document.id))
        .outerjoin(Document, Document.subject_id == Subject.id)
        .group_by(Subject.id)
        .order_by(Subject.id)
    )
    return [(subject, count) for subject, count in result.all()]


async def get_subject(db: AsyncSession, subject_id: int) -> Subject | None:
    result = await db.execute(select(Subject).where(Subject.id == subject_id))
    return result.scalar_one_or_none()


async def get_or_create_subject(db: AsyncSession, name: str) -> Subject:
    subject = await get_subject_by_name(db, name)
    if subject is not None:
        return subject

    subject = Subject(name=name)
    db.add(subject)
    await db.flush()
    return subject


async def create_session(db: AsyncSession, subject_id: int, mode: str = "textbook") -> ChatSession:
    session = ChatSession(subject_id=subject_id, mode=mode)
    db.add(session)
    await db.flush()
    return session


async def get_session(db: AsyncSession, session_id: str) -> ChatSession | None:
    result = await db.execute(select(ChatSession).where(ChatSession.id == session_id))
    return result.scalar_one_or_none()


async def get_history(db: AsyncSession, session_id: str) -> list[Message]:
    result = await db.execute(select(Message).where(Message.session_id == session_id).order_by(Message.id))
    return list(result.scalars().all())


@dataclass(frozen=True)
class SessionPreview:
    session_id: str
    subject_id: int
    message_count: int
    last_message_preview: str
    last_active: datetime
    teaching_mode: str


async def list_sessions_for_subject(db: AsyncSession, subject_id: int, limit: int = 20) -> list[SessionPreview]:
    """Recent sessions for a subject, newest first, with a preview -- powers
    the sidebar's conversation history list (see SessionHistory.tsx).
    Sessions with zero messages (abandoned before any turn completed) are
    excluded by the inner join against `aggregates` below.
    """
    aggregates = (
        select(
            Message.session_id.label("session_id"),
            func.count(Message.id).label("message_count"),
            func.max(Message.created_at).label("last_active"),
        )
        .group_by(Message.session_id)
        .subquery()
    )

    last_user_message = (
        select(
            Message.session_id.label("session_id"),
            Message.content.label("content"),
            func.row_number().over(partition_by=Message.session_id, order_by=Message.id.desc()).label("rn"),
        )
        .where(Message.role == "user")
        .subquery()
    )

    result = await db.execute(
        select(
            ChatSession.id,
            ChatSession.subject_id,
            ChatSession.mode,
            aggregates.c.message_count,
            aggregates.c.last_active,
            last_user_message.c.content,
        )
        .join(aggregates, aggregates.c.session_id == ChatSession.id)
        .outerjoin(
            last_user_message,
            (last_user_message.c.session_id == ChatSession.id) & (last_user_message.c.rn == 1),
        )
        .where(ChatSession.subject_id == subject_id)
        .order_by(aggregates.c.last_active.desc())
        .limit(limit)
    )

    return [
        SessionPreview(
            session_id=session_id,
            subject_id=session_subject_id,
            message_count=message_count,
            last_message_preview=(content or "")[:80],
            last_active=last_active,
            teaching_mode=mode,
        )
        for session_id, session_subject_id, mode, message_count, last_active, content in result.all()
    ]


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
    event_type: str = "generation",
    retrieval_best_score: float | None = None,
    retrieval_chunk_count: int | None = None,
    guardrail_triggered: bool | None = None,
) -> UsageEvent:
    event = UsageEvent(
        session_id=session_id,
        subject_id=subject_id,
        event_type=event_type,
        provider=provider,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        retrieval_best_score=retrieval_best_score,
        retrieval_chunk_count=retrieval_chunk_count,
        guardrail_triggered=guardrail_triggered,
    )
    db.add(event)
    await db.flush()
    return event


async def get_session_usage(db: AsyncSession, session_id: str) -> list[UsageEvent]:
    """All usage_events rows for one session, oldest first -- the aggregation
    source for GET /api/sessions/{session_id}/usage."""
    result = await db.execute(select(UsageEvent).where(UsageEvent.session_id == session_id).order_by(UsageEvent.id))
    return list(result.scalars().all())


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


# --- Parent-view aggregation queries (Phase 6) -----------------------------
#
# These read usage_events for the parent-only cost/retrieval-health
# endpoints (technical-design.md: "the full breakdown ... lives on the
# parent-only view"). Score-distribution buckets match api/parent.py's
# documented thresholds: excellent >= 0.8, good >= 0.65, fair >= 0.55,
# poor < 0.55.


@dataclass(frozen=True)
class UsageSummary:
    total_cost_usd: float
    # "Sessions"/"turns" are approximated from usage_events rather than
    # joined against chat_sessions/messages -- a session that only ever hit
    # the "no documents uploaded" guardrail (handle_turn's very first check,
    # before any usage_event is logged) won't be counted. Acceptable v1
    # simplification: this view is a cost/health dashboard, not an audit log.
    total_sessions: int
    total_turns: int
    cost_by_event_type: dict[str, float]
    cost_by_subject: list[tuple[str, float]]


@dataclass(frozen=True)
class SessionSummary:
    session_id: str
    subject_name: str
    created_at: datetime
    cost_usd: float
    turn_count: int
    event_count: int


@dataclass(frozen=True)
class RetrievalHealth:
    total_queries: int
    guardrail_triggered_count: int
    avg_best_score: float | None
    excellent_count: int
    good_count: int
    fair_count: int
    poor_count: int


async def get_usage_summary(db: AsyncSession, period: Period) -> UsageSummary:
    """Aggregated usage across all sessions within `period`."""
    cutoff = _period_cutoff(period)
    filters = [] if cutoff is None else [UsageEvent.created_at >= cutoff]

    totals_result = await db.execute(
        select(
            func.coalesce(func.sum(UsageEvent.cost_usd), 0.0),
            func.count(func.distinct(UsageEvent.session_id)),
            func.count(case((UsageEvent.event_type == "generation", 1))),
        ).where(*filters)
    )
    total_cost_usd, total_sessions, total_turns = totals_result.one()

    by_event_type_result = await db.execute(
        select(UsageEvent.event_type, func.sum(UsageEvent.cost_usd)).where(*filters).group_by(UsageEvent.event_type)
    )
    cost_by_event_type = {event_type: float(cost) for event_type, cost in by_event_type_result.all()}

    by_subject_result = await db.execute(
        select(Subject.name, func.sum(UsageEvent.cost_usd))
        .join(Subject, Subject.id == UsageEvent.subject_id)
        .where(*filters)
        .group_by(Subject.name)
    )
    cost_by_subject = [(name, float(cost)) for name, cost in by_subject_result.all()]

    return UsageSummary(
        total_cost_usd=float(total_cost_usd),
        total_sessions=total_sessions,
        total_turns=total_turns,
        cost_by_event_type=cost_by_event_type,
        cost_by_subject=cost_by_subject,
    )


async def get_session_list(db: AsyncSession, period: Period, limit: int, offset: int) -> list[SessionSummary]:
    """Sessions created within `period`, newest first, with their subject
    name and aggregated cost/turn/event counts (an outer join, so a session
    with zero usage_events still appears, with zeroes)."""
    cutoff = _period_cutoff(period)
    filters = [] if cutoff is None else [ChatSession.created_at >= cutoff]

    result = await db.execute(
        select(
            ChatSession.id,
            Subject.name,
            ChatSession.created_at,
            func.coalesce(func.sum(UsageEvent.cost_usd), 0.0),
            func.count(case((UsageEvent.event_type == "generation", 1))),
            func.count(UsageEvent.id),
        )
        .join(Subject, Subject.id == ChatSession.subject_id)
        .outerjoin(UsageEvent, UsageEvent.session_id == ChatSession.id)
        .where(*filters)
        .group_by(ChatSession.id, Subject.name, ChatSession.created_at)
        .order_by(ChatSession.created_at.desc())
        .limit(limit)
        .offset(offset)
    )

    return [
        SessionSummary(
            session_id=session_id,
            subject_name=subject_name,
            created_at=created_at,
            cost_usd=float(cost_usd),
            turn_count=turn_count,
            event_count=event_count,
        )
        for session_id, subject_name, created_at, cost_usd, turn_count, event_count in result.all()
    ]


async def get_retrieval_health(db: AsyncSession, period: Period) -> RetrievalHealth:
    """Retrieval-quality metrics aggregated from event_type="retrieval" rows."""
    cutoff = _period_cutoff(period)
    filters = [UsageEvent.event_type == "retrieval"]
    if cutoff is not None:
        filters.append(UsageEvent.created_at >= cutoff)

    score = UsageEvent.retrieval_best_score
    result = await db.execute(
        select(
            func.count(),
            func.coalesce(func.sum(case((UsageEvent.guardrail_triggered.is_(True), 1), else_=0)), 0),
            func.avg(score),
            func.count(case((score >= 0.8, 1))),
            func.count(case(((score >= 0.65) & (score < 0.8), 1))),
            func.count(case(((score >= 0.55) & (score < 0.65), 1))),
            func.count(case((score < 0.55, 1))),
        ).where(*filters)
    )
    total, triggered, avg_score, excellent, good, fair, poor = result.one()

    return RetrievalHealth(
        total_queries=total,
        guardrail_triggered_count=triggered,
        avg_best_score=None if avg_score is None else float(avg_score),
        excellent_count=excellent,
        good_count=good,
        fair_count=fair,
        poor_count=poor,
    )
