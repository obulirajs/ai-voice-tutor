from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.storage import (
    append_message,
    create_document,
    create_session,
    get_history,
    get_or_create_subject,
    get_session,
    get_session_usage,
    get_subject,
    init_db,
    list_sessions_for_subject,
    list_subjects,
    log_usage_event,
)
from app.storage.models import Base

Scenario = Callable[[AsyncSession], Coroutine[Any, Any, None]]


def run(scenario: Scenario) -> None:
    """Run one test scenario against a fresh in-memory SQLite engine, then dispose it."""

    async def wrapper() -> None:
        engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        async with session_factory() as db:
            await scenario(db)

        await engine.dispose()

    asyncio.run(wrapper())


def test_init_db_creates_tables() -> None:
    async def scenario() -> None:
        engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        await init_db(engine)  # no prior create_all — this call must create the schema itself

        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        async with session_factory() as db:
            # Would raise if the subjects table doesn't exist.
            subject = await get_or_create_subject(db, "french")
            assert subject.id is not None

        await engine.dispose()

    asyncio.run(scenario())


def test_get_or_create_subject_is_idempotent() -> None:
    async def scenario(db: AsyncSession) -> None:
        first = await get_or_create_subject(db, "french")
        second = await get_or_create_subject(db, "french")

        assert first.id == second.id
        assert first.name == "french"

    run(scenario)


def test_get_or_create_subject_distinguishes_names() -> None:
    async def scenario(db: AsyncSession) -> None:
        french = await get_or_create_subject(db, "french")
        maths = await get_or_create_subject(db, "maths")

        assert french.id != maths.id

    run(scenario)


def test_create_and_get_session_round_trips() -> None:
    async def scenario(db: AsyncSession) -> None:
        subject = await get_or_create_subject(db, "french")

        created = await create_session(db, subject.id)
        fetched = await get_session(db, created.id)

        assert fetched is not None
        assert fetched.id == created.id
        assert fetched.subject_id == subject.id

    run(scenario)


def test_get_session_returns_none_for_unknown_id() -> None:
    async def scenario(db: AsyncSession) -> None:
        assert await get_session(db, "does-not-exist") is None

    run(scenario)


def test_append_message_and_get_history_preserves_order() -> None:
    async def scenario(db: AsyncSession) -> None:
        subject = await get_or_create_subject(db, "french")
        session = await create_session(db, subject.id)

        await append_message(db, session.id, role="user", content="Bonjour")
        await append_message(db, session.id, role="assistant", content="Bonjour ! Ca va ?")
        await append_message(db, session.id, role="user", content="Bien, merci")

        history = await get_history(db, session.id)

        assert [m.role for m in history] == ["user", "assistant", "user"]
        assert [m.content for m in history] == ["Bonjour", "Bonjour ! Ca va ?", "Bien, merci"]

    run(scenario)


def test_get_history_is_scoped_to_session() -> None:
    async def scenario(db: AsyncSession) -> None:
        subject = await get_or_create_subject(db, "french")
        session_a = await create_session(db, subject.id)
        session_b = await create_session(db, subject.id)

        await append_message(db, session_a.id, role="user", content="from A")
        await append_message(db, session_b.id, role="user", content="from B")

        history_a = await get_history(db, session_a.id)

        assert [m.content for m in history_a] == ["from A"]

    run(scenario)


def test_log_usage_event_stores_all_fields() -> None:
    async def scenario(db: AsyncSession) -> None:
        subject = await get_or_create_subject(db, "french")
        session = await create_session(db, subject.id)

        event = await log_usage_event(
            db,
            session_id=session.id,
            subject_id=subject.id,
            provider="anthropic",
            model="claude-sonnet-5",
            input_tokens=100,
            output_tokens=40,
            cost_usd=0.0006,
            latency_ms=250.5,
        )

        assert event.id is not None
        assert event.session_id == session.id
        assert event.subject_id == subject.id
        assert event.provider == "anthropic"
        assert event.model == "claude-sonnet-5"
        assert event.input_tokens == 100
        assert event.output_tokens == 40
        assert event.cost_usd == 0.0006
        assert event.latency_ms == 250.5

    run(scenario)


def test_log_usage_event_defaults_event_type_to_generation() -> None:
    async def scenario(db: AsyncSession) -> None:
        subject = await get_or_create_subject(db, "french")
        session = await create_session(db, subject.id)

        event = await log_usage_event(
            db,
            session_id=session.id,
            subject_id=subject.id,
            provider="anthropic",
            model="claude-sonnet-5",
            input_tokens=100,
            output_tokens=40,
            cost_usd=0.0006,
        )

        assert event.event_type == "generation"
        assert event.retrieval_best_score is None
        assert event.retrieval_chunk_count is None
        assert event.guardrail_triggered is None

    run(scenario)


def test_log_usage_event_stores_retrieval_fields() -> None:
    async def scenario(db: AsyncSession) -> None:
        subject = await get_or_create_subject(db, "french")
        session = await create_session(db, subject.id)

        event = await log_usage_event(
            db,
            session_id=session.id,
            subject_id=subject.id,
            provider="ollama",
            model="nomic-embed-text",
            input_tokens=None,
            output_tokens=None,
            cost_usd=0.0,
            event_type="retrieval",
            retrieval_best_score=0.42,
            retrieval_chunk_count=3,
            guardrail_triggered=False,
        )

        assert event.event_type == "retrieval"
        assert event.retrieval_best_score == pytest.approx(0.42)
        assert event.retrieval_chunk_count == 3
        assert event.guardrail_triggered is False

    run(scenario)


def test_get_subject_returns_row_by_id() -> None:
    async def scenario(db: AsyncSession) -> None:
        created = await get_or_create_subject(db, "french")

        fetched = await get_subject(db, created.id)

        assert fetched is not None
        assert fetched.id == created.id
        assert fetched.name == "french"

    run(scenario)


def test_get_subject_returns_none_for_unknown_id() -> None:
    async def scenario(db: AsyncSession) -> None:
        assert await get_subject(db, 999) is None

    run(scenario)


def test_log_usage_event_allows_zero_cost_for_ollama() -> None:
    async def scenario(db: AsyncSession) -> None:
        subject = await get_or_create_subject(db, "french")
        session = await create_session(db, subject.id)

        event = await log_usage_event(
            db,
            session_id=session.id,
            subject_id=subject.id,
            provider="ollama",
            model="llama3.2:1b",
            input_tokens=100,
            output_tokens=40,
            cost_usd=0.0,
            latency_ms=None,
        )

        assert event.cost_usd == 0.0
        assert event.latency_ms is None

    run(scenario)


def test_list_subjects_reports_zero_documents_for_new_subject() -> None:
    async def scenario(db: AsyncSession) -> None:
        await get_or_create_subject(db, "french")

        rows = await list_subjects(db)

        assert len(rows) == 1
        subject, count = rows[0]
        assert subject.name == "french"
        assert count == 0

    run(scenario)


def test_list_subjects_counts_documents_per_subject() -> None:
    async def scenario(db: AsyncSession) -> None:
        french = await get_or_create_subject(db, "french")
        await get_or_create_subject(db, "maths")

        await create_document(
            db, subject_id=french.id, filename="ch1.pdf", collection="subject_1", page_count=5
        )
        await create_document(
            db, subject_id=french.id, filename="ch2.pdf", collection="subject_1", page_count=3
        )

        rows = await list_subjects(db)
        counts_by_name = {subject.name: count for subject, count in rows}

        assert counts_by_name == {"french": 2, "maths": 0}

    run(scenario)


def test_get_session_usage_returns_events_in_order() -> None:
    async def scenario(db: AsyncSession) -> None:
        subject = await get_or_create_subject(db, "french")
        session = await create_session(db, subject.id)

        await log_usage_event(
            db,
            session_id=session.id,
            subject_id=subject.id,
            provider="anthropic",
            model="claude-sonnet-5",
            input_tokens=100,
            output_tokens=40,
            cost_usd=0.001,
            event_type="generation",
        )
        await log_usage_event(
            db,
            session_id=session.id,
            subject_id=subject.id,
            provider="ollama",
            model="nomic-embed-text",
            input_tokens=20,
            output_tokens=None,
            cost_usd=0.0,
            event_type="embedding",
        )

        events = await get_session_usage(db, session.id)

        assert [e.event_type for e in events] == ["generation", "embedding"]

    run(scenario)


def test_get_session_usage_is_scoped_to_session() -> None:
    async def scenario(db: AsyncSession) -> None:
        subject = await get_or_create_subject(db, "french")
        session_a = await create_session(db, subject.id)
        session_b = await create_session(db, subject.id)

        await log_usage_event(
            db,
            session_id=session_a.id,
            subject_id=subject.id,
            provider="anthropic",
            model="claude-sonnet-5",
            input_tokens=100,
            output_tokens=40,
            cost_usd=0.001,
        )
        await log_usage_event(
            db,
            session_id=session_b.id,
            subject_id=subject.id,
            provider="anthropic",
            model="claude-sonnet-5",
            input_tokens=200,
            output_tokens=80,
            cost_usd=0.002,
        )

        events_a = await get_session_usage(db, session_a.id)

        assert len(events_a) == 1
        assert events_a[0].input_tokens == 100

    run(scenario)


def test_get_session_usage_returns_empty_list_for_unknown_session() -> None:
    async def scenario(db: AsyncSession) -> None:
        assert await get_session_usage(db, "does-not-exist") == []

    run(scenario)


# --- list_sessions_for_subject (conversation history sidebar) -------------


def test_list_sessions_for_subject_returns_preview_data() -> None:
    async def scenario(db: AsyncSession) -> None:
        subject = await get_or_create_subject(db, "french")
        session = await create_session(db, subject.id, mode="teacher")

        await append_message(db, session.id, role="user", content="Bonjour" * 20)  # > 80 chars
        await append_message(db, session.id, role="assistant", content="Salut !")

        previews = await list_sessions_for_subject(db, subject.id)

        assert len(previews) == 1
        preview = previews[0]
        assert preview.session_id == session.id
        assert preview.subject_id == subject.id
        assert preview.message_count == 2
        assert preview.last_message_preview == ("Bonjour" * 20)[:80]
        assert preview.teaching_mode == "teacher"

    run(scenario)


def test_list_sessions_for_subject_orders_newest_first() -> None:
    async def scenario(db: AsyncSession) -> None:
        subject = await get_or_create_subject(db, "french")
        older = await create_session(db, subject.id)
        await append_message(db, older.id, role="user", content="first")

        newer = await create_session(db, subject.id)
        await append_message(db, newer.id, role="user", content="second")

        previews = await list_sessions_for_subject(db, subject.id)

        assert [p.session_id for p in previews] == [newer.id, older.id]

    run(scenario)


def test_list_sessions_for_subject_excludes_empty_sessions() -> None:
    async def scenario(db: AsyncSession) -> None:
        subject = await get_or_create_subject(db, "french")
        await create_session(db, subject.id)  # never gets a message

        with_message = await create_session(db, subject.id)
        await append_message(db, with_message.id, role="user", content="hi")

        previews = await list_sessions_for_subject(db, subject.id)

        assert [p.session_id for p in previews] == [with_message.id]

    run(scenario)


def test_list_sessions_for_subject_returns_empty_for_subject_with_no_sessions() -> None:
    async def scenario(db: AsyncSession) -> None:
        subject = await get_or_create_subject(db, "french")

        assert await list_sessions_for_subject(db, subject.id) == []

    run(scenario)
