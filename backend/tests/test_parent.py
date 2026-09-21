from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Coroutine, Iterator
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.main import app
from app.storage import create_session, get_db, get_or_create_subject
from app.storage.models import Base, UsageEvent

Seed = Callable[[AsyncSession], Coroutine[Any, Any, None]]


@pytest.fixture
def make_client() -> Iterator[Callable[[Seed | None], TestClient]]:
    """Builds a TestClient backed by a fresh in-memory SQLite engine, whose
    optional `seed` coroutine runs once, on the first request the client
    makes -- there's no HTTP path that writes usage_events, so tests seed
    the DB directly instead (mirrors override_get_db in test_api.py, plus a
    one-shot seed hook run on the same event loop as the requests)."""

    def _make(seed: Seed | None = None) -> TestClient:
        engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        seeded = False

        async def _get_db() -> AsyncIterator[AsyncSession]:
            nonlocal seeded
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            async with session_factory() as session:
                if not seeded and seed is not None:
                    await seed(session)
                    await session.commit()
                    seeded = True
                yield session
                await session.commit()

        app.dependency_overrides[get_db] = _get_db
        return TestClient(app)

    yield _make
    app.dependency_overrides.pop(get_db, None)


async def _seed_basic_usage(db: AsyncSession) -> None:
    """One subject, one session, one generation event (0.02) and one
    embedding event (0.001)."""
    subject = await get_or_create_subject(db, "French")
    session = await create_session(db, subject.id)
    db.add(
        UsageEvent(
            session_id=session.id,
            subject_id=subject.id,
            event_type="generation",
            provider="anthropic",
            model="claude",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.02,
        )
    )
    db.add(
        UsageEvent(
            session_id=session.id,
            subject_id=subject.id,
            event_type="embedding",
            provider="anthropic",
            model="embed",
            input_tokens=10,
            output_tokens=0,
            cost_usd=0.001,
        )
    )


@pytest.mark.parametrize(
    "path",
    ["/api/parent/usage/summary", "/api/parent/usage/sessions", "/api/parent/retrieval-health", "/api/parent/settings"],
)
def test_parent_endpoints_403_without_role_header(
    make_client: Callable[[Seed | None], TestClient], path: str
) -> None:
    client = make_client(None)

    response = client.get(path)

    assert response.status_code == 403


@pytest.mark.parametrize(
    "path",
    ["/api/parent/usage/summary", "/api/parent/usage/sessions", "/api/parent/retrieval-health", "/api/parent/settings"],
)
def test_parent_endpoints_403_for_student_role(
    make_client: Callable[[Seed | None], TestClient], path: str
) -> None:
    client = make_client(None)

    response = client.get(path, headers={"X-Tutor-Role": "student"})

    assert response.status_code == 403


def test_usage_summary_aggregates_cost_for_parent_role(
    make_client: Callable[[Seed | None], TestClient],
) -> None:
    client = make_client(_seed_basic_usage)

    response = client.get("/api/parent/usage/summary", params={"period": "all"}, headers={"X-Tutor-Role": "parent"})

    assert response.status_code == 200
    body = response.json()
    assert body["total_cost_usd"] == pytest.approx(0.021)
    assert body["total_sessions"] == 1
    assert body["total_turns"] == 1  # only "generation" events count as a turn
    assert body["cost_by_event_type"]["generation"] == pytest.approx(0.02)
    assert body["cost_by_event_type"]["embedding"] == pytest.approx(0.001)
    assert body["cost_by_event_type"]["retrieval"] == 0.0
    assert body["cost_by_subject"] == [{"subject_name": "French", "cost_usd": pytest.approx(0.021)}]


async def _seed_usage_across_periods(db: AsyncSession) -> None:
    """One event from just now, one from 10 days ago -- inside "week" for
    only the first, inside "month" for both."""
    subject = await get_or_create_subject(db, "French")
    session = await create_session(db, subject.id)
    db.add(
        UsageEvent(
            session_id=session.id,
            subject_id=subject.id,
            event_type="generation",
            provider="anthropic",
            model="claude",
            cost_usd=0.01,
            created_at=datetime.now(timezone.utc),
        )
    )
    db.add(
        UsageEvent(
            session_id=session.id,
            subject_id=subject.id,
            event_type="generation",
            provider="anthropic",
            model="claude",
            cost_usd=0.05,
            created_at=datetime.now(timezone.utc) - timedelta(days=10),
        )
    )


def test_usage_summary_period_filtering(make_client: Callable[[Seed | None], TestClient]) -> None:
    client = make_client(_seed_usage_across_periods)

    week = client.get(
        "/api/parent/usage/summary", params={"period": "week"}, headers={"X-Tutor-Role": "parent"}
    ).json()
    all_time = client.get(
        "/api/parent/usage/summary", params={"period": "all"}, headers={"X-Tutor-Role": "parent"}
    ).json()

    assert week["total_cost_usd"] == pytest.approx(0.01)
    assert all_time["total_cost_usd"] == pytest.approx(0.06)


async def _seed_two_sessions(db: AsyncSession) -> None:
    subject = await get_or_create_subject(db, "French")
    older = await create_session(db, subject.id)
    older.created_at = datetime.now(timezone.utc) - timedelta(hours=2)
    newer = await create_session(db, subject.id)
    db.add(
        UsageEvent(
            session_id=older.id,
            subject_id=subject.id,
            event_type="generation",
            provider="anthropic",
            model="claude",
            cost_usd=0.01,
        )
    )
    db.add(
        UsageEvent(
            session_id=newer.id,
            subject_id=subject.id,
            event_type="generation",
            provider="anthropic",
            model="claude",
            cost_usd=0.02,
        )
    )


def test_usage_sessions_lists_newest_first_with_aggregated_cost(
    make_client: Callable[[Seed | None], TestClient],
) -> None:
    client = make_client(_seed_two_sessions)

    response = client.get(
        "/api/parent/usage/sessions", params={"period": "all"}, headers={"X-Tutor-Role": "parent"}
    )

    assert response.status_code == 200
    sessions = response.json()
    assert len(sessions) == 2
    assert sessions[0]["cost_usd"] == pytest.approx(0.02)  # newer session first
    assert sessions[1]["cost_usd"] == pytest.approx(0.01)
    assert all(s["subject_name"] == "French" for s in sessions)


def test_usage_sessions_respects_limit_and_offset(
    make_client: Callable[[Seed | None], TestClient],
) -> None:
    client = make_client(_seed_two_sessions)

    first_page = client.get(
        "/api/parent/usage/sessions",
        params={"period": "all", "limit": 1, "offset": 0},
        headers={"X-Tutor-Role": "parent"},
    ).json()
    second_page = client.get(
        "/api/parent/usage/sessions",
        params={"period": "all", "limit": 1, "offset": 1},
        headers={"X-Tutor-Role": "parent"},
    ).json()

    assert len(first_page) == 1
    assert len(second_page) == 1
    assert first_page[0]["session_id"] != second_page[0]["session_id"]


async def _seed_retrieval_scores(db: AsyncSession) -> None:
    subject = await get_or_create_subject(db, "French")
    session = await create_session(db, subject.id)
    scores_and_triggers = [
        (0.9, False),  # excellent
        (0.7, False),  # good
        (0.6, False),  # fair
        (0.4, True),  # poor, guardrail triggered
    ]
    for score, triggered in scores_and_triggers:
        db.add(
            UsageEvent(
                session_id=session.id,
                subject_id=subject.id,
                event_type="retrieval",
                provider="anthropic",
                model="embed",
                cost_usd=0.0,
                retrieval_best_score=score,
                retrieval_chunk_count=3,
                guardrail_triggered=triggered,
            )
        )


def test_retrieval_health_reports_score_distribution(
    make_client: Callable[[Seed | None], TestClient],
) -> None:
    client = make_client(_seed_retrieval_scores)

    response = client.get(
        "/api/parent/retrieval-health", params={"period": "all"}, headers={"X-Tutor-Role": "parent"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["total_queries"] == 4
    assert body["guardrail_triggered_count"] == 1
    assert body["guardrail_trigger_rate"] == pytest.approx(0.25)
    assert body["avg_best_score"] == pytest.approx((0.9 + 0.7 + 0.6 + 0.4) / 4)
    assert body["score_distribution"] == {"excellent": 1, "good": 1, "fair": 1, "poor": 1}


def test_retrieval_health_with_no_data_returns_zeroed_response(
    make_client: Callable[[Seed | None], TestClient],
) -> None:
    client = make_client(None)

    response = client.get(
        "/api/parent/retrieval-health", params={"period": "all"}, headers={"X-Tutor-Role": "parent"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["total_queries"] == 0
    assert body["guardrail_trigger_rate"] == 0.0
    assert body["avg_best_score"] is None


def test_parent_settings_returns_hardcoded_response(
    make_client: Callable[[Seed | None], TestClient],
) -> None:
    client = make_client(None)

    response = client.get("/api/parent/settings", headers={"X-Tutor-Role": "parent"})

    assert response.status_code == 200
    assert response.json() == {"web_search_enabled": False}
