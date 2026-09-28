from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.api import diagram as diagram_module
from app.main import app
from app.models import (
    Message,
    ModelProvider,
    ModelResponse,
    Usage,
    get_diagram_model_provider,
)
from app.storage import get_db
from app.storage.models import Base, UsageEvent

SessionFactory = async_sessionmaker[AsyncSession]


class FakeDiagramProvider(ModelProvider):
    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.call_count = 0
        self.received: list[Message] = []

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return "qwen2.5-coder:3b"

    async def generate(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
    ) -> ModelResponse:
        self.call_count += 1
        self.received = messages
        return ModelResponse(content=self._reply, usage=Usage(input_tokens=40, output_tokens=8))


@pytest.fixture
def override_get_db() -> Iterator[SessionFactory]:
    """Same pattern as test_api.py's fixture, but also yields the
    session_factory so a test can open its own session afterward to assert
    on rows the request wrote (e.g. a usage_events row). Table creation
    happens inside the override itself so it runs on the same event loop
    TestClient uses to dispatch requests -- aiosqlite connections can't
    cross event loops.
    """
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async def _get_db() -> AsyncIterator[AsyncSession]:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        async with session_factory() as session:
            yield session
            await session.commit()

    app.dependency_overrides[get_db] = _get_db
    yield session_factory
    app.dependency_overrides.pop(get_db, None)


@pytest.fixture(autouse=True)
def clear_diagram_cache() -> Iterator[None]:
    diagram_module._cache.clear()
    yield
    diagram_module._cache.clear()


def test_diagram_endpoint_returns_sanitized_mermaid(override_get_db: SessionFactory) -> None:
    fake = FakeDiagramProvider("flowchart TD\nA[Evaporation] --> B[Condensation]")
    app.dependency_overrides[get_diagram_model_provider] = lambda: fake
    client = TestClient(app)

    try:
        response = client.post("/api/diagram", json={"message_text": "Water evaporates, then condenses."})
    finally:
        app.dependency_overrides.pop(get_diagram_model_provider, None)

    assert response.status_code == 200
    body = response.json()
    assert body["mermaid"] == "flowchart TD\nA[Evaporation] --> B[Condensation]"
    assert body["reason"] is None
    assert fake.call_count == 1


def test_diagram_endpoint_returns_not_applicable_for_none(override_get_db: SessionFactory) -> None:
    fake = FakeDiagramProvider("NONE")
    app.dependency_overrides[get_diagram_model_provider] = lambda: fake
    client = TestClient(app)

    try:
        response = client.post("/api/diagram", json={"message_text": "Le chat means cat."})
    finally:
        app.dependency_overrides.pop(get_diagram_model_provider, None)

    assert response.status_code == 200
    body = response.json()
    assert body["mermaid"] is None
    assert body["reason"] == "not_applicable"


def test_diagram_endpoint_includes_previous_error_in_the_prompt(override_get_db: SessionFactory) -> None:
    fake = FakeDiagramProvider("flowchart TD\nA[Start] --> B[End]")
    app.dependency_overrides[get_diagram_model_provider] = lambda: fake
    client = TestClient(app)

    try:
        client.post(
            "/api/diagram",
            json={"message_text": "The process starts, then ends.", "previous_error": "Unexpected token"},
        )
    finally:
        app.dependency_overrides.pop(get_diagram_model_provider, None)

    assert "Unexpected token" in fake.received[0]["content"]


def test_diagram_endpoint_cache_hit_skips_a_second_provider_call(override_get_db: SessionFactory) -> None:
    fake = FakeDiagramProvider("flowchart TD\nA[Start] --> B[End]")
    app.dependency_overrides[get_diagram_model_provider] = lambda: fake
    client = TestClient(app)

    try:
        payload = {"message_text": "The process starts, then ends, repeatably."}
        first = client.post("/api/diagram", json=payload)
        second = client.post("/api/diagram", json=payload)
    finally:
        app.dependency_overrides.pop(get_diagram_model_provider, None)

    assert first.json() == second.json()
    assert fake.call_count == 1


def test_diagram_endpoint_disabled_provider_makes_no_call(override_get_db: SessionFactory) -> None:
    app.dependency_overrides[get_diagram_model_provider] = lambda: None
    client = TestClient(app)

    try:
        response = client.post("/api/diagram", json={"message_text": "Anything at all."})
    finally:
        app.dependency_overrides.pop(get_diagram_model_provider, None)

    assert response.status_code == 200
    body = response.json()
    assert body["mermaid"] is None
    assert body["reason"] == "disabled"


def test_diagram_endpoint_writes_a_diagram_usage_event(override_get_db: SessionFactory) -> None:
    fake = FakeDiagramProvider("flowchart TD\nA[Start] --> B[End]")
    app.dependency_overrides[get_diagram_model_provider] = lambda: fake
    client = TestClient(app)

    try:
        response = client.post("/api/diagram", json={"message_text": "The process starts, then ends."})
        assert response.status_code == 200
    finally:
        app.dependency_overrides.pop(get_diagram_model_provider, None)

    async def _fetch_diagram_events() -> list[UsageEvent]:
        async with override_get_db() as session:
            result = await session.execute(select(UsageEvent).where(UsageEvent.event_type == "diagram"))
            return list(result.scalars().all())

    events = asyncio.run(_fetch_diagram_events())
    assert len(events) == 1
    assert events[0].provider == "ollama"
    assert events[0].model == "qwen2.5-coder:3b"
    assert events[0].input_tokens == 40
    assert events[0].output_tokens == 8


def test_diagram_endpoint_disabled_provider_writes_no_usage_event(override_get_db: SessionFactory) -> None:
    app.dependency_overrides[get_diagram_model_provider] = lambda: None
    client = TestClient(app)

    try:
        client.post("/api/diagram", json={"message_text": "Anything at all."})
    finally:
        app.dependency_overrides.pop(get_diagram_model_provider, None)

    async def _fetch_diagram_events() -> list[UsageEvent]:
        async with override_get_db() as session:
            result = await session.execute(select(UsageEvent).where(UsageEvent.event_type == "diagram"))
            return list(result.scalars().all())

    events = asyncio.run(_fetch_diagram_events())
    assert events == []
