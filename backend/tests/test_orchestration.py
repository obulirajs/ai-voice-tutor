from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.models import Message, ModelProvider, ModelResponse, Usage
from app.orchestration import handle_turn
from app.storage import get_history, get_or_create_subject
from app.storage.models import Base, UsageEvent

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


class FakeProvider(ModelProvider):
    def __init__(
        self,
        reply: str,
        *,
        provider: str = "anthropic",
        model: str = "claude-sonnet-5",
        input_tokens: int | None = 1000,
        output_tokens: int | None = 500,
    ) -> None:
        self._reply = reply
        self._provider = provider
        self._model = model
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self.received: list[Message] = []

    @property
    def provider_name(self) -> str:
        return self._provider

    @property
    def model_name(self) -> str:
        return self._model

    def generate(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse:
        self.received = messages
        return ModelResponse(
            content=self._reply,
            usage=Usage(input_tokens=self._input_tokens, output_tokens=self._output_tokens),
        )


def test_handle_turn_creates_session_when_none_given() -> None:
    async def scenario(db: AsyncSession) -> None:
        provider = FakeProvider(reply="Bonjour !")

        result = await handle_turn(db, provider, session_id=None, user_message="Bonjour")

        assert result.reply == "Bonjour !"
        assert result.session_id

    run(scenario)


def test_handle_turn_defaults_new_sessions_to_general_subject() -> None:
    async def scenario(db: AsyncSession) -> None:
        provider = FakeProvider(reply="ok")

        await handle_turn(db, provider, session_id=None, user_message="hi")

        subject = await get_or_create_subject(db, "general")
        assert subject.name == "general"

    run(scenario)


def test_handle_turn_appends_user_and_assistant_messages() -> None:
    async def scenario(db: AsyncSession) -> None:
        provider = FakeProvider(reply="Bonjour ! Ca va ?")

        result = await handle_turn(db, provider, session_id=None, user_message="Bonjour")

        history = await get_history(db, result.session_id)
        assert [(m.role, m.content) for m in history] == [
            ("user", "Bonjour"),
            ("assistant", "Bonjour ! Ca va ?"),
        ]

    run(scenario)


def test_handle_turn_continues_existing_session_with_prior_history() -> None:
    async def scenario(db: AsyncSession) -> None:
        provider = FakeProvider(reply="reply 1")
        first = await handle_turn(db, provider, session_id=None, user_message="turn 1")

        provider2 = FakeProvider(reply="reply 2")
        await handle_turn(db, provider2, session_id=first.session_id, user_message="turn 2")

        assert provider2.received == [
            {"role": "user", "content": "turn 1"},
            {"role": "assistant", "content": "reply 1"},
            {"role": "user", "content": "turn 2"},
        ]

    run(scenario)


def test_handle_turn_raises_lookup_error_for_unknown_session() -> None:
    async def scenario(db: AsyncSession) -> None:
        provider = FakeProvider(reply="unused")

        with pytest.raises(LookupError):
            await handle_turn(db, provider, session_id="does-not-exist", user_message="hi")

    run(scenario)


def test_handle_turn_logs_usage_event_with_estimated_cost() -> None:
    async def scenario(db: AsyncSession) -> None:
        provider = FakeProvider(
            reply="ok",
            provider="anthropic",
            model="claude-sonnet-5",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
        )

        result = await handle_turn(db, provider, session_id=None, user_message="hi")

        rows = (await db.execute(select(UsageEvent).where(UsageEvent.session_id == result.session_id))).scalars().all()

        assert len(rows) == 1
        event = rows[0]
        assert event.provider == "anthropic"
        assert event.model == "claude-sonnet-5"
        assert event.input_tokens == 1_000_000
        assert event.output_tokens == 1_000_000
        # $2/MTok in + $10/MTok out, at 1M tokens each => $2 + $10 = $12
        assert event.cost_usd == pytest.approx(12.0)

    run(scenario)


def test_handle_turn_costs_zero_for_ollama() -> None:
    async def scenario(db: AsyncSession) -> None:
        provider = FakeProvider(
            reply="ok",
            provider="ollama",
            model="llama3.2:1b",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
        )

        result = await handle_turn(db, provider, session_id=None, user_message="hi")

        rows = (await db.execute(select(UsageEvent).where(UsageEvent.session_id == result.session_id))).scalars().all()

        assert rows[0].cost_usd == 0.0

    run(scenario)
