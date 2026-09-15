"""Orchestration module — the core harness.

Public surface: handle_turn(), the session-state loop that loads-or-creates
a session, appends the turn to history, calls the Model Provider, estimates
cost, and logs the usage_event. Depends on app.models' and app.storage's
interfaces only, never a concrete adapter — see technical-design.md's
dependency-inversion convention.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import cast

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Message as ProviderMessage
from app.models import ModelProvider
from app.storage import (
    ChatSession,
    append_message,
    create_session,
    get_history,
    get_or_create_subject,
    get_session,
    log_usage_event,
)

from .pricing import estimate_cost_usd

__all__ = ["TurnResult", "handle_turn"]

DEFAULT_SUBJECT = "general"


@dataclass(frozen=True)
class TurnResult:
    reply: str
    session_id: str


async def handle_turn(
    db: AsyncSession,
    provider: ModelProvider,
    *,
    session_id: str | None,
    user_message: str,
    subject: str = DEFAULT_SUBJECT,
) -> TurnResult:
    """Run one conversational turn: history in, Model Provider call, usage logged."""
    session = await _load_or_create_session(db, session_id, subject)

    prior_messages = await get_history(db, session.id)
    await append_message(db, session.id, role="user", content=user_message)

    provider_messages: list[ProviderMessage] = [
        cast(ProviderMessage, {"role": m.role, "content": m.content}) for m in prior_messages
    ]
    provider_messages.append(cast(ProviderMessage, {"role": "user", "content": user_message}))

    started = time.perf_counter()
    response = provider.generate(provider_messages)
    latency_ms = (time.perf_counter() - started) * 1000

    await append_message(db, session.id, role="assistant", content=response.content)

    cost_usd = estimate_cost_usd(provider.model_name, response.usage.input_tokens, response.usage.output_tokens)
    await log_usage_event(
        db,
        session_id=session.id,
        subject_id=session.subject_id,
        provider=provider.provider_name,
        model=provider.model_name,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
    )

    return TurnResult(reply=response.content, session_id=session.id)


async def _load_or_create_session(db: AsyncSession, session_id: str | None, subject: str) -> ChatSession:
    if session_id is not None:
        session = await get_session(db, session_id)
        if session is None:
            raise LookupError(f"Unknown session_id: {session_id!r}")
        return session

    subject_row = await get_or_create_subject(db, subject)
    return await create_session(db, subject_row.id)
