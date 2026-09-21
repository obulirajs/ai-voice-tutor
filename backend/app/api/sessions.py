from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.storage import (
    get_db,
    get_history,
    get_session,
    get_session_usage,
    list_sessions_for_subject,
)

router = APIRouter()


class UsageEventResponse(BaseModel):
    event_type: str
    provider: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float
    created_at: datetime


class SessionUsageResponse(BaseModel):
    session_id: str
    total_cost_usd: float
    total_input_tokens: int
    total_output_tokens: int
    event_count: int
    events: list[UsageEventResponse]


@router.get("/sessions/{session_id}/usage", response_model=SessionUsageResponse)
async def session_usage(
    session_id: str,
    db: AsyncSession = Depends(get_db),  # noqa: B008
) -> SessionUsageResponse:
    session = await get_session(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Unknown session_id: {session_id!r}")

    events = await get_session_usage(db, session_id)

    return SessionUsageResponse(
        session_id=session_id,
        total_cost_usd=sum(event.cost_usd for event in events),
        total_input_tokens=sum(event.input_tokens or 0 for event in events),
        total_output_tokens=sum(event.output_tokens or 0 for event in events),
        event_count=len(events),
        events=[
            UsageEventResponse(
                event_type=event.event_type,
                provider=event.provider,
                model=event.model,
                input_tokens=event.input_tokens,
                output_tokens=event.output_tokens,
                cost_usd=event.cost_usd,
                created_at=event.created_at,
            )
            for event in events
        ],
    )


class SessionPreviewResponse(BaseModel):
    session_id: str
    subject_id: int
    message_count: int
    last_message_preview: str
    last_active: datetime
    teaching_mode: str


@router.get("/subjects/{subject_id}/sessions", response_model=list[SessionPreviewResponse])
async def list_sessions(
    subject_id: int,
    limit: int = 20,
    db: AsyncSession = Depends(get_db),  # noqa: B008
) -> list[SessionPreviewResponse]:
    previews = await list_sessions_for_subject(db, subject_id, limit=limit)
    return [
        SessionPreviewResponse(
            session_id=p.session_id,
            subject_id=p.subject_id,
            message_count=p.message_count,
            last_message_preview=p.last_message_preview,
            last_active=p.last_active,
            teaching_mode=p.teaching_mode,
        )
        for p in previews
    ]


class MessageResponse(BaseModel):
    role: str
    content: str


@router.get("/sessions/{session_id}/messages", response_model=list[MessageResponse])
async def get_session_messages(
    session_id: str,
    db: AsyncSession = Depends(get_db),  # noqa: B008
) -> list[MessageResponse]:
    session = await get_session(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Unknown session_id: {session_id!r}")

    messages = await get_history(db, session_id)
    return [MessageResponse(role=m.role, content=m.content) for m in messages]
