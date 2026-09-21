"""Parent-only usage/cost/retrieval-health endpoints.

technical-design.md: the parent role sees "per-request/session/weekly cost,
token counts, retrieval health" -- data the student role never gets, gated
on the backend (development-plan.md Phase 6 exit: "the parent role changes
what the API returns, not just what's displayed"). Every endpoint here
depends on get_role() and 403s outright for anything but Role.PARENT --
this is a visibility gate for a single-household deployment, not real auth
(see api/role.py).
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.storage import (
    Period,
    get_db,
    get_retrieval_health,
    get_session_list,
    get_usage_summary,
)

from .role import Role, get_role

router = APIRouter()


def _require_parent(role: Role) -> None:
    if role != Role.PARENT:
        raise HTTPException(status_code=403, detail="This endpoint is parent-only.")


class CostByEventType(BaseModel):
    generation: float = 0.0
    embedding: float = 0.0
    retrieval: float = 0.0
    asr: float = 0.0
    tts: float = 0.0


class CostBySubject(BaseModel):
    subject_name: str
    cost_usd: float


class UsageSummaryResponse(BaseModel):
    total_cost_usd: float
    total_sessions: int
    total_turns: int
    cost_by_event_type: CostByEventType
    cost_by_subject: list[CostBySubject]


@router.get("/parent/usage/summary", response_model=UsageSummaryResponse)
async def usage_summary(
    period: Period = "week",
    role: Role = Depends(get_role),  # noqa: B008
    db: AsyncSession = Depends(get_db),  # noqa: B008
) -> UsageSummaryResponse:
    _require_parent(role)
    summary = await get_usage_summary(db, period)

    return UsageSummaryResponse(
        total_cost_usd=summary.total_cost_usd,
        total_sessions=summary.total_sessions,
        total_turns=summary.total_turns,
        cost_by_event_type=CostByEventType(**summary.cost_by_event_type),
        cost_by_subject=[
            CostBySubject(subject_name=name, cost_usd=cost) for name, cost in summary.cost_by_subject
        ],
    )


class SessionSummaryResponse(BaseModel):
    session_id: str
    subject_name: str
    created_at: datetime
    cost_usd: float
    turn_count: int
    event_count: int


@router.get("/parent/usage/sessions", response_model=list[SessionSummaryResponse])
async def usage_sessions(
    period: Period = "week",
    limit: int = 20,
    offset: int = 0,
    role: Role = Depends(get_role),  # noqa: B008
    db: AsyncSession = Depends(get_db),  # noqa: B008
) -> list[SessionSummaryResponse]:
    _require_parent(role)
    sessions = await get_session_list(db, period, limit, offset)

    return [
        SessionSummaryResponse(
            session_id=s.session_id,
            subject_name=s.subject_name,
            created_at=s.created_at,
            cost_usd=s.cost_usd,
            turn_count=s.turn_count,
            event_count=s.event_count,
        )
        for s in sessions
    ]


class ScoreDistribution(BaseModel):
    excellent: int
    good: int
    fair: int
    poor: int


class RetrievalHealthResponse(BaseModel):
    total_queries: int
    guardrail_triggered_count: int
    guardrail_trigger_rate: float
    avg_best_score: float | None
    score_distribution: ScoreDistribution


@router.get("/parent/retrieval-health", response_model=RetrievalHealthResponse)
async def retrieval_health(
    period: Period = "week",
    role: Role = Depends(get_role),  # noqa: B008
    db: AsyncSession = Depends(get_db),  # noqa: B008
) -> RetrievalHealthResponse:
    _require_parent(role)
    health = await get_retrieval_health(db, period)

    trigger_rate = health.guardrail_triggered_count / health.total_queries if health.total_queries else 0.0

    return RetrievalHealthResponse(
        total_queries=health.total_queries,
        guardrail_triggered_count=health.guardrail_triggered_count,
        guardrail_trigger_rate=trigger_rate,
        avg_best_score=health.avg_best_score,
        score_distribution=ScoreDistribution(
            excellent=health.excellent_count,
            good=health.good_count,
            fair=health.fair_count,
            poor=health.poor_count,
        ),
    )


class ParentSettingsResponse(BaseModel):
    # Hardcoded for now -- Phase 6.5 makes this a real, persisted setting.
    web_search_enabled: bool = False


@router.get("/parent/settings", response_model=ParentSettingsResponse)
async def parent_settings(
    role: Role = Depends(get_role),  # noqa: B008
) -> ParentSettingsResponse:
    _require_parent(role)
    return ParentSettingsResponse()
