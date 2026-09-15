from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import ModelProvider, get_provider
from app.orchestration import handle_turn
from app.storage import get_db

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    reply: str
    session_id: str


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    provider: ModelProvider = Depends(get_provider),  # noqa: B008
    db: AsyncSession = Depends(get_db),  # noqa: B008
) -> ChatResponse:
    try:
        result = await handle_turn(
            db,
            provider,
            session_id=request.session_id,
            user_message=request.message,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return ChatResponse(reply=result.reply, session_id=result.session_id)
