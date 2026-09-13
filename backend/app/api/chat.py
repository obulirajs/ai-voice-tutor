from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.models import Message, ModelProvider, get_provider

router = APIRouter()


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, provider: ModelProvider = Depends(get_provider)) -> ChatResponse:  # noqa: B008
    messages: list[Message] = [{"role": "user", "content": request.message}]
    response = provider.generate(messages)
    return ChatResponse(reply=response.content)
