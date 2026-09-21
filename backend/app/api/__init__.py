"""API module — FastAPI routers exposed to app.main.

Depends on app.models', app.storage's, and app.orchestration's interfaces,
never on a concrete adapter — see technical-design.md's dependency-inversion
convention.
"""

from __future__ import annotations

from .chat import router as chat_router
from .documents import router as documents_router
from .parent import router as parent_router
from .sessions import router as sessions_router
from .subjects import router as subjects_router
from .tts import router as tts_router
from .voice import router as voice_router

__all__ = [
    "chat_router",
    "documents_router",
    "parent_router",
    "sessions_router",
    "subjects_router",
    "tts_router",
    "voice_router",
]
