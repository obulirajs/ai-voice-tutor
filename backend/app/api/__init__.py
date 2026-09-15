"""API module — FastAPI routers exposed to app.main.

Depends on app.models', app.storage's, and app.orchestration's interfaces,
never on a concrete adapter — see technical-design.md's dependency-inversion
convention.
"""

from __future__ import annotations

from .chat import router as chat_router
from .documents import router as documents_router

__all__ = ["chat_router", "documents_router"]
