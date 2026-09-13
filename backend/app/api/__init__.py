"""API module — FastAPI routers exposed to app.main.

Depends on app.models' interface (ModelProvider, get_provider), never on a
concrete adapter — see technical-design.md's dependency-inversion convention.
"""

from __future__ import annotations

from .chat import router as chat_router

__all__ = ["chat_router"]
