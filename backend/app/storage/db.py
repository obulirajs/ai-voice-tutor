from __future__ import annotations

import os
from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from .models import Base

_DEFAULT_DATABASE_URL = "sqlite+aiosqlite:///./tutor.db"

engine: AsyncEngine = create_async_engine(os.getenv("DATABASE_URL", _DEFAULT_DATABASE_URL))
_session_factory = async_sessionmaker(engine, expire_on_commit=False)


async def init_db(target_engine: AsyncEngine | None = None) -> None:
    """Create all tables. Called from app.main's startup lifespan."""
    active_engine = target_engine or engine
    async with active_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding a session; commits on success, rolls back on error."""
    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
