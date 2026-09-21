from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import (
    chat_router,
    documents_router,
    parent_router,
    sessions_router,
    subjects_router,
    tts_router,
    voice_router,
)
from app.storage import init_db

# Load the .env file
load_dotenv()

# Vite dev server origins -- see frontend/vite.config.ts's dev proxy, which
# forwards /api and /health to this app so CORS shouldn't matter in practice,
# but this covers direct calls (e.g. a future non-proxied deployment).
_DEV_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
    await init_db()
    yield


app = FastAPI(title="Project Tutor", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_DEV_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(chat_router, prefix="/api")
app.include_router(documents_router, prefix="/api")
app.include_router(subjects_router, prefix="/api")
app.include_router(sessions_router, prefix="/api")
app.include_router(voice_router, prefix="/api")
app.include_router(parent_router, prefix="/api")
app.include_router(tts_router, prefix="/api")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
