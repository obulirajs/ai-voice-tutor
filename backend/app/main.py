from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

from app.api import chat_router, documents_router
from app.storage import init_db

# Load the .env file
load_dotenv()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
    await init_db()
    yield


app = FastAPI(title="Project Tutor", lifespan=lifespan)
app.include_router(chat_router)
app.include_router(documents_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
