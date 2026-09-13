from dotenv import load_dotenv
from fastapi import FastAPI

from app.api import chat_router

load_dotenv()


app = FastAPI(title="Project Tutor")
app.include_router(chat_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
