"""Regression test for the event-loop-blocking bug in the chat path:
app.orchestration.handle_turn's ModelProvider.generate() call and
app.knowledge.retrieval's EmbeddingProvider.embed() call used to be
synchronous calls made from inside an async request handler, which blocks
FastAPI's single event loop for the whole call -- confirmed in practice by
a hung GET /health while a live chat reply was still generating.

Fixed by making the adapters themselves genuinely async (ollama.AsyncClient,
anthropic.AsyncAnthropic) rather than wrapping a sync call in a threadpool
at the call site -- see app/models/ollama_adapter.py, anthropic_adapter.py,
ollama_embedding_adapter.py, anthropic_vision_adapter.py, and
tesseract_vision_adapter.py (which still uses asyncio.to_thread internally,
since pytesseract has no async client to switch to).

This test proves the fix holds: a slow (mocked, 2s) chat generation must
not stop GET /health -- issued concurrently against the same server -- from
responding immediately.

Deliberately does NOT use FastAPI's TestClient: TestClient dispatches each
call through its own short-lived anyio portal/event loop, so two concurrent
TestClient calls don't actually share one event loop the way real
concurrent connections to one running server do -- it can't reproduce this
bug at all (confirmed while writing this test: a version of SlowProvider
that blocked with time.sleep() still passed under TestClient). A real
uvicorn server in a background thread, hit with real HTTP requests, is what
actually exercises the single event loop the bug lives in.
"""

from __future__ import annotations

import asyncio
import socket
import threading
import time
from typing import Any

import httpx
import pymupdf
import pytest
import uvicorn
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

import app.storage.db as db_module
from app.knowledge import Chunk, SearchResult, VectorStore, get_vector_store
from app.main import app
from app.models import (
    EmbeddingProvider,
    EmbeddingResponse,
    Message,
    ModelProvider,
    ModelResponse,
    Usage,
    VisionProvider,
    VisionResponse,
    get_embedding_provider,
    get_provider,
    get_vision_provider,
)

_GENERATION_DELAY_S = 2.0


class SlowProvider(ModelProvider):
    """A ModelProvider whose generate() awaits for _GENERATION_DELAY_S,
    simulating a real async Ollama/Anthropic network call that's slow --
    not a sync call, since that's exactly the bug this test guards against."""

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return "fake-model"

    async def generate(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
    ) -> ModelResponse:
        await asyncio.sleep(_GENERATION_DELAY_S)
        return ModelResponse(content="Bonjour !", usage=Usage(input_tokens=3, output_tokens=5))


class FakeEmbeddingProvider(EmbeddingProvider):
    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return "fake-embed"

    async def embed(self, texts: list[str]) -> EmbeddingResponse:
        embeddings = [[float(len(t) % 7), float(hash(t) % 11)] for t in texts]
        return EmbeddingResponse(embeddings=embeddings, usage=Usage(input_tokens=len(texts) * 10, output_tokens=None))


class FakeVisionProvider(VisionProvider):
    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return "fake-vision"

    async def transcribe_image(self, image_bytes: bytes, media_type: str, instructions: str) -> VisionResponse:
        return VisionResponse(text="unused", usage=Usage(input_tokens=200, output_tokens=50))


class FakeVectorStore(VectorStore):
    """Accepts real upsert_chunks() calls during ingestion -- used only to
    get a document stored for the subject before the chat call, which then
    switches the override to FixedResultVectorStore for a deterministic
    retrieval score."""

    def __init__(self) -> None:
        self.collections: dict[str, list[Chunk]] = {}

    async def upsert_chunks(self, collection: str, chunks: list[Chunk]) -> None:
        self.collections.setdefault(collection, []).extend(chunks)

    async def search(self, collection: str, query_embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        return []

    async def delete_document_chunks(self, collection: str, document_id: int) -> None:
        raise NotImplementedError


class FixedResultVectorStore(VectorStore):
    def __init__(self, hits: list[SearchResult]) -> None:
        self._hits = hits

    async def upsert_chunks(self, collection: str, chunks: list[Chunk]) -> None:
        raise NotImplementedError

    async def search(self, collection: str, query_embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        return self._hits[:top_k]

    async def delete_document_chunks(self, collection: str, document_id: int) -> None:
        raise NotImplementedError


def _build_test_pdf() -> bytes:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Bonjour, comment ca va ?")
    return document.tobytes()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class _NoSignalHandlerServer(uvicorn.Server):
    """uvicorn.Server.run() tries to install signal handlers, which only
    works on the main thread -- this test runs the server in a background
    thread instead, so that step is skipped."""

    def install_signal_handlers(self) -> None:
        pass


@pytest.fixture
def live_server(monkeypatch: pytest.MonkeyPatch):
    """Runs the real app under a real uvicorn server in a background
    thread, against an isolated in-memory database (via monkeypatching
    app.storage.db's engine/session_factory -- the lifespan's init_db()
    call isn't reachable through FastAPI's dependency_overrides, so this
    is the only way to keep it off the real dev tutor.db). Yields the
    server's base URL.
    """
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(db_module, "engine", engine)
    monkeypatch.setattr(db_module, "_session_factory", session_factory)

    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = _NoSignalHandlerServer(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.02)
    assert server.started, "uvicorn server did not start in time"

    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def test_health_stays_responsive_during_a_slow_chat_generation(live_server: str) -> None:
    app.dependency_overrides[get_embedding_provider] = lambda: FakeEmbeddingProvider()
    app.dependency_overrides[get_vision_provider] = lambda: FakeVisionProvider()
    app.dependency_overrides[get_vector_store] = lambda: FakeVectorStore()
    app.dependency_overrides[get_provider] = lambda: SlowProvider()

    try:
        with httpx.Client(base_url=live_server, timeout=30.0) as setup_client:
            # A document must exist for the subject before /chat
            # retrieves+answers (otherwise handle_turn's "no documents"
            # guardrail short-circuits before ever calling SlowProvider).
            setup_client.post(
                "/api/subjects/general/documents",
                files={"file": ("chapter1.pdf", _build_test_pdf(), "application/pdf")},
            )
        app.dependency_overrides[get_vector_store] = lambda: FixedResultVectorStore(
            [SearchResult(text="Le present tense chunk.", document_id=1, page_number=3, distance=0.1)]
        )

        chat_result: dict[str, Any] = {}

        def send_chat() -> None:
            with httpx.Client(base_url=live_server, timeout=_GENERATION_DELAY_S + 10) as chat_client:
                chat_result["response"] = chat_client.post("/api/chat", json={"message": "Bonjour"})

        thread = threading.Thread(target=send_chat)
        thread.start()
        # Give the request time to actually reach SlowProvider.generate()'s
        # sleep before racing /health against it.
        time.sleep(0.3)

        with httpx.Client(base_url=live_server, timeout=5.0) as health_client:
            health_started = time.perf_counter()
            health_response = health_client.get("/health")
            health_elapsed = time.perf_counter() - health_started

        thread.join(timeout=_GENERATION_DELAY_S + 10)
    finally:
        app.dependency_overrides.pop(get_provider, None)
        app.dependency_overrides.pop(get_embedding_provider, None)
        app.dependency_overrides.pop(get_vision_provider, None)
        app.dependency_overrides.pop(get_vector_store, None)

    assert health_response.status_code == 200
    # Well under _GENERATION_DELAY_S -- proves /health wasn't queued behind
    # the still-running chat generation.
    assert health_elapsed < 1.0

    assert not thread.is_alive()
    chat_response = chat_result["response"]
    assert chat_response.status_code == 200
    assert chat_response.json()["reply"] == "Bonjour !"
