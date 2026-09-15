from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

import pymupdf
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

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
from app.storage import get_db
from app.storage.models import Base


class FakeProvider(ModelProvider):
    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.received: list[Message] = []

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return "fake-model"

    def generate(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse:
        self.received = messages
        return ModelResponse(content=self._reply, usage=Usage(input_tokens=3, output_tokens=5))


@pytest.fixture
def override_get_db() -> Iterator[None]:
    """Point get_db at a fresh in-memory SQLite engine for the duration of one test.

    Table creation happens inside the override itself (not via a separate
    asyncio.run()) so it runs on the same event loop TestClient uses to
    dispatch requests — aiosqlite connections can't cross event loops.
    """
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async def _get_db() -> AsyncIterator[AsyncSession]:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        async with session_factory() as session:
            yield session
            await session.commit()

    app.dependency_overrides[get_db] = _get_db
    yield
    app.dependency_overrides.pop(get_db, None)


def test_chat_returns_provider_reply(override_get_db: None) -> None:
    fake = FakeProvider(reply="Bonjour ! Comment puis-je t'aider ?")
    app.dependency_overrides[get_provider] = lambda: fake
    client = TestClient(app)

    try:
        response = client.post("/chat", json={"message": "Bonjour"})
    finally:
        app.dependency_overrides.pop(get_provider, None)

    assert response.status_code == 200
    body = response.json()
    assert body["reply"] == "Bonjour ! Comment puis-je t'aider ?"
    assert isinstance(body["session_id"], str) and body["session_id"]
    assert fake.received == [{"role": "user", "content": "Bonjour"}]


def test_chat_reuses_existing_session(override_get_db: None) -> None:
    fake = FakeProvider(reply="ok")
    app.dependency_overrides[get_provider] = lambda: fake
    client = TestClient(app)

    try:
        first = client.post("/chat", json={"message": "Bonjour"}).json()
        second = client.post("/chat", json={"message": "Ca va ?", "session_id": first["session_id"]}).json()
    finally:
        app.dependency_overrides.pop(get_provider, None)

    assert second["session_id"] == first["session_id"]
    assert fake.received == [
        {"role": "user", "content": "Bonjour"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "Ca va ?"},
    ]


def test_chat_unknown_session_id_returns_404(override_get_db: None) -> None:
    fake = FakeProvider(reply="unused")
    app.dependency_overrides[get_provider] = lambda: fake
    client = TestClient(app)

    try:
        response = client.post("/chat", json={"message": "hi", "session_id": "does-not-exist"})
    finally:
        app.dependency_overrides.pop(get_provider, None)

    assert response.status_code == 404


def test_chat_requires_message_field(override_get_db: None) -> None:
    app.dependency_overrides[get_provider] = lambda: FakeProvider(reply="unused")
    client = TestClient(app)

    try:
        response = client.post("/chat", json={})
    finally:
        app.dependency_overrides.pop(get_provider, None)

    assert response.status_code == 422


class FakeEmbeddingProvider(EmbeddingProvider):
    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return "fake-embed"

    def embed(self, texts: list[str]) -> EmbeddingResponse:
        embeddings = [[float(len(t) % 7), float(hash(t) % 11)] for t in texts]
        return EmbeddingResponse(embeddings=embeddings, usage=Usage(input_tokens=len(texts) * 10, output_tokens=None))


class FakeVisionProvider(VisionProvider):
    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return "fake-vision"

    def transcribe_image(self, image_bytes: bytes, media_type: str, instructions: str) -> VisionResponse:
        return VisionResponse(text="unused", usage=Usage(input_tokens=200, output_tokens=50))


class FakeVectorStore(VectorStore):
    def __init__(self) -> None:
        self.collections: dict[str, list[Chunk]] = {}

    async def upsert_chunks(self, collection: str, chunks: list[Chunk]) -> None:
        self.collections.setdefault(collection, []).extend(chunks)

    async def search(self, collection: str, query_embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        def squared_distance(chunk: Chunk) -> float:
            return sum((a - b) ** 2 for a, b in zip(chunk.embedding, query_embedding, strict=True))

        ranked = sorted(self.collections.get(collection, []), key=squared_distance)[:top_k]
        return [
            SearchResult(
                text=chunk.text,
                document_id=chunk.document_id,
                page_number=chunk.page_number,
                distance=squared_distance(chunk),
            )
            for chunk in ranked
        ]

    async def delete_document_chunks(self, collection: str, document_id: int) -> None:
        remaining = [c for c in self.collections.get(collection, []) if c.document_id != document_id]
        self.collections[collection] = remaining


def _build_test_pdf() -> bytes:
    document = pymupdf.open()
    page = document.new_page(width=400, height=600)
    page.insert_text((50, 50), "Le present tense chunk. Le present s'utilise pour des actions habituelles.")
    pdf_bytes = document.tobytes()
    document.close()
    return pdf_bytes


def test_upload_document_returns_consistency_check(override_get_db: None) -> None:
    app.dependency_overrides[get_embedding_provider] = lambda: FakeEmbeddingProvider()
    app.dependency_overrides[get_vision_provider] = lambda: FakeVisionProvider()
    app.dependency_overrides[get_vector_store] = lambda: FakeVectorStore()
    client = TestClient(app)

    try:
        response = client.post(
            "/subjects/french/documents",
            files={"file": ("chapter5.pdf", _build_test_pdf(), "application/pdf")},
        )
    finally:
        app.dependency_overrides.pop(get_embedding_provider, None)
        app.dependency_overrides.pop(get_vision_provider, None)
        app.dependency_overrides.pop(get_vector_store, None)

    assert response.status_code == 200
    body = response.json()
    assert body["chunk_count"] >= 1
    assert body["consistency_check"]["sampled"] == body["consistency_check"]["passed"]
    assert body["consistency_check"]["failed"] == []


def test_upload_document_rejects_non_pdf(override_get_db: None) -> None:
    app.dependency_overrides[get_embedding_provider] = lambda: FakeEmbeddingProvider()
    app.dependency_overrides[get_vision_provider] = lambda: FakeVisionProvider()
    app.dependency_overrides[get_vector_store] = lambda: FakeVectorStore()
    client = TestClient(app)

    try:
        response = client.post(
            "/subjects/french/documents",
            files={"file": ("notes.txt", b"hello", "text/plain")},
        )
    finally:
        app.dependency_overrides.pop(get_embedding_provider, None)
        app.dependency_overrides.pop(get_vision_provider, None)
        app.dependency_overrides.pop(get_vector_store, None)

    assert response.status_code == 415


def test_upload_document_rejects_pdf_content_type_with_non_pdf_bytes(override_get_db: None) -> None:
    """A spoofed Content-Type header alone must not be enough -- the actual
    file signature (magic bytes) is checked too.
    """
    app.dependency_overrides[get_embedding_provider] = lambda: FakeEmbeddingProvider()
    app.dependency_overrides[get_vision_provider] = lambda: FakeVisionProvider()
    app.dependency_overrides[get_vector_store] = lambda: FakeVectorStore()
    client = TestClient(app)

    try:
        response = client.post(
            "/subjects/french/documents",
            files={"file": ("fake.pdf", b"this is not really a pdf", "application/pdf")},
        )
    finally:
        app.dependency_overrides.pop(get_embedding_provider, None)
        app.dependency_overrides.pop(get_vision_provider, None)
        app.dependency_overrides.pop(get_vector_store, None)

    assert response.status_code == 415


def test_upload_document_rejects_oversized_file(override_get_db: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UPLOAD_MAX_SIZE_MB", "0.0001")  # ~100 bytes, smaller than the test PDF
    app.dependency_overrides[get_embedding_provider] = lambda: FakeEmbeddingProvider()
    app.dependency_overrides[get_vision_provider] = lambda: FakeVisionProvider()
    app.dependency_overrides[get_vector_store] = lambda: FakeVectorStore()
    client = TestClient(app)

    try:
        response = client.post(
            "/subjects/french/documents",
            files={"file": ("chapter5.pdf", _build_test_pdf(), "application/pdf")},
        )
    finally:
        app.dependency_overrides.pop(get_embedding_provider, None)
        app.dependency_overrides.pop(get_vision_provider, None)
        app.dependency_overrides.pop(get_vector_store, None)

    assert response.status_code == 413


def test_upload_document_rejects_corrupt_pdf(override_get_db: None) -> None:
    """Bytes that pass the magic-byte check (start with %PDF-) but aren't a
    real, openable PDF structure.
    """
    app.dependency_overrides[get_embedding_provider] = lambda: FakeEmbeddingProvider()
    app.dependency_overrides[get_vision_provider] = lambda: FakeVisionProvider()
    app.dependency_overrides[get_vector_store] = lambda: FakeVectorStore()
    client = TestClient(app)

    try:
        response = client.post(
            "/subjects/french/documents",
            files={"file": ("corrupt.pdf", b"%PDF-1.4\nnot actually a valid pdf body", "application/pdf")},
        )
    finally:
        app.dependency_overrides.pop(get_embedding_provider, None)
        app.dependency_overrides.pop(get_vision_provider, None)
        app.dependency_overrides.pop(get_vector_store, None)

    assert response.status_code == 422


def test_upload_document_rejects_pdf_with_no_extractable_content(override_get_db: None) -> None:
    document = pymupdf.open()
    document.new_page(width=400, height=600)  # blank: no text, no images
    blank_pdf_bytes = document.tobytes()
    document.close()

    app.dependency_overrides[get_embedding_provider] = lambda: FakeEmbeddingProvider()
    app.dependency_overrides[get_vision_provider] = lambda: FakeVisionProvider()
    app.dependency_overrides[get_vector_store] = lambda: FakeVectorStore()
    client = TestClient(app)

    try:
        response = client.post(
            "/subjects/french/documents",
            files={"file": ("blank.pdf", blank_pdf_bytes, "application/pdf")},
        )
    finally:
        app.dependency_overrides.pop(get_embedding_provider, None)
        app.dependency_overrides.pop(get_vision_provider, None)
        app.dependency_overrides.pop(get_vector_store, None)

    assert response.status_code == 422


def test_upload_document_rejects_duplicate_content_hash(override_get_db: None) -> None:
    shared_store = FakeVectorStore()
    app.dependency_overrides[get_embedding_provider] = lambda: FakeEmbeddingProvider()
    app.dependency_overrides[get_vision_provider] = lambda: FakeVisionProvider()
    app.dependency_overrides[get_vector_store] = lambda: shared_store
    client = TestClient(app)
    pdf_bytes = _build_test_pdf()

    try:
        first = client.post(
            "/subjects/french/documents",
            files={"file": ("chapter5.pdf", pdf_bytes, "application/pdf")},
        )
        assert first.status_code == 200
        first_document_id = first.json()["document_id"]

        # Same bytes, different filename -- still a duplicate by content.
        second = client.post(
            "/subjects/french/documents",
            files={"file": ("chapter5-renamed.pdf", pdf_bytes, "application/pdf")},
        )
    finally:
        app.dependency_overrides.pop(get_embedding_provider, None)
        app.dependency_overrides.pop(get_vision_provider, None)
        app.dependency_overrides.pop(get_vector_store, None)

    assert second.status_code == 409
    assert str(first_document_id) in second.json()["detail"]


def test_upload_document_allows_same_content_in_different_subjects(override_get_db: None) -> None:
    app.dependency_overrides[get_embedding_provider] = lambda: FakeEmbeddingProvider()
    app.dependency_overrides[get_vision_provider] = lambda: FakeVisionProvider()
    app.dependency_overrides[get_vector_store] = lambda: FakeVectorStore()
    client = TestClient(app)
    pdf_bytes = _build_test_pdf()

    try:
        french = client.post(
            "/subjects/french/documents",
            files={"file": ("chapter5.pdf", pdf_bytes, "application/pdf")},
        )
        spanish = client.post(
            "/subjects/spanish/documents",
            files={"file": ("chapter5.pdf", pdf_bytes, "application/pdf")},
        )
    finally:
        app.dependency_overrides.pop(get_embedding_provider, None)
        app.dependency_overrides.pop(get_vision_provider, None)
        app.dependency_overrides.pop(get_vector_store, None)

    assert french.status_code == 200
    assert spanish.status_code == 200


def test_upload_document_rejects_concurrent_ingestion_for_same_subject(override_get_db: None) -> None:
    from app.api import documents as documents_module

    app.dependency_overrides[get_embedding_provider] = lambda: FakeEmbeddingProvider()
    app.dependency_overrides[get_vision_provider] = lambda: FakeVisionProvider()
    app.dependency_overrides[get_vector_store] = lambda: FakeVectorStore()
    client = TestClient(app)

    documents_module._ingesting_subjects.add("french")
    try:
        response = client.post(
            "/subjects/french/documents",
            files={"file": ("chapter5.pdf", _build_test_pdf(), "application/pdf")},
        )
    finally:
        documents_module._ingesting_subjects.discard("french")
        app.dependency_overrides.pop(get_embedding_provider, None)
        app.dependency_overrides.pop(get_vision_provider, None)
        app.dependency_overrides.pop(get_vector_store, None)

    assert response.status_code == 429
    assert response.headers.get("retry-after") is not None


def test_delete_document_removes_row_and_chunks_allowing_clean_reingest(override_get_db: None) -> None:
    shared_store = FakeVectorStore()
    app.dependency_overrides[get_embedding_provider] = lambda: FakeEmbeddingProvider()
    app.dependency_overrides[get_vision_provider] = lambda: FakeVisionProvider()
    app.dependency_overrides[get_vector_store] = lambda: shared_store
    client = TestClient(app)

    try:
        upload_body = client.post(
            "/subjects/french/documents",
            files={"file": ("chapter5.pdf", _build_test_pdf(), "application/pdf")},
        ).json()
        document_id = upload_body["document_id"]
        collection = f"subject_{1}"

        delete_response = client.delete(f"/subjects/french/documents/{document_id}")

        assert delete_response.status_code == 204
        assert shared_store.collections.get(collection, []) == []

        missing_response = client.delete(f"/subjects/french/documents/{document_id}")
        assert missing_response.status_code == 404
    finally:
        app.dependency_overrides.pop(get_embedding_provider, None)
        app.dependency_overrides.pop(get_vision_provider, None)
        app.dependency_overrides.pop(get_vector_store, None)


def test_delete_document_rejects_wrong_subject(override_get_db: None) -> None:
    app.dependency_overrides[get_embedding_provider] = lambda: FakeEmbeddingProvider()
    app.dependency_overrides[get_vision_provider] = lambda: FakeVisionProvider()
    app.dependency_overrides[get_vector_store] = lambda: FakeVectorStore()
    client = TestClient(app)

    try:
        upload_body = client.post(
            "/subjects/french/documents",
            files={"file": ("chapter5.pdf", _build_test_pdf(), "application/pdf")},
        ).json()
        document_id = upload_body["document_id"]

        response = client.delete(f"/subjects/spanish/documents/{document_id}")
    finally:
        app.dependency_overrides.pop(get_embedding_provider, None)
        app.dependency_overrides.pop(get_vision_provider, None)
        app.dependency_overrides.pop(get_vector_store, None)

    assert response.status_code == 404


def test_sanity_check_endpoint_reports_page_match_and_out_of_scope(override_get_db: None) -> None:
    shared_store = FakeVectorStore()
    app.dependency_overrides[get_embedding_provider] = lambda: FakeEmbeddingProvider()
    app.dependency_overrides[get_vision_provider] = lambda: FakeVisionProvider()
    app.dependency_overrides[get_vector_store] = lambda: shared_store
    client = TestClient(app)

    try:
        upload_body = client.post(
            "/subjects/french/documents",
            files={"file": ("chapter5.pdf", _build_test_pdf(), "application/pdf")},
        ).json()
        document_id = upload_body["document_id"]

        response = client.post(
            f"/subjects/french/documents/{document_id}/sanity-check",
            json={
                "questions": [
                    {
                        "question": "Le present tense",
                        "expected_answer": "It's used for habitual actions.",
                        "source": "p.1",
                        "type": "grammar",
                    },
                    {
                        "question": "What does the book say about Victor Hugo?",
                        "expected_answer": "Not covered.",
                        "source": "Not in the textbook",
                        "type": "out-of-scope",
                    },
                ]
            },
        )
    finally:
        app.dependency_overrides.pop(get_embedding_provider, None)
        app.dependency_overrides.pop(get_vision_provider, None)
        app.dependency_overrides.pop(get_vector_store, None)

    assert response.status_code == 200
    body = response.json()
    assert body["document_id"] == document_id
    assert len(body["results"]) == 2

    out_of_scope = next(r for r in body["results"] if r["type"] == "out-of-scope")
    assert out_of_scope["page_match"] is None

    assert body["summary"]["overall"]["total"] == 2
    assert body["summary"]["by_type"]["out-of-scope"]["page_match_unknown"] == 1


def test_sanity_check_endpoint_404s_for_unknown_document(override_get_db: None) -> None:
    app.dependency_overrides[get_embedding_provider] = lambda: FakeEmbeddingProvider()
    app.dependency_overrides[get_vision_provider] = lambda: FakeVisionProvider()
    app.dependency_overrides[get_vector_store] = lambda: FakeVectorStore()
    client = TestClient(app)

    try:
        response = client.post(
            "/subjects/french/documents/999/sanity-check",
            json={"questions": [{"question": "q", "expected_answer": "a", "source": "p.1", "type": "grammar"}]},
        )
    finally:
        app.dependency_overrides.pop(get_embedding_provider, None)
        app.dependency_overrides.pop(get_vision_provider, None)
        app.dependency_overrides.pop(get_vector_store, None)

    assert response.status_code == 404
