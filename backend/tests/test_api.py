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
        self.received_temperature: float | None = None

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
        temperature: float | None = None,
    ) -> ModelResponse:
        self.received = messages
        self.received_temperature = temperature
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


def test_chat_refuses_when_no_documents_uploaded_for_subject(override_get_db: None) -> None:
    fake = FakeProvider(reply="unused")
    app.dependency_overrides[get_provider] = lambda: fake
    client = TestClient(app)

    try:
        response = client.post("/api/chat", json={"message": "Bonjour"})
    finally:
        app.dependency_overrides.pop(get_provider, None)

    assert response.status_code == 200
    body = response.json()
    assert body["reply"] == (
        "No textbook has been uploaded for this subject yet. Please upload your study material first."
    )
    assert body["sources"] == []
    assert body["visual_directives"] == []
    assert fake.received == []


def test_chat_returns_grounded_provider_reply_with_sources(override_get_db: None) -> None:
    app.dependency_overrides[get_embedding_provider] = lambda: FakeEmbeddingProvider()
    app.dependency_overrides[get_vision_provider] = lambda: FakeVisionProvider()
    app.dependency_overrides[get_vector_store] = lambda: FakeVectorStore()
    fake = FakeProvider(reply="Bonjour ! Comment puis-je t'aider ?")
    app.dependency_overrides[get_provider] = lambda: fake
    client = TestClient(app)

    try:
        # A document must exist for the subject before /chat will retrieve+answer.
        client.post(
            "/api/subjects/general/documents",
            files={"file": ("chapter1.pdf", _build_test_pdf(), "application/pdf")},
        )
        # Swap in a deterministic retrieval score for the /chat call itself.
        app.dependency_overrides[get_vector_store] = lambda: FixedResultVectorStore(
            [SearchResult(text="Le present tense chunk.", document_id=1, page_number=3, distance=0.1)]
        )

        response = client.post("/api/chat", json={"message": "Bonjour"})
    finally:
        app.dependency_overrides.pop(get_provider, None)
        app.dependency_overrides.pop(get_embedding_provider, None)
        app.dependency_overrides.pop(get_vision_provider, None)
        app.dependency_overrides.pop(get_vector_store, None)

    assert response.status_code == 200
    body = response.json()
    assert body["reply"] == "Bonjour ! Comment puis-je t'aider ?"
    assert isinstance(body["session_id"], str) and body["session_id"]
    assert len(body["sources"]) == 1
    assert body["sources"][0]["page_number"] == 3
    assert body["sources"][0]["score"] == pytest.approx(0.9)
    assert body["visual_directives"] == []
    assert fake.received[0]["role"] == "system"
    assert fake.received[-1] == {"role": "user", "content": "Bonjour"}
    assert fake.received_temperature is None


def test_chat_extracts_visual_directives_from_latex_reply(override_get_db: None) -> None:
    app.dependency_overrides[get_embedding_provider] = lambda: FakeEmbeddingProvider()
    app.dependency_overrides[get_vision_provider] = lambda: FakeVisionProvider()
    app.dependency_overrides[get_vector_store] = lambda: FakeVectorStore()
    fake = FakeProvider(reply="The **theorem** says $$a^2 + b^2 = c^2$$.")
    app.dependency_overrides[get_provider] = lambda: fake
    client = TestClient(app)

    try:
        client.post(
            "/api/subjects/general/documents",
            files={"file": ("chapter1.pdf", _build_test_pdf(), "application/pdf")},
        )
        app.dependency_overrides[get_vector_store] = lambda: FixedResultVectorStore(
            [SearchResult(text="Le present tense chunk.", document_id=1, page_number=3, distance=0.1)]
        )

        response = client.post("/api/chat", json={"message": "Explique le theoreme"})
    finally:
        app.dependency_overrides.pop(get_provider, None)
        app.dependency_overrides.pop(get_embedding_provider, None)
        app.dependency_overrides.pop(get_vision_provider, None)
        app.dependency_overrides.pop(get_vector_store, None)

    assert response.status_code == 200
    body = response.json()
    assert body["visual_directives"] == [
        {"directive_type": "formula", "content": "a^2 + b^2 = c^2", "label": None},
        {"directive_type": "highlight", "content": "theorem", "label": None},
    ]


def test_chat_reuses_existing_session(override_get_db: None) -> None:
    app.dependency_overrides[get_embedding_provider] = lambda: FakeEmbeddingProvider()
    app.dependency_overrides[get_vision_provider] = lambda: FakeVisionProvider()
    app.dependency_overrides[get_vector_store] = lambda: FakeVectorStore()
    fake = FakeProvider(reply="ok")
    app.dependency_overrides[get_provider] = lambda: fake
    client = TestClient(app)

    try:
        client.post(
            "/api/subjects/general/documents",
            files={"file": ("chapter1.pdf", _build_test_pdf(), "application/pdf")},
        )
        app.dependency_overrides[get_vector_store] = lambda: FixedResultVectorStore(
            [SearchResult(text="Le present tense chunk.", document_id=1, page_number=1, distance=0.1)]
        )

        first = client.post("/api/chat", json={"message": "Bonjour"}).json()
        second = client.post("/api/chat", json={"message": "Ca va ?", "session_id": first["session_id"]}).json()
    finally:
        app.dependency_overrides.pop(get_provider, None)
        app.dependency_overrides.pop(get_embedding_provider, None)
        app.dependency_overrides.pop(get_vision_provider, None)
        app.dependency_overrides.pop(get_vector_store, None)

    assert second["session_id"] == first["session_id"]
    assert fake.received[0]["role"] == "system"
    assert fake.received[1:] == [
        {"role": "user", "content": "Bonjour"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "Ca va ?"},
    ]


def test_chat_unknown_session_id_returns_404(override_get_db: None) -> None:
    fake = FakeProvider(reply="unused")
    app.dependency_overrides[get_provider] = lambda: fake
    client = TestClient(app)

    try:
        response = client.post("/api/chat", json={"message": "hi", "session_id": "does-not-exist"})
    finally:
        app.dependency_overrides.pop(get_provider, None)

    assert response.status_code == 404


def test_chat_requires_message_field(override_get_db: None) -> None:
    app.dependency_overrides[get_provider] = lambda: FakeProvider(reply="unused")
    client = TestClient(app)

    try:
        response = client.post("/api/chat", json={})
    finally:
        app.dependency_overrides.pop(get_provider, None)

    assert response.status_code == 422


def test_chat_defaults_to_textbook_mode(override_get_db: None) -> None:
    app.dependency_overrides[get_embedding_provider] = lambda: FakeEmbeddingProvider()
    app.dependency_overrides[get_vision_provider] = lambda: FakeVisionProvider()
    app.dependency_overrides[get_vector_store] = lambda: FakeVectorStore()
    fake = FakeProvider(reply="unused")
    app.dependency_overrides[get_provider] = lambda: fake
    client = TestClient(app)

    try:
        client.post(
            "/api/subjects/general/documents",
            files={"file": ("chapter1.pdf", _build_test_pdf(), "application/pdf")},
        )
        app.dependency_overrides[get_vector_store] = lambda: FixedResultVectorStore(
            [SearchResult(text="Le present tense chunk.", document_id=1, page_number=3, distance=0.1)]
        )

        # mode omitted entirely -- must behave the same as mode="textbook".
        response = client.post("/api/chat", json={"message": "Bonjour"})
    finally:
        app.dependency_overrides.pop(get_provider, None)
        app.dependency_overrides.pop(get_embedding_provider, None)
        app.dependency_overrides.pop(get_vision_provider, None)
        app.dependency_overrides.pop(get_vector_store, None)

    assert response.status_code == 200
    system_content = fake.received[0]["content"]
    assert "MUST answer strictly from the passages" in system_content
    assert "experienced, friendly classroom teacher" not in system_content


def test_chat_teacher_mode_changes_system_prompt(override_get_db: None) -> None:
    app.dependency_overrides[get_embedding_provider] = lambda: FakeEmbeddingProvider()
    app.dependency_overrides[get_vision_provider] = lambda: FakeVisionProvider()
    app.dependency_overrides[get_vector_store] = lambda: FakeVectorStore()
    fake = FakeProvider(reply="unused")
    app.dependency_overrides[get_provider] = lambda: fake
    client = TestClient(app)

    try:
        client.post(
            "/api/subjects/general/documents",
            files={"file": ("chapter1.pdf", _build_test_pdf(), "application/pdf")},
        )
        app.dependency_overrides[get_vector_store] = lambda: FixedResultVectorStore(
            [SearchResult(text="Le present tense chunk.", document_id=1, page_number=3, distance=0.1)]
        )

        response = client.post("/api/chat", json={"message": "Explique", "mode": "teacher"})
    finally:
        app.dependency_overrides.pop(get_provider, None)
        app.dependency_overrides.pop(get_embedding_provider, None)
        app.dependency_overrides.pop(get_vision_provider, None)
        app.dependency_overrides.pop(get_vector_store, None)

    assert response.status_code == 200
    system_content = fake.received[0]["content"]
    assert "experienced, friendly classroom teacher" in system_content
    assert "**💡 Beyond the textbook:**" in system_content


def test_chat_rejects_invalid_mode(override_get_db: None) -> None:
    app.dependency_overrides[get_provider] = lambda: FakeProvider(reply="unused")
    client = TestClient(app)

    try:
        response = client.post("/api/chat", json={"message": "Bonjour", "mode": "bogus"})
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


class FixedResultVectorStore(VectorStore):
    """Returns the same fixed search results regardless of query embedding --
    used for /chat tests that need a precise, deterministic retrieval score
    rather than FakeVectorStore's real (hash-based) embedding distance."""

    def __init__(self, hits: list[SearchResult]) -> None:
        self._hits = hits

    async def upsert_chunks(self, collection: str, chunks: list[Chunk]) -> None:
        raise NotImplementedError

    async def search(self, collection: str, query_embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        return self._hits[:top_k]

    async def delete_document_chunks(self, collection: str, document_id: int) -> None:
        raise NotImplementedError


class QuestionAwareEmbeddingProvider(EmbeddingProvider):
    """Embeds a "covered" question to [1.0, 0.0] and everything else to
    [0.0, 1.0], so a paired vector store can deterministically return a good
    match for one and a poor (guardrail-triggering) match for the other --
    used for the /subjects/{subject_id}/qa-check tests, which need two
    different retrieval outcomes from the same overridden dependency within
    one request."""

    def __init__(self, covered_substring: str) -> None:
        self._covered_substring = covered_substring

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return "fake-embed"

    def embed(self, texts: list[str]) -> EmbeddingResponse:
        embeddings = [[1.0, 0.0] if self._covered_substring in t else [0.0, 1.0] for t in texts]
        return EmbeddingResponse(embeddings=embeddings, usage=Usage(input_tokens=len(texts) * 10, output_tokens=None))


class FixedDistanceVectorStore(VectorStore):
    """Pairs with QuestionAwareEmbeddingProvider: returns one chunk whose
    distance from the query embedding is 0.0 (score 1.0, easily clears the
    guardrail) for [1.0, 0.0] queries, and 2.0 (score -1.0, well below it)
    for anything else."""

    def __init__(self, chunk_text: str, page_number: int) -> None:
        self._chunk_text = chunk_text
        self._page_number = page_number

    async def upsert_chunks(self, collection: str, chunks: list[Chunk]) -> None:
        raise NotImplementedError

    async def search(self, collection: str, query_embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        distance = sum((a - b) ** 2 for a, b in zip([1.0, 0.0], query_embedding, strict=True))
        return [
            SearchResult(text=self._chunk_text, document_id=1, page_number=self._page_number, distance=distance)
        ]

    async def delete_document_chunks(self, collection: str, document_id: int) -> None:
        raise NotImplementedError


def _build_test_pdf() -> bytes:
    document = pymupdf.open()
    page = document.new_page(width=400, height=600)
    page.insert_text((50, 50), "Le present tense chunk. Le present s'utilise pour des actions habituelles.")
    pdf_bytes = document.tobytes()
    document.close()
    return pdf_bytes


# --- GET /subjects/{subject_id}/qa-check (Phase 3, prompt 3) ---


def test_qa_check_endpoint_reports_pass_and_guardrail_refusal(override_get_db: None) -> None:
    app.dependency_overrides[get_embedding_provider] = lambda: FakeEmbeddingProvider()
    app.dependency_overrides[get_vision_provider] = lambda: FakeVisionProvider()
    app.dependency_overrides[get_vector_store] = lambda: FakeVectorStore()
    client = TestClient(app)

    try:
        # A document must exist for the subject before qa-check will retrieve+answer.
        # Fresh in-memory DB per test -- "french" is the first subject created, so id=1.
        client.post(
            "/api/subjects/french/documents",
            files={"file": ("chapter1.pdf", _build_test_pdf(), "application/pdf")},
        )

        provider = FakeProvider(
            reply="Le festival interculturel de Nantes et les Journées Interculturelles sont mentionnés."
        )
        app.dependency_overrides[get_provider] = lambda: provider
        app.dependency_overrides[get_embedding_provider] = lambda: QuestionAwareEmbeddingProvider(
            covered_substring="festivals"
        )
        app.dependency_overrides[get_vector_store] = lambda: FixedDistanceVectorStore(
            chunk_text="Le festival interculturel de Nantes est un evenement annuel.", page_number=12
        )

        response = client.get("/api/subjects/1/qa-check")
    finally:
        app.dependency_overrides.pop(get_provider, None)
        app.dependency_overrides.pop(get_embedding_provider, None)
        app.dependency_overrides.pop(get_vision_provider, None)
        app.dependency_overrides.pop(get_vector_store, None)

    assert response.status_code == 200
    body = response.json()
    assert body["subject_id"] == 1
    assert len(body["checks"]) == 2

    covered_check, uncovered_check = body["checks"]
    assert covered_check["passed"] is True
    assert "interculturel" in covered_check["response_preview"]
    assert len(covered_check["sources"]) == 1
    assert covered_check["sources"][0]["page_number"] == 12

    # qa-check calls handle_turn with temperature=0 for deterministic eval output.
    assert provider.received_temperature == 0.0

    assert uncovered_check["passed"] is True
    assert uncovered_check["sources"] == []

    assert body["all_passed"] is True


def test_qa_check_endpoint_reports_failure_when_covered_reply_misses_keywords(override_get_db: None) -> None:
    app.dependency_overrides[get_embedding_provider] = lambda: FakeEmbeddingProvider()
    app.dependency_overrides[get_vision_provider] = lambda: FakeVisionProvider()
    app.dependency_overrides[get_vector_store] = lambda: FakeVectorStore()
    client = TestClient(app)

    try:
        client.post(
            "/api/subjects/french/documents",
            files={"file": ("chapter1.pdf", _build_test_pdf(), "application/pdf")},
        )

        # Reply doesn't mention any of the expected substrings -- should fail.
        app.dependency_overrides[get_provider] = lambda: FakeProvider(reply="I don't know.")
        app.dependency_overrides[get_embedding_provider] = lambda: QuestionAwareEmbeddingProvider(
            covered_substring="festivals"
        )
        app.dependency_overrides[get_vector_store] = lambda: FixedDistanceVectorStore(
            chunk_text="unrelated text", page_number=1
        )

        response = client.get("/api/subjects/1/qa-check")
    finally:
        app.dependency_overrides.pop(get_provider, None)
        app.dependency_overrides.pop(get_embedding_provider, None)
        app.dependency_overrides.pop(get_vision_provider, None)
        app.dependency_overrides.pop(get_vector_store, None)

    assert response.status_code == 200
    body = response.json()
    assert body["checks"][0]["passed"] is False
    assert body["all_passed"] is False


def test_qa_check_endpoint_404s_for_unknown_subject(override_get_db: None) -> None:
    client = TestClient(app)

    response = client.get("/api/subjects/999/qa-check")

    assert response.status_code == 404


def test_upload_document_returns_consistency_check(override_get_db: None) -> None:
    app.dependency_overrides[get_embedding_provider] = lambda: FakeEmbeddingProvider()
    app.dependency_overrides[get_vision_provider] = lambda: FakeVisionProvider()
    app.dependency_overrides[get_vector_store] = lambda: FakeVectorStore()
    client = TestClient(app)

    try:
        response = client.post(
            "/api/subjects/french/documents",
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
            "/api/subjects/french/documents",
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
            "/api/subjects/french/documents",
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
            "/api/subjects/french/documents",
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
            "/api/subjects/french/documents",
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
            "/api/subjects/french/documents",
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
            "/api/subjects/french/documents",
            files={"file": ("chapter5.pdf", pdf_bytes, "application/pdf")},
        )
        assert first.status_code == 200
        first_document_id = first.json()["document_id"]

        # Same bytes, different filename -- still a duplicate by content.
        second = client.post(
            "/api/subjects/french/documents",
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
            "/api/subjects/french/documents",
            files={"file": ("chapter5.pdf", pdf_bytes, "application/pdf")},
        )
        spanish = client.post(
            "/api/subjects/spanish/documents",
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
            "/api/subjects/french/documents",
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
            "/api/subjects/french/documents",
            files={"file": ("chapter5.pdf", _build_test_pdf(), "application/pdf")},
        ).json()
        document_id = upload_body["document_id"]
        collection = f"subject_{1}"

        delete_response = client.delete(f"/api/subjects/french/documents/{document_id}")

        assert delete_response.status_code == 204
        assert shared_store.collections.get(collection, []) == []

        missing_response = client.delete(f"/api/subjects/french/documents/{document_id}")
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
            "/api/subjects/french/documents",
            files={"file": ("chapter5.pdf", _build_test_pdf(), "application/pdf")},
        ).json()
        document_id = upload_body["document_id"]

        response = client.delete(f"/api/subjects/spanish/documents/{document_id}")
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
            "/api/subjects/french/documents",
            files={"file": ("chapter5.pdf", _build_test_pdf(), "application/pdf")},
        ).json()
        document_id = upload_body["document_id"]

        response = client.post(
            f"/api/subjects/french/documents/{document_id}/sanity-check",
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
            "/api/subjects/french/documents/999/sanity-check",
            json={"questions": [{"question": "q", "expected_answer": "a", "source": "p.1", "type": "grammar"}]},
        )
    finally:
        app.dependency_overrides.pop(get_embedding_provider, None)
        app.dependency_overrides.pop(get_vision_provider, None)
        app.dependency_overrides.pop(get_vector_store, None)

    assert response.status_code == 404


# --- GET /api/subjects (Phase 4a) ---


def test_list_subjects_endpoint_returns_document_counts(override_get_db: None) -> None:
    app.dependency_overrides[get_embedding_provider] = lambda: FakeEmbeddingProvider()
    app.dependency_overrides[get_vision_provider] = lambda: FakeVisionProvider()
    app.dependency_overrides[get_vector_store] = lambda: FakeVectorStore()
    client = TestClient(app)

    try:
        client.post(
            "/api/subjects/french/documents",
            files={"file": ("chapter1.pdf", _build_test_pdf(), "application/pdf")},
        )
        response = client.get("/api/subjects")
    finally:
        app.dependency_overrides.pop(get_embedding_provider, None)
        app.dependency_overrides.pop(get_vision_provider, None)
        app.dependency_overrides.pop(get_vector_store, None)

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["name"] == "french"
    assert body[0]["document_count"] == 1
    assert "created_at" in body[0]


def test_list_subjects_endpoint_returns_empty_list_when_no_subjects(override_get_db: None) -> None:
    client = TestClient(app)

    response = client.get("/api/subjects")

    assert response.status_code == 200
    assert response.json() == []


# --- GET /api/subjects/{subject_id}/documents (Phase 4a) ---


def test_list_subject_documents_endpoint_returns_documents(override_get_db: None) -> None:
    app.dependency_overrides[get_embedding_provider] = lambda: FakeEmbeddingProvider()
    app.dependency_overrides[get_vision_provider] = lambda: FakeVisionProvider()
    app.dependency_overrides[get_vector_store] = lambda: FakeVectorStore()
    client = TestClient(app)

    try:
        upload_body = client.post(
            "/api/subjects/french/documents",
            files={"file": ("chapter1.pdf", _build_test_pdf(), "application/pdf")},
        ).json()

        # Fresh in-memory DB per test -- "french" is the first subject created, so id=1.
        response = client.get("/api/subjects/1/documents")
    finally:
        app.dependency_overrides.pop(get_embedding_provider, None)
        app.dependency_overrides.pop(get_vision_provider, None)
        app.dependency_overrides.pop(get_vector_store, None)

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["id"] == upload_body["document_id"]
    assert body[0]["filename"] == "chapter1.pdf"
    assert body[0]["page_count"] == upload_body["page_count"]
    assert body[0]["scanned_page_count"] == upload_body["scanned_page_count"]
    assert body[0]["chunk_count"] == upload_body["chunk_count"]
    assert "created_at" in body[0]


def test_list_subject_documents_endpoint_404s_for_unknown_subject(override_get_db: None) -> None:
    client = TestClient(app)

    response = client.get("/api/subjects/999/documents")

    assert response.status_code == 404


# --- ChatRequest.subject (Phase 4a) ---


def test_chat_creates_session_for_requested_subject(override_get_db: None) -> None:
    app.dependency_overrides[get_embedding_provider] = lambda: FakeEmbeddingProvider()
    app.dependency_overrides[get_vision_provider] = lambda: FakeVisionProvider()
    app.dependency_overrides[get_vector_store] = lambda: FakeVectorStore()
    fake = FakeProvider(reply="unused")
    app.dependency_overrides[get_provider] = lambda: fake
    client = TestClient(app)

    try:
        response = client.post("/api/chat", json={"message": "Bonjour", "subject": "french"})
        subjects = client.get("/api/subjects").json()
    finally:
        app.dependency_overrides.pop(get_provider, None)
        app.dependency_overrides.pop(get_embedding_provider, None)
        app.dependency_overrides.pop(get_vision_provider, None)
        app.dependency_overrides.pop(get_vector_store, None)

    assert response.status_code == 200
    assert response.json()["session_id"]
    assert [s["name"] for s in subjects] == ["french"]


def test_chat_ignores_subject_field_when_session_id_given(override_get_db: None) -> None:
    app.dependency_overrides[get_embedding_provider] = lambda: FakeEmbeddingProvider()
    app.dependency_overrides[get_vision_provider] = lambda: FakeVisionProvider()
    app.dependency_overrides[get_vector_store] = lambda: FakeVectorStore()
    fake = FakeProvider(reply="unused")
    app.dependency_overrides[get_provider] = lambda: fake
    client = TestClient(app)

    try:
        first = client.post("/api/chat", json={"message": "Bonjour", "subject": "french"}).json()
        second = client.post(
            "/api/chat",
            json={"message": "Ca va ?", "session_id": first["session_id"], "subject": "maths"},
        )
        subjects = client.get("/api/subjects").json()
    finally:
        app.dependency_overrides.pop(get_provider, None)
        app.dependency_overrides.pop(get_embedding_provider, None)
        app.dependency_overrides.pop(get_vision_provider, None)
        app.dependency_overrides.pop(get_vector_store, None)

    assert second.status_code == 200
    assert second.json()["session_id"] == first["session_id"]
    # "maths" must never get created -- the existing session's subject wins.
    assert [s["name"] for s in subjects] == ["french"]


# --- GET /api/sessions/{session_id}/usage (Phase 4a) ---


def test_session_usage_endpoint_aggregates_events_for_session(override_get_db: None) -> None:
    app.dependency_overrides[get_embedding_provider] = lambda: FakeEmbeddingProvider()
    app.dependency_overrides[get_vision_provider] = lambda: FakeVisionProvider()
    app.dependency_overrides[get_vector_store] = lambda: FakeVectorStore()
    fake = FakeProvider(reply="Bonjour !")
    app.dependency_overrides[get_provider] = lambda: fake
    client = TestClient(app)

    try:
        client.post(
            "/api/subjects/general/documents",
            files={"file": ("chapter1.pdf", _build_test_pdf(), "application/pdf")},
        )
        app.dependency_overrides[get_vector_store] = lambda: FixedResultVectorStore(
            [SearchResult(text="Le present tense chunk.", document_id=1, page_number=3, distance=0.1)]
        )

        chat_response = client.post("/api/chat", json={"message": "Bonjour"})
        session_id = chat_response.json()["session_id"]

        response = client.get(f"/api/sessions/{session_id}/usage")
    finally:
        app.dependency_overrides.pop(get_provider, None)
        app.dependency_overrides.pop(get_embedding_provider, None)
        app.dependency_overrides.pop(get_vision_provider, None)
        app.dependency_overrides.pop(get_vector_store, None)

    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == session_id
    assert body["event_count"] == 3
    # FakeEmbeddingProvider charges 10 input tokens per text (1 query) + FakeProvider's 3.
    assert body["total_input_tokens"] == 13
    assert body["total_output_tokens"] == 5
    assert body["total_cost_usd"] == pytest.approx(0.0)
    assert {event["event_type"] for event in body["events"]} == {"retrieval", "embedding", "generation"}


def test_session_usage_endpoint_404s_for_unknown_session(override_get_db: None) -> None:
    client = TestClient(app)

    response = client.get("/api/sessions/does-not-exist/usage")

    assert response.status_code == 404


# --- GET /api/subjects/{subject_id}/sessions and /api/sessions/{id}/messages ---


def test_list_sessions_endpoint_returns_preview_for_subject(override_get_db: None) -> None:
    app.dependency_overrides[get_embedding_provider] = lambda: FakeEmbeddingProvider()
    app.dependency_overrides[get_vision_provider] = lambda: FakeVisionProvider()
    app.dependency_overrides[get_vector_store] = lambda: FakeVectorStore()
    fake = FakeProvider(reply="Bonjour !")
    app.dependency_overrides[get_provider] = lambda: fake
    client = TestClient(app)

    try:
        client.post(
            "/api/subjects/general/documents",
            files={"file": ("chapter1.pdf", _build_test_pdf(), "application/pdf")},
        )
        app.dependency_overrides[get_vector_store] = lambda: FixedResultVectorStore(
            [SearchResult(text="Le present tense chunk.", document_id=1, page_number=3, distance=0.1)]
        )

        first = client.post("/api/chat", json={"message": "Bonjour"}).json()
        second = client.post("/api/chat", json={"message": "Autre question", "subject": "general"}).json()

        response = client.get("/api/subjects/1/sessions")
    finally:
        app.dependency_overrides.pop(get_provider, None)
        app.dependency_overrides.pop(get_embedding_provider, None)
        app.dependency_overrides.pop(get_vision_provider, None)
        app.dependency_overrides.pop(get_vector_store, None)

    assert response.status_code == 200
    body = response.json()
    assert {s["session_id"] for s in body} == {first["session_id"], second["session_id"]}
    session_entry = next(s for s in body if s["session_id"] == first["session_id"])
    assert session_entry["subject_id"] == 1
    assert session_entry["message_count"] == 2
    assert session_entry["last_message_preview"] == "Bonjour"
    assert session_entry["teaching_mode"] == "textbook"


def test_list_sessions_endpoint_returns_empty_list_for_subject_with_no_sessions(override_get_db: None) -> None:
    client = TestClient(app)

    response = client.get("/api/subjects/999/sessions")

    assert response.status_code == 200
    assert response.json() == []


def test_get_session_messages_endpoint_returns_messages_in_order(override_get_db: None) -> None:
    app.dependency_overrides[get_embedding_provider] = lambda: FakeEmbeddingProvider()
    app.dependency_overrides[get_vision_provider] = lambda: FakeVisionProvider()
    app.dependency_overrides[get_vector_store] = lambda: FakeVectorStore()
    fake = FakeProvider(reply="Bonjour !")
    app.dependency_overrides[get_provider] = lambda: fake
    client = TestClient(app)

    try:
        client.post(
            "/api/subjects/general/documents",
            files={"file": ("chapter1.pdf", _build_test_pdf(), "application/pdf")},
        )
        app.dependency_overrides[get_vector_store] = lambda: FixedResultVectorStore(
            [SearchResult(text="Le present tense chunk.", document_id=1, page_number=3, distance=0.1)]
        )

        chat_response = client.post("/api/chat", json={"message": "Bonjour"}).json()
        session_id = chat_response["session_id"]

        response = client.get(f"/api/sessions/{session_id}/messages")
    finally:
        app.dependency_overrides.pop(get_provider, None)
        app.dependency_overrides.pop(get_embedding_provider, None)
        app.dependency_overrides.pop(get_vision_provider, None)
        app.dependency_overrides.pop(get_vector_store, None)

    assert response.status_code == 200
    body = response.json()
    assert body == [
        {"role": "user", "content": "Bonjour"},
        {"role": "assistant", "content": "Bonjour !"},
    ]


def test_get_session_messages_endpoint_404s_for_unknown_session(override_get_db: None) -> None:
    client = TestClient(app)

    response = client.get("/api/sessions/does-not-exist/messages")

    assert response.status_code == 404


# --- app.main: /api prefix, /health, CORS (Phase 4a) ---


def test_health_endpoint_stays_unprefixed() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_cors_allows_configured_vite_dev_origin() -> None:
    client = TestClient(app)

    response = client.get("/health", headers={"Origin": "http://localhost:5173"})

    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_cors_omits_header_for_unlisted_origin() -> None:
    client = TestClient(app)

    response = client.get("/health", headers={"Origin": "http://evil.example.com"})

    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers
