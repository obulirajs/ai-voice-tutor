from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

import pymupdf
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool
from starlette.testclient import TestClient

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
from app.voice import (
    ASRProvider,
    SynthesisResult,
    TranscriptionResult,
    TTSProvider,
    get_asr_provider,
    get_tts_provider,
)


class FakeProvider(ModelProvider):
    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.call_count = 0
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
        temperature: float | None = None,
    ) -> ModelResponse:
        self.call_count += 1
        self.received = messages
        return ModelResponse(content=self._reply, usage=Usage(input_tokens=3, output_tokens=5))


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
        return []

    async def delete_document_chunks(self, collection: str, document_id: int) -> None:
        raise NotImplementedError


class FixedResultVectorStore(VectorStore):
    """Returns the same fixed search results regardless of query embedding --
    a precise, deterministic retrieval score for the grounded-turn test."""

    def __init__(self, hits: list[SearchResult]) -> None:
        self._hits = hits

    async def upsert_chunks(self, collection: str, chunks: list[Chunk]) -> None:
        raise NotImplementedError

    async def search(self, collection: str, query_embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        return self._hits[:top_k]

    async def delete_document_chunks(self, collection: str, document_id: int) -> None:
        raise NotImplementedError


class FakeASRProvider(ASRProvider):
    def __init__(self, text: str) -> None:
        self._text = text
        self.transcribe_call_count = 0
        self.received_audio: list[bytes] = []

    @property
    def provider_name(self) -> str:
        return "fake-asr"

    @property
    def model_name(self) -> str:
        return "fake-whisper"

    async def transcribe(self, audio_data: bytes, language_hint: str | None = None) -> TranscriptionResult:
        self.transcribe_call_count += 1
        self.received_audio.append(audio_data)
        return TranscriptionResult(text=self._text, language="en", confidence=0.95, duration_seconds=1.2)


class FakeTTSProvider(TTSProvider):
    def __init__(self) -> None:
        self.synthesize_call_count = 0
        self.received_text: list[str] = []
        self.received_voice: list[str | None] = []

    @property
    def provider_name(self) -> str:
        return "fake-tts"

    @property
    def voice_name(self) -> str:
        return "fake-voice"

    async def synthesize(self, text: str, *, voice: str | None = None) -> SynthesisResult:
        self.synthesize_call_count += 1
        self.received_text.append(text)
        self.received_voice.append(voice)
        return SynthesisResult(audio_data=b"FAKE_WAV_BYTES", content_type="audio/wav", sample_rate=16000, duration_seconds=0.5)


def _build_test_pdf() -> bytes:
    document = pymupdf.open()
    page = document.new_page(width=400, height=600)
    page.insert_text((50, 50), "Le present tense chunk. Le present s'utilise pour des actions habituelles.")
    pdf_bytes = document.tobytes()
    document.close()
    return pdf_bytes


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


def _clear_overrides() -> None:
    for dep in (get_provider, get_embedding_provider, get_vision_provider, get_vector_store, get_asr_provider, get_tts_provider):
        app.dependency_overrides.pop(dep, None)


# --- init / ready handshake ---


def test_voice_ws_init_creates_session_and_sends_ready(override_get_db: None) -> None:
    client = TestClient(app)

    with client.websocket_connect("/api/voice") as ws:
        ws.send_json({"type": "init", "subject": "general"})
        ready = ws.receive_json()

    assert ready["type"] == "ready"
    assert isinstance(ready["session_id"], str) and ready["session_id"]


def test_voice_ws_init_reuses_existing_session_id(override_get_db: None) -> None:
    client = TestClient(app)

    with client.websocket_connect("/api/voice") as ws:
        ws.send_json({"type": "init", "subject": "general"})
        first_ready = ws.receive_json()

    with client.websocket_connect("/api/voice") as ws:
        ws.send_json({"type": "init", "session_id": first_ready["session_id"]})
        second_ready = ws.receive_json()

    assert second_ready["session_id"] == first_ready["session_id"]


def test_voice_ws_init_unknown_session_id_returns_error(override_get_db: None) -> None:
    client = TestClient(app)

    with client.websocket_connect("/api/voice") as ws:
        ws.send_json({"type": "init", "session_id": "does-not-exist"})
        response = ws.receive_json()

    assert response["type"] == "error"
    assert "does-not-exist" in response["detail"]


def test_voice_ws_malformed_init_message_returns_error(override_get_db: None) -> None:
    client = TestClient(app)

    with client.websocket_connect("/api/voice") as ws:
        ws.send_json({"type": "not_init"})
        response = ws.receive_json()

    assert response["type"] == "error"


# --- full voice turn: grounded question ---


def test_voice_ws_full_turn_returns_grounded_reply_and_audio(override_get_db: None) -> None:
    app.dependency_overrides[get_embedding_provider] = lambda: FakeEmbeddingProvider()
    app.dependency_overrides[get_vision_provider] = lambda: FakeVisionProvider()
    app.dependency_overrides[get_vector_store] = lambda: FakeVectorStore()
    client = TestClient(app)

    reply_text = "Le present tense est utilise pour des actions habituelles. (page 7)"
    fake_provider = FakeProvider(reply=reply_text)
    fake_asr = FakeASRProvider(text="How does the present tense work?")
    fake_tts = FakeTTSProvider()

    try:
        # A document must exist for the subject before the turn will retrieve+answer.
        client.post(
            "/api/subjects/french/documents",
            files={"file": ("chapter1.pdf", _build_test_pdf(), "application/pdf")},
        )

        app.dependency_overrides[get_provider] = lambda: fake_provider
        app.dependency_overrides[get_asr_provider] = lambda: fake_asr
        app.dependency_overrides[get_tts_provider] = lambda: fake_tts
        app.dependency_overrides[get_vector_store] = lambda: FixedResultVectorStore(
            [SearchResult(text="Le present tense chunk.", document_id=1, page_number=7, distance=0.1)]
        )

        with client.websocket_connect("/api/voice") as ws:
            ws.send_json({"type": "init", "subject": "french"})
            ready = ws.receive_json()
            session_id = ready["session_id"]

            ws.send_json({"type": "audio_start"})
            ws.send_bytes(b"fake-webm-opus-bytes")
            ws.send_json({"type": "audio_end"})

            response = ws.receive_json()
            audio_frame = ws.receive_bytes()
            turn_complete = ws.receive_json()

        usage_response = client.get(f"/api/sessions/{session_id}/usage")
    finally:
        _clear_overrides()

    assert response["type"] == "response"
    assert response["transcription"] == "How does the present tense work?"
    assert response["reply"] == reply_text
    assert response["session_id"] == session_id
    assert response["is_command"] is False
    assert response["command_type"] is None
    assert response["new_subject"] is None
    assert response["sources"] == [{"page_number": 7, "score": pytest.approx(0.9)}]

    assert audio_frame == b"FAKE_WAV_BYTES"
    assert turn_complete == {"type": "turn_complete"}

    assert fake_asr.transcribe_call_count == 1
    assert fake_asr.received_audio == [b"fake-webm-opus-bytes"]

    # TTS receives the spoken-form-preprocessed reply -- the "(page 7)"
    # citation is visual, not spoken, so it should be stripped before TTS.
    assert fake_tts.synthesize_call_count == 1
    assert "(page 7)" not in fake_tts.received_text[0]
    assert "page 7" not in fake_tts.received_text[0]

    usage_body = usage_response.json()
    event_types = {event["event_type"] for event in usage_body["events"]}
    assert "asr" in event_types
    assert "tts" in event_types


def test_voice_ws_empty_audio_returns_error_and_stays_open(override_get_db: None) -> None:
    app.dependency_overrides[get_embedding_provider] = lambda: FakeEmbeddingProvider()
    app.dependency_overrides[get_vision_provider] = lambda: FakeVisionProvider()
    app.dependency_overrides[get_vector_store] = lambda: FakeVectorStore()
    client = TestClient(app)

    fake_asr = FakeASRProvider(text="unused")

    try:
        app.dependency_overrides[get_provider] = lambda: FakeProvider(reply="unused")
        app.dependency_overrides[get_asr_provider] = lambda: fake_asr
        app.dependency_overrides[get_tts_provider] = lambda: FakeTTSProvider()

        with client.websocket_connect("/api/voice") as ws:
            ws.send_json({"type": "init", "subject": "general"})
            ws.receive_json()

            ws.send_json({"type": "audio_start"})
            ws.send_json({"type": "audio_end"})
            first = ws.receive_json()
            assert first["type"] == "error"
            assert "No audio" in first["detail"]
            assert fake_asr.transcribe_call_count == 0

            # The connection must still be usable for a subsequent turn.
            ws.send_json({"type": "audio_start"})
            ws.send_bytes(b"some-audio")
            ws.send_json({"type": "audio_end"})
            second = ws.receive_json()
            assert second["type"] == "response"
            assert fake_asr.transcribe_call_count == 1
    finally:
        _clear_overrides()


def test_voice_ws_audio_bytes_before_audio_start_returns_error(override_get_db: None) -> None:
    client = TestClient(app)

    with client.websocket_connect("/api/voice") as ws:
        ws.send_json({"type": "init", "subject": "general"})
        ws.receive_json()

        ws.send_bytes(b"stray audio bytes")
        response = ws.receive_json()

    assert response["type"] == "error"
    assert "audio_start" in response["detail"]


def test_voice_ws_response_includes_visual_directives(override_get_db: None) -> None:
    app.dependency_overrides[get_embedding_provider] = lambda: FakeEmbeddingProvider()
    app.dependency_overrides[get_vision_provider] = lambda: FakeVisionProvider()
    app.dependency_overrides[get_vector_store] = lambda: FakeVectorStore()
    client = TestClient(app)

    reply_text = "The **present tense** is used for habitual actions."
    fake_provider = FakeProvider(reply=reply_text)
    fake_asr = FakeASRProvider(text="How does the present tense work?")
    fake_tts = FakeTTSProvider()

    try:
        client.post(
            "/api/subjects/french/documents",
            files={"file": ("chapter1.pdf", _build_test_pdf(), "application/pdf")},
        )

        app.dependency_overrides[get_provider] = lambda: fake_provider
        app.dependency_overrides[get_asr_provider] = lambda: fake_asr
        app.dependency_overrides[get_tts_provider] = lambda: fake_tts
        app.dependency_overrides[get_vector_store] = lambda: FixedResultVectorStore(
            [SearchResult(text="Le present tense chunk.", document_id=1, page_number=7, distance=0.1)]
        )

        with client.websocket_connect("/api/voice") as ws:
            ws.send_json({"type": "init", "subject": "french"})
            ws.receive_json()

            ws.send_json({"type": "audio_start"})
            ws.send_bytes(b"fake-webm-opus-bytes")
            ws.send_json({"type": "audio_end"})

            response = ws.receive_json()
            ws.receive_bytes()
            ws.receive_json()
    finally:
        _clear_overrides()

    assert response["visual_directives"] == [
        {"directive_type": "highlight", "content": "present tense", "label": None}
    ]


def test_voice_ws_init_voice_is_passed_to_tts_provider(override_get_db: None) -> None:
    app.dependency_overrides[get_embedding_provider] = lambda: FakeEmbeddingProvider()
    app.dependency_overrides[get_vision_provider] = lambda: FakeVisionProvider()
    app.dependency_overrides[get_vector_store] = lambda: FakeVectorStore()
    client = TestClient(app)

    fake_asr = FakeASRProvider(text="How does the present tense work?")
    fake_tts = FakeTTSProvider()

    try:
        client.post(
            "/api/subjects/french/documents",
            files={"file": ("chapter1.pdf", _build_test_pdf(), "application/pdf")},
        )

        app.dependency_overrides[get_provider] = lambda: FakeProvider(reply="unused")
        app.dependency_overrides[get_asr_provider] = lambda: fake_asr
        app.dependency_overrides[get_tts_provider] = lambda: fake_tts
        app.dependency_overrides[get_vector_store] = lambda: FixedResultVectorStore(
            [SearchResult(text="Le present tense chunk.", document_id=1, page_number=7, distance=0.1)]
        )

        with client.websocket_connect("/api/voice") as ws:
            ws.send_json({"type": "init", "subject": "french", "voice": "en-IN-NeerjaNeural"})
            ws.receive_json()

            ws.send_json({"type": "audio_start"})
            ws.send_bytes(b"fake-webm-opus-bytes")
            ws.send_json({"type": "audio_end"})

            ws.receive_json()
            ws.receive_bytes()
            ws.receive_json()
    finally:
        _clear_overrides()

    assert fake_tts.received_voice == ["en-IN-NeerjaNeural"]


def test_voice_ws_init_without_voice_passes_none_to_tts_provider(override_get_db: None) -> None:
    client = TestClient(app)

    fake_tts = FakeTTSProvider()

    try:
        app.dependency_overrides[get_provider] = lambda: FakeProvider(reply="unused")
        app.dependency_overrides[get_asr_provider] = lambda: FakeASRProvider(text="switch to Physics")
        app.dependency_overrides[get_tts_provider] = lambda: fake_tts

        with client.websocket_connect("/api/voice") as ws:
            ws.send_json({"type": "init", "subject": "general"})
            ws.receive_json()

            ws.send_json({"type": "audio_start"})
            ws.send_bytes(b"fake-audio")
            ws.send_json({"type": "audio_end"})

            ws.receive_json()
            ws.receive_bytes()
            ws.receive_json()
    finally:
        _clear_overrides()

    assert fake_tts.received_voice == [None]


# --- command routing through the WebSocket ---


def test_voice_ws_command_routes_without_hitting_llm(override_get_db: None) -> None:
    app.dependency_overrides[get_embedding_provider] = lambda: FakeEmbeddingProvider()
    app.dependency_overrides[get_vision_provider] = lambda: FakeVisionProvider()
    app.dependency_overrides[get_vector_store] = lambda: FakeVectorStore()
    client = TestClient(app)

    fake_provider = FakeProvider(reply="unused")
    fake_asr = FakeASRProvider(text="switch to French")
    fake_tts = FakeTTSProvider()

    try:
        # Establish "french" as a real subject first (as uploading a
        # textbook would in practice), so the switch command has something
        # to fuzzy-match against.
        client.post(
            "/api/subjects/french/documents",
            files={"file": ("chapter1.pdf", _build_test_pdf(), "application/pdf")},
        )

        app.dependency_overrides[get_provider] = lambda: fake_provider
        app.dependency_overrides[get_asr_provider] = lambda: fake_asr
        app.dependency_overrides[get_tts_provider] = lambda: fake_tts

        with client.websocket_connect("/api/voice") as ws:
            ws.send_json({"type": "init", "subject": "general"})
            ready = ws.receive_json()
            original_session_id = ready["session_id"]

            ws.send_json({"type": "audio_start"})
            ws.send_bytes(b"fake-audio")
            ws.send_json({"type": "audio_end"})

            response = ws.receive_json()
            ws.receive_bytes()
            ws.receive_json()
    finally:
        _clear_overrides()

    assert response["type"] == "response"
    assert response["is_command"] is True
    assert response["command_type"] == "switch_subject"
    assert response["transcription"] == "switch to French"
    assert "french" in response["reply"].lower()
    # A successful switch starts a fresh session under the new subject.
    assert response["session_id"] != original_session_id
    assert response["new_subject"] is not None
    assert response["new_subject"].lower() == "french"

    # No LLM call and no TTS text should still have been produced for the
    # command's own confirmation text.
    assert fake_provider.call_count == 0
    assert fake_tts.synthesize_call_count == 1
    assert fake_tts.received_text[0] == response["reply"]


def test_voice_ws_command_unknown_subject_does_not_change_session(override_get_db: None) -> None:
    client = TestClient(app)

    fake_asr = FakeASRProvider(text="switch to Physics")

    try:
        app.dependency_overrides[get_provider] = lambda: FakeProvider(reply="unused")
        app.dependency_overrides[get_asr_provider] = lambda: fake_asr
        app.dependency_overrides[get_tts_provider] = lambda: FakeTTSProvider()

        with client.websocket_connect("/api/voice") as ws:
            ws.send_json({"type": "init", "subject": "general"})
            ready = ws.receive_json()
            original_session_id = ready["session_id"]

            ws.send_json({"type": "audio_start"})
            ws.send_bytes(b"fake-audio")
            ws.send_json({"type": "audio_end"})

            response = ws.receive_json()
            ws.receive_bytes()
            ws.receive_json()
    finally:
        _clear_overrides()

    assert response["is_command"] is True
    assert response["command_type"] == "switch_subject"
    assert response["session_id"] == original_session_id
    assert response["new_subject"] is None
    assert "Physics" in response["reply"]
