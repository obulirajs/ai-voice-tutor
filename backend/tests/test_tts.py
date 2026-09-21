from __future__ import annotations

from starlette.testclient import TestClient

from app.main import app
from app.voice import SynthesisResult, TTSProvider, get_tts_provider


class FakeTTSProvider(TTSProvider):
    def __init__(self) -> None:
        self.received_text: list[str] = []
        self.received_voice: list[str | None] = []

    @property
    def provider_name(self) -> str:
        return "fake-tts"

    @property
    def voice_name(self) -> str:
        return "fake-voice"

    async def synthesize(self, text: str, *, voice: str | None = None) -> SynthesisResult:
        self.received_text.append(text)
        self.received_voice.append(voice)
        return SynthesisResult(audio_data=b"FAKE_AUDIO_BYTES", content_type="audio/mpeg", sample_rate=None, duration_seconds=None)


def test_tts_endpoint_returns_audio_bytes() -> None:
    fake = FakeTTSProvider()
    app.dependency_overrides[get_tts_provider] = lambda: fake
    client = TestClient(app)

    try:
        response = client.post("/api/tts", json={"text": "Hello there"})
    finally:
        app.dependency_overrides.pop(get_tts_provider, None)

    assert response.status_code == 200
    assert response.content == b"FAKE_AUDIO_BYTES"
    assert response.headers["content-type"] == "audio/mpeg"
    assert fake.received_text == ["Hello there"]


def test_tts_endpoint_strips_citations_before_synthesis() -> None:
    fake = FakeTTSProvider()
    app.dependency_overrides[get_tts_provider] = lambda: fake
    client = TestClient(app)

    try:
        client.post("/api/tts", json={"text": "The answer is 42 (page 7)."})
    finally:
        app.dependency_overrides.pop(get_tts_provider, None)

    assert "(page 7)" not in fake.received_text[0]


def test_tts_endpoint_with_voice_passes_voice_to_provider() -> None:
    fake = FakeTTSProvider()
    app.dependency_overrides[get_tts_provider] = lambda: fake
    client = TestClient(app)

    try:
        response = client.post("/api/tts", json={"text": "Hello", "voice": "en-IN-NeerjaNeural"})
    finally:
        app.dependency_overrides.pop(get_tts_provider, None)

    assert response.status_code == 200
    assert fake.received_voice == ["en-IN-NeerjaNeural"]


def test_tts_endpoint_defaults_voice_to_none() -> None:
    fake = FakeTTSProvider()
    app.dependency_overrides[get_tts_provider] = lambda: fake
    client = TestClient(app)

    try:
        client.post("/api/tts", json={"text": "Hello"})
    finally:
        app.dependency_overrides.pop(get_tts_provider, None)

    assert fake.received_voice == [None]


def test_voices_endpoint_returns_available_voices() -> None:
    client = TestClient(app)

    response = client.get("/api/voices")

    assert response.status_code == 200
    body = response.json()
    assert len(body) > 0
    for voice in body:
        assert {"id", "name", "language", "gender", "description"} <= voice.keys()
    assert any(voice["id"] == "en-US-AriaNeural" for voice in body)
