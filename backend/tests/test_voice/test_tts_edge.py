from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping

import pytest

from app.voice.tts_edge import EdgeTTSAdapter


class _FakeCommunicate:
    """Stands in for edge_tts.Communicate -- same .stream() shape (an async
    iterator of chunk dicts), no real network call."""

    def __init__(self, chunks: list[Mapping[str, object]], error: Exception | None = None) -> None:
        self._chunks = chunks
        self._error = error

    async def stream(self) -> AsyncIterator[Mapping[str, object]]:
        if self._error is not None:
            raise self._error
        for chunk in self._chunks:
            yield chunk


def test_construct_adapter() -> None:
    adapter = EdgeTTSAdapter(voice="en-US-AriaNeural")

    assert adapter.provider_name == "edge-tts"
    assert adapter.voice_name == "en-US-AriaNeural"


def test_synthesize_collects_audio_chunks_and_ignores_boundaries() -> None:
    chunks: list[Mapping[str, object]] = [
        {"type": "audio", "data": b"abc"},
        {"type": "WordBoundary", "offset": 0, "duration": 1},
        {"type": "audio", "data": b"def"},
    ]

    def factory(text: str, voice: str) -> _FakeCommunicate:
        return _FakeCommunicate(chunks)

    adapter = EdgeTTSAdapter(voice="en-US-AriaNeural", communicate_factory=factory)  # type: ignore[arg-type]

    result = asyncio.run(adapter.synthesize("Bonjour"))

    assert result.audio_data == b"abcdef"
    assert result.content_type == "audio/mpeg"
    assert result.sample_rate is None


def test_synthesize_voice_param_overrides_adapter_default() -> None:
    received_voices: list[str] = []

    def factory(text: str, voice: str) -> _FakeCommunicate:
        received_voices.append(voice)
        return _FakeCommunicate([{"type": "audio", "data": b"abc"}])

    adapter = EdgeTTSAdapter(voice="en-US-AriaNeural", communicate_factory=factory)  # type: ignore[arg-type]

    asyncio.run(adapter.synthesize("Bonjour", voice="fr-FR-DeniseNeural"))

    assert received_voices == ["fr-FR-DeniseNeural"]


def test_synthesize_without_voice_param_uses_adapter_default() -> None:
    received_voices: list[str] = []

    def factory(text: str, voice: str) -> _FakeCommunicate:
        received_voices.append(voice)
        return _FakeCommunicate([{"type": "audio", "data": b"abc"}])

    adapter = EdgeTTSAdapter(voice="en-US-AriaNeural", communicate_factory=factory)  # type: ignore[arg-type]

    asyncio.run(adapter.synthesize("Bonjour"))

    assert received_voices == ["en-US-AriaNeural"]


def test_synthesize_raises_on_empty_audio() -> None:
    def factory(text: str, voice: str) -> _FakeCommunicate:
        return _FakeCommunicate([])

    adapter = EdgeTTSAdapter(communicate_factory=factory)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="no audio"):
        asyncio.run(adapter.synthesize("Bonjour"))


def test_synthesize_wraps_stream_errors() -> None:
    def factory(text: str, voice: str) -> _FakeCommunicate:
        return _FakeCommunicate([], error=ConnectionError("network down"))

    adapter = EdgeTTSAdapter(communicate_factory=factory)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="failed to synthesize"):
        asyncio.run(adapter.synthesize("Bonjour"))


@pytest.mark.slow
def test_real_synthesis_smoke() -> None:
    """Requires network access to Microsoft's edge-tts service."""
    adapter = EdgeTTSAdapter()

    result = asyncio.run(adapter.synthesize("Hello"))

    assert len(result.audio_data) > 0
    assert result.content_type == "audio/mpeg"
