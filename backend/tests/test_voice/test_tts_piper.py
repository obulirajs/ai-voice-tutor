from __future__ import annotations

import asyncio
import io
import wave
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

# Skips this whole module gracefully if piper-tts isn't installed (its wheel
# is primarily Linux -- see technical-design.md) rather than failing import.
pytest.importorskip("piper")

import numpy as np

from app.voice.tts_piper import PiperAdapter


@dataclass
class _FakeChunk:
    sample_rate: int
    sample_width: int
    sample_channels: int
    audio_int16_bytes: bytes
    audio_int16_array: np.ndarray


class _FakeVoice:
    """Stands in for piper.PiperVoice -- same .synthesize() shape (an
    iterable of audio chunks), no real model load."""

    def __init__(self, chunks: list[_FakeChunk], error: Exception | None = None) -> None:
        self._chunks = chunks
        self._error = error
        self.config = SimpleNamespace(sample_rate=22050)

    def synthesize(self, text: str) -> object:
        if self._error is not None:
            raise self._error
        return iter(self._chunks)


def test_construct_adapter() -> None:
    adapter = PiperAdapter(voice="en_US-lessac-medium")

    assert adapter.provider_name == "piper"
    assert adapter.voice_name == "en_US-lessac-medium"


def test_synthesize_produces_valid_wav_bytes() -> None:
    samples = np.zeros(100, dtype=np.int16)
    chunk = _FakeChunk(
        sample_rate=22050,
        sample_width=2,
        sample_channels=1,
        audio_int16_bytes=samples.tobytes(),
        audio_int16_array=samples,
    )
    fake_voice = _FakeVoice([chunk])
    adapter = PiperAdapter(voice_instance=fake_voice)  # type: ignore[arg-type]

    result = asyncio.run(adapter.synthesize("Bonjour"))

    assert result.content_type == "audio/wav"
    assert result.sample_rate == 22050
    assert result.duration_seconds == pytest.approx(100 / 22050)

    with wave.open(io.BytesIO(result.audio_data)) as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getframerate() == 22050
        assert wav_file.getsampwidth() == 2


def test_synthesize_ignores_voice_param() -> None:
    samples = np.zeros(100, dtype=np.int16)
    chunk = _FakeChunk(
        sample_rate=22050,
        sample_width=2,
        sample_channels=1,
        audio_int16_bytes=samples.tobytes(),
        audio_int16_array=samples,
    )
    fake_voice = _FakeVoice([chunk])
    adapter = PiperAdapter(voice="en_US-lessac-medium", voice_instance=fake_voice)  # type: ignore[arg-type]

    result = asyncio.run(adapter.synthesize("Bonjour", voice="fr-FR-DeniseNeural"))

    # Piper can't swap voices per-call -- falls back to the adapter's own
    # configured voice, so this just needs to succeed rather than error.
    assert result.content_type == "audio/wav"
    assert adapter.voice_name == "en_US-lessac-medium"


def test_synthesize_wraps_errors() -> None:
    fake_voice = _FakeVoice([], error=RuntimeError("model exploded"))
    adapter = PiperAdapter(voice_instance=fake_voice)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="failed to synthesize"):
        asyncio.run(adapter.synthesize("Bonjour"))


@pytest.mark.slow
def test_real_synthesis_smoke() -> None:
    """Requires downloading the real Piper voice model on first run."""
    adapter = PiperAdapter()

    result = asyncio.run(adapter.synthesize("Hello"))

    assert len(result.audio_data) > 0
    assert result.content_type == "audio/wav"
