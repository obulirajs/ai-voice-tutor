from __future__ import annotations

import asyncio
import io
import wave
from dataclasses import dataclass

import pytest

from app.voice.asr_whisper import FasterWhisperAdapter


@dataclass
class _FakeSegment:
    text: str
    avg_logprob: float


@dataclass
class _FakeInfo:
    language: str
    duration: float


class _FakeWhisperModel:
    """Stands in for faster_whisper.WhisperModel -- same .transcribe() shape
    (returns an iterable of segments plus an info object), no real model load."""

    def __init__(self, segments: list[_FakeSegment], info: _FakeInfo, error: Exception | None = None) -> None:
        self._segments = segments
        self._info = info
        self._error = error

    def transcribe(
        self, audio_path: str, language: str | None = None, beam_size: int = 1
    ) -> tuple[object, _FakeInfo]:
        if self._error is not None:
            raise self._error
        return iter(self._segments), self._info


def test_construct_adapter_does_not_load_model() -> None:
    # Constructing shouldn't touch the network/filesystem -- lazy load only.
    adapter = FasterWhisperAdapter(model_size="base", device="cpu")

    assert adapter.provider_name == "faster-whisper"
    assert adapter.model_name == "base"


def test_transcribe_returns_joined_text_and_metadata() -> None:
    fake_model = _FakeWhisperModel(
        segments=[
            _FakeSegment(text=" Bonjour", avg_logprob=-0.1),
            _FakeSegment(text=" le monde", avg_logprob=-0.3),
        ],
        info=_FakeInfo(language="fr", duration=2.5),
    )
    adapter = FasterWhisperAdapter(model=fake_model)  # type: ignore[arg-type]

    result = asyncio.run(adapter.transcribe(b"fake audio bytes"))

    assert result.text == "Bonjour le monde"
    assert result.language == "fr"
    assert result.duration_seconds == 2.5
    assert result.confidence is not None
    assert 0 < result.confidence <= 1


def test_transcribe_with_no_segments_has_no_confidence() -> None:
    fake_model = _FakeWhisperModel(segments=[], info=_FakeInfo(language="en", duration=0.0))
    adapter = FasterWhisperAdapter(model=fake_model)  # type: ignore[arg-type]

    result = asyncio.run(adapter.transcribe(b""))

    assert result.text == ""
    assert result.confidence is None


def test_transcribe_wraps_errors_in_runtime_error() -> None:
    fake_model = _FakeWhisperModel(
        segments=[],
        info=_FakeInfo(language="en", duration=0.0),
        error=RuntimeError("bad audio"),
    )
    adapter = FasterWhisperAdapter(model=fake_model)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="failed to transcribe"):
        asyncio.run(adapter.transcribe(b"not valid audio"))


@pytest.mark.slow
def test_real_transcription_smoke() -> None:
    """Requires the real faster-whisper model (downloads ~150MB on first run)."""
    adapter = FasterWhisperAdapter(model_size="base", device="cpu")

    # A tiny silent WAV clip -- proves the real pipeline runs end to end,
    # not testing transcription accuracy.
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * 16000)

    result = asyncio.run(adapter.transcribe(buffer.getvalue()))

    assert isinstance(result.text, str)
    assert result.duration_seconds is not None
