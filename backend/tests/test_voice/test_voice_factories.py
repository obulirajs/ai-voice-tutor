from __future__ import annotations

import sys

import pytest

from app.voice import get_asr_provider, get_tts_provider
from app.voice.asr_whisper import FasterWhisperAdapter
from app.voice.tts_edge import EdgeTTSAdapter
from app.voice.tts_piper import PiperAdapter


def test_get_asr_provider_defaults_to_whisper(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ASR_PROVIDER", raising=False)

    provider = get_asr_provider()

    assert isinstance(provider, FasterWhisperAdapter)
    assert provider.provider_name == "faster-whisper"


def test_get_asr_provider_reads_model_size_and_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASR_PROVIDER", "whisper")
    monkeypatch.setenv("WHISPER_MODEL_SIZE", "small")
    monkeypatch.setenv("WHISPER_DEVICE", "cpu")

    provider = get_asr_provider()

    assert isinstance(provider, FasterWhisperAdapter)
    assert provider.model_name == "small"


def test_get_asr_provider_rejects_unknown_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASR_PROVIDER", "bogus")

    with pytest.raises(ValueError):
        get_asr_provider()


def test_get_tts_provider_defaults_to_edge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TTS_PROVIDER", raising=False)

    provider = get_tts_provider()

    assert isinstance(provider, EdgeTTSAdapter)
    assert provider.provider_name == "edge-tts"


def test_get_tts_provider_reads_edge_voice(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TTS_PROVIDER", "edge")
    monkeypatch.setenv("EDGE_TTS_VOICE", "en-GB-SoniaNeural")

    provider = get_tts_provider()

    assert isinstance(provider, EdgeTTSAdapter)
    assert provider.voice_name == "en-GB-SoniaNeural"


def test_get_tts_provider_returns_piper_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TTS_PROVIDER", "piper")
    monkeypatch.setenv("PIPER_VOICE", "en_US-lessac-medium")

    provider = get_tts_provider()

    assert isinstance(provider, PiperAdapter)
    assert provider.voice_name == "en_US-lessac-medium"


def test_get_tts_provider_rejects_unknown_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TTS_PROVIDER", "bogus")

    with pytest.raises(ValueError):
        get_tts_provider()


def test_get_tts_provider_piper_missing_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """If piper-tts isn't installed, get_tts_provider(piper) should raise
    ValueError (a config problem), not ImportError (an internal one)."""
    monkeypatch.setenv("TTS_PROVIDER", "piper")
    # sys.modules[name] = None is the documented way to make `import name`
    # raise ImportError, without needing to fake __import__ itself.
    monkeypatch.setitem(sys.modules, "piper", None)  # type: ignore[misc]

    with pytest.raises(ValueError, match="piper-tts is not installed"):
        get_tts_provider()
