"""Voice module — ASR/TTS abstraction and the spoken-form preprocessor.

Public surface: the `ASRProvider`/`TTSProvider` interfaces, their result
types, `SpokenFormPreprocessor`, and their `get_*()` factories, which read
the environment and return the configured adapter. Callers (orchestration,
the voice WebSocket endpoint) must depend on these, never import a concrete
adapter directly — the same dependency-inversion convention as
app.models (technical-design.md).
"""

from __future__ import annotations

import os

from app.config import VOICE_DEFAULTS

from .asr_whisper import FasterWhisperAdapter
from .base import ASRProvider, SynthesisResult, TranscriptionResult, TTSProvider
from .spoken_form import SpokenFormPreprocessor
from .tts_edge import EdgeTTSAdapter
from .tts_piper import PiperAdapter

__all__ = [
    "ASRProvider",
    "SpokenFormPreprocessor",
    "SynthesisResult",
    "TTSProvider",
    "TranscriptionResult",
    "get_asr_provider",
    "get_tts_provider",
]

# Local/offline ASR is lightweight enough to stay the default on this CPU
# (technical-design.md's hardware assessment keeps ASR/TTS local while the
# LLM call itself defaults to the cloud). edge-tts, not Piper, is the
# practical TTS default on Windows -- Piper's wheel support there is
# inconsistent enough not to rely on, and the app already needs internet
# for the Anthropic API. See app.config.VOICE_DEFAULTS for the fallback
# values below.


def get_asr_provider() -> ASRProvider:
    """Build the ASR adapter configured via the ASR_PROVIDER env var."""
    provider = os.getenv("ASR_PROVIDER", VOICE_DEFAULTS.asr_provider).lower()

    if provider == "whisper":
        return FasterWhisperAdapter(
            model_size=os.getenv("WHISPER_MODEL_SIZE", VOICE_DEFAULTS.whisper_model_size),
            device=os.getenv("WHISPER_DEVICE", VOICE_DEFAULTS.whisper_device),
        )

    raise ValueError(f"Unknown ASR_PROVIDER: {provider!r} (expected 'whisper')")


def get_tts_provider() -> TTSProvider:
    """Build the TTS adapter configured via the TTS_PROVIDER env var."""
    provider = os.getenv("TTS_PROVIDER", VOICE_DEFAULTS.tts_provider).lower()

    if provider == "edge":
        return EdgeTTSAdapter(voice=os.getenv("EDGE_TTS_VOICE", VOICE_DEFAULTS.edge_voice))

    if provider == "piper":
        try:
            import piper  # noqa: F401
        except ImportError as exc:
            raise ValueError(
                "TTS_PROVIDER=piper but piper-tts is not installed. Run `pip install "
                "piper-tts`, or set TTS_PROVIDER=edge."
            ) from exc
        return PiperAdapter(
            voice=os.getenv("PIPER_VOICE", VOICE_DEFAULTS.piper_voice),
            data_dir=os.getenv("PIPER_DATA_DIR", VOICE_DEFAULTS.piper_data_dir),
        )

    raise ValueError(f"Unknown TTS_PROVIDER: {provider!r} (expected 'edge' or 'piper')")
