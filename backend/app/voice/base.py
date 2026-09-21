from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: str | None
    confidence: float | None
    duration_seconds: float | None


@dataclass(frozen=True)
class SynthesisResult:
    audio_data: bytes
    # e.g. "audio/wav" (Piper) or "audio/mpeg" (edge-tts, which outputs MP3
    # natively) -- lets callers/consumers handle either without transcoding.
    content_type: str
    sample_rate: int | None
    duration_seconds: float | None


class ASRProvider(ABC):
    """Common interface every speech-to-text adapter (faster-whisper, ...)
    implements. Orchestration and the voice WebSocket endpoint depend on this
    interface, never on a concrete adapter — see technical-design.md's
    dependency-inversion convention, same pattern as ModelProvider.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Short adapter name (e.g. "faster-whisper") — logged on usage_events."""
        raise NotImplementedError

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The configured model name/size — logged on usage_events."""
        raise NotImplementedError

    @abstractmethod
    async def transcribe(self, audio_data: bytes, language_hint: str | None = None) -> TranscriptionResult:
        """Transcribe one clip of audio (any container faster-whisper decodes:
        WAV, WebM, MP3, ...) to text. language_hint, when given, skips
        language auto-detection.
        """
        raise NotImplementedError


class TTSProvider(ABC):
    """Common interface every text-to-speech adapter (Piper, edge-tts, ...)
    implements — the mirror of ASRProvider.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Short adapter name (e.g. "piper", "edge-tts") — logged on usage_events."""
        raise NotImplementedError

    @property
    @abstractmethod
    def voice_name(self) -> str:
        """The configured voice — logged on usage_events."""
        raise NotImplementedError

    @abstractmethod
    async def synthesize(self, text: str, *, voice: str | None = None) -> SynthesisResult:
        """Synthesize spoken audio for one piece of text. Callers should run
        text through SpokenFormPreprocessor first -- this method reads text
        literally.

        voice, when given, overrides the adapter's configured default voice
        for this call only (see api/tts.py's voice picker and the voice
        WebSocket's per-connection voice selection). An adapter that can't
        honor per-call voice selection (Piper: swapping voices means loading
        a different local model) ignores it and falls back to its own
        configured voice.
        """
        raise NotImplementedError
