from __future__ import annotations

import asyncio
import math
import os
import tempfile
from typing import TYPE_CHECKING

from .base import ASRProvider, TranscriptionResult

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

_DEFAULT_MODEL_SIZE = "base"
_DEFAULT_DEVICE = "cpu"
# Greedy decoding -- fine for conversational speech and meaningfully faster
# than the library's beam_size=5 default on a CPU-only machine.
_BEAM_SIZE = 1


class FasterWhisperAdapter(ASRProvider):
    """Local ASR adapter backed by faster-whisper (CTranslate2). Default
    local/offline speech-to-text -- see technical-design.md's hardware
    assessment: ASR stays local since it's lightweight enough for this CPU.
    """

    def __init__(
        self,
        model_size: str = _DEFAULT_MODEL_SIZE,
        device: str = _DEFAULT_DEVICE,
        model: WhisperModel | None = None,
    ) -> None:
        self._model_size = model_size
        self._device = device
        # Lazy-loaded on first transcribe() -- the model download (~150MB for
        # "base") and load are too slow to do at construction/import time.
        # `model` lets tests inject a fake without touching the real thing.
        self._model = model

    @property
    def provider_name(self) -> str:
        return "faster-whisper"

    @property
    def model_name(self) -> str:
        return self._model_size

    def _get_model(self) -> WhisperModel:
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError(
                    "faster-whisper is not installed. Run `pip install faster-whisper`."
                ) from exc
            try:
                self._model = WhisperModel(self._model_size, device=self._device)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load faster-whisper model {self._model_size!r} on device "
                    f"{self._device!r}: {exc}"
                ) from exc
        return self._model

    async def transcribe(self, audio_data: bytes, language_hint: str | None = None) -> TranscriptionResult:
        return await asyncio.to_thread(self._transcribe_sync, audio_data, language_hint)

    def _transcribe_sync(self, audio_data: bytes, language_hint: str | None) -> TranscriptionResult:
        model = self._get_model()

        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as tmp_file:
                tmp_file.write(audio_data)
                tmp_path = tmp_file.name

            segments_iter, info = model.transcribe(tmp_path, language=language_hint, beam_size=_BEAM_SIZE)
            segments = list(segments_iter)
        except Exception as exc:
            raise RuntimeError(f"faster-whisper failed to transcribe audio: {exc}") from exc
        finally:
            if tmp_path is not None:
                os.unlink(tmp_path)

        text = "".join(segment.text for segment in segments).strip()
        confidences = [math.exp(segment.avg_logprob) for segment in segments]
        confidence = sum(confidences) / len(confidences) if confidences else None

        return TranscriptionResult(
            text=text,
            language=info.language,
            confidence=confidence,
            duration_seconds=info.duration,
        )
