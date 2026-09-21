from __future__ import annotations

import asyncio
import io
import wave
from pathlib import Path
from typing import TYPE_CHECKING

from .base import SynthesisResult, TTSProvider

if TYPE_CHECKING:
    from piper import PiperVoice

_DEFAULT_VOICE = "en_US-lessac-medium"
_DEFAULT_DATA_DIR = "./piper_voices"


class PiperAdapter(TTSProvider):
    """Local TTS adapter backed by Piper (ONNX) -- offline, fast on CPU, the
    architecture.md default. If piper-tts isn't installed (its wheel is
    primarily Linux; see technical-design.md), this stays importable and
    raises a clear error only when actually used, so a Windows dev machine
    running TTS_PROVIDER=edge is unaffected.
    """

    def __init__(
        self,
        voice: str = _DEFAULT_VOICE,
        data_dir: str = _DEFAULT_DATA_DIR,
        voice_instance: PiperVoice | None = None,
    ) -> None:
        self._voice = voice
        self._data_dir = Path(data_dir)
        # Lazy-loaded (and downloaded) on first synthesize() call. `voice_instance`
        # lets tests inject a fake without touching the real model/network.
        self._voice_instance = voice_instance

    @property
    def provider_name(self) -> str:
        return "piper"

    @property
    def voice_name(self) -> str:
        return self._voice

    def _get_voice(self) -> PiperVoice:
        if self._voice_instance is None:
            try:
                from piper import PiperVoice
                from piper.download_voices import download_voice
            except ImportError as exc:
                raise RuntimeError(
                    "piper-tts is not installed. Run `pip install piper-tts`, or set "
                    "TTS_PROVIDER=edge to use edge-tts instead."
                ) from exc

            self._data_dir.mkdir(parents=True, exist_ok=True)
            model_path = self._data_dir / f"{self._voice}.onnx"
            config_path = self._data_dir / f"{self._voice}.onnx.json"
            try:
                if not (model_path.exists() and config_path.exists()):
                    download_voice(self._voice, self._data_dir)
                self._voice_instance = PiperVoice.load(model_path, config_path, download_dir=self._data_dir)
            except Exception as exc:
                raise RuntimeError(f"Failed to load Piper voice {self._voice!r}: {exc}") from exc
        return self._voice_instance

    async def synthesize(self, text: str, *, voice: str | None = None) -> SynthesisResult:
        # Per-call voice override isn't supported -- swapping voices means
        # loading a different local ONNX model, not a request parameter.
        # Falls back to this adapter's configured voice (see base.py).
        return await asyncio.to_thread(self._synthesize_sync, text)

    def _synthesize_sync(self, text: str) -> SynthesisResult:
        voice = self._get_voice()

        sample_rate: int | None = None
        sample_width = 2
        channels = 1
        pcm_chunks: list[bytes] = []
        total_samples = 0

        try:
            for chunk in voice.synthesize(text):
                sample_rate = chunk.sample_rate
                sample_width = chunk.sample_width
                channels = chunk.sample_channels
                pcm_chunks.append(chunk.audio_int16_bytes)
                total_samples += len(chunk.audio_int16_array)
        except Exception as exc:
            raise RuntimeError(f"Piper failed to synthesize speech: {exc}") from exc

        if sample_rate is None:
            sample_rate = voice.config.sample_rate

        wav_bytes = _pcm_to_wav(b"".join(pcm_chunks), sample_rate, sample_width, channels)
        duration_seconds = total_samples / sample_rate if sample_rate else None

        return SynthesisResult(
            audio_data=wav_bytes,
            content_type="audio/wav",
            sample_rate=sample_rate,
            duration_seconds=duration_seconds,
        )


def _pcm_to_wav(pcm_data: bytes, sample_rate: int, sample_width: int, channels: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)
    return buffer.getvalue()
