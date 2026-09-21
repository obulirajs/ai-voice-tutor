from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from .base import SynthesisResult, TTSProvider

if TYPE_CHECKING:
    import edge_tts

_DEFAULT_VOICE = "en-US-AriaNeural"


class EdgeTTSAdapter(TTSProvider):
    """Cloud TTS adapter backed by Microsoft Edge's read-aloud engine over
    HTTPS -- the practical default on Windows (technical-design.md):
    cross-platform, good voice quality, async-native, and the app already
    requires internet for the Anthropic API. Outputs MP3 (edge-tts's native
    format) rather than WAV -- see SynthesisResult.content_type.
    """

    def __init__(
        self,
        voice: str = _DEFAULT_VOICE,
        communicate_factory: Callable[[str, str], edge_tts.Communicate] | None = None,
    ) -> None:
        self._voice = voice
        # Defaults to the real edge_tts.Communicate; tests inject a fake
        # factory so synthesis tests don't need a real network call.
        self._communicate_factory = communicate_factory

    @property
    def provider_name(self) -> str:
        return "edge-tts"

    @property
    def voice_name(self) -> str:
        return self._voice

    def _make_communicate(self, text: str, voice: str) -> edge_tts.Communicate:
        if self._communicate_factory is not None:
            return self._communicate_factory(text, voice)
        try:
            import edge_tts
        except ImportError as exc:
            raise RuntimeError("edge-tts is not installed. Run `pip install edge-tts`.") from exc
        return edge_tts.Communicate(text, voice)

    async def synthesize(self, text: str, *, voice: str | None = None) -> SynthesisResult:
        communicate = self._make_communicate(text, voice or self._voice)

        audio_chunks: list[bytes] = []
        try:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_chunks.append(chunk["data"])
        except Exception as exc:
            raise RuntimeError(f"edge-tts failed to synthesize speech: {exc}") from exc

        audio_data = b"".join(audio_chunks)
        if not audio_data:
            raise RuntimeError("edge-tts returned no audio data")

        return SynthesisResult(
            audio_data=audio_data,
            content_type="audio/mpeg",
            sample_rate=None,
            duration_seconds=None,
        )
