from __future__ import annotations

import base64
from typing import Any

import anthropic

from app.config import MODEL_DEFAULTS

from .base import Usage
from .vision_base import VisionProvider, VisionResponse


class AnthropicVisionProvider(VisionProvider):
    """OCR/diagram-transcription adapter backed by the Anthropic API's
    vision support. Uses AsyncAnthropic so transcribe_image() is a genuine
    async network call rather than a sync one that would block the event
    loop -- ingesting a scan-heavy PDF makes many of these calls.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        client: anthropic.AsyncAnthropic | None = None,
    ) -> None:
        self._model = model
        self._client = client or anthropic.AsyncAnthropic(api_key=api_key)

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def model_name(self) -> str:
        return self._model

    async def transcribe_image(self, image_bytes: bytes, media_type: str, instructions: str) -> VisionResponse:
        encoded = base64.standard_b64encode(image_bytes).decode("ascii")
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": MODEL_DEFAULTS.anthropic_vision_max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": media_type, "data": encoded},
                        },
                        {"type": "text", "text": instructions},
                    ],
                }
            ],
        }
        result = await self._client.messages.create(**kwargs)
        text = "".join(block.text for block in result.content if block.type == "text")
        return VisionResponse(
            text=text,
            usage=Usage(input_tokens=result.usage.input_tokens, output_tokens=result.usage.output_tokens),
        )
