from __future__ import annotations

import base64
from typing import Any

import anthropic

from .base import Usage
from .vision_base import VisionProvider, VisionResponse

_DEFAULT_MAX_TOKENS = 2048


class AnthropicVisionProvider(VisionProvider):
    """OCR/diagram-transcription adapter backed by the Anthropic API's vision support."""

    def __init__(
        self,
        model: str,
        api_key: str,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self._model = model
        self._client = client or anthropic.Anthropic(api_key=api_key)

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def model_name(self) -> str:
        return self._model

    def transcribe_image(self, image_bytes: bytes, media_type: str, instructions: str) -> VisionResponse:
        encoded = base64.standard_b64encode(image_bytes).decode("ascii")
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": _DEFAULT_MAX_TOKENS,
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
        result = self._client.messages.create(**kwargs)
        text = "".join(block.text for block in result.content if block.type == "text")
        return VisionResponse(
            text=text,
            usage=Usage(input_tokens=result.usage.input_tokens, output_tokens=result.usage.output_tokens),
        )
