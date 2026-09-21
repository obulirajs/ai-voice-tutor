from __future__ import annotations

from typing import Any

import ollama

from .base import Message, ModelProvider, ModelResponse, Usage


class OllamaProvider(ModelProvider):
    """Local LLM adapter backed by the Ollama Python client."""

    def __init__(self, model: str, client: ollama.Client | None = None) -> None:
        self._model = model
        self._client = client or ollama.Client()

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

    def generate(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
    ) -> ModelResponse:
        kwargs: dict[str, Any] = {"model": self._model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        if temperature is not None:
            kwargs["options"] = {"temperature": temperature}

        result = self._client.chat(**kwargs)
        message = result["message"]

        return ModelResponse(
            content=message.get("content", ""),
            tool_calls=message.get("tool_calls") or [],
            usage=Usage(
                input_tokens=result.get("prompt_eval_count"),
                output_tokens=result.get("eval_count"),
            ),
        )
