from __future__ import annotations

from typing import Any

import anthropic

from .base import Message, ModelProvider, ModelResponse, Usage

_DEFAULT_MAX_TOKENS = 1024


class AnthropicProvider(ModelProvider):
    """Cloud LLM adapter backed by the Anthropic API — the default live provider."""

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

    def generate(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
    ) -> ModelResponse:
        system, turns = _split_system(messages)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": _DEFAULT_MAX_TOKENS,
            "messages": turns,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools
        if temperature is not None:
            kwargs["temperature"] = temperature

        result = self._client.messages.create(**kwargs)

        content = "".join(block.text for block in result.content if block.type == "text")
        tool_calls = [
            {"id": block.id, "name": block.name, "input": block.input}
            for block in result.content
            if block.type == "tool_use"
        ]

        return ModelResponse(
            content=content,
            tool_calls=tool_calls,
            usage=Usage(
                input_tokens=result.usage.input_tokens,
                output_tokens=result.usage.output_tokens,
            ),
        )


def _split_system(messages: list[Message]) -> tuple[str | None, list[Message]]:
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    turns = [m for m in messages if m["role"] != "system"]
    system = "\n\n".join(system_parts) if system_parts else None
    return system, turns
