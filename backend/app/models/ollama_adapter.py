from __future__ import annotations

import re
from typing import Any

import ollama

from app.config import MODEL_DEFAULTS

from .base import Message, ModelProvider, ModelResponse, Usage

# Defensive cleanup: some thinking-capable models (e.g. Qwen3.5) can still
# emit a <think>...</think> reasoning block inline in the message content
# even when think=False was requested -- never let that leak into a reply
# shown to the student. DOTALL so the block's own newlines are matched;
# IGNORECASE since not every model/version lowercases the tag consistently.
_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def _strip_think_tags(content: str) -> str:
    return _THINK_BLOCK.sub("", content).strip()


class OllamaProvider(ModelProvider):
    """Local LLM adapter backed by the Ollama Python client's AsyncClient --
    generate() is a genuine async network call (not a sync call run in a
    threadpool), so it can't block the event loop other requests share.
    """

    def __init__(
        self,
        model: str,
        think: bool = MODEL_DEFAULTS.ollama_think,
        client: ollama.AsyncClient | None = None,
    ) -> None:
        self._model = model
        self._think = think
        self._client = client or ollama.AsyncClient()

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
    ) -> ModelResponse:
        kwargs: dict[str, Any] = {"model": self._model, "messages": messages, "think": self._think}
        if tools:
            kwargs["tools"] = tools
        if temperature is not None:
            kwargs["options"] = {"temperature": temperature}

        result = await self._client.chat(**kwargs)
        message = result["message"]

        return ModelResponse(
            content=_strip_think_tags(message.get("content", "")),
            tool_calls=message.get("tool_calls") or [],
            usage=Usage(
                input_tokens=result.get("prompt_eval_count"),
                output_tokens=result.get("eval_count"),
            ),
        )
