from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(frozen=True)
class ModelResponse:
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)


class ModelProvider(ABC):
    """Common interface every LLM adapter (Ollama, Anthropic, ...) implements.

    Orchestration and API code depend on this interface, never on a concrete
    adapter — see technical-design.md's dependency-inversion convention.
    """

    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse:
        raise NotImplementedError
