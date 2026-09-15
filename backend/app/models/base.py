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

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Short adapter name (e.g. "anthropic", "ollama") — logged on usage_events."""
        raise NotImplementedError

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The configured model name — logged on usage_events and used for cost lookup."""
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse:
        raise NotImplementedError
