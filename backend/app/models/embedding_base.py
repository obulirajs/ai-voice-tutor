from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .base import Usage


@dataclass(frozen=True)
class EmbeddingResponse:
    embeddings: list[list[float]]
    usage: Usage = field(default_factory=Usage)


class EmbeddingProvider(ABC):
    """Common interface for embedding adapters — the same adapter pattern as
    ModelProvider, applied to the embedding model (architecture.md: "same
    pattern reused for the embedding model"). Ollama-only for now: the
    Anthropic API has no embeddings endpoint.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def model_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def embed(self, texts: list[str]) -> EmbeddingResponse:
        """Return one embedding vector per input text (same order), plus token usage."""
        raise NotImplementedError
