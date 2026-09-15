from __future__ import annotations

from abc import ABC, abstractmethod


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
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text, same order."""
        raise NotImplementedError
