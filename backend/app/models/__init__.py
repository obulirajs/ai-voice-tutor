"""Model Provider module — the LLM abstraction.

Public surface: the `ModelProvider`/`EmbeddingProvider`/`VisionProvider`
interfaces, their message/response types, and their `get_*()` factories,
which read the environment and return the configured adapter. Callers
(orchestration, ingestion, api) must depend on these, never import a
concrete adapter directly — that's the dependency-inversion convention
from technical-design.md.
"""

from __future__ import annotations

import os

from .anthropic_adapter import AnthropicProvider
from .anthropic_vision_adapter import AnthropicVisionProvider
from .base import Message, ModelProvider, ModelResponse, Usage
from .embedding_base import EmbeddingProvider
from .ollama_adapter import OllamaProvider
from .ollama_embedding_adapter import OllamaEmbeddingProvider
from .vision_base import VisionProvider

__all__ = [
    "EmbeddingProvider",
    "Message",
    "ModelProvider",
    "ModelResponse",
    "Usage",
    "VisionProvider",
    "get_embedding_provider",
    "get_provider",
    "get_vision_provider",
]

# Anthropic is the default active provider for live conversation; Ollama
# stays wired in behind the same interface for offline/dev use. See
# CLAUDE.md's non-negotiable conventions.
_DEFAULT_PROVIDER = "anthropic"
_DEFAULT_MODEL = "llama3.2:1b"


def get_provider() -> ModelProvider:
    """Build the Model Provider adapter configured via the MODEL_PROVIDER env var."""
    provider = os.getenv("MODEL_PROVIDER", _DEFAULT_PROVIDER).lower()

    if provider == "ollama":
        return OllamaProvider(model=os.getenv("OLLAMA_MODEL", _DEFAULT_MODEL))

    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required when MODEL_PROVIDER=anthropic")
        return AnthropicProvider(
            model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5"),
            api_key=api_key,
        )

    raise ValueError(f"Unknown MODEL_PROVIDER: {provider!r} (expected 'ollama' or 'anthropic')")


def get_embedding_provider() -> EmbeddingProvider:
    """Build the embedding adapter configured via the EMBEDDING_PROVIDER env var.

    Ollama-only for now — the Anthropic API has no embeddings endpoint.
    """
    provider = os.getenv("EMBEDDING_PROVIDER", "ollama").lower()

    if provider == "ollama":
        return OllamaEmbeddingProvider(model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"))

    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {provider!r} (expected 'ollama')")


def get_vision_provider() -> VisionProvider:
    """Build the vision adapter configured via the VISION_PROVIDER env var."""
    provider = os.getenv("VISION_PROVIDER", "anthropic").lower()

    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required when VISION_PROVIDER=anthropic")
        return AnthropicVisionProvider(
            model=os.getenv("ANTHROPIC_VISION_MODEL", os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")),
            api_key=api_key,
        )

    raise ValueError(f"Unknown VISION_PROVIDER: {provider!r} (expected 'anthropic')")
