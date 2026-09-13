"""Model Provider module — the LLM abstraction.

Public surface: the `ModelProvider` interface, its message/response types,
and `get_provider()`, which reads MODEL_PROVIDER from the environment and
returns the configured adapter. Callers (orchestration, api) must depend on
these, never import `OllamaProvider`/`AnthropicProvider` directly — that's
the dependency-inversion convention from technical-design.md.
"""

from __future__ import annotations

import os

from .anthropic_adapter import AnthropicProvider
from .base import Message, ModelProvider, ModelResponse, Usage
from .ollama_adapter import OllamaProvider

__all__ = [
    "Message",
    "ModelProvider",
    "ModelResponse",
    "Usage",
    "get_provider",
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
