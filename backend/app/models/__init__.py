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

from app.config import DIAGRAM_DEFAULTS, MODEL_DEFAULTS

from .anthropic_adapter import AnthropicProvider
from .anthropic_vision_adapter import AnthropicVisionProvider
from .base import Message, ModelProvider, ModelResponse, Usage
from .embedding_base import EmbeddingProvider, EmbeddingResponse
from .ollama_adapter import OllamaProvider
from .ollama_embedding_adapter import OllamaEmbeddingProvider
from .pricing import estimate_cost_usd
from .tesseract_vision_adapter import TesseractVisionProvider
from .vision_base import VisionProvider, VisionResponse

__all__ = [
    "EmbeddingProvider",
    "EmbeddingResponse",
    "Message",
    "ModelProvider",
    "ModelResponse",
    "TesseractVisionProvider",
    "Usage",
    "VisionProvider",
    "VisionResponse",
    "estimate_cost_usd",
    "get_diagram_model_provider",
    "get_embedding_provider",
    "get_provider",
    "get_vision_provider",
]

def _read_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def get_provider() -> ModelProvider:
    """Build the Model Provider adapter configured via the MODEL_PROVIDER env var.

    Anthropic is the default active provider for live conversation; Ollama
    stays wired in behind the same interface for offline/dev use. See
    CLAUDE.md's non-negotiable conventions.
    """
    provider = os.getenv("MODEL_PROVIDER", MODEL_DEFAULTS.provider).lower()

    if provider == "ollama":
        return OllamaProvider(
            model=os.getenv("OLLAMA_MODEL", MODEL_DEFAULTS.ollama_model),
            think=_read_bool_env("OLLAMA_THINK", MODEL_DEFAULTS.ollama_think),
        )

    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required when MODEL_PROVIDER=anthropic")
        return AnthropicProvider(
            model=os.getenv("ANTHROPIC_MODEL", MODEL_DEFAULTS.anthropic_model),
            api_key=api_key,
        )

    raise ValueError(f"Unknown MODEL_PROVIDER: {provider!r} (expected 'ollama' or 'anthropic')")


def get_embedding_provider() -> EmbeddingProvider:
    """Build the embedding adapter configured via the EMBEDDING_PROVIDER env var.

    Ollama-only for now — the Anthropic API has no embeddings endpoint.
    """
    provider = os.getenv("EMBEDDING_PROVIDER", MODEL_DEFAULTS.embedding_provider).lower()

    if provider == "ollama":
        return OllamaEmbeddingProvider(model=os.getenv("OLLAMA_EMBEDDING_MODEL", MODEL_DEFAULTS.ollama_embedding_model))

    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {provider!r} (expected 'ollama')")


def get_vision_provider() -> VisionProvider:
    """Build the vision adapter configured via the VISION_PROVIDER env var.

    Tesseract (free, local OCR) is the default -- see
    tesseract_vision_adapter.py. Anthropic's vision API remains available
    as an opt-in for pages Tesseract handles poorly (complex diagrams), at
    real per-page API cost.
    """
    provider = os.getenv("VISION_PROVIDER", MODEL_DEFAULTS.vision_provider).lower()

    if provider == "tesseract":
        langs = os.getenv("TESSERACT_LANGS", MODEL_DEFAULTS.tesseract_langs)
        tesseract_cmd = os.getenv("TESSERACT_CMD") or None
        return TesseractVisionProvider(langs=langs, tesseract_cmd=tesseract_cmd)

    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required when VISION_PROVIDER=anthropic")
        return AnthropicVisionProvider(
            model=os.getenv("ANTHROPIC_VISION_MODEL", os.getenv("ANTHROPIC_MODEL", MODEL_DEFAULTS.anthropic_vision_model)),
            api_key=api_key,
        )

    raise ValueError(f"Unknown VISION_PROVIDER: {provider!r} (expected 'tesseract' or 'anthropic')")


def get_diagram_model_provider() -> ModelProvider | None:
    """Build the Model Provider adapter for on-demand diagram generation
    (see app.orchestration.diagrams), configured via the DIAGRAM_PROVIDER
    env var. Reuses the same Ollama/Anthropic adapters as get_provider() --
    just with a smaller, diagram-specific model -- rather than a new
    interface. Returns None when DIAGRAM_PROVIDER=off (the feature is
    disabled entirely).
    """
    provider = os.getenv("DIAGRAM_PROVIDER", DIAGRAM_DEFAULTS.provider).lower()

    if provider == "off":
        return None

    if provider == "ollama":
        return OllamaProvider(
            model=os.getenv("OLLAMA_DIAGRAM_MODEL", DIAGRAM_DEFAULTS.ollama_model),
            think=_read_bool_env("OLLAMA_THINK", MODEL_DEFAULTS.ollama_think),
        )

    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required when DIAGRAM_PROVIDER=anthropic")
        return AnthropicProvider(
            model=os.getenv("ANTHROPIC_DIAGRAM_MODEL", DIAGRAM_DEFAULTS.anthropic_model),
            api_key=api_key,
        )

    raise ValueError(f"Unknown DIAGRAM_PROVIDER: {provider!r} (expected 'ollama', 'anthropic', or 'off')")
