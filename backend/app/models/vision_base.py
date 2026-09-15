from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .base import Usage


@dataclass(frozen=True)
class VisionResponse:
    text: str
    usage: Usage = field(default_factory=Usage)


class VisionProvider(ABC):
    """Common interface for vision-capable adapters, used by ingestion's OCR
    pass over scanned pages and diagrams/formulae.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Short adapter name (e.g. "anthropic") — logged on usage_events."""
        raise NotImplementedError

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The configured model name — logged on usage_events and used for cost lookup."""
        raise NotImplementedError

    @abstractmethod
    def transcribe_image(self, image_bytes: bytes, media_type: str, instructions: str) -> VisionResponse:
        """Return the model's text transcription of one image, per instructions, plus token usage."""
        raise NotImplementedError
