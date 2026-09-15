from __future__ import annotations

from abc import ABC, abstractmethod


class VisionProvider(ABC):
    """Common interface for vision-capable adapters, used by ingestion's OCR
    pass over scanned pages and diagrams/formulae.
    """

    @abstractmethod
    def transcribe_image(self, image_bytes: bytes, media_type: str, instructions: str) -> str:
        """Return the model's text transcription of one image, per instructions."""
        raise NotImplementedError
