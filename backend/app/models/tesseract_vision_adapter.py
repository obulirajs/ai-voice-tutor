"""Local OCR adapter using Tesseract -- zero API cost, no network required.

Implements the same VisionProvider interface as AnthropicVisionProvider so
callers (the ingestion pipeline) don't know or care which backend is used.
Tesseract handles printed text well (textbook pages, typed documents). For
complex diagrams/formulas, the Anthropic vision adapter remains available
as an opt-in alternative via VISION_PROVIDER=anthropic in .env.
"""

from __future__ import annotations

import io

from PIL import Image

from .base import Usage
from .vision_base import VisionProvider, VisionResponse


class TesseractVisionProvider(VisionProvider):
    """OCR adapter backed by Tesseract (pytesseract).

    Supports English and French out of the box (the two CBSE subjects
    currently in use). Additional languages/traineddata (e.g. "equ" for
    math formula recognition) can be configured via the TESSERACT_LANGS
    env var, e.g. "eng+fra+equ".
    """

    def __init__(self, langs: str = "eng+fra", tesseract_cmd: str | None = None) -> None:
        import pytesseract  # import here to fail fast with a clear error

        self._pytesseract = pytesseract
        self._langs = langs

        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

        try:
            pytesseract.get_tesseract_version()
        except Exception as exc:
            raise RuntimeError(
                "Tesseract is not installed or not on PATH. "
                "Install it: Windows -> https://github.com/UB-Mannheim/tesseract/wiki "
                "(then set TESSERACT_CMD in .env to the installed tesseract.exe path if "
                "it's not on PATH), Linux -> 'sudo apt install tesseract-ocr', "
                "macOS -> 'brew install tesseract'"
            ) from exc

    @property
    def provider_name(self) -> str:
        return "tesseract"

    @property
    def model_name(self) -> str:
        return f"tesseract-{self._langs}"

    def transcribe_image(self, image_bytes: bytes, media_type: str, instructions: str) -> VisionResponse:
        image = Image.open(io.BytesIO(image_bytes))
        text = self._pytesseract.image_to_string(image, lang=self._langs)
        # Tesseract is local -- no token usage, zero cost.
        return VisionResponse(text=text, usage=Usage(input_tokens=0, output_tokens=0))
