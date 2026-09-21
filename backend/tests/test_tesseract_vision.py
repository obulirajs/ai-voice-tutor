from __future__ import annotations

import io
import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image, ImageDraw, ImageFont

from app.models.tesseract_vision_adapter import TesseractVisionProvider
from app.models.vision_base import VisionProvider

# TESSERACT_CMD / PATH / the common Windows install location, in that order
# -- mirrors get_vision_provider()'s own lookup, so this test suite isn't
# wrongly skipped on a machine where tesseract.exe exists but isn't on PATH
# (the common case right after installing it on Windows).
_WINDOWS_DEFAULT_INSTALL = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def _find_tesseract_cmd() -> str | None:
    if cmd := os.environ.get("TESSERACT_CMD"):
        return cmd
    if shutil.which("tesseract"):
        return None  # already resolvable without an explicit path
    if Path(_WINDOWS_DEFAULT_INSTALL).is_file():
        return _WINDOWS_DEFAULT_INSTALL
    return None


_TESSERACT_CMD = _find_tesseract_cmd()
_TESSERACT_AVAILABLE = bool(shutil.which("tesseract")) or _TESSERACT_CMD is not None

skip_without_tesseract = pytest.mark.skipif(not _TESSERACT_AVAILABLE, reason="Tesseract not installed")


def test_tesseract_vision_provider_implements_interface() -> None:
    assert issubclass(TesseractVisionProvider, VisionProvider)


@skip_without_tesseract
def test_tesseract_vision_provider_exposes_name_and_model() -> None:
    provider = TesseractVisionProvider(langs="eng+fra", tesseract_cmd=_TESSERACT_CMD)

    assert provider.provider_name == "tesseract"
    assert provider.model_name == "tesseract-eng+fra"


@skip_without_tesseract
def test_tesseract_vision_provider_transcribes_simple_image() -> None:
    provider = TesseractVisionProvider(langs="eng", tesseract_cmd=_TESSERACT_CMD)

    # A large explicit font size -- PIL's tiny default bitmap font renders
    # too small/pixelated for Tesseract to read reliably.
    image = Image.new("RGB", (400, 80), color="white")
    draw = ImageDraw.Draw(image)
    draw.text((10, 20), "Hello World", fill="black", font=ImageFont.load_default(size=32))
    buf = io.BytesIO()
    image.save(buf, format="PNG")

    response = provider.transcribe_image(buf.getvalue(), "image/png", "transcribe this page")

    assert "Hello" in response.text
    assert "World" in response.text


@skip_without_tesseract
def test_tesseract_vision_provider_usage_is_always_zero() -> None:
    provider = TesseractVisionProvider(langs="eng", tesseract_cmd=_TESSERACT_CMD)

    image = Image.new("RGB", (200, 60), color="white")
    buf = io.BytesIO()
    image.save(buf, format="PNG")

    response = provider.transcribe_image(buf.getvalue(), "image/png", "transcribe this page")

    assert response.usage.input_tokens == 0
    assert response.usage.output_tokens == 0


def test_tesseract_vision_provider_raises_when_tesseract_missing() -> None:
    with (
        patch("pytesseract.get_tesseract_version", side_effect=OSError("not found")),
        pytest.raises(RuntimeError, match="Tesseract is not installed"),
    ):
        TesseractVisionProvider(langs="eng")


def test_tesseract_vision_provider_sets_explicit_command_path() -> None:
    fake_pytesseract = MagicMock()
    with patch.dict("sys.modules", {"pytesseract": fake_pytesseract}):
        TesseractVisionProvider(langs="eng", tesseract_cmd="/custom/path/tesseract")

    assert fake_pytesseract.pytesseract.tesseract_cmd == "/custom/path/tesseract"
