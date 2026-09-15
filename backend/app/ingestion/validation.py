"""Upload validation — checks applied to raw bytes before they ever reach
the ingestion pipeline. See technical-design.md's Guardrails section:
"Upload validation: file size/type limits, and treating uploaded content as
untrusted input." Pure functions, no I/O, no database/provider access, so
they're cheap to run before doing any real work.
"""

from __future__ import annotations

import hashlib

import pymupdf

_PDF_MAGIC_BYTES = b"%PDF-"


class InvalidPdfError(Exception):
    """Raised when uploaded bytes can't be opened as a PDF, or open but have
    no extractable content (text or an embedded image) on any page.
    """


def looks_like_pdf(data: bytes) -> bool:
    """Magic-byte check on the actual file signature -- not the client-supplied
    filename extension or Content-Type header, both of which are trivially
    spoofable.
    """
    return data.startswith(_PDF_MAGIC_BYTES)


def compute_content_hash(data: bytes) -> str:
    """SHA-256 hex digest of raw file bytes, used to detect a duplicate
    upload (exact re-upload or the same file under a different name).
    """
    return hashlib.sha256(data).hexdigest()


def open_validated_pdf(pdf_bytes: bytes) -> pymupdf.Document:
    """Opens pdf_bytes and confirms it has at least one page with real
    content, raising InvalidPdfError otherwise. Reused as the single
    "is this actually a usable PDF" check -- callers should not need to
    open a PyMuPDF document any other way just to validate it.
    """
    try:
        document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise InvalidPdfError(f"Could not open file as a PDF: {exc}") from exc

    if document.page_count < 1:
        raise InvalidPdfError("PDF has no pages")

    has_content = any(
        document[page_number].get_text().strip() or document[page_number].get_image_info()
        for page_number in range(document.page_count)
    )
    if not has_content:
        raise InvalidPdfError("PDF has no extractable text or images on any page")

    return document
