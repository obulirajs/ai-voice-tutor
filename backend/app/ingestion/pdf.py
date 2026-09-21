from __future__ import annotations

import unicodedata
from dataclasses import dataclass

import pymupdf

# Heuristic thresholds for scanned-vs-text-layer detection.
#
# A page whose real word count is low *and* that carries meaningful embedded
# image content is treated as needing the OCR/vision path. Word count (not
# extracted-character count) is the primary signal: a page can have a
# handful of caption/label words that still add up to 40+ characters while
# the content anyone actually cares about (job ads, logos, a poster) only
# exists as pixels in an image.
#
# Image coverage is computed excluding any image whose bounding box covers
# almost the entire page — these textbooks lay a full-page decorative
# border/background image on nearly every page, and including it would make
# every page register as "high image coverage" regardless of actual content.
_MIN_WORDS_PER_PAGE = 30
_CONTENT_IMAGE_COVERAGE_RATIO = 0.15
_BACKGROUND_IMAGE_AREA_RATIO = 0.95

# A second, independent signal a word-count check can't see: some PDFs
# embed a Type3 (or otherwise custom-encoded) font with no usable
# ToUnicode mapping. pymupdf still extracts a normal-looking word count from
# these pages, but the "words" decode to private-use-area/dingbat code
# points instead of real characters -- unreadable to a human and useless to
# an embedding model. A page whose extracted text is mostly this kind of
# garbage needs the OCR/vision path just as much as a genuinely scanned one.
_GARBLED_TEXT_RATIO_THRESHOLD = 0.3
# "C*" covers control/private-use/surrogate/unassigned categories; "So" is
# "Symbol, other" (dingbats and similar blocks), which is where a broken
# font's misdecoded glyphs typically land.
_GARBLED_CATEGORY_PREFIXES = ("C", "So")


def garbled_text_ratio(text: str) -> float:
    """Fraction of non-whitespace characters in `text` that look like
    font-encoding garbage (private-use area, symbol/dingbat blocks, control
    characters) rather than ordinary letters/digits/punctuation."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    garbled = sum(
        1 for c in chars if unicodedata.category(c).startswith(_GARBLED_CATEGORY_PREFIXES) or ord(c) >= 0xE000
    )
    return garbled / len(chars)


# Text-quality scoring: a broader, per-page check than garbled_text_ratio
# alone. Runs on extracted text from EITHER path (text-layer or OCR/vision)
# before chunking, so a page that passed page_is_scanned() as "text-layer"
# but still extracted poorly (e.g. a font-encoding issue below the
# page_is_scanned garbled-ratio threshold, or just sparse/broken text) can
# still trigger an OCR fallback -- see pipeline.py's per-page quality gate.
_QUALITY_MIN_WORD_COUNT = 5
_QUALITY_TARGET_AVG_WORD_LENGTH = 5.0
_QUALITY_GOOD_THRESHOLD = 0.60


@dataclass(frozen=True)
class TextQualityResult:
    """Per-page text quality assessment."""

    score: float  # 0.0 (garbage) to 1.0 (clean text)
    garbled_ratio: float  # reuses garbled_text_ratio()
    word_count: int
    avg_word_length: float
    alpha_ratio: float  # fraction of alpha chars among non-whitespace
    verdict: str  # "good" | "poor" | "empty"


def assess_text_quality(text: str) -> TextQualityResult:
    """Scores extracted page text on three independent signals a
    garbled-ratio check alone can miss: how alphabetic the content is,
    whether word lengths look like real prose, and whether there's enough
    of it to judge at all.
    """
    garbled = garbled_text_ratio(text)
    words = text.split()
    word_count = len(words)

    if word_count < _QUALITY_MIN_WORD_COUNT:
        return TextQualityResult(
            score=0.0,
            garbled_ratio=garbled,
            word_count=word_count,
            avg_word_length=0.0,
            alpha_ratio=0.0,
            verdict="empty",
        )

    avg_word_length = sum(len(w) for w in words) / word_count
    non_whitespace = [c for c in text if not c.isspace()]
    alpha_ratio = sum(1 for c in non_whitespace if c.isalpha()) / len(non_whitespace) if non_whitespace else 0.0

    garbled_score = 1.0 - min(garbled, 1.0)
    alpha_score = alpha_ratio
    length_score = max(0.0, 1.0 - abs(avg_word_length - _QUALITY_TARGET_AVG_WORD_LENGTH) / 10.0)
    score = 0.4 * garbled_score + 0.4 * alpha_score + 0.2 * length_score

    verdict = "good" if score >= _QUALITY_GOOD_THRESHOLD else "poor"
    return TextQualityResult(
        score=score,
        garbled_ratio=garbled,
        word_count=word_count,
        avg_word_length=avg_word_length,
        alpha_ratio=alpha_ratio,
        verdict=verdict,
    )


def load_pdf(pdf_bytes: bytes) -> pymupdf.Document:
    return pymupdf.open(stream=pdf_bytes, filetype="pdf")


def _bbox_area(bbox: tuple[float, float, float, float]) -> float:
    return max(bbox[2] - bbox[0], 0) * max(bbox[3] - bbox[1], 0)


def page_is_scanned(page: pymupdf.Page) -> bool:
    """Word-count-vs-image-coverage heuristic, plus a garbled-text check.

    Returns True when either:
      - the page has few real words of extracted text but substantial
        embedded image content (excluding full-page background images) —
        the signature of a page whose actual content (a scanned document, a
        poster, a logo grid) lives only in an image, even though a caption
        or label happens to produce enough raw characters to look like a
        normal text-layer page under a pure character-count check; or
      - the extracted text itself is mostly garbled font-encoding output
        (see garbled_text_ratio) — a broken/custom font can produce a high
        word count that is nonetheless unusable, which the count-vs-image
        check alone can't see.
    """
    words = page.get_text("words")
    word_count = len(words)

    if garbled_text_ratio(" ".join(w[4] for w in words)) >= _GARBLED_TEXT_RATIO_THRESHOLD:
        return True

    page_area = page.rect.width * page.rect.height
    if page_area <= 0:
        return False

    images = page.get_image_info()
    if not images:
        return False

    non_background = [img for img in images if _bbox_area(img["bbox"]) < _BACKGROUND_IMAGE_AREA_RATIO * page_area]
    # If every image on the page covers almost the whole page, there's no
    # separate decorative layer to discard -- it's a single full-page image
    # (a classic whole-page scan), so treat it as content rather than
    # excluding the only image present.
    content_images = non_background if non_background else images

    content_image_ratio = sum(_bbox_area(img["bbox"]) for img in content_images) / page_area

    return word_count < _MIN_WORDS_PER_PAGE and content_image_ratio >= _CONTENT_IMAGE_COVERAGE_RATIO


def extract_page_text(page: pymupdf.Page) -> str:
    return page.get_text()


def render_page_png(page: pymupdf.Page, dpi: int = 200) -> bytes:
    pixmap = page.get_pixmap(dpi=dpi)
    return pixmap.tobytes("png")
