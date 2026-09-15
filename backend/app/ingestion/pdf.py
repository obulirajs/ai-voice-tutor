from __future__ import annotations

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


def load_pdf(pdf_bytes: bytes) -> pymupdf.Document:
    return pymupdf.open(stream=pdf_bytes, filetype="pdf")


def _bbox_area(bbox: tuple[float, float, float, float]) -> float:
    return max(bbox[2] - bbox[0], 0) * max(bbox[3] - bbox[1], 0)


def page_is_scanned(page: pymupdf.Page) -> bool:
    """Word-count vs. content-image-coverage heuristic.

    Returns True when the page has few real words of extracted text but
    substantial embedded image content (excluding full-page background
    images) — the signature of a page whose actual content (a scanned
    document, a poster, a logo grid) lives only in an image, even though a
    caption or label happens to produce enough raw characters to look like
    a normal text-layer page under a pure character-count check.
    """
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
    word_count = len(page.get_text("words"))

    return word_count < _MIN_WORDS_PER_PAGE and content_image_ratio >= _CONTENT_IMAGE_COVERAGE_RATIO


def extract_page_text(page: pymupdf.Page) -> str:
    return page.get_text()


def render_page_png(page: pymupdf.Page, dpi: int = 200) -> bytes:
    pixmap = page.get_pixmap(dpi=dpi)
    return pixmap.tobytes("png")
