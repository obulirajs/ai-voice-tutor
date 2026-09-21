from __future__ import annotations

import time
from dataclasses import dataclass

import pymupdf
from sqlalchemy.ext.asyncio import AsyncSession

from app.knowledge import (
    Chunk,
    ConsistencyCheckResult,
    VectorStore,
    check_ingestion_consistency,
    check_subject_language_mismatch,
)
from app.models import EmbeddingProvider, VisionProvider, estimate_cost_usd
from app.storage import create_document, log_usage_event

from . import pdf
from .chunking import chunk_text

_VISION_INSTRUCTIONS = (
    "Transcribe this scanned textbook page into plain text, preserving reading "
    "order. Render any mathematical formula in simple LaTeX-like notation inline "
    "with the surrounding text (e.g. \\(x^2\\)), and briefly describe any diagram "
    "in [brackets] where it appears in the page."
)


@dataclass(frozen=True)
class PageQualitySummary:
    """Aggregate of the per-page assess_text_quality() results from one
    ingestion run, for the frontend to warn on rather than silently
    ingesting unreadable content (see quality_warning below)."""

    total_pages: int
    good_pages: int
    poor_pages: int
    empty_pages: int
    ocr_fallback_pages: int
    avg_score: float
    worst_page: int | None  # 1-indexed page number
    worst_score: float


@dataclass(frozen=True)
class IngestResult:
    document_id: int
    page_count: int
    scanned_page_count: int
    chunk_count: int
    consistency_check: ConsistencyCheckResult
    subject_mismatch_warning: str | None = None
    quality_summary: PageQualitySummary | None = None
    quality_warning: str | None = None


def collection_for_subject(subject_id: int) -> str:
    return f"subject_{subject_id}"


async def _log_call_usage(
    db: AsyncSession,
    *,
    subject_id: int,
    event_type: str,
    provider_name: str,
    model_name: str,
    input_tokens: int | None,
    output_tokens: int | None,
    latency_ms: float,
) -> None:
    """Logs one usage_events row for a single vision or embedding call made
    during ingestion. Ingestion has no chat session, so session_id is None --
    otherwise the same provider/model/tokens/cost/latency shape as the chat
    path's usage logging in app.orchestration (event_type "vision" or
    "embedding" here, vs. "generation"/"embedding"/"retrieval" there).
    """
    cost_usd = estimate_cost_usd(model_name, input_tokens, output_tokens)
    await log_usage_event(
        db,
        session_id=None,
        subject_id=subject_id,
        event_type=event_type,
        provider=provider_name,
        model=model_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
    )


async def _run_vision(
    db: AsyncSession,
    vision_provider: VisionProvider,
    page: pymupdf.Page,
    *,
    subject_id: int,
) -> tuple[str, pdf.TextQualityResult]:
    """Renders one page, runs it through the vision provider, logs the
    usage event, and scores the result -- the shared body of both the
    is_scanned path and the poor-text-layer OCR fallback below."""
    image_bytes = pdf.render_page_png(page)
    started = time.perf_counter()
    vision_response = vision_provider.transcribe_image(image_bytes, "image/png", _VISION_INSTRUCTIONS)
    latency_ms = (time.perf_counter() - started) * 1000
    await _log_call_usage(
        db,
        subject_id=subject_id,
        event_type="vision",
        provider_name=vision_provider.provider_name,
        model_name=vision_provider.model_name,
        input_tokens=vision_response.usage.input_tokens,
        output_tokens=vision_response.usage.output_tokens,
        latency_ms=latency_ms,
    )
    text = vision_response.text
    return text, pdf.assess_text_quality(text)


def _summarize_page_quality(
    page_qualities: list[pdf.TextQualityResult], ocr_fallback_count: int
) -> tuple[PageQualitySummary | None, str | None]:
    if not page_qualities:
        return None, None

    good = sum(1 for q in page_qualities if q.verdict == "good")
    poor = sum(1 for q in page_qualities if q.verdict == "poor")
    empty = sum(1 for q in page_qualities if q.verdict == "empty")
    avg_score = sum(q.score for q in page_qualities) / len(page_qualities)
    worst_index = min(range(len(page_qualities)), key=lambda i: page_qualities[i].score)

    summary = PageQualitySummary(
        total_pages=len(page_qualities),
        good_pages=good,
        poor_pages=poor,
        empty_pages=empty,
        ocr_fallback_pages=ocr_fallback_count,
        avg_score=round(avg_score, 3),
        worst_page=worst_index + 1,
        worst_score=round(page_qualities[worst_index].score, 3),
    )

    warnings: list[str] = []
    if poor > 0 and poor / len(page_qualities) > 0.3:
        warnings.append(f"{poor} of {len(page_qualities)} pages had poor text quality and may not be searchable.")
    if empty > 0:
        warnings.append(f"{empty} pages produced no usable text.")
    if avg_score < 0.5:
        warnings.append(
            f"Overall text extraction quality is low (avg score {avg_score:.2f}). Retrieval may be unreliable."
        )

    return summary, " ".join(warnings) if warnings else None


async def ingest_pdf(
    db: AsyncSession,
    embedding_provider: EmbeddingProvider,
    vision_provider: VisionProvider,
    vector_store: VectorStore,
    *,
    subject_id: int,
    subject_name: str,
    filename: str,
    pdf_bytes: bytes,
    content_hash: str | None = None,
) -> IngestResult:
    """Upload -> scan detection -> OCR/vision -> chunk -> embed -> store -> consistency-check.

    The post-ingestion retrieval-quality check here is an automatic,
    zero-input internal consistency check (does each sampled chunk find
    itself again?) — a Knowledge-module concern, not an ingestion one; see
    app.knowledge.consistency. Grading against a curated golden-question
    set is a separate, decoupled concern — see the
    POST /subjects/{subject}/documents/{document_id}/sanity-check endpoint,
    which calls app.knowledge.golden_qa directly and never touches this
    function.

    Also runs a lightweight, non-blocking subject/language sanity check
    (app.knowledge.check_subject_language_mismatch) after chunks are stored
    and before the consistency check -- catches an upload to the wrong
    subject (e.g. an English document under "french") without an LLM call.
    """
    document = pdf.load_pdf(pdf_bytes)
    collection = collection_for_subject(subject_id)

    scanned_page_count = 0
    chunk_texts: list[str] = []
    chunk_pages: list[int] = []
    chunk_is_ocr: list[bool] = []
    page_qualities: list[pdf.TextQualityResult] = []
    ocr_fallback_count = 0

    for page_number in range(1, document.page_count + 1):
        page = document[page_number - 1]
        is_scanned = pdf.page_is_scanned(page)
        used_ocr_fallback = False

        if is_scanned:
            scanned_page_count += 1
            page_text, quality = await _run_vision(db, vision_provider, page, subject_id=subject_id)
        else:
            page_text = pdf.extract_page_text(page)
            quality = pdf.assess_text_quality(page_text)

            if quality.verdict != "good":
                # A text-layer page page_is_scanned() didn't flag, but whose
                # extraction still came out unreadable -- OCR fallback is
                # free with the default local Tesseract provider (see
                # get_vision_provider()), so try it and keep whichever
                # version scores higher rather than chunking known-bad text.
                ocr_text, ocr_quality = await _run_vision(db, vision_provider, page, subject_id=subject_id)
                if ocr_quality.score > quality.score:
                    page_text, quality = ocr_text, ocr_quality
                used_ocr_fallback = True
                ocr_fallback_count += 1

        page_qualities.append(quality)

        # Empty pages (e.g. a blank divider page) have nothing worth
        # embedding -- skip chunking rather than store zero-signal vectors.
        if quality.verdict != "empty":
            for chunk in chunk_text(page_text):
                chunk_texts.append(chunk)
                chunk_pages.append(page_number)
                chunk_is_ocr.append(is_scanned or used_ocr_fallback)

    quality_summary, quality_warning = _summarize_page_quality(page_qualities, ocr_fallback_count)

    document_row = await create_document(
        db,
        subject_id=subject_id,
        filename=filename,
        collection=collection,
        page_count=document.page_count,
        scanned_page_count=scanned_page_count,
        chunk_count=len(chunk_texts),
        content_hash=content_hash,
    )

    chunks: list[Chunk] = []
    if chunk_texts:
        started = time.perf_counter()
        embedding_response = embedding_provider.embed(chunk_texts)
        latency_ms = (time.perf_counter() - started) * 1000
        await _log_call_usage(
            db,
            subject_id=subject_id,
            event_type="embedding",
            provider_name=embedding_provider.provider_name,
            model_name=embedding_provider.model_name,
            input_tokens=embedding_response.usage.input_tokens,
            output_tokens=embedding_response.usage.output_tokens,
            latency_ms=latency_ms,
        )
        chunks = [
            Chunk(text=text, embedding=embedding, document_id=document_row.id, page_number=page_number)
            for text, embedding, page_number in zip(
                chunk_texts, embedding_response.embeddings, chunk_pages, strict=True
            )
        ]
        await vector_store.upsert_chunks(collection, chunks)

    subject_mismatch_warning = check_subject_language_mismatch(subject_name, chunk_texts)

    consistency_check = await check_ingestion_consistency(vector_store, collection, chunks, chunk_is_ocr)

    return IngestResult(
        document_id=document_row.id,
        page_count=document.page_count,
        scanned_page_count=scanned_page_count,
        chunk_count=len(chunk_texts),
        consistency_check=consistency_check,
        subject_mismatch_warning=subject_mismatch_warning,
        quality_summary=quality_summary,
        quality_warning=quality_warning,
    )
