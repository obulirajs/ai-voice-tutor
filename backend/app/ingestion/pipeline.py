from __future__ import annotations

import time
from dataclasses import dataclass

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
class IngestResult:
    document_id: int
    page_count: int
    scanned_page_count: int
    chunk_count: int
    consistency_check: ConsistencyCheckResult
    subject_mismatch_warning: str | None = None


def collection_for_subject(subject_id: int) -> str:
    return f"subject_{subject_id}"


async def _log_call_usage(
    db: AsyncSession,
    *,
    subject_id: int,
    provider_name: str,
    model_name: str,
    input_tokens: int | None,
    output_tokens: int | None,
    latency_ms: float,
) -> None:
    """Logs one usage_events row for a single vision or embedding call made
    during ingestion. Ingestion has no chat session, so session_id is None --
    otherwise the same provider/model/tokens/cost/latency shape as the chat
    path's usage logging in app.orchestration.
    """
    cost_usd = estimate_cost_usd(model_name, input_tokens, output_tokens)
    await log_usage_event(
        db,
        session_id=None,
        subject_id=subject_id,
        provider=provider_name,
        model=model_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
    )


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

    for page_number in range(1, document.page_count + 1):
        page = document[page_number - 1]
        is_scanned = pdf.page_is_scanned(page)

        if is_scanned:
            scanned_page_count += 1
            image_bytes = pdf.render_page_png(page)
            started = time.perf_counter()
            vision_response = vision_provider.transcribe_image(image_bytes, "image/png", _VISION_INSTRUCTIONS)
            latency_ms = (time.perf_counter() - started) * 1000
            page_text = vision_response.text
            await _log_call_usage(
                db,
                subject_id=subject_id,
                provider_name=vision_provider.provider_name,
                model_name=vision_provider.model_name,
                input_tokens=vision_response.usage.input_tokens,
                output_tokens=vision_response.usage.output_tokens,
                latency_ms=latency_ms,
            )
        else:
            page_text = pdf.extract_page_text(page)

        for chunk in chunk_text(page_text):
            chunk_texts.append(chunk)
            chunk_pages.append(page_number)
            chunk_is_ocr.append(is_scanned)

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
    )
