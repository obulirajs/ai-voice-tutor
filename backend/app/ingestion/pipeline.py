from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.knowledge import (
    Chunk,
    ConsistencyCheckResult,
    VectorStore,
    check_ingestion_consistency,
)
from app.models import EmbeddingProvider, VisionProvider
from app.storage import create_document

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


def collection_for_subject(subject_id: int) -> str:
    return f"subject_{subject_id}"


async def ingest_pdf(
    db: AsyncSession,
    embedding_provider: EmbeddingProvider,
    vision_provider: VisionProvider,
    vector_store: VectorStore,
    *,
    subject_id: int,
    filename: str,
    pdf_bytes: bytes,
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
            page_text = vision_provider.transcribe_image(image_bytes, "image/png", _VISION_INSTRUCTIONS)
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
    )

    chunks: list[Chunk] = []
    if chunk_texts:
        embeddings = embedding_provider.embed(chunk_texts)
        chunks = [
            Chunk(text=text, embedding=embedding, document_id=document_row.id, page_number=page_number)
            for text, embedding, page_number in zip(chunk_texts, embeddings, chunk_pages, strict=True)
        ]
        await vector_store.upsert_chunks(collection, chunks)

    consistency_check = await check_ingestion_consistency(vector_store, collection, chunks, chunk_is_ocr)

    return IngestResult(
        document_id=document_row.id,
        page_count=document.page_count,
        scanned_page_count=scanned_page_count,
        chunk_count=len(chunk_texts),
        consistency_check=consistency_check,
    )
