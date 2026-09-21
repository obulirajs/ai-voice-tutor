from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any
from unittest.mock import patch

import pymupdf
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.ingestion import ingest_pdf
from app.knowledge import Chunk, SearchResult, VectorStore
from app.models import (
    EmbeddingProvider,
    EmbeddingResponse,
    Usage,
    VisionProvider,
    VisionResponse,
)
from app.storage import get_or_create_subject
from app.storage.models import Base

Scenario = Callable[[AsyncSession], Coroutine[Any, Any, None]]


def run(scenario: Scenario) -> None:
    async def wrapper() -> None:
        engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        async with session_factory() as db:
            await scenario(db)

        await engine.dispose()

    asyncio.run(wrapper())

# A realistic sample of the actual corruption diagnose_retrieval.py found in
# the Science-NCERT PDF: pymupdf reports a normal-looking word count via
# newline-separated tokens, but they decode to private-use/dingbat code
# points, not real characters. Can't be reproduced by inserting text into a
# real PDF page (pymupdf substitutes a replacement glyph for code points a
# font doesn't support, rather than preserving them) -- so these tests mock
# pdf.extract_page_text() directly, exactly what's needed to exercise the
# pipeline's *response* to bad extraction, independent of PDF internals.
_GARBLED_TEXT = (
    "❙\n\x00✁\n✂\n✄\n\x00✂\n12\n✶✁✆✁\n✝\n"
    "✞✟✠✡⚛⚜\n✠\n✌✍\n⚛✍\n✡\n✎✏\n"
    "✡\n✑✒⚜✠\n✌✍\n❆\n❝\n✓\n❆\n❝\n"
)

_CLEAN_TEXT = (
    "Le present tense is used for habitual actions and general truths. "
    "Il decrit une action qui se repete regulierement dans le temps present."
)


class FakeEmbeddingProvider(EmbeddingProvider):
    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return "fake-embed"

    def embed(self, texts: list[str]) -> EmbeddingResponse:
        return EmbeddingResponse(
            embeddings=[[float(len(t))] for t in texts], usage=Usage(input_tokens=len(texts) * 10, output_tokens=None)
        )


class FakeVisionProvider(VisionProvider):
    def __init__(self, transcription: str) -> None:
        self._transcription = transcription
        self.calls: list[tuple[bytes, str, str]] = []

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return "fake-vision"

    def transcribe_image(self, image_bytes: bytes, media_type: str, instructions: str) -> VisionResponse:
        self.calls.append((image_bytes, media_type, instructions))
        return VisionResponse(text=self._transcription, usage=Usage(input_tokens=200, output_tokens=50))


class FakeVectorStore(VectorStore):
    def __init__(self) -> None:
        self.collections: dict[str, list[Chunk]] = {}

    async def upsert_chunks(self, collection: str, chunks: list[Chunk]) -> None:
        self.collections.setdefault(collection, []).extend(chunks)

    async def search(self, collection: str, query_embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        return [
            SearchResult(text=c.text, document_id=c.document_id, page_number=c.page_number, distance=0.0)
            for c in self.collections.get(collection, [])[:top_k]
        ]

    async def delete_document_chunks(self, collection: str, document_id: int) -> None:
        self.collections[collection] = [c for c in self.collections.get(collection, []) if c.document_id != document_id]


def _build_single_page_pdf(text: str) -> bytes:
    """One page with a real, plain text layer -- no images, so
    page_is_scanned() sees genuine words and classifies it as text-layer
    regardless of what pdf.extract_page_text() is mocked to return later."""
    document = pymupdf.open()
    page = document.new_page(width=400, height=600)
    page.insert_text((50, 50), text)
    pdf_bytes = document.tobytes()
    document.close()
    return pdf_bytes


def test_ingest_pdf_clean_pdf_has_no_quality_warning() -> None:
    async def scenario(db: AsyncSession) -> None:
        subject = await get_or_create_subject(db, "french")

        result = await ingest_pdf(
            db,
            FakeEmbeddingProvider(),
            FakeVisionProvider(transcription="unused"),
            FakeVectorStore(),
            subject_id=subject.id,
            subject_name="french",
            filename="clean.pdf",
            pdf_bytes=_build_single_page_pdf(_CLEAN_TEXT),
        )

        assert result.quality_warning is None
        assert result.quality_summary is not None
        assert result.quality_summary.good_pages == 1
        assert result.quality_summary.poor_pages == 0
        assert result.quality_summary.empty_pages == 0

    run(scenario)


def test_ingest_pdf_garbled_text_layer_triggers_ocr_fallback() -> None:
    async def scenario(db: AsyncSession) -> None:
        subject = await get_or_create_subject(db, "science")
        vision = FakeVisionProvider(transcription=_CLEAN_TEXT)

        with patch("app.ingestion.pipeline.pdf.extract_page_text", return_value=_GARBLED_TEXT):
            result = await ingest_pdf(
                db,
                FakeEmbeddingProvider(),
                vision,
                FakeVectorStore(),
                subject_id=subject.id,
                subject_name="science",
                filename="garbled.pdf",
                pdf_bytes=_build_single_page_pdf(_CLEAN_TEXT),
            )

        # page_is_scanned() reads the real (clean) page directly, not the
        # mocked extract_page_text() -- so this page is classified as
        # text-layer, and the fallback is the quality gate's doing, not the
        # scanned-page heuristic.
        assert result.scanned_page_count == 0
        assert len(vision.calls) == 1
        assert result.quality_summary is not None
        assert result.quality_summary.ocr_fallback_pages == 1
        # The fallback's clean transcription won out over the garbled
        # extraction, so the page reads as good, not poor.
        assert result.quality_summary.good_pages == 1
        assert result.quality_warning is None

    run(scenario)


def test_ingest_pdf_all_pages_garbled_and_ocr_also_poor_sets_quality_warning() -> None:
    async def scenario(db: AsyncSession) -> None:
        subject = await get_or_create_subject(db, "science")
        # The OCR fallback is just as garbled as the original extraction --
        # neither version is usable.
        vision = FakeVisionProvider(transcription=_GARBLED_TEXT)

        with patch("app.ingestion.pipeline.pdf.extract_page_text", return_value=_GARBLED_TEXT):
            result = await ingest_pdf(
                db,
                FakeEmbeddingProvider(),
                vision,
                FakeVectorStore(),
                subject_id=subject.id,
                subject_name="science",
                filename="all_garbled.pdf",
                pdf_bytes=_build_single_page_pdf(_CLEAN_TEXT),
            )

        assert result.quality_summary is not None
        assert result.quality_summary.poor_pages == 1
        assert result.quality_warning is not None
        assert "poor text quality" in result.quality_warning or "low" in result.quality_warning

    run(scenario)
