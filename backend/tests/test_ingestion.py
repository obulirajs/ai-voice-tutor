from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

import pymupdf
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.ingestion import ingest_pdf
from app.ingestion.chunking import chunk_text
from app.ingestion.pdf import garbled_text_ratio, load_pdf, page_is_scanned
from app.knowledge import Chunk, SearchResult, VectorStore
from app.models import (
    EmbeddingProvider,
    EmbeddingResponse,
    Usage,
    VisionProvider,
    VisionResponse,
)
from app.storage import get_or_create_subject, list_documents
from app.storage.models import Base, UsageEvent

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


# A small vocabulary a FakeEmbeddingProvider scores keyword-presence
# against, so retrieval in tests behaves meaningfully (routes a question to
# the chunk that actually shares its keyword) without a real embedding model.
_VOCAB = ["present", "compose", "vocabulaire", "diagramme"]


class FakeEmbeddingProvider(EmbeddingProvider):
    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return "fake-embed"

    def embed(self, texts: list[str]) -> EmbeddingResponse:
        embeddings = [[float(word in text.lower()) for word in _VOCAB] for text in texts]
        return EmbeddingResponse(embeddings=embeddings, usage=Usage(input_tokens=len(texts) * 10, output_tokens=None))


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
    """In-memory VectorStore fake, mirroring how test_api.py fakes ModelProvider."""

    def __init__(self) -> None:
        self.collections: dict[str, list[Chunk]] = {}

    async def upsert_chunks(self, collection: str, chunks: list[Chunk]) -> None:
        self.collections.setdefault(collection, []).extend(chunks)

    async def search(self, collection: str, query_embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        def squared_distance(chunk: Chunk) -> float:
            return sum((a - b) ** 2 for a, b in zip(chunk.embedding, query_embedding, strict=True))

        ranked = sorted(self.collections.get(collection, []), key=squared_distance)[:top_k]
        return [
            SearchResult(
                text=chunk.text,
                document_id=chunk.document_id,
                page_number=chunk.page_number,
                distance=squared_distance(chunk),
            )
            for chunk in ranked
        ]

    async def delete_document_chunks(self, collection: str, document_id: int) -> None:
        remaining = [c for c in self.collections.get(collection, []) if c.document_id != document_id]
        self.collections[collection] = remaining


def _build_test_pdf() -> bytes:
    """Page 1: real text layer. Page 2: scanned (full-page image, no text)."""
    document = pymupdf.open()

    text_page = document.new_page(width=400, height=600)
    text_page.insert_text(
        (50, 50),
        "Le present tense chunk. Le present s'utilise pour des actions habituelles.",
    )

    scanned_page = document.new_page(width=400, height=600)
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 400, 600))
    pixmap.set_rect(pixmap.irect, (200, 200, 200))
    scanned_page.insert_image(scanned_page.rect, pixmap=pixmap)

    pdf_bytes = document.tobytes()
    document.close()
    return pdf_bytes


def test_page_is_scanned_detects_text_vs_image_pages() -> None:
    document = load_pdf(_build_test_pdf())

    assert page_is_scanned(document[0]) is False
    assert page_is_scanned(document[1]) is True


def _build_page(
    document: pymupdf.Document,
    *,
    background: bool,
    caption_words: str,
    content_image_bboxes: list[tuple[float, float, float, float]],
) -> pymupdf.Page:
    """Builds a 612x792 page shaped like this textbook's real layout: an
    optional full-page decorative background image, a short caption/label
    line of real text, and zero or more smaller "content" images placed at
    the given boxes -- mirroring the low-word-count-plus-embedded-images
    pattern seen on the real scanned pages this heuristic needs to catch.
    """
    page = document.new_page(width=612, height=792)

    if background:
        bg_pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 612, 792))
        bg_pixmap.set_rect(bg_pixmap.irect, (240, 240, 240))
        page.insert_image(page.rect, pixmap=bg_pixmap)

    if caption_words:
        page.insert_text((50, 50), caption_words)

    for bbox in content_image_bboxes:
        pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 100, 100))
        pixmap.set_rect(pixmap.irect, (100, 100, 100))
        page.insert_image(pymupdf.Rect(*bbox), pixmap=pixmap)

    return page


def test_page_is_scanned_flags_low_word_count_page_with_incidental_caption_text() -> None:
    """Regression test for the Phase 2 golden-QA finding: a page with only a
    caption/label line of real text (well over the old 40-character
    threshold) but whose actual content -- a media-logo grid, a poster --
    lives entirely in embedded images must still be flagged as scanned, not
    waved through as text-layer because the caption alone passed a raw
    character-count check.
    """
    document = pymupdf.open()
    # Mirrors PDF page 56 (printed p.48, "media logos"): ~27 words of
    # caption/label text, a full-page background, and several content
    # images covering well over the 0.15 coverage threshold.
    page = _build_page(
        document,
        background=True,
        caption_words="Les medias Je decouvre un hebdomadaire un quotidien des chaines de television un blog une publicite",
        content_image_bboxes=[
            (50, 100, 250, 300),
            (300, 100, 550, 300),
            (50, 350, 250, 550),
            (300, 350, 550, 550),
        ],
    )

    assert page_is_scanned(page) is True
    document.close()


def test_page_is_scanned_leaves_text_heavy_illustrated_page_as_text_layer() -> None:
    """A genuinely text-heavy page (a real dialogue/explanation, well above
    the word-count threshold) must stay classified as text-layer even when
    it carries a full-page decorative background plus a couple of
    illustrative images, so ordinary lesson pages don't needlessly go
    through the slower, paid vision path.
    """
    document = pymupdf.open()
    paragraph = (
        "Pauline et Ali discutent de leurs projets apres le baccalaureat. "
        "Ali cherche un travail a mi-temps pendant ses etudes universitaires. "
        "Pauline lui suggere de contacter le CROUS pour les bourses disponibles. "
        "Ils parlent aussi des logements etudiants pres du campus."
    )
    page = _build_page(
        document,
        background=True,
        caption_words=paragraph,
        content_image_bboxes=[(50, 550, 250, 700)],
    )

    assert page_is_scanned(page) is False
    document.close()


def test_garbled_text_ratio_distinguishes_real_text_from_font_encoding_garbage() -> None:
    """Regression coverage for the diagnose_retrieval.py finding: the NCERT
    Science PDF's body text uses a Type3 font with no usable ToUnicode map,
    so pymupdf decodes it to private-use/dingbat code points instead of
    real characters."""
    assert garbled_text_ratio("Bonjour, comment ca va aujourd'hui?") < 0.05
    assert garbled_text_ratio("❙\x00✁✂✄\x00✂✶☎✆☎✝✞✟✠✡☛☞✠✌✍☛✍✡✎✏✡✑✒☞✠✌✍") > 0.9
    assert garbled_text_ratio("") == 0.0


class _FakeWordsPage:
    """Minimal stand-in for pymupdf.Page covering only what page_is_scanned()
    reads before its garbled-text check short-circuits (get_text()) -- a
    real Type3-font-corrupted PDF page isn't something insert_text() can
    reproduce (pymupdf substitutes a fallback glyph for codepoints missing
    from the font instead of preserving them), so this fakes the extraction
    result directly instead."""

    def __init__(self, words: list[str]) -> None:
        self._words = words

    def get_text(self, option: str | None = None) -> object:
        if option == "words":
            return [(0.0, 0.0, 10.0, 10.0, word, 0, 0, index) for index, word in enumerate(self._words)]
        return " ".join(self._words)


def test_page_is_scanned_flags_garbled_font_encoded_text() -> None:
    """A page whose text layer decodes to font-encoding garbage must be
    flagged as needing OCR even though its word count clears
    _MIN_WORDS_PER_PAGE -- the real-world case this heuristic was missing
    before the diagnose_retrieval.py investigation.
    """
    garbled_words = ["❙", "✁✂✄", "☎✆✝✞✟", "✠✡☛☞✌", "✍✎✏✑✒"] * 10
    page = _FakeWordsPage(garbled_words)

    assert page_is_scanned(page) is True  # type: ignore[arg-type]


def test_chunk_text_splits_on_paragraph_boundaries() -> None:
    text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."

    chunks = chunk_text(text, max_chars=1000)

    assert chunks == ["Paragraph one.\n\nParagraph two.\n\nParagraph three."]


def test_chunk_text_packs_up_to_max_chars_then_splits() -> None:
    # A + B fit together under max_chars (400+2+400=802 <= 900); adding C
    # would overflow it (802+2+400=1204 > 900), so C starts a new chunk.
    text = "A" * 400 + "\n\n" + "B" * 400 + "\n\n" + "C" * 400

    chunks = chunk_text(text, max_chars=900)

    assert len(chunks) == 2
    assert chunks[0] == "A" * 400 + "\n\n" + "B" * 400
    assert chunks[1] == "C" * 400


def test_chunk_text_returns_empty_list_for_blank_text() -> None:
    assert chunk_text("   \n\n  ") == []


def test_ingest_pdf_extracts_text_page_directly_without_vision_call() -> None:
    async def scenario(db: AsyncSession) -> None:
        subject = await get_or_create_subject(db, "french")
        # A real placeholder sentence, not literally "unused" -- the quality
        # gate now skips chunking a page whose text scores "empty" (fewer
        # than 5 words), and a single word would otherwise trigger that.
        vision = FakeVisionProvider(transcription="Diagramme illustrant la conjugaison du present.")

        result = await ingest_pdf(
            db,
            FakeEmbeddingProvider(),
            vision,
            FakeVectorStore(),
            subject_id=subject.id,
            subject_name="french",
            filename="chapter5.pdf",
            pdf_bytes=_build_test_pdf(),
        )

        assert result.page_count == 2
        assert result.scanned_page_count == 1
        assert result.chunk_count >= 2
        # Only the scanned page (page 2) should have gone through vision.
        assert len(vision.calls) == 1

    run(scenario)


def test_ingest_pdf_records_document_row() -> None:
    async def scenario(db: AsyncSession) -> None:
        subject = await get_or_create_subject(db, "french")

        result = await ingest_pdf(
            db,
            FakeEmbeddingProvider(),
            FakeVisionProvider(transcription="Diagramme illustrant la conjugaison."),
            FakeVectorStore(),
            subject_id=subject.id,
            subject_name="french",
            filename="chapter5.pdf",
            pdf_bytes=_build_test_pdf(),
        )

        documents = await list_documents(db, subject.id)

        assert len(documents) == 1
        assert documents[0].id == result.document_id
        assert documents[0].filename == "chapter5.pdf"
        assert documents[0].page_count == 2
        assert documents[0].scanned_page_count == 1

    run(scenario)


def test_ingest_pdf_runs_consistency_check_and_it_reaches_the_result() -> None:
    async def scenario(db: AsyncSession) -> None:
        subject = await get_or_create_subject(db, "french")

        result = await ingest_pdf(
            db,
            FakeEmbeddingProvider(),
            FakeVisionProvider(transcription="Diagramme illustrant la conjugaison."),
            FakeVectorStore(),
            subject_id=subject.id,
            subject_name="french",
            filename="chapter5.pdf",
            pdf_bytes=_build_test_pdf(),
        )

        assert result.consistency_check.sampled > 0
        assert result.consistency_check.sampled == result.consistency_check.passed
        assert result.consistency_check.failed == []

    run(scenario)


def test_ingest_pdf_consistency_check_reports_failure_when_storage_is_broken() -> None:
    class BrokenVectorStore(FakeVectorStore):
        """upsert_chunks works normally, but search() always returns content
        unrelated to the query -- simulates a corrupted/mismatched embedding
        between storage and retrieval. The failure must be reported in the
        result, not swallowed.
        """

        async def search(self, collection: str, query_embedding: list[float], top_k: int = 5) -> list[SearchResult]:
            return [
                SearchResult(text="totally unrelated content", document_id=None, page_number=None, distance=99.0)
            ]

    async def scenario(db: AsyncSession) -> None:
        subject = await get_or_create_subject(db, "french")

        result = await ingest_pdf(
            db,
            FakeEmbeddingProvider(),
            FakeVisionProvider(transcription="Diagramme illustrant la conjugaison."),
            BrokenVectorStore(),
            subject_id=subject.id,
            subject_name="french",
            filename="chapter5.pdf",
            pdf_bytes=_build_test_pdf(),
        )

        assert result.consistency_check.sampled > 0
        assert result.consistency_check.passed == 0
        assert len(result.consistency_check.failed) == result.consistency_check.sampled

    run(scenario)


def test_ingest_pdf_transcribes_scanned_page_via_vision_provider() -> None:
    async def scenario(db: AsyncSession) -> None:
        subject = await get_or_create_subject(db, "french")
        vision = FakeVisionProvider(transcription="Diagramme illustrant la conjugaison du present.")

        result = await ingest_pdf(
            db,
            FakeEmbeddingProvider(),
            vision,
            FakeVectorStore(),
            subject_id=subject.id,
            subject_name="french",
            filename="chapter5.pdf",
            pdf_bytes=_build_test_pdf(),
        )

        assert result.scanned_page_count == 1
        image_bytes, media_type, _instructions = vision.calls[0]
        assert media_type == "image/png"
        assert image_bytes.startswith(b"\x89PNG")

    run(scenario)


def test_ingest_pdf_logs_usage_events_for_vision_and_embedding_calls() -> None:
    """Observability must cover ingestion, not just the chat path: one
    usage_events row per vision call (per scanned page) and one per
    embedding call (one batched call over all chunks), even though no chat
    question has been asked yet.
    """

    async def scenario(db: AsyncSession) -> None:
        subject = await get_or_create_subject(db, "french")

        result = await ingest_pdf(
            db,
            FakeEmbeddingProvider(),
            FakeVisionProvider(transcription="Diagramme illustrant la conjugaison."),
            FakeVectorStore(),
            subject_id=subject.id,
            subject_name="french",
            filename="chapter5.pdf",
            pdf_bytes=_build_test_pdf(),
        )

        rows = (await db.execute(select(UsageEvent).order_by(UsageEvent.id))).scalars().all()

        assert len(rows) == 2
        assert all(row.session_id is None for row in rows)
        assert all(row.subject_id == subject.id for row in rows)

        vision_events = [r for r in rows if r.provider == "fake" and r.model == "fake-vision"]
        assert len(vision_events) == 1
        assert vision_events[0].event_type == "vision"
        assert vision_events[0].input_tokens == 200
        assert vision_events[0].output_tokens == 50
        assert vision_events[0].cost_usd == 0.0  # not in the pricing table, same as an unpriced Ollama model
        assert vision_events[0].latency_ms is not None

        embedding_events = [r for r in rows if r.provider == "fake" and r.model == "fake-embed"]
        assert len(embedding_events) == 1
        assert embedding_events[0].event_type == "embedding"
        assert embedding_events[0].input_tokens == result.chunk_count * 10
        assert embedding_events[0].output_tokens is None
        assert embedding_events[0].cost_usd == 0.0
        assert embedding_events[0].latency_ms is not None

    run(scenario)


def _build_english_test_pdf() -> bytes:
    """Six pages of unambiguously English text -- each page's extracted text
    becomes its own chunk (chunking is per-page), so this reliably produces
    >=5 chunks for the subject/language sanity check to sample from.
    """
    sentences = [
        "This chapter reviews ordinary English grammar and vocabulary for beginner students.",
        "Students should practice these verb conjugations every single day after class.",
        "The next lesson introduces new adjectives and common expressions used daily.",
        "Homework this week focuses on reading comprehension and simple dictation exercises.",
        "Remember to review the pronunciation guide before the listening test on Friday.",
        "Class participation and written assignments both count toward the final grade.",
    ]
    document = pymupdf.open()
    for sentence in sentences:
        page = document.new_page(width=612, height=200)
        page.insert_text((50, 50), sentence * 3)  # comfortably above the min-signal threshold

    pdf_bytes = document.tobytes()
    document.close()
    return pdf_bytes


def _build_french_test_pdf() -> bytes:
    """Six pages of genuine accented French text -- unlike _build_test_pdf()'s
    unaccented pseudo-French fixture, this carries the character-range
    signal the language heuristic actually looks for.
    """
    sentences = [
        "Le professeur explique la conjugaison des verbes réguliers en français.",
        "Les élèves répètent les phrases après le professeur chaque matin.",
        "Cette leçon présente le vocabulaire lié à la nourriture et aux repas.",
        "Il faut réviser les leçons précédentes avant l'examen de vendredi prochain.",
        "La grammaire française comprend plusieurs règles particulières à apprendre.",
        "Chaque élève doit répondre aux questions posées à la fin du chapitre.",
    ]
    document = pymupdf.open()
    for sentence in sentences:
        page = document.new_page(width=612, height=200)
        page.insert_text((50, 50), sentence * 3)

    pdf_bytes = document.tobytes()
    document.close()
    return pdf_bytes


def test_ingest_pdf_warns_on_english_content_uploaded_to_french_subject() -> None:
    async def scenario(db: AsyncSession) -> None:
        subject = await get_or_create_subject(db, "french")

        result = await ingest_pdf(
            db,
            FakeEmbeddingProvider(),
            FakeVisionProvider(transcription="unused"),
            FakeVectorStore(),
            subject_id=subject.id,
            subject_name="french",
            filename="wrong-subject.pdf",
            pdf_bytes=_build_english_test_pdf(),
        )

        assert result.chunk_count >= 5
        assert result.subject_mismatch_warning is not None
        assert "English" in result.subject_mismatch_warning
        assert "French" in result.subject_mismatch_warning

    run(scenario)


def test_ingest_pdf_no_warning_for_french_content_uploaded_to_french_subject() -> None:
    async def scenario(db: AsyncSession) -> None:
        subject = await get_or_create_subject(db, "french")

        result = await ingest_pdf(
            db,
            FakeEmbeddingProvider(),
            FakeVisionProvider(transcription="unused"),
            FakeVectorStore(),
            subject_id=subject.id,
            subject_name="french",
            filename="chapter5.pdf",
            pdf_bytes=_build_french_test_pdf(),
        )

        assert result.chunk_count >= 5
        assert result.subject_mismatch_warning is None

    run(scenario)
