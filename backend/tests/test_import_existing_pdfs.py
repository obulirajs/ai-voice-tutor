from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.storage import create_document, get_document_pdf_path, get_or_create_subject
from app.storage.models import Base
from scripts.import_existing_pdfs import import_existing_pdfs

Scenario = Callable[[AsyncSession], Coroutine[Any, Any, None]]


def run(scenario: Scenario) -> None:
    """Run one test scenario against a fresh in-memory SQLite engine, then dispose it."""

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


def test_hash_match_copies_pdf_to_documents_dir(tmp_path: Path) -> None:
    textbooks_dir = tmp_path / "textbooks_input"
    textbooks_dir.mkdir()
    pdf_bytes = b"%PDF-1.4 fake content for hashing"
    # Deliberately a different name than the document's own filename --
    # matching must be by content hash, not name.
    (textbooks_dir / "French_Entre_Jeunes.pdf").write_bytes(pdf_bytes)
    content_hash = hashlib.sha256(pdf_bytes).hexdigest()

    async def scenario(db: AsyncSession) -> None:
        subject = await get_or_create_subject(db, "french")
        document = await create_document(
            db,
            subject_id=subject.id,
            filename="chapter1.pdf",
            collection="subject_1",
            page_count=1,
            content_hash=content_hash,
        )

        summary = await import_existing_pdfs(db, textbooks_dir=textbooks_dir)

        assert summary.matched == [document.id]
        assert summary.already_stored == []
        assert summary.no_match == []
        assert get_document_pdf_path(document.id).read_bytes() == pdf_bytes

    run(scenario)


def test_filename_only_match_is_not_copied(tmp_path: Path) -> None:
    """A candidate file with the SAME NAME as the document's original
    filename but different content must not be treated as a match -- only
    the content hash counts."""
    textbooks_dir = tmp_path / "textbooks_input"
    textbooks_dir.mkdir()
    (textbooks_dir / "chapter1.pdf").write_bytes(b"%PDF-1.4 completely different bytes")

    async def scenario(db: AsyncSession) -> None:
        subject = await get_or_create_subject(db, "french")
        document = await create_document(
            db,
            subject_id=subject.id,
            filename="chapter1.pdf",
            collection="subject_1",
            page_count=1,
            content_hash=hashlib.sha256(b"the real original bytes").hexdigest(),
        )

        summary = await import_existing_pdfs(db, textbooks_dir=textbooks_dir)

        assert summary.matched == []
        assert summary.no_match == [(document.id, "chapter1.pdf")]
        assert not get_document_pdf_path(document.id).exists()

    run(scenario)


def test_second_run_reports_already_stored(tmp_path: Path) -> None:
    textbooks_dir = tmp_path / "textbooks_input"
    textbooks_dir.mkdir()
    pdf_bytes = b"%PDF-1.4 fake content for hashing"
    (textbooks_dir / "chapter1.pdf").write_bytes(pdf_bytes)
    content_hash = hashlib.sha256(pdf_bytes).hexdigest()

    async def scenario(db: AsyncSession) -> None:
        subject = await get_or_create_subject(db, "french")
        document = await create_document(
            db,
            subject_id=subject.id,
            filename="chapter1.pdf",
            collection="subject_1",
            page_count=1,
            content_hash=content_hash,
        )

        first = await import_existing_pdfs(db, textbooks_dir=textbooks_dir)
        assert first.matched == [document.id]

        second = await import_existing_pdfs(db, textbooks_dir=textbooks_dir)
        assert second.matched == []
        assert second.already_stored == [document.id]
        assert second.no_match == []

    run(scenario)


def test_dry_run_reports_match_without_copying(tmp_path: Path) -> None:
    textbooks_dir = tmp_path / "textbooks_input"
    textbooks_dir.mkdir()
    pdf_bytes = b"%PDF-1.4 fake content for hashing"
    (textbooks_dir / "chapter1.pdf").write_bytes(pdf_bytes)
    content_hash = hashlib.sha256(pdf_bytes).hexdigest()

    async def scenario(db: AsyncSession) -> None:
        subject = await get_or_create_subject(db, "french")
        document = await create_document(
            db,
            subject_id=subject.id,
            filename="chapter1.pdf",
            collection="subject_1",
            page_count=1,
            content_hash=content_hash,
        )

        summary = await import_existing_pdfs(db, textbooks_dir=textbooks_dir, dry_run=True)

        assert summary.matched == [document.id]
        assert not get_document_pdf_path(document.id).exists()

    run(scenario)
