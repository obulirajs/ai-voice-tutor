"""One-off script: back-fills stored PDFs for Document rows created before
the Visual Companion's page-view feature existed, by SHA-256 hash-matching
them against files already sitting in backend/textbooks_input/ -- no
re-upload needed.

Run from the backend directory:

    python -m scripts.import_existing_pdfs [--dry-run]

Matches ONLY by content hash (Document.content_hash, the same SHA-256 the
upload endpoint computes for duplicate detection) -- never by filename, so a
renamed copy of the same file still matches, and a same-named but different
file does not. Idempotent: a document whose PDF is already stored is
reported as "already stored" and left untouched, so running this twice (or
after uploading more textbooks) is always safe.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Allow running this script directly (`python scripts/import_existing_pdfs.py`)
# from the backend directory without installing the app as a package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import STORAGE_DEFAULTS
from app.ingestion import compute_content_hash
from app.storage import get_document_pdf_path
from app.storage.models import Document

DEFAULT_TEXTBOOKS_DIR = Path(__file__).resolve().parent.parent / "textbooks_input"


@dataclass
class ImportSummary:
    matched: list[int] = field(default_factory=list)
    already_stored: list[int] = field(default_factory=list)
    no_match: list[tuple[int, str]] = field(default_factory=list)


async def import_existing_pdfs(
    db: AsyncSession, *, textbooks_dir: Path, dry_run: bool = False
) -> ImportSummary:
    """Back-fills stored PDFs for every Document row with no PDF on disk yet.

    Hashes every *.pdf in `textbooks_dir` once, then matches each
    PDF-less Document by content_hash -- never by filename. A document
    whose PDF is already stored (get_document_pdf_path(document.id) exists)
    is reported as already-stored and skipped entirely, which is what makes
    running this script twice safe.
    """
    summary = ImportSummary()

    candidate_files = sorted(textbooks_dir.glob("*.pdf")) if textbooks_dir.exists() else []
    hash_to_file = {compute_content_hash(f.read_bytes()): f for f in candidate_files}

    result = await db.execute(select(Document))
    documents = list(result.scalars().all())

    for document in documents:
        pdf_path = get_document_pdf_path(document.id)
        if pdf_path.exists():
            summary.already_stored.append(document.id)
            continue

        source_file = hash_to_file.get(document.content_hash) if document.content_hash else None
        if source_file is None:
            summary.no_match.append((document.id, document.filename))
            continue

        if not dry_run:
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_file, pdf_path)
        summary.matched.append(document.id)

    return summary


def _print_summary(summary: ImportSummary, *, dry_run: bool) -> None:
    suffix = " (dry run, not copied)" if dry_run else ""
    print(f"Matched: {len(summary.matched)}{suffix}")
    for document_id in summary.matched:
        print(f"  - document {document_id}")

    print(f"Already stored: {len(summary.already_stored)}")
    for document_id in summary.already_stored:
        print(f"  - document {document_id}")

    print(f"No match (need re-upload): {len(summary.no_match)}")
    for document_id, filename in summary.no_match:
        print(f"  - document {document_id}: {filename!r}")


async def _main(dry_run: bool) -> None:
    load_dotenv()

    engine = create_async_engine(os.getenv("DATABASE_URL", STORAGE_DEFAULTS.database_url))
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with session_factory() as db:
        summary = await import_existing_pdfs(db, textbooks_dir=DEFAULT_TEXTBOOKS_DIR, dry_run=dry_run)

    await engine.dispose()
    _print_summary(summary, dry_run=dry_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Report matches without copying files.")
    args = parser.parse_args()
    asyncio.run(_main(args.dry_run))
