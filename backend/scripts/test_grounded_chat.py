"""One-off script: exercise the full retrieval-grounded chat loop
end-to-end against the real databases (tutor.db + vector_store.db) --
retrieve -> grounded prompt -> Model Provider -> cited reply, or the
grounding guardrail's refusal. For Raj to eyeball; not an automated test
(see tests/test_orchestration.py and tests/test_api.py for those, and
GET /subjects/{subject_id}/qa-check for a pass/fail version of this same
idea).

Calls app.orchestration.handle_turn() directly (same as
scripts/query_vector_store.py calls the knowledge layer directly) rather
than going over HTTP, so it also works without a server running. Makes
real Model Provider and embedding calls -- this has a real cost per run.

Runs three scenarios:
  a. a covered question -- expect a grounded reply with page citations.
  b. an uncovered question -- expect the grounding guardrail's refusal.
  c. a multi-turn follow-up -- expect conversation history to carry over
     and the reply to still cite sources.

Requires a subject/document row in tutor.db matching the ingested
collection in vector_store.db (see restore/ingest first if missing) and a
working Model Provider (ANTHROPIC_API_KEY / MODEL_PROVIDER in .env).

Run from the backend directory:

    ./.venv/Scripts/python.exe scripts/test_grounded_chat.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Allow running this script directly (`python scripts/test_grounded_chat.py`)
# from the backend directory without installing the app as a package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.knowledge import get_vector_store
from app.models import get_embedding_provider, get_provider
from app.orchestration import TurnResult, handle_turn

# Matches the subject this project's French textbook was ingested under
# (see app.ingestion.pipeline.collection_for_subject -- its subject_id's
# collection is what vector_store.db actually holds the 195 chunks under).
SUBJECT_NAME = "French"

COVERED_QUESTION = "Quels sont les festivals mentionnés dans le livre?"
UNCOVERED_QUESTION = "What is the capital of Japan?"
FOLLOWUP_SEED_QUESTION = "Quels sont les jours fériés en France?"
FOLLOWUP_QUESTION = "Can you explain more about that?"


def _print_turn(label: str, question: str, result: TurnResult) -> None:
    guardrail_triggered = not result.sources
    print("=" * 80)
    print(f"[{label}] {question}")
    print(f"session_id: {result.session_id}")
    print(f"guardrail_triggered: {guardrail_triggered}")
    print(f"reply: {result.reply}")
    if result.sources:
        print("sources:")
        for source in result.sources:
            print(f"  - page={source.page_number} score={source.score:.3f}")
    print()


async def main() -> None:
    load_dotenv()

    provider = get_provider()
    embedding_provider = get_embedding_provider()
    vector_store = get_vector_store()

    database_url = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./tutor.db")
    engine = create_async_engine(database_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with session_factory() as db:
        covered = await handle_turn(
            db,
            provider,
            embedding_provider,
            vector_store,
            session_id=None,
            user_message=COVERED_QUESTION,
            subject=SUBJECT_NAME,
        )
        await db.commit()
        _print_turn("a. covered question", COVERED_QUESTION, covered)

        uncovered = await handle_turn(
            db,
            provider,
            embedding_provider,
            vector_store,
            session_id=None,
            user_message=UNCOVERED_QUESTION,
            subject=SUBJECT_NAME,
        )
        await db.commit()
        _print_turn("b. uncovered question", UNCOVERED_QUESTION, uncovered)

        seed = await handle_turn(
            db,
            provider,
            embedding_provider,
            vector_store,
            session_id=None,
            user_message=FOLLOWUP_SEED_QUESTION,
            subject=SUBJECT_NAME,
        )
        await db.commit()
        _print_turn("c. multi-turn (seed)", FOLLOWUP_SEED_QUESTION, seed)

        followup = await handle_turn(
            db,
            provider,
            embedding_provider,
            vector_store,
            session_id=seed.session_id,
            user_message=FOLLOWUP_QUESTION,
            subject=SUBJECT_NAME,
        )
        await db.commit()
        _print_turn("c. multi-turn (follow-up)", FOLLOWUP_QUESTION, followup)

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
