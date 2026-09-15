"""One-off script: run a few French questions straight through the
VectorStore interface, no chat endpoint involved, to eyeball retrieval
quality directly against whatever is currently ingested.

Run from the backend directory:

    ./.venv/Scripts/python.exe scripts/query_vector_store.py

Uses the same get_embedding_provider()/get_vector_store() factories the app
itself calls (app.knowledge.golden_qa.run_golden_qa_check does the same
embed-then-search dance) -- never touches an adapter directly.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Allow running this script directly (`python scripts/query_vector_store.py`)
# from the backend directory without installing the app as a package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from app.knowledge import get_vector_store  # noqa: E402
from app.models import get_embedding_provider  # noqa: E402

# The French textbook's collection from this project's per-subject
# namespacing (collection_for_subject(subject_id) in app.ingestion.pipeline).
COLLECTION = "subject_1"

QUESTIONS = [
    "What details appear on the 'Le Monde en Fete' festival poster on page 107?",
    "What job advertisements appear on page 27 of the textbook?",
    "What is the futur anterieur of the verb DONNER for 'nous'?",
]


async def main() -> None:
    load_dotenv()

    embedding_provider = get_embedding_provider()
    vector_store = get_vector_store()

    for question in QUESTIONS:
        (query_embedding,) = embedding_provider.embed([question])
        results = await vector_store.search(COLLECTION, query_embedding, top_k=3)

        print("=" * 80)
        print(f"QUERY: {question}")
        for rank, result in enumerate(results, start=1):
            similarity = 1.0 - result.distance
            snippet = result.text[:150].replace("\n", " | ")
            print(f"  #{rank}  page={result.page_number}  similarity={similarity:.3f}")
            print(f"       {snippet}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
