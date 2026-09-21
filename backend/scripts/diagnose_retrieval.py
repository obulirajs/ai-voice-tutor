"""Standalone diagnostic: why is the grounding guardrail firing on every
question ("This topic isn't covered in your textbook")?

Walks the whole retrieval path end to end against the real databases
(tutor.db + vector_store.db), in the order a failure would actually occur:
Ollama reachable? -> nomic-embed-text pulled? -> what's actually stored in
each DB? -> does embedding a query produce a real vector? -> does searching
each subject's collection surface anything above the score threshold? ->
do the collection names ingestion wrote match what retrieval reads? It also
flags a failure mode none of the above questions catch on their own:
chunks whose *text* is present but corrupted (garbled character output from
a broken PDF font encoding) — embeddings on garbage text produce garbage
similarity scores even though every plumbing step "worked".

Run from the backend directory:

    ./.venv/Scripts/python.exe scripts/diagnose_retrieval.py
"""

from __future__ import annotations

import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

# Allow running this script directly (`python scripts/diagnose_retrieval.py`)
# from the backend directory without installing the app as a package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ollama
import sqlite_vec
from dotenv import load_dotenv
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.ingestion.pdf import garbled_text_ratio
from app.ingestion.pipeline import collection_for_subject
from app.knowledge import get_retrieval_score_threshold, get_vector_store
from app.knowledge.retrieval import RetrievalService
from app.knowledge.store import collection_table
from app.models import get_embedding_provider
from app.storage.models import Document, Subject

TUTOR_DB_PATH = "tutor.db"
VECTOR_DB_PATH = "vector_store.db"
OLLAMA_HOST = "http://127.0.0.1:11434"
EMBEDDING_MODEL = "nomic-embed-text"
EXPECTED_EMBEDDING_DIM = 768

FRENCH_QUESTION = "Quels sont les festivals mentionnés dans le livre?"
SCIENCE_QUESTION = "What is the formula for glucose?"
OFF_TOPIC_QUESTION = "What is the capital of Japan?"


def _safe_print(text: str = "") -> None:
    """Print without crashing on Windows consoles whose default codepage
    (cp1252) can't encode every character a garbled chunk might contain."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode(sys.stdout.encoding or "ascii", errors="replace").decode(sys.stdout.encoding or "ascii"))


@dataclass
class RetrievalCheck:
    subject_name: str
    question: str
    best_score: float
    passes_threshold: bool
    top_snippet: str
    garbled_ratio: float


def check_ollama() -> tuple[bool, bool]:
    _safe_print("--- 1. Ollama availability ---")
    try:
        client = ollama.Client(host=OLLAMA_HOST)
        models = client.list()
    except Exception as exc:  # noqa: BLE001 - diagnostic script, report and continue
        _safe_print(f"Ollama running: no ({exc})")
        _safe_print("nomic-embed-text available: no (Ollama unreachable)")
        return False, False

    names = [m.model or "" for m in models.models]
    has_embed_model = any(name.startswith(EMBEDDING_MODEL) for name in names)
    _safe_print("Ollama running: yes")
    _safe_print(f"nomic-embed-text available: {'yes' if has_embed_model else 'no'} (models: {', '.join(names) or 'none'})")
    _safe_print("")
    return True, has_embed_model


async def check_databases() -> tuple[list[tuple[int, str, int]], dict[str, int], dict[str, float]]:
    _safe_print("--- 2. tutor.db + vector_store.db contents ---")

    engine = create_async_engine(f"sqlite+aiosqlite:///./{TUTOR_DB_PATH}")
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    subjects: list[tuple[int, str, int]] = []
    async with session_factory() as db:
        result = await db.execute(
            select(Subject.id, Subject.name, func.count(Document.id))
            .outerjoin(Document, Document.subject_id == Subject.id)
            .group_by(Subject.id)
            .order_by(Subject.id)
        )
        for subject_id, name, doc_count in result.all():
            subjects.append((subject_id, name, doc_count))
    await engine.dispose()

    for subject_id, name, doc_count in subjects:
        _safe_print(f"  subject {subject_id} {name!r}: {doc_count} document(s)")

    conn = sqlite3.connect(VECTOR_DB_PATH)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    chunk_counts: dict[str, int] = {}
    for collection, count in conn.execute("SELECT collection, COUNT(*) FROM chunks GROUP BY collection"):
        chunk_counts[collection] = count

    for collection, count in sorted(chunk_counts.items()):
        _safe_print(f"  collection {collection!r}: {count} chunk(s)")

    _safe_print("")
    _safe_print("  Corpus-wide text quality (every chunk, not just what one query retrieves):")
    corpus_garbled: dict[str, float] = {}
    for collection in sorted(chunk_counts):
        rows = conn.execute("SELECT text FROM chunks WHERE collection = ?", (collection,)).fetchall()
        ratios = [garbled_text_ratio(text) for (text,) in rows]
        garbled_chunks = sum(1 for r in ratios if r > 0.3)
        avg_ratio = garbled_chunks / len(ratios) if ratios else 0.0
        corpus_garbled[collection] = avg_ratio
        _safe_print(f"    {collection!r}: {garbled_chunks}/{len(ratios)} chunks ({avg_ratio:.0%}) look like garbled font-encoding output")
    conn.close()
    _safe_print("")
    return subjects, chunk_counts, corpus_garbled


def check_embedding() -> tuple[bool, str]:
    _safe_print("--- 3. Direct embedding sanity check ---")
    try:
        provider = get_embedding_provider()
        response = provider.embed(["What is glucose?"])
        (vector,) = response.embeddings
    except Exception as exc:  # noqa: BLE001
        _safe_print(f"Embedding call FAILED: {exc}")
        _safe_print("")
        return False, f"FAILED ({exc})"

    non_zero = any(abs(v) > 1e-9 for v in vector)
    dim_ok = len(vector) == EXPECTED_EMBEDDING_DIM
    _safe_print(f"dimension: {len(vector)} (expected {EXPECTED_EMBEDDING_DIM})")
    _safe_print(f"non-zero: {non_zero}")
    _safe_print(f"first 5 values: {vector[:5]}")
    ok = non_zero and dim_ok
    summary = f"OK (dim={len(vector)}, non-zero)" if ok else f"FAILED (dim={len(vector)}, non_zero={non_zero})"
    _safe_print(f"embedding sanity: {summary}")
    _safe_print("")
    return ok, summary


async def check_retrieval(
    subjects: list[tuple[int, str, int]], chunk_counts: dict[str, int]
) -> list[RetrievalCheck]:
    _safe_print("--- 4. Direct retrieval test ---")
    embedding_provider = get_embedding_provider()
    vector_store = get_vector_store()
    service = RetrievalService(embedding_provider, vector_store)
    threshold = get_retrieval_score_threshold()

    questions_by_subject = {
        "french": FRENCH_QUESTION,
        "science": SCIENCE_QUESTION,
    }

    results: list[RetrievalCheck] = []
    for subject_id, name, _doc_count in subjects:
        collection = collection_for_subject(subject_id)
        if chunk_counts.get(collection, 0) == 0:
            _safe_print(f"[{name}] collection {collection!r} has no chunks -- skipping retrieval test")
            continue

        question = questions_by_subject.get(name.lower(), f"What is the first lesson in {name}?")
        result = await service.retrieve(question, collection, top_k=5)

        _safe_print(f"[{name}] {question!r}")
        garbled_ratios = []
        for rank, chunk in enumerate(result.chunks, start=1):
            ratio = garbled_text_ratio(chunk.text)
            garbled_ratios.append(ratio)
            snippet = chunk.text[:80].replace("\n", " | ")
            flag = "  <-- GARBLED TEXT" if ratio > 0.3 else ""
            _safe_print(f"  #{rank} score={chunk.score:.3f} page={chunk.page_number} garbled={ratio:.0%}{flag}")
            _safe_print(f"       {snippet!r}")

        # The max across returned chunks, not just #1 -- a clean chunk (e.g.
        # an OCR'd cover page) can rank first even when the rest of the
        # collection's text is garbled, which would otherwise hide the
        # corruption from this per-query check.
        worst_garbled = max(garbled_ratios) if garbled_ratios else 0.0
        top_snippet = result.chunks[0].text[:80] if result.chunks else ""
        passes = result.best_score >= threshold
        _safe_print(f"  best_score={result.best_score:.3f} vs threshold={threshold:.2f}: {'PASS' if passes else 'FAIL'}")
        _safe_print("")
        results.append(
            RetrievalCheck(
                subject_name=name,
                question=question,
                best_score=result.best_score,
                passes_threshold=passes,
                top_snippet=top_snippet,
                garbled_ratio=worst_garbled,
            )
        )

    # Off-topic control question: should fail the threshold for every subject.
    for subject_id, name, _doc_count in subjects:
        collection = collection_for_subject(subject_id)
        if chunk_counts.get(collection, 0) == 0:
            continue
        result = await service.retrieve(OFF_TOPIC_QUESTION, collection, top_k=3)
        passes = result.best_score >= threshold
        _safe_print(
            f"[{name}] off-topic control {OFF_TOPIC_QUESTION!r}: best_score={result.best_score:.3f} "
            f"({'unexpectedly PASSED -- guardrail would NOT fire' if passes else 'correctly below threshold'})"
        )
    _safe_print("")
    return results


def check_collection_alignment(subjects: list[tuple[int, str, int]], chunk_counts: dict[str, int]) -> tuple[bool, str]:
    _safe_print("--- 5. Collection name alignment ---")
    mismatches: list[str] = []
    for subject_id, name, _doc_count in subjects:
        expected = collection_for_subject(subject_id)
        table = collection_table(expected)
        count = chunk_counts.get(expected, 0)
        _safe_print(f"  subject {subject_id} {name!r} -> collection_for_subject() = {expected!r} (table {table!r}), {count} chunk(s) found under that name")
        if count == 0:
            mismatches.append(f"{name}: expected collection {expected!r} has 0 chunks")

    ok = not mismatches
    summary = "OK" if ok else f"MISMATCH ({'; '.join(mismatches)})"
    _safe_print(f"alignment: {summary}")
    _safe_print("")
    return ok, summary


async def main() -> None:
    load_dotenv()

    ollama_running, embed_model_available = check_ollama()
    subjects, chunk_counts, corpus_garbled = await check_databases()
    embedding_ok, embedding_summary = check_embedding()
    retrieval_results = await check_retrieval(subjects, chunk_counts) if embedding_ok else []
    alignment_ok, alignment_summary = check_collection_alignment(subjects, chunk_counts)

    threshold = get_retrieval_score_threshold()

    _safe_print("=== RETRIEVAL DIAGNOSTIC SUMMARY ===")
    _safe_print(f"Ollama running: {'yes' if ollama_running else 'no'}")
    _safe_print(f"nomic-embed-text available: {'yes' if embed_model_available else 'no'}")
    _safe_print(f"Subjects in tutor.db: {[(name, doc_count) for _sid, name, doc_count in subjects]}")
    _safe_print(f"Collections in vector_store.db: {sorted(chunk_counts.items())}")
    _safe_print(f"Corpus-wide garbled text: {[(c, f'{r:.0%}') for c, r in sorted(corpus_garbled.items())]}")
    _safe_print(f"Collection name alignment: {alignment_summary}")
    _safe_print(f"Embedding sanity: {embedding_summary}")

    root_causes: list[str] = []
    if not ollama_running:
        root_causes.append("Ollama is not running -- embeddings cannot be computed at all.")
    elif not embed_model_available:
        root_causes.append(f"{EMBEDDING_MODEL} is not pulled in Ollama -- embed calls will fail or use the wrong model.")
    if not alignment_ok:
        root_causes.append(f"Collection name mismatch: {alignment_summary}")

    for check in retrieval_results:
        label = f"{check.subject_name} retrieval"
        collection = collection_for_subject(next(sid for sid, name, _dc in subjects if name == check.subject_name))
        subject_corpus_garbled = corpus_garbled.get(collection, 0.0)
        _safe_print(f"{label}: best_score={check.best_score:.2f} ({'PASS' if check.passes_threshold else 'FAIL'} vs {threshold:.2f} threshold)")
        if not check.passes_threshold:
            if subject_corpus_garbled > 0.3 or check.garbled_ratio > 0.3:
                root_causes.append(
                    f"{check.subject_name}: {subject_corpus_garbled:.0%} of chunks in this subject's collection are "
                    f"garbled font-encoding output (not real text) -- the source PDF has a broken/Type3 font "
                    f"encoding that app.ingestion.pdf.page_is_scanned() doesn't detect (word count looks normal, "
                    f"so these pages are never routed to OCR). Embeddings on garbage text produce meaningless "
                    f"similarity scores. Snippet of a retrieved chunk: {check.top_snippet[:60]!r}"
                )
            else:
                root_causes.append(
                    f"{check.subject_name}: best retrieval score ({check.best_score:.2f}) is genuinely below "
                    f"the {threshold:.2f} threshold on readable text -- either sparse coverage of this question "
                    f"or the threshold is miscalibrated for this content."
                )

    _safe_print("=== ROOT CAUSE ===")
    if root_causes:
        for cause in root_causes:
            _safe_print(f"- {cause}")
    else:
        _safe_print("- No issue found: Ollama is up, collections align, and all tested subjects clear the threshold.")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
