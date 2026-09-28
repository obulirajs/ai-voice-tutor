"""One-off benchmarking script: run backend/eval/tutor_eval_set.json through
the full retrieval-grounded chat loop (app.orchestration.handle_turn --
the same code path as POST /chat) for one Ollama model, and record
per-question metrics to a JSONL file for later grading.

Not part of the app; a throwaway harness for the local-model comparison
requested in this eval (see backend/eval/reports/). Reuses
scripts/test_grounded_chat.py's real-DB setup pattern.

The model itself is NOT selected by --model here -- that would only
duplicate what MODEL_PROVIDER/OLLAMA_MODEL in backend/.env already
configure, and get_provider() must read the same env every other code path
reads (dependency inversion -- see CLAUDE.md). --model instead labels the
output header and is asserted against whatever get_provider() actually
constructs, so a stale .env edit fails loudly instead of silently mislabeling
a run.

Run from the backend directory, once per model, as a fresh process (so
config reloads from .env):

    ./.venv/Scripts/python.exe scripts/compare_tutor_models.py \\
        --model llama3.2:1b --out eval/results/llama3.2-1b.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Allow running this script directly from the backend directory without
# installing the app as a package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import ORCHESTRATION_DEFAULTS
from app.knowledge import get_vector_store
from app.models import get_embedding_provider, get_provider
from app.orchestration import TurnResult, handle_turn

EVAL_SET_PATH = Path(__file__).resolve().parent.parent / "eval" / "tutor_eval_set.json"
WARMUP_QUESTION = "Bonjour, comment vas-tu ?"
WARMUP_SUBJECT = "French"
PER_ITEM_TIMEOUT_S = 180.0

_THINK_LEAK = re.compile(r"<\s*/?\s*think\s*>", re.IGNORECASE)
_COLUMN_SPLIT = re.compile(r"\s{2,}")


def _ollama_version() -> str:
    try:
        out = subprocess.run(["ollama", "--version"], capture_output=True, text=True, timeout=15)
        return out.stdout.strip() or out.stderr.strip()
    except Exception as exc:  # pragma: no cover -- diagnostic path only
        return f"<error: {exc}>"


def _ollama_ps_rows() -> list[dict[str, str]]:
    """Parses `ollama ps` into a list of {NAME, ID, SIZE, PROCESSOR, UNTIL} dicts."""
    try:
        out = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=15)
    except Exception:  # pragma: no cover -- diagnostic path only
        return []
    lines = [line for line in out.stdout.splitlines() if line.strip()]
    if not lines:
        return []
    header = _COLUMN_SPLIT.split(lines[0].strip())
    rows = []
    for line in lines[1:]:
        fields = _COLUMN_SPLIT.split(line.strip())
        rows.append(dict(zip(header, fields, strict=False)))
    return rows


def _ollama_stop_all_loaded() -> list[str]:
    stopped = []
    for row in _ollama_ps_rows():
        name = row.get("NAME")
        if name:
            subprocess.run(["ollama", "stop", name], capture_output=True, text=True, timeout=30)
            stopped.append(name)
    return stopped


def _processor_for(model: str) -> str | None:
    for row in _ollama_ps_rows():
        if row.get("NAME") == model:
            return row.get("PROCESSOR")
    return None


def _load_eval_set() -> list[dict]:
    return json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))


def _load_done_ids(out_path: Path) -> tuple[list[str], list[str]]:
    """Returns (existing_lines, done_item_ids) so a rerun can append rather
    than duplicate work already written to out_path."""
    if not out_path.exists():
        return [], []
    lines = out_path.read_text(encoding="utf-8").splitlines()
    done_ids = []
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("type") == "item_result":
            done_ids.append(record["id"])
    return lines, done_ids


def _append_jsonl(out_path: Path, record: dict) -> None:
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _guardrail_refused(reply: str, sources: list) -> bool:
    """No sources means the guardrail fired before the model was ever
    called; the model can also self-refuse (with sources still attached,
    per prompt_builder's "related but doesn't answer" instruction) by
    emitting the exact canned not-covered sentence."""
    not_covered = "This topic isn't covered in your textbook. Try asking about something from your syllabus."
    return not sources or not_covered.lower() in reply.lower()


async def _run_one(
    db: AsyncSession,
    provider,
    embedding_provider,
    vector_store,
    item: dict,
    mode: str,
) -> dict:
    started = time.perf_counter()
    error: str | None = None
    reply = ""
    sources: list[dict] = []
    try:
        result: TurnResult = await asyncio.wait_for(
            handle_turn(
                db,
                provider,
                embedding_provider,
                vector_store,
                session_id=None,
                user_message=item["question"],
                subject=item["subject"],
                temperature=0.0,
                mode=mode,
            ),
            timeout=PER_ITEM_TIMEOUT_S,
        )
        await db.commit()
        reply = result.reply
        sources = [{"page_number": s.page_number, "score": s.score} for s in result.sources]
    except TimeoutError:
        error = f"timeout after {PER_ITEM_TIMEOUT_S}s"
    except Exception as exc:  # noqa: BLE001 -- record any adapter/runtime error and keep going
        error = f"{type(exc).__name__}: {exc}"

    latency_ms = (time.perf_counter() - started) * 1000

    return {
        "type": "item_result",
        "id": item["id"],
        "item_type": item["type"],
        "subject": item["subject"],
        "question": item["question"],
        "reply": reply,
        "sources": sources,
        "guardrail_refused": _guardrail_refused(reply, sources) if not error else None,
        "latency_ms": round(latency_ms, 1),
        "word_count": len(reply.split()) if reply else 0,
        "contains_think_tag": bool(_THINK_LEAK.search(reply)) if reply else False,
        "error": error,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Ollama model name; asserted against get_provider()'s model")
    parser.add_argument("--out", required=True, type=Path, help="Output JSONL path")
    parser.add_argument("--mode", default=ORCHESTRATION_DEFAULTS.default_teaching_mode, choices=["textbook", "teacher"])
    args = parser.parse_args()

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    provider = get_provider()
    if provider.model_name != args.model:
        raise SystemExit(
            f"--model {args.model!r} doesn't match the configured provider's model "
            f"{provider.model_name!r} -- update OLLAMA_MODEL in backend/.env first."
        )
    embedding_provider = get_embedding_provider()
    vector_store = get_vector_store()

    database_url = "sqlite+aiosqlite:///./tutor.db"
    engine = create_async_engine(database_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    existing_lines, done_ids = _load_done_ids(args.out)

    if not existing_lines:
        print(f"[{args.model}] stopping any loaded Ollama models...")
        _ollama_stop_all_loaded()

        print(f"[{args.model}] sending warm-up question (not recorded)...")
        async with session_factory() as db:
            try:
                await asyncio.wait_for(
                    handle_turn(
                        db,
                        provider,
                        embedding_provider,
                        vector_store,
                        session_id=None,
                        user_message=WARMUP_QUESTION,
                        subject=WARMUP_SUBJECT,
                        temperature=0.0,
                        mode=args.mode,
                    ),
                    timeout=PER_ITEM_TIMEOUT_S,
                )
                await db.commit()
            except Exception as exc:  # noqa: BLE001
                header = {
                    "type": "run_header",
                    "model": args.model,
                    "mode": args.mode,
                    "ollama_version": _ollama_version(),
                    "processor": None,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "fatal_error": f"warm-up call failed: {type(exc).__name__}: {exc}",
                }
                _append_jsonl(args.out, header)
                print(f"[{args.model}] FATAL during warm-up: {exc}")
                await engine.dispose()
                return

        processor = _processor_for(args.model)
        header = {
            "type": "run_header",
            "model": args.model,
            "mode": args.mode,
            "ollama_version": _ollama_version(),
            "processor": processor,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "fatal_error": None,
        }
        _append_jsonl(args.out, header)
        print(f"[{args.model}] warm-up done. processor={processor}")

    items = _load_eval_set()
    remaining = [item for item in items if item["id"] not in done_ids]
    print(f"[{args.model}] {len(done_ids)} items already done, {len(remaining)} remaining")

    for i, item in enumerate(remaining, start=1):
        async with session_factory() as db:
            record = await _run_one(db, provider, embedding_provider, vector_store, item, args.mode)
        _append_jsonl(args.out, record)
        status = "ERROR" if record["error"] else ("REFUSED" if record["guardrail_refused"] else "answered")
        print(f"[{args.model}] ({i}/{len(remaining)}) {item['id']}: {status} in {record['latency_ms']:.0f}ms")

    await engine.dispose()
    print(f"[{args.model}] done -> {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
