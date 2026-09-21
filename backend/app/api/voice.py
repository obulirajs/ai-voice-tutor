"""Voice WebSocket endpoint — turn-based (not streaming) for v1.

Protocol:
  1. Client connects, sends a text frame {"type": "init", "session_id"?,
     "subject"?, "voice"?} -- same session_id/subject semantics as POST
     /api/chat; voice is an optional TTS voice id (see api/tts.py's
     AVAILABLE_VOICES) used for every turn on this connection, defaulting
     to the TTS provider's own configured voice when omitted.
  2. Server resolves/creates the session and replies {"type": "ready",
     "session_id": ...}.
  3. Per turn: client sends {"type": "audio_start"}, one or more binary
     frames of audio, then {"type": "audio_end"}. Server runs the voice
     pipeline (ASR -> command check or handle_turn -> spoken-form ->
     TTS) and replies with a {"type": "response", "transcription",
     "reply", "sources", "session_id", "is_command", "command_type",
     "new_subject", "visual_directives"} JSON frame (new_subject is set
     only when a switch_subject command actually resolved; visual_directives
     is always [] for a command turn), a binary frame of synthesized
     audio, then {"type": "turn_complete"}.
  4. Any step failing sends {"type": "error", "detail": ...} -- the
     connection stays open for the next turn (an init failure is the one
     exception: there's no session to keep the connection open for).

Depends on app.models', app.knowledge's, app.voice's, and app.orchestration's
interfaces only, never a concrete adapter — same dependency-inversion
convention as chat.py.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

from fastapi import APIRouter, Depends, WebSocket
from pydantic import BaseModel, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.knowledge import VectorStore, get_vector_store
from app.models import (
    EmbeddingProvider,
    ModelProvider,
    get_embedding_provider,
    get_provider,
)
from app.orchestration import DEFAULT_SUBJECT, handle_voice_turn
from app.storage import (
    create_session,
    get_db,
    get_or_create_subject,
    get_session,
    log_usage_event,
)
from app.voice import (
    ASRProvider,
    SpokenFormPreprocessor,
    TTSProvider,
    get_asr_provider,
    get_tts_provider,
)

router = APIRouter()

_spoken_form = SpokenFormPreprocessor()


@dataclass(frozen=True)
class _InitResult:
    session_id: str
    # The student's selected TTS voice for this connection (see Settings'
    # voice picker), or None to use the TTS provider's own default.
    voice: str | None


class _InitMessage(BaseModel):
    type: str
    session_id: str | None = None
    subject: str | None = None
    voice: str | None = None


@router.websocket("/voice")
async def voice_websocket(
    websocket: WebSocket,
    provider: ModelProvider = Depends(get_provider),  # noqa: B008
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),  # noqa: B008
    vector_store: VectorStore = Depends(get_vector_store),  # noqa: B008
    asr_provider: ASRProvider = Depends(get_asr_provider),  # noqa: B008
    tts_provider: TTSProvider = Depends(get_tts_provider),  # noqa: B008
    db: AsyncSession = Depends(get_db),  # noqa: B008
) -> None:
    await websocket.accept()

    init_result = await _handle_init(websocket, db)
    if init_result is None:
        await websocket.close()
        return
    session_id = init_result.session_id
    voice = init_result.voice

    audio_chunks: list[bytes] = []
    receiving_audio = False

    while True:
        message = await websocket.receive()
        if message.get("type") == "websocket.disconnect":
            return

        text = message.get("text")
        data = message.get("bytes")

        if text is not None:
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "detail": "Malformed JSON frame."})
                continue

            frame_type = payload.get("type")
            if frame_type == "audio_start":
                audio_chunks = []
                receiving_audio = True
            elif frame_type == "audio_end":
                receiving_audio = False
                audio_data = b"".join(audio_chunks)
                audio_chunks = []
                try:
                    session_id = await _run_turn(
                        websocket,
                        db,
                        provider,
                        embedding_provider,
                        vector_store,
                        asr_provider,
                        tts_provider,
                        audio_data=audio_data,
                        session_id=session_id,
                        voice=voice,
                    )
                except Exception as exc:  # noqa: BLE001 -- deliberately broad: report and keep the socket open
                    await websocket.send_json({"type": "error", "detail": f"Turn failed: {exc}"})
            else:
                await websocket.send_json({"type": "error", "detail": f"Unexpected message type: {frame_type!r}"})
        elif data is not None:
            if not receiving_audio:
                await websocket.send_json({"type": "error", "detail": "Received audio bytes before audio_start."})
                continue
            audio_chunks.append(data)


async def _handle_init(websocket: WebSocket, db: AsyncSession) -> _InitResult | None:
    """Parses the init frame and resolves/creates its session. Returns the
    session_id and selected voice, or None (having already sent an
    {"type": "error"} frame) if the init message was malformed or named an
    unknown session_id.
    """
    message = await websocket.receive()
    if message.get("type") == "websocket.disconnect":
        return None

    text = message.get("text")
    try:
        if text is None:
            raise ValueError("Expected a text frame for the init message.")
        payload = json.loads(text)
        if payload.get("type") != "init":
            raise ValueError(f"Expected type='init', got {payload.get('type')!r}")
        init_message = _InitMessage.model_validate(payload)
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        await websocket.send_json({"type": "error", "detail": f"Invalid init message: {exc}"})
        return None

    try:
        if init_message.session_id is not None:
            session = await get_session(db, init_message.session_id)
            if session is None:
                raise LookupError(f"Unknown session_id: {init_message.session_id!r}")
        else:
            subject_row = await get_or_create_subject(db, init_message.subject or DEFAULT_SUBJECT)
            session = await create_session(db, subject_row.id)
        await db.commit()
    except LookupError as exc:
        await websocket.send_json({"type": "error", "detail": str(exc)})
        return None
    except Exception as exc:  # noqa: BLE001 -- report to the client instead of letting the socket die raw
        await websocket.send_json({"type": "error", "detail": f"Failed to start voice session: {exc}"})
        return None

    await websocket.send_json({"type": "ready", "session_id": session.id})
    return _InitResult(session_id=session.id, voice=init_message.voice)


async def _run_turn(
    websocket: WebSocket,
    db: AsyncSession,
    provider: ModelProvider,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    asr_provider: ASRProvider,
    tts_provider: TTSProvider,
    *,
    audio_data: bytes,
    session_id: str,
    voice: str | None = None,
) -> str:
    """Runs one full voice turn (ASR -> command/handle_turn -> spoken-form ->
    TTS). Returns the session_id to use for the next turn (unchanged unless
    this turn was a switch_subject command). Each pipeline step reports its
    own failure via an {"type": "error"} frame and returns early, leaving
    the connection open for the next turn.
    """
    try:
        if not audio_data:
            await websocket.send_json({"type": "error", "detail": "No audio received for this turn."})
            return session_id

        try:
            started = time.perf_counter()
            transcription = await asr_provider.transcribe(audio_data)
            asr_latency_ms = (time.perf_counter() - started) * 1000
        except Exception as exc:  # noqa: BLE001 -- report to the client and keep the socket open
            await websocket.send_json({"type": "error", "detail": f"Speech recognition failed: {exc}"})
            return session_id

        await _log_voice_usage(
            db,
            session_id=session_id,
            provider_name=asr_provider.provider_name,
            model_name=asr_provider.model_name,
            latency_ms=asr_latency_ms,
            event_type="asr",
        )

        try:
            turn_result = await handle_voice_turn(
                db,
                provider,
                embedding_provider,
                vector_store,
                session_id=session_id,
                transcribed_text=transcription.text,
            )
        except LookupError as exc:
            await websocket.send_json({"type": "error", "detail": str(exc)})
            return session_id
        except Exception as exc:  # noqa: BLE001 -- report to the client and keep the socket open
            await websocket.send_json({"type": "error", "detail": f"Failed to process turn: {exc}"})
            return session_id

        spoken_text = _spoken_form.preprocess(turn_result.reply)

        try:
            started = time.perf_counter()
            synthesis = await tts_provider.synthesize(spoken_text, voice=voice)
            tts_latency_ms = (time.perf_counter() - started) * 1000
        except Exception as exc:  # noqa: BLE001 -- report to the client and keep the socket open
            await websocket.send_json({"type": "error", "detail": f"Speech synthesis failed: {exc}"})
            return turn_result.session_id

        await _log_voice_usage(
            db,
            session_id=turn_result.session_id,
            provider_name=tts_provider.provider_name,
            model_name=tts_provider.voice_name,
            latency_ms=tts_latency_ms,
            event_type="tts",
        )

        await websocket.send_json(
            {
                "type": "response",
                "transcription": transcription.text,
                "reply": turn_result.reply,
                "sources": [
                    {"page_number": s.page_number, "score": s.score} for s in turn_result.sources
                ],
                "session_id": turn_result.session_id,
                "is_command": turn_result.is_command,
                "command_type": turn_result.command_type,
                # Set only when a switch_subject command actually resolved --
                # the client uses this to update its active subject.
                "new_subject": turn_result.new_subject,
                "visual_directives": [
                    {"directive_type": d.directive_type, "content": d.content, "label": d.label}
                    for d in turn_result.visual_directives
                ],
            }
        )
        await websocket.send_bytes(synthesis.audio_data)
        await websocket.send_json({"type": "turn_complete"})

        return turn_result.session_id
    finally:
        # Each turn is its own durability boundary -- unlike an HTTP request,
        # a WebSocket connection can stay open for many turns, so relying on
        # get_db()'s single end-of-connection commit would risk losing every
        # turn's history/usage_events if the process dies mid-conversation.
        await db.commit()


async def _log_voice_usage(
    db: AsyncSession,
    *,
    session_id: str,
    provider_name: str,
    model_name: str,
    latency_ms: float,
    event_type: str,
) -> None:
    session = await get_session(db, session_id)

    await log_usage_event(
        db,
        session_id=session_id,
        subject_id=session.subject_id if session is not None else None,
        provider=provider_name,
        model=model_name,
        input_tokens=None,
        output_tokens=None,
        cost_usd=0.0,
        latency_ms=latency_ms,
        event_type=event_type,
    )
