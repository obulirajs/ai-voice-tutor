"""On-demand text-to-speech — lets the frontend request spoken audio for
any piece of text (a replay button on a chat message, a voice-picker
preview) outside the turn-based voice WebSocket's own pipeline. Also
exposes the curated list of selectable TTS voices.

Depends on app.voice's TTSProvider interface only, never a concrete
adapter — same dependency-inversion convention as voice.py.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel

from app.voice import SpokenFormPreprocessor, TTSProvider, get_tts_provider

router = APIRouter()

_spoken_form = SpokenFormPreprocessor()

# A small, curated subset of edge-tts's full voice catalog -- enough
# variety (accent, gender) for a student to find one they like without an
# overwhelming picker. Piper ignores `voice` (see tts_piper.py), so this
# list is only meaningful for the edge-tts provider.
AVAILABLE_VOICES = [
    {
        "id": "en-US-AriaNeural",
        "name": "Aria",
        "language": "English",
        "gender": "Female",
        "description": "Friendly and clear",
    },
    {
        "id": "en-US-GuyNeural",
        "name": "Guy",
        "language": "English",
        "gender": "Male",
        "description": "Calm and conversational",
    },
    {
        "id": "en-IN-NeerjaNeural",
        "name": "Neerja",
        "language": "English (India)",
        "gender": "Female",
        "description": "Indian English accent",
    },
    {
        "id": "en-IN-PrabhatNeural",
        "name": "Prabhat",
        "language": "English (India)",
        "gender": "Male",
        "description": "Indian English accent",
    },
    {
        "id": "fr-FR-DeniseNeural",
        "name": "Denise",
        "language": "French",
        "gender": "Female",
        "description": "Native French speaker",
    },
    {
        "id": "fr-FR-HenriNeural",
        "name": "Henri",
        "language": "French",
        "gender": "Male",
        "description": "Native French speaker",
    },
]


class TTSRequest(BaseModel):
    text: str
    voice: str | None = None


@router.post("/tts")
async def text_to_speech(
    request: TTSRequest,
    tts_provider: TTSProvider = Depends(get_tts_provider),  # noqa: B008
) -> Response:
    """Synthesize speech for `request.text`, run through the same
    spoken-form preprocessing the voice pipeline applies (strips citations/
    markdown, converts math notation to words) before handing it to TTS.
    """
    spoken_text = _spoken_form.preprocess(request.text)
    result = await tts_provider.synthesize(spoken_text, voice=request.voice)
    return Response(content=result.audio_data, media_type=result.content_type)


class VoiceOptionResponse(BaseModel):
    id: str
    name: str
    language: str
    gender: str
    description: str


@router.get("/voices", response_model=list[VoiceOptionResponse])
async def list_voices() -> list[VoiceOptionResponse]:
    return [VoiceOptionResponse(**voice) for voice in AVAILABLE_VOICES]
