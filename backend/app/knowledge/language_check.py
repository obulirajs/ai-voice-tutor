from __future__ import annotations

import random

# Character-range heuristic, not a real language detector or LLM call: cheap
# enough to run on every ingestion as a sanity check, not a grounding
# guardrail. Devanagari has no overlap with Latin script, so its presence is
# an unambiguous Hindi signal; French-accented Latin letters are likewise
# rare-to-absent in ordinary English text. Plain ASCII letters with neither
# signal present are treated as English.
_FRENCH_ACCENT_CHARS = set("àâäéèêëîïôöùûüÿçœæÀÂÄÉÈÊËÎÏÔÖÙÛÜŸÇŒÆ")
_DEVANAGARI_RANGE = ("ऀ", "ॿ")
_MIN_SIGNAL_CHARS = 3
_SAMPLE_SIZE = 5

# Which language each subject's content is expected to be in. Subjects not
# listed here (nothing beyond French exists yet -- see development-plan.md's
# "Phase 2 of the vision") simply skip the check rather than guess.
_SUBJECT_EXPECTED_LANGUAGE: dict[str, str] = {
    "french": "french",
}

_LANGUAGE_DISPLAY_NAMES: dict[str, str] = {
    "french": "French",
    "hindi": "Hindi",
    "english": "English",
}


def detect_chunk_language(text: str) -> str | None:
    """Character-range guess at one chunk's dominant language.

    Returns None when the chunk has no strong signal either way (too short,
    mostly numbers/punctuation/whitespace) rather than force a guess.
    """
    low, high = _DEVANAGARI_RANGE
    devanagari = sum(1 for ch in text if low <= ch <= high)
    if devanagari >= _MIN_SIGNAL_CHARS:
        return "hindi"

    french_accents = sum(1 for ch in text if ch in _FRENCH_ACCENT_CHARS)
    if french_accents >= _MIN_SIGNAL_CHARS:
        return "french"

    ascii_letters = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    if ascii_letters >= _MIN_SIGNAL_CHARS:
        return "english"

    return None


def check_subject_language_mismatch(subject: str, chunk_texts: list[str]) -> str | None:
    """Samples up to 5 random chunks from a just-ingested document and warns
    (never blocks) when their dominant language doesn't match what the
    subject implies -- e.g. predominantly English or Hindi content uploaded
    under the "french" subject.

    Returns None whenever there's nothing confident/actionable to report:
    an unrecognized subject, no chunks, no chunk with a strong-enough
    language signal, or a dominant language that isn't a clear majority of
    the sample (a genuinely mixed-language sample -- common in a
    language-teaching text -- shouldn't trigger a false alarm).
    """
    expected = _SUBJECT_EXPECTED_LANGUAGE.get(subject.strip().lower())
    if expected is None or not chunk_texts:
        return None

    sample = random.sample(chunk_texts, k=min(_SAMPLE_SIZE, len(chunk_texts)))
    detected = [lang for text in sample if (lang := detect_chunk_language(text)) is not None]
    if not detected:
        return None

    dominant = max(set(detected), key=detected.count)
    if dominant == expected or detected.count(dominant) <= len(detected) / 2:
        return None

    dominant_name = _LANGUAGE_DISPLAY_NAMES.get(dominant, dominant.title())
    expected_name = _LANGUAGE_DISPLAY_NAMES.get(expected, expected.title())
    return (
        f"Uploaded content appears to be predominantly {dominant_name} — expected {expected_name}. "
        "Please verify this is the correct subject."
    )
