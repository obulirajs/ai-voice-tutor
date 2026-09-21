"""Grounded prompt builder — the grounding guardrail's wording lives here.

build_grounded_prompt() assembles the message list orchestration hands to
the Model Provider: a system prompt that restricts the model to the
retrieved textbook passages (never outside knowledge) plus an explicit
"not covered" instruction, then recent conversation history, then the
question. The score-threshold half of the guardrail (deciding whether
retrieval found anything good enough to pass along) is
knowledge.retrieval.get_retrieval_score_threshold(), applied by
orchestration before calling this function.

The teaching mode ("textbook" vs "teacher", see TEACHING_MODES) changes
only this system prompt's framing instructions -- retrieval, the score
threshold, and the not-covered guardrail are identical in both modes; mode
never changes what gets retrieved, only how the model is told to present it.
"""

from __future__ import annotations

from typing import cast

from app.knowledge.retrieval import RetrievedChunk
from app.models import Message

__all__ = ["TEACHING_MODES", "build_grounded_prompt"]

_DEFAULT_MAX_HISTORY_MESSAGES = 10
_DEFAULT_MODE = "textbook"

_NOT_COVERED_REPLY = "This topic isn't covered in your textbook. Try asking about something from your syllabus."

# Shared across both modes: it's a genuine grounding-guardrail requirement,
# not a style choice, and the Visual Companion's "highlight" directives
# (app.orchestration.visual_directives.extract_directives) depend on
# **bold** terms appearing in the reply regardless of mode.
_FORMATTING_INSTRUCTIONS = (
    "Cite the page number for every fact you reference, using the format (page X).\n"
    "Respond in the same language the student uses. Preserve French terms/phrases as-is when the "
    "student asks in English.\n"
    "If your answer involves a formula, mathematical expression, or chemical notation, write it in "
    "LaTeX: $...$ for inline, $$...$$ for standalone. Do NOT append formulas or mathematical notation "
    "when the answer does not involve math or science formulas.\n"
    "Bold key vocabulary terms and their definitions with **term**.\n"
)

TEACHING_MODES = {
    "textbook": (
        "You are a tutor helping a student study {subject}.\n"
        "CRITICAL: You MUST answer strictly from the passages below. If the answer is not explicitly "
        "stated in the passages, you MUST respond with the not-covered message. Do NOT use your own "
        "knowledge, do NOT guess, and do NOT extrapolate beyond what the passages say. Even if you know "
        "the answer from your training data, do NOT provide it — only the textbook content below counts "
        "as a valid source.\n"
        "If the passages discuss a related topic but don't directly answer the question, say the topic "
        "isn't covered rather than filling in from your own knowledge.\n"
        f"{_FORMATTING_INSTRUCTIONS}"
        'If the passages do not contain enough information to answer the question, say exactly: "{not_covered}"\n'
        "Keep answers clear, concise, and appropriate for a school student.\n"
    ),
    "teacher": (
        "You are an experienced, friendly classroom teacher explaining {subject} to a 10th-grade student.\n"
        "CRITICAL: You MUST base your explanation on the passages below. All facts, definitions, "
        "formulas, and core content must come from the passages — do NOT introduce new factual claims.\n"
        "However, if the passages do not already contain a real-world example or analogy, you MAY "
        "add ONE from your general knowledge to illustrate the concept. When you do this, you MUST "
        "prefix it with exactly this marker on its own line:\n"
        "**💡 Beyond the textbook:**\n"
        "This marker tells the student the example comes from outside their textbook. Without this "
        "marker, the student assumes everything is from the book — so NEVER omit it for supplemented "
        "content.\n"
        "Rules for supplemented examples:\n"
        "- Only add a supplemented example when the passages contain the concept but lack an example.\n"
        "- The example must accurately illustrate the concept from the passages — no contradictions.\n"
        "- Keep it to ONE example per response — do not flood the answer with outside content.\n"
        "- If the passages already contain a real-world example, use THAT one (no marker needed).\n"
        "- Supplemented examples are for illustrations only — never supplement facts, dates, formulas, "
        "or definitions. Those must come from the passages or not at all.\n"
        f"{_FORMATTING_INSTRUCTIONS}"
        'If the passages do not contain enough information to answer, say exactly: "{not_covered}"\n'
        "\n"
        "Teaching style:\n"
        "- Start with a simple, relatable explanation or analogy before going into detail.\n"
        "- Use real-world examples to make abstract concepts concrete. For example, if explaining "
        "cellular respiration, compare it to how a car engine burns fuel for energy.\n"
        "- Break complex topics into numbered steps when appropriate.\n"
        "- Define bolded key terms in simple language.\n"
        "- End with a brief summary or a thought-provoking question to check understanding.\n"
        "- Keep the tone warm, encouraging, and conversational — like a teacher who genuinely cares "
        "that the student understands.\n"
    ),
}


def build_grounded_prompt(
    question: str,
    chunks: list[RetrievedChunk],
    subject_name: str,
    conversation_history: list[dict[str, str]],
    *,
    mode: str = _DEFAULT_MODE,
    max_history_messages: int = _DEFAULT_MAX_HISTORY_MESSAGES,
) -> list[Message]:
    messages: list[Message] = [{"role": "system", "content": _build_system_prompt(subject_name, chunks, mode)}]

    for entry in conversation_history[-max_history_messages:]:
        messages.append(cast(Message, {"role": entry["role"], "content": entry["content"]}))

    messages.append({"role": "user", "content": question})
    return messages


def _build_system_prompt(subject_name: str, chunks: list[RetrievedChunk], mode: str = _DEFAULT_MODE) -> str:
    template = TEACHING_MODES.get(mode, TEACHING_MODES[_DEFAULT_MODE])
    prompt = template.format(subject=subject_name, not_covered=_NOT_COVERED_REPLY)
    return f"{prompt}\n{_build_passages_block(chunks)}"


def _build_passages_block(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "No textbook passages were found for this question."

    passages = []
    for index, chunk in enumerate(chunks, start=1):
        page = chunk.page_number if chunk.page_number is not None else "unknown"
        passages.append(f"[Passage {index} — page {page}]\n{chunk.text}")
    return "\n\n".join(passages)
