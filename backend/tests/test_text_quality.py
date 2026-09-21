from __future__ import annotations

from app.ingestion.pdf import assess_text_quality

# Matches the real Science-NCERT PDF corruption pattern found by
# diagnose_retrieval.py -- pymupdf reports a normal-looking word count via
# newline-separated tokens, but the tokens decode to private-use/dingbat
# code points, not real characters.
_GARBLED_SAMPLE = (
    "❙\n\x00✁\n✂\n✄\n\x00✂\n12\n✶✁✆✁\n✝\n"
    "✞✟✠✡⚛⚜\n✠\n✌✍\n⚛✍\n✡\n✎✏\n"
    "✡\n✑✒⚜✠\n✌✍\n❆\n❝\n✓\n❆\n❝\n"
)


def test_assess_text_quality_clean_english_scores_good() -> None:
    text = (
        "The mitochondria is the powerhouse of the cell. It converts nutrients "
        "into energy through a process called cellular respiration, which every "
        "living organism depends on to survive."
    )
    result = assess_text_quality(text)

    assert result.score > 0.8
    assert result.verdict == "good"


def test_assess_text_quality_clean_french_scores_good() -> None:
    text = (
        "J'aime beaucoup cette chanson. Céline Dion a chantée cette chanson célèbre. "
        "As-tu vu le film dont l'histoire m'a beaucoup plu?"
    )
    result = assess_text_quality(text)

    assert result.score > 0.8
    assert result.verdict == "good"


def test_assess_text_quality_math_formulas_mixed_in_scores_good() -> None:
    text = (
        "The kinetic energy of an object is given by the formula KE = (1/2)mv^2 "
        "where m is mass and v is velocity. This is a fundamental concept in "
        "physics used throughout the chapter."
    )
    result = assess_text_quality(text)

    assert result.score > 0.6
    assert result.verdict == "good"


def test_assess_text_quality_garbled_font_encoding_scores_poor() -> None:
    result = assess_text_quality(_GARBLED_SAMPLE)

    assert result.score < 0.5
    assert result.verdict == "poor"


def test_assess_text_quality_mixed_garbled_and_real_scores_poor() -> None:
    text = "The cat sat on the mat. " + _GARBLED_SAMPLE + " more garbage follows here as well"
    result = assess_text_quality(text)

    assert result.verdict == "poor"


def test_assess_text_quality_nearly_empty_is_empty() -> None:
    result = assess_text_quality("a b c")

    assert result.verdict == "empty"
    assert result.score == 0.0


def test_assess_text_quality_empty_string_is_empty() -> None:
    result = assess_text_quality("")

    assert result.verdict == "empty"
    assert result.score == 0.0
    assert result.word_count == 0
