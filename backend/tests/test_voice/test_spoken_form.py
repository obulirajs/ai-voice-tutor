from __future__ import annotations

from app.voice import SpokenFormPreprocessor

preprocessor = SpokenFormPreprocessor()


def test_exponent_squared() -> None:
    assert preprocessor.preprocess("x^2") == "x squared"


def test_exponent_cubed() -> None:
    assert preprocessor.preprocess("x^3") == "x cubed"


def test_exponent_general_power() -> None:
    assert preprocessor.preprocess("x^n") == "x to the power of n"


def test_exponent_multi_digit() -> None:
    assert preprocessor.preprocess("2^10") == "2 to the power of 10"


def test_square_root() -> None:
    assert preprocessor.preprocess("√4") == "square root of 4"


def test_square_root_with_parens() -> None:
    assert preprocessor.preprocess("√(16)") == "square root of 16"


def test_square_root_decimal() -> None:
    assert preprocessor.preprocess("√2.5") == "square root of 2.5"


def test_chemical_formula_water() -> None:
    assert preprocessor.preprocess("H₂O") == "H 2 O"


def test_chemical_formula_carbon_dioxide() -> None:
    assert preprocessor.preprocess("CO₂") == "C O 2"


def test_chemical_formula_ascii_digits() -> None:
    assert preprocessor.preprocess("H2O") == "H 2 O"


def test_chemical_formula_iron_oxide() -> None:
    assert preprocessor.preprocess("Fe2O3") == "Fe 2 O 3"


def test_common_fraction_one_half() -> None:
    assert preprocessor.preprocess("1/2") == "one half"


def test_common_fraction_three_quarters() -> None:
    assert preprocessor.preprocess("3/4") == "three quarters"


def test_common_fraction_two_thirds() -> None:
    assert preprocessor.preprocess("2/3") == "two thirds"


def test_uncommon_fraction_falls_back_to_over() -> None:
    assert preprocessor.preprocess("3/7") == "3 over 7"


def test_fraction_does_not_match_dates_or_larger_numbers() -> None:
    # Three-digit numbers either side shouldn't be treated as a fraction.
    assert preprocessor.preprocess("100/200") == "100/200"


def test_plus_operator() -> None:
    assert preprocessor.preprocess("2 + 2") == "2 plus 2"


def test_equals_operator() -> None:
    assert preprocessor.preprocess("2 + 2 = 4") == "2 plus 2 equals 4"


def test_exit_criteria_example() -> None:
    assert preprocessor.preprocess("x^2 + √4 = 8") == "x squared plus square root of 4 equals 8"


def test_markdown_bold_stripped() -> None:
    assert preprocessor.preprocess("This is **important** text.") == "This is important text."


def test_markdown_italic_stripped() -> None:
    assert preprocessor.preprocess("This is *emphasized* text.") == "This is emphasized text."


def test_markdown_underscore_italic_stripped() -> None:
    assert preprocessor.preprocess("This is _emphasized_ text.") == "This is emphasized text."


def test_markdown_heading_stripped() -> None:
    assert preprocessor.preprocess("# Chapter One\nSome text.") == "Chapter One\nSome text."


def test_markdown_bullets_stripped() -> None:
    result = preprocessor.preprocess("Topics:\n- avoir\n- etre\n* faire")
    assert result == "Topics:\navoir\netre\nfaire"


def test_markdown_link_keeps_text_drops_url() -> None:
    result = preprocessor.preprocess("See [the textbook](https://example.com/book.pdf) for more.")
    assert result == "See the textbook for more."


def test_markdown_inline_code_stripped() -> None:
    assert preprocessor.preprocess("Use the `avoir` verb.") == "Use the avoir verb."


def test_bracket_citation_stripped() -> None:
    result = preprocessor.preprocess("Le passe compose est utilise. [Passage 1 — page 12]")
    assert result == "Le passe compose est utilise."


def test_paren_page_citation_stripped() -> None:
    result = preprocessor.preprocess("Le passe compose est utilise (page 12).")
    assert result == "Le passe compose est utilise."


def test_paren_p_dot_citation_stripped() -> None:
    result = preprocessor.preprocess("Some fact (p. 45) follows.")
    assert result == "Some fact follows."


def test_combined_markdown_and_math() -> None:
    result = preprocessor.preprocess("**Formula:** x^2 is read as x squared, and 1/2 is a half.")
    assert result == "Formula: x squared is read as x squared, and one half is a half."


def test_plain_text_unaffected() -> None:
    text = "Bonjour, comment allez-vous aujourd'hui?"
    assert preprocessor.preprocess(text) == text


def test_whitespace_cleanup_collapses_double_spaces() -> None:
    assert preprocessor.preprocess("Hello    world") == "Hello world"

    assert preprocessor.preprocess("Hello , world .") == "Hello, world."


# --- blank-line collapsing (TTS engines truncate at a blank-line boundary) ---


def test_blank_lines_collapsed() -> None:
    """TTS engines choke on blank lines — they must be collapsed to single newlines."""
    text = "First paragraph about forces.\n\n\nSecond paragraph about motion.\n\n\n\nThird paragraph."
    result = preprocessor.preprocess(text)
    assert "\n\n" not in result
    assert "First paragraph about forces.\nSecond paragraph about motion.\nThird paragraph." == result


def test_markdown_heading_blank_lines() -> None:
    """After stripping '## Heading\\n\\n', the blank line must not survive."""
    text = "## Forces\n\nForce equals mass times acceleration.\n\n## Energy\n\nEnergy is the ability to do work."
    result = preprocessor.preprocess(text)
    assert "\n\n" not in result
    assert "Forces" in result
    assert "Energy" in result


# --- LaTeX conversion (the LLM writes formulas in LaTeX per the system prompt) ---


def test_inline_latex_delimiters_stripped() -> None:
    """$...$ delimiters must be removed, not read as 'dollar'."""
    result = preprocessor.preprocess("$F = ma$")
    assert "f equals ma" in result.lower()
    assert "$" not in result


def test_display_latex_delimiters_stripped() -> None:
    """$$...$$ display math delimiters must be removed."""
    result = preprocessor.preprocess("$$E = mc^2$$")
    assert "$$" not in result
    assert "$" not in result


def test_latex_frac() -> None:
    """\\frac{1}{2} → 'one half', \\frac{a}{b} → 'a over b'."""
    assert "one half" in preprocessor.preprocess(r"$\frac{1}{2}$")
    assert "a over b" in preprocessor.preprocess(r"$\frac{a}{b}$")


def test_latex_sqrt() -> None:
    """\\sqrt{16} → 'square root of 16'."""
    assert "square root of 16" in preprocessor.preprocess(r"$\sqrt{16}$")


def test_latex_times() -> None:
    """\\times → 'times'."""
    result = preprocessor.preprocess(r"$F = m \times a$")
    assert "times" in result
    assert r"\times" not in result


def test_latex_greek_letters() -> None:
    """Greek letter commands → spoken names."""
    assert "pi" in preprocessor.preprocess(r"$\pi$")
    assert "theta" in preprocessor.preprocess(r"$\theta$")
    assert "alpha" in preprocessor.preprocess(r"$\alpha$")


def test_latex_power() -> None:
    """x^{2} → 'x squared'."""
    result = preprocessor.preprocess(r"$x^{2}$")
    assert "squared" in result


def test_no_leftover_backslashes() -> None:
    """No raw LaTeX commands should survive preprocessing."""
    result = preprocessor.preprocess(r"$\frac{1}{2} \times \pi \approx 1.57$")
    assert "\\" not in result


def test_no_leftover_braces() -> None:
    """No LaTeX grouping braces should survive preprocessing."""
    result = preprocessor.preprocess(r"$\frac{a+b}{c}$")
    assert "{" not in result
    assert "}" not in result


def test_complex_formula() -> None:
    """A realistic CBSE science formula converts to something speakable."""
    result = preprocessor.preprocess(r"The kinetic energy is $KE = \frac{1}{2}mv^{2}$")
    assert "$" not in result
    assert "\\" not in result
    assert "one half" in result or "1 over 2" in result
    assert "squared" in result
