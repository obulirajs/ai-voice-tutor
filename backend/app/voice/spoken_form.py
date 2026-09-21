from __future__ import annotations

import re

__all__ = ["SpokenFormPreprocessor"]

# Common fractions read as words; anything else falls back to "N over D"
# (architecture.md: "x^2" -> "x squared" is the same idea applied to fractions).
_COMMON_FRACTIONS = {
    (1, 2): "one half",
    (1, 3): "one third",
    (2, 3): "two thirds",
    (1, 4): "one quarter",
    (3, 4): "three quarters",
    (1, 5): "one fifth",
    (2, 5): "two fifths",
    (3, 5): "three fifths",
    (4, 5): "four fifths",
}

_SUBSCRIPT_DIGITS = "₀₁₂₃₄₅₆₇₈₉"
_SUPERSCRIPT_DIGITS = "⁰¹²³⁴⁵⁶⁷⁸⁹"
_DIGIT_TRANSLATION = str.maketrans(_SUBSCRIPT_DIGITS + _SUPERSCRIPT_DIGITS, "0123456789" * 2)

# A chemical-formula-like token: two or more repeats of an element symbol
# (capital letter + optional lowercase letter) each optionally followed by a
# digit count, e.g. "H2O", "CO2", "Fe2O3".
_FORMULA_TOKEN = re.compile(r"\b(?:[A-Z][a-z]?\d*){2,}\b")
_FORMULA_PART = re.compile(r"[A-Z][a-z]?|\d+")

_EXPONENT = re.compile(r"\b(\w+)\^(\w+)\b")
_SQUARE_ROOT = re.compile(r"√\(?\s*(\d+(?:\.\d+)?)\s*\)?")
_FRACTION = re.compile(r"(?<!\d)(\d{1,2})/(\d{1,2})(?!\d)")

# LaTeX delimiters: $$...$$ (display math) and $...$ (inline math). The
# system prompt (prompt_builder.py) instructs the model to write formulas in
# LaTeX, so a reply's math needs converting to words before it ever reaches
# the plain-text converters below -- those expect "x^2", not "$x^2$" or
# "\frac{1}{2}".
_DISPLAY_LATEX = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
_INLINE_LATEX = re.compile(r"\$(.+?)\$")

# LaTeX commands with their own spoken-form conversion (structured, not a
# simple word swap).
_LATEX_FRAC = re.compile(r"\\frac\s*\{([^}]*)\}\s*\{([^}]*)\}")
_LATEX_SQRT = re.compile(r"\\sqrt\s*\{([^}]*)\}")
_LATEX_POWER = re.compile(r"\{([^}]*)\}\s*\^\s*\{([^}]*)\}")
_LATEX_SIMPLE_POWER = re.compile(r"(\w)\s*\^\s*\{([^}]*)\}")
_LATEX_SIMPLE_POWER2 = re.compile(r"(\w)\s*\^\s*(\w)")
_LATEX_SUBSCRIPT = re.compile(r"(\w)\s*_\s*\{([^}]*)\}")
_LATEX_SUBSCRIPT2 = re.compile(r"(\w)\s*_\s*(\w)")

# LaTeX commands that map to a fixed spoken word/phrase -- applied as plain
# substring replacement after the structured conversions above.
_LATEX_COMMANDS: dict[str, str] = {
    r"\times": " times ",
    r"\cdot": " times ",
    r"\div": " divided by ",
    r"\pm": " plus or minus ",
    r"\mp": " minus or plus ",
    r"\leq": " is less than or equal to ",
    r"\geq": " is greater than or equal to ",
    r"\neq": " is not equal to ",
    r"\approx": " is approximately equal to ",
    r"\infty": " infinity ",
    r"\pi": " pi ",
    r"\theta": " theta ",
    r"\alpha": " alpha ",
    r"\beta": " beta ",
    r"\gamma": " gamma ",
    r"\delta": " delta ",
    r"\lambda": " lambda ",
    r"\mu": " mu ",
    r"\sigma": " sigma ",
    r"\omega": " omega ",
    r"\sum": " the sum of ",
    r"\int": " the integral of ",
    r"\partial": " partial ",
    r"\nabla": " del ",
    r"\rightarrow": " goes to ",
    r"\leftarrow": " from ",
    r"\Rightarrow": " implies ",
    r"\left": "",
    r"\right": "",
    r"\text": "",
    r"\mathrm": "",
    r"\mathbf": "",
    r"\mathit": "",
}

# LLM replies follow the grounded-prompt's citation instruction ("cite the
# page number ... using the format (page X)") or the passages block's own
# "[Passage N — page P]" header -- both are visual, not spoken.
_BRACKET_CITATION = re.compile(r"\[\s*Passage\s+\d+\s*[-—]\s*page\s+\d+\s*\]", re.IGNORECASE)
_PAREN_CITATION = re.compile(r"\(\s*p(?:age)?s?\.?\s*\d+(?:\s*[-–]\s*\d+)?\s*\)", re.IGNORECASE)

_HEADING = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_BULLET = re.compile(r"^[ \t]*[-*+]\s+", re.MULTILINE)
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC_STAR = re.compile(r"\*(.+?)\*")
_ITALIC_UNDERSCORE = re.compile(r"_(.+?)_")
_INLINE_CODE = re.compile(r"`(.+?)`")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")

_PLUS = re.compile(r"\s*\+\s*")
_EQUALS = re.compile(r"\s*=\s*")

_EXTRA_SPACE = re.compile(r"[ \t]+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,!?])")
# Two or more newlines (with optional whitespace between them) -- a
# paragraph break. TTS engines (edge-tts included) treat a blank line as a
# segment boundary and can truncate synthesis there, so these collapse to a
# single newline (still a natural pause) rather than surviving as-is.
_CONSECUTIVE_NEWLINES = re.compile(r"\n\s*\n")
_LEADING_TRAILING_WS = re.compile(r"^[ \t]+|[ \t]+$", re.MULTILINE)


class SpokenFormPreprocessor:
    """Text-to-text transform applied to a reply before TTS.

    Converts notation that would sound wrong read literally (LaTeX math
    delimiters/commands, plain-text exponents, chemical subscripts, square
    roots, fractions), strips formatting that's visual only (markdown,
    source citations), and collapses blank lines (a paragraph break TTS
    engines can otherwise treat as a truncation point). Doesn't aim for
    perfect coverage -- unusual notation is just read literally, which is
    an acceptable v1 gap (architecture.md's spoken-form preprocessor note).
    """

    def preprocess(self, text: str) -> str:
        text = self._convert_latex(text)
        text = self._strip_citations(text)
        text = self._strip_markdown(text)
        text = self._convert_chemical_formulas(text)
        text = self._convert_square_roots(text)
        text = self._convert_exponents(text)
        text = self._convert_fractions(text)
        text = self._convert_operators(text)
        return self._cleanup_whitespace(text)

    def _convert_latex(self, text: str) -> str:
        """Unwrap $...$/$$...$$ math delimiters and convert the LaTeX inside
        them to spoken words.

        Runs first, before any other transform: LaTeX has its own syntax for
        exponents/fractions ("\\frac{1}{2}", "x^{2}") that must become words
        here rather than falling through to the plain-text converters below,
        which expect bare "1/2" or "x^2" with no braces or delimiters.

        The command/power/subscript conversions below apply only to the text
        captured *inside* a math delimiter, not the reply as a whole --
        running them globally would also rewrite unrelated prose (markdown's
        underscore italics, e.g. "_emphasized_", look exactly like a LaTeX
        subscript to a bare regex).
        """
        # Display math before inline -- the inline pattern would otherwise
        # match one half of a "$$...$$" pair as its own "$...$" expression.
        text = _DISPLAY_LATEX.sub(lambda m: self._convert_latex_expression(m.group(1)), text)
        text = _INLINE_LATEX.sub(lambda m: self._convert_latex_expression(m.group(1)), text)
        return text

    def _convert_latex_expression(self, expr: str) -> str:
        """Converts the content of a single $...$/$$...$$ expression -- LaTeX
        commands and structure only, never touching text outside it.
        """

        def _frac_to_spoken(match: re.Match[str]) -> str:
            numerator, denominator = match.group(1).strip(), match.group(2).strip()
            try:
                spoken = _COMMON_FRACTIONS.get((int(numerator), int(denominator)))
                if spoken is not None:
                    return spoken
            except ValueError:
                pass
            return f"{numerator} over {denominator}"

        expr = _LATEX_FRAC.sub(_frac_to_spoken, expr)
        expr = _LATEX_SQRT.sub(lambda m: f"square root of {m.group(1).strip()}", expr)

        def _power_to_spoken(base: str, exponent: str) -> str:
            exponent = exponent.strip()
            if exponent == "2":
                return f"{base} squared"
            if exponent == "3":
                return f"{base} cubed"
            return f"{base} to the power of {exponent}"

        expr = _LATEX_POWER.sub(lambda m: _power_to_spoken(m.group(1).strip(), m.group(2)), expr)
        expr = _LATEX_SIMPLE_POWER.sub(lambda m: _power_to_spoken(m.group(1), m.group(2)), expr)
        expr = _LATEX_SIMPLE_POWER2.sub(lambda m: _power_to_spoken(m.group(1), m.group(2)), expr)

        expr = _LATEX_SUBSCRIPT.sub(lambda m: f"{m.group(1)} sub {m.group(2).strip()}", expr)
        expr = _LATEX_SUBSCRIPT2.sub(lambda m: f"{m.group(1)} sub {m.group(2)}", expr)

        for command, spoken in _LATEX_COMMANDS.items():
            expr = expr.replace(command, spoken)

        # Leftover LaTeX grouping braces the conversions above didn't consume
        # (a literal escaped "\{"/"\}" is left alone).
        expr = re.sub(r"(?<!\\)[{}]", " ", expr)
        # Spacing commands with no useful spoken form.
        expr = re.sub(r"\\[,;:!]|\\quad|\\qquad|\\hspace\{[^}]*\}|\\vspace\{[^}]*\}", " ", expr)
        # Any remaining \commandname we don't know how to speak -- better to
        # drop it than read "backslash foo" aloud.
        expr = re.sub(r"\\[a-zA-Z]+", " ", expr)

        return expr

    def _strip_citations(self, text: str) -> str:
        text = _BRACKET_CITATION.sub("", text)
        text = _PAREN_CITATION.sub("", text)
        return text

    def _strip_markdown(self, text: str) -> str:
        text = _HEADING.sub("", text)
        text = _BULLET.sub("", text)
        text = _LINK.sub(r"\1", text)
        text = _BOLD.sub(r"\1", text)
        text = _ITALIC_STAR.sub(r"\1", text)
        text = _ITALIC_UNDERSCORE.sub(r"\1", text)
        text = _INLINE_CODE.sub(r"\1", text)
        return text

    def _convert_chemical_formulas(self, text: str) -> str:
        text = text.translate(_DIGIT_TRANSLATION)

        def replace(match: re.Match[str]) -> str:
            parts = _FORMULA_PART.findall(match.group(0))
            return " ".join(parts)

        return _FORMULA_TOKEN.sub(replace, text)

    def _convert_square_roots(self, text: str) -> str:
        return _SQUARE_ROOT.sub(lambda m: f"square root of {m.group(1)}", text)

    def _convert_exponents(self, text: str) -> str:
        def replace(match: re.Match[str]) -> str:
            base, exponent = match.group(1), match.group(2)
            if exponent == "2":
                return f"{base} squared"
            if exponent == "3":
                return f"{base} cubed"
            return f"{base} to the power of {exponent}"

        return _EXPONENT.sub(replace, text)

    def _convert_fractions(self, text: str) -> str:
        def replace(match: re.Match[str]) -> str:
            numerator, denominator = int(match.group(1)), int(match.group(2))
            spoken = _COMMON_FRACTIONS.get((numerator, denominator))
            if spoken is not None:
                return spoken
            return f"{numerator} over {denominator}"

        return _FRACTION.sub(replace, text)

    def _convert_operators(self, text: str) -> str:
        text = _PLUS.sub(" plus ", text)
        text = _EQUALS.sub(" equals ", text)
        return text

    def _cleanup_whitespace(self, text: str) -> str:
        text = _LEADING_TRAILING_WS.sub("", text)
        text = _CONSECUTIVE_NEWLINES.sub("\n", text)
        text = _EXTRA_SPACE.sub(" ", text)
        text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
        return text.strip()
