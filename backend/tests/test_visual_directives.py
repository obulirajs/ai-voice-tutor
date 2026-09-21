from __future__ import annotations

from app.orchestration.visual_directives import VisualDirective, extract_directives


def test_display_math_extracted_as_formula() -> None:
    reply, directives = extract_directives("The formula is $$x^2 + y^2 = r^2$$ as shown.")

    assert reply == "The formula is $$x^2 + y^2 = r^2$$ as shown."
    assert directives == [VisualDirective(directive_type="formula", content="x^2 + y^2 = r^2")]


def test_inline_math_extracted_as_formula() -> None:
    _, directives = extract_directives("We know that $E = mc^2$ describes energy.")

    assert directives == [VisualDirective(directive_type="formula", content="E = mc^2")]


def test_dollar_amounts_are_not_extracted_as_math() -> None:
    _, directives = extract_directives("The book costs $5.00, or $100 if imported.")

    assert directives == []


def test_dollar_amounts_with_no_space_between_are_not_paired_as_math() -> None:
    _, directives = extract_directives("It costs $5 or $10 depending on the edition.")

    assert directives == []


def test_citation_extracted_as_text_block() -> None:
    _, directives = extract_directives("Paris is the capital [Passage 2 — page 14].")

    assert directives == [
        VisualDirective(directive_type="text_block", content="[Passage 2 — page 14]", label="page 14")
    ]


def test_bold_terms_extracted_as_highlights() -> None:
    _, directives = extract_directives("The **subjonctif** and **indicatif** are moods in French.")

    assert directives == [
        VisualDirective(directive_type="highlight", content="subjonctif"),
        VisualDirective(directive_type="highlight", content="indicatif"),
    ]


def test_bold_highlights_capped_at_five() -> None:
    reply = " ".join(f"**term{i}**" for i in range(8))
    _, directives = extract_directives(reply)

    assert len(directives) == 5
    assert [d.content for d in directives] == [f"term{i}" for i in range(5)]


def test_no_directives_for_plain_text() -> None:
    reply, directives = extract_directives("This is a plain sentence with no special notation.")

    assert reply == "This is a plain sentence with no special notation."
    assert directives == []


def test_supplemented_marker_extracted_as_supplemented_directive() -> None:
    reply = "**💡 Beyond the textbook:** Think of it like a solar panel charging a battery."
    _, directives = extract_directives(reply)

    assert (
        VisualDirective(
            directive_type="supplemented", content="Example from outside the textbook", label="model knowledge"
        )
        in directives
    )


def test_no_supplemented_directive_without_marker() -> None:
    _, directives = extract_directives("Photosynthesis converts light energy into chemical energy.")

    assert not any(d.directive_type == "supplemented" for d in directives)


def test_mixed_directive_types_extracted_together() -> None:
    reply = (
        "The **Pythagorean theorem** states $$a^2 + b^2 = c^2$$, "
        "as shown in [Passage 1 — page 5]. It costs $9.99 to print."
    )
    _, directives = extract_directives(reply)

    assert VisualDirective(directive_type="formula", content="a^2 + b^2 = c^2") in directives
    assert VisualDirective(directive_type="highlight", content="Pythagorean theorem") in directives
    assert (
        VisualDirective(directive_type="text_block", content="[Passage 1 — page 5]", label="page 5") in directives
    )
    assert not any(d.content == "9.99" for d in directives)
