"""Tests for the heading list read from a saved Azure parse."""

from __future__ import annotations

from typing import Any

from sec_rag.ingestion.headings import Heading, extract_headings


def _paragraph(text: str, offset: int, page: int, role: str | None) -> dict[str, Any]:
    """Build one Azure paragraph with its span and start page."""
    paragraph: dict[str, Any] = {
        "content": text,
        "spans": [{"offset": offset, "length": len(text)}],
        "boundingRegions": [{"pageNumber": page}],
    }
    if role is not None:
        paragraph["role"] = role
    return paragraph


def test_levels_come_from_the_sections_tree(
    worked_example_parse: dict[str, Any],
) -> None:
    """Each heading's level is the depth of the section it opens; body text is skipped."""
    headings = extract_headings(worked_example_parse)

    assert [(heading.text, heading.level) for heading in headings] == [
        ("PART I", 1),
        ("Item 1. Business", 2),
        ("Item 1A. Risk Factors", 2),
        ("PART II", 1),
        ("Item 5. Market for ...", 2),
    ]
    assert [heading.page for heading in headings] == [4, 4, 10, 13, 13]


def test_offset_is_the_raw_position() -> None:
    """A heading's offset is where its markdown line starts in Azure's content."""
    content = "x" * 42 + "## PART II\n\nBody."
    raw = {
        "content": content,
        "paragraphs": [_paragraph("PART II", 42, 1, "title")],
        "sections": [{"elements": ["/paragraphs/0"]}],
    }

    [heading] = extract_headings(raw)

    assert heading == Heading(offset=42, page=1, level=0, text="PART II")
    assert content[heading.offset :].startswith("## PART II")


def test_headings_are_returned_in_reading_order() -> None:
    """Headings are sorted by raw position, whatever order Azure lists them in."""
    raw = {
        "content": "",
        "paragraphs": [
            _paragraph("Second", 50, 2, "sectionHeading"),
            _paragraph("First", 5, 1, "title"),
        ],
        "sections": [],
    }

    assert [heading.text for heading in extract_headings(raw)] == ["First", "Second"]


def test_depth_zero_is_a_real_level() -> None:
    """A heading opening the top section has level 0, not 'no level'."""
    raw = {
        "content": "",
        "paragraphs": [_paragraph("Root", 0, 1, "title")],
        "sections": [{"elements": ["/paragraphs/0"]}],
    }

    [heading] = extract_headings(raw)

    assert heading.level == 0


def test_heading_outside_the_tree_has_no_level() -> None:
    """A heading that opens no section is kept, with level None."""
    raw = {
        "content": "",
        "paragraphs": [_paragraph("Stray heading", 0, 1, "sectionHeading")],
        "sections": [],
    }

    [heading] = extract_headings(raw)

    assert heading.level is None


def test_heading_text_whitespace_is_tidied() -> None:
    """Line breaks and repeated spaces inside a heading become single spaces."""
    raw = {
        "content": "",
        "paragraphs": [_paragraph("Item 7.\nManagement's   Discussion", 0, 1, "title")],
        "sections": [],
    }

    [heading] = extract_headings(raw)

    assert heading.text == "Item 7. Management's Discussion"
