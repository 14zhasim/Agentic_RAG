"""Tests for chunk_filing: the Stage 1.3 chunking rules on hand-made pages.

Every test builds its own small filing with `_filing`, so the page being cut
is visible in the test. Settings are tiny (floor 10 tokens) so the text stays
readable; each "long" stretch below is well over 10 tokens and each "short"
one well under, so no test depends on an exact token count.
"""

from __future__ import annotations

from typing import Any

from sec_rag.chunking.chunk import chunk_filing, count_tokens, tidy

DOC = "3M_2018_10K"
SETTINGS = {"floor_tokens": 10, "ceiling_tokens": 40, "overlap_cap_tokens": 6}

# 15-16 tokens each: comfortably over the floor of 10.
LONG_A = "Revenue grew in every segment this year because demand for industrial products stayed strong."
LONG_B = "Operating margins improved as the company cut costs and raised prices across its main markets."
LONG_C = "Cash flow from operations funded dividends, share repurchases and several small acquisitions."
# About 3 tokens: well under the floor.
SHORT = "See below."


def _filing(
    page_texts: list[str],
    headings: tuple[str, ...] = (),
    tables: tuple[str, ...] = (),
    figures: tuple[str, ...] = (),
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build a saved parse and `load_pages`-style pages from page strings.

    Pages are joined by one newline in `content`; an empty string is a blank
    page with no span. Headings, tables and figures are given as exact text
    and located by their first occurrence in `content`, so each must be
    unique in the test.
    """
    content = "\n".join(page_texts)
    raw_pages: list[dict[str, Any]] = []
    pages: list[dict[str, Any]] = []
    offset = 0
    for index, text in enumerate(page_texts):
        start: int | None = offset if text else None
        raw_page: dict[str, Any] = {"pageNumber": index + 1}
        if text:
            raw_page["spans"] = [{"offset": offset, "length": len(text)}]
        raw_pages.append(raw_page)
        pages.append(
            {
                "doc_name": DOC,
                "company": "3M",
                "year": 2018,
                "doc_type": "10K",
                "page_index": index,
                "start_offset": start,
                "text": text,
            }
        )
        offset += len(text) + 1

    def span(text: str) -> list[dict[str, int]]:
        return [{"offset": content.index(text), "length": len(text)}]

    raw = {
        "content": content,
        "pages": raw_pages,
        "paragraphs": [
            {
                "role": "sectionHeading",
                "content": heading,
                "spans": span(heading),
                "boundingRegions": [{"pageNumber": 1}],
            }
            for heading in headings
        ],
        "sections": [],
        "tables": [{"spans": span(table)} for table in tables],
        "figures": [{"spans": span(figure)} for figure in figures],
    }
    return raw, pages


def _chunk(raw: dict[str, Any], pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run the chunker and check rule 1 on every chunk it returns."""
    chunks = chunk_filing(raw, pages, SETTINGS)
    _assert_one_page(chunks, pages)
    return chunks


def _assert_one_page(chunks: list[dict[str, Any]], pages: list[dict[str, Any]]) -> None:
    """Build Order 1.3: every chunk's own text lies inside its one page."""
    for chunk in chunks:
        page = pages[chunk["page_index"]]
        page_end = page["start_offset"] + len(page["text"])
        assert page["start_offset"] <= chunk["start_offset"]
        assert chunk["start_offset"] < chunk["end_offset"] <= page_end


def _own_text(raw: dict[str, Any], chunk: dict[str, Any]) -> str:
    return raw["content"][chunk["start_offset"] : chunk["end_offset"]]


def test_single_page_of_prose_is_one_chunk() -> None:
    raw, pages = _filing([f"   {LONG_A}   "])

    [chunk] = _chunk(raw, pages)

    assert chunk["chunk_id"] == "3M_2018_10K:p0:c0"
    assert chunk["kind"] == "prose"
    assert (chunk["company"], chunk["year"], chunk["doc_type"]) == ("3M", 2018, "10K")
    assert chunk["page_index"] == 0
    # Offsets are trimmed to the visible text.
    assert _own_text(raw, chunk) == LONG_A
    assert chunk["text"] == LONG_A
    assert chunk["carried_text"] == ""
    assert chunk["token_count"] == count_tokens(LONG_A)
    assert chunk["heading_path"] == []


def test_table_is_its_own_chunk_and_prose_either_side_is_joined() -> None:
    table = "<table><tr><td>Revenue</td><td>32,765</td></tr></table>"
    raw, pages = _filing([f"{LONG_A}\n\n{table}\n\n{LONG_B}"], tables=(table,))

    prose, table_chunk = _chunk(raw, pages)

    assert prose["chunk_id"].endswith("c0") and prose["kind"] == "prose"
    assert prose["text"] == f"{LONG_A}\n\n{LONG_B}"
    assert table_chunk["chunk_id"].endswith("c1") and table_chunk["kind"] == "table"
    assert table_chunk["text"] == table
    assert _own_text(raw, table_chunk) == table


def test_figure_with_text_is_a_chunk_and_empty_figure_is_skipped() -> None:
    figure = "<figure>\n<figcaption>Sales by region</figcaption>\n\nUS 40%\n</figure>"
    logo = "<figure>\n</figure>"
    raw, pages = _filing([f"{logo}\n\n{LONG_A}\n\n{figure}"], figures=(logo, figure))

    chunks = _chunk(raw, pages)

    assert [chunk["kind"] for chunk in chunks] == ["prose", "figure"]
    assert chunks[1]["text"] == figure
    # The empty figure is neither a chunk nor left behind in the prose.
    assert "figure" not in chunks[0]["text"]


def test_heading_cut_kept_when_both_sides_clear_the_floor() -> None:
    raw, pages = _filing(
        [f"## Results\n\n{LONG_A}\n\n## Outlook\n\n{LONG_B}"],
        headings=("## Results", "## Outlook"),
    )

    first, second = _chunk(raw, pages)

    assert first["text"] == f"## Results\n\n{LONG_A}"
    # Heading lines stay in the chunk's text.
    assert second["text"] == f"## Outlook\n\n{LONG_B}"


def test_heading_cut_rejected_when_the_piece_before_is_under_the_floor() -> None:
    raw, pages = _filing(
        [f"{SHORT}\n\n## Outlook\n\n{LONG_B}"], headings=("## Outlook",)
    )

    [chunk] = _chunk(raw, pages)

    assert chunk["text"] == f"{SHORT}\n\n## Outlook\n\n{LONG_B}"


def test_heading_cut_rejected_when_the_remainder_is_under_the_floor() -> None:
    raw, pages = _filing(
        [f"{LONG_A}\n\n## Outlook\n\n{SHORT}"], headings=("## Outlook",)
    )

    [chunk] = _chunk(raw, pages)

    assert chunk["text"] == f"{LONG_A}\n\n## Outlook\n\n{SHORT}"


def test_headings_tried_left_to_right() -> None:
    """`## Risks` is rejected only because `## Outlook` was kept first.

    Measured from the page start, the piece before `## Risks` would be long
    enough; measured from the kept `## Outlook` cut, it's only a short line.
    """
    raw, pages = _filing(
        [f"## Results\n\n{LONG_A}\n\n## Outlook\n\n{SHORT}\n\n## Risks\n\n{LONG_C}"],
        headings=("## Results", "## Outlook", "## Risks"),
    )

    first, second = _chunk(raw, pages)

    assert first["text"] == f"## Results\n\n{LONG_A}"
    assert second["text"] == f"## Outlook\n\n{SHORT}\n\n## Risks\n\n{LONG_C}"


def test_heading_inside_a_table_is_not_a_cut_point() -> None:
    table = "<table><tr><td>\n## Inside\n</td></tr></table>"
    raw, pages = _filing(
        [f"{LONG_A}\n\n{table}\n\n{LONG_B}"],
        headings=("## Inside",),
        tables=(table,),
    )

    chunks = _chunk(raw, pages)

    assert [chunk["kind"] for chunk in chunks] == ["prose", "table"]


def test_small_page_is_still_its_own_chunk() -> None:
    raw, pages = _filing([LONG_A, SHORT])

    chunks = _chunk(raw, pages)

    assert [chunk["text"] for chunk in chunks] == [LONG_A, SHORT]
    assert [chunk["page_index"] for chunk in chunks] == [0, 1]


def test_blank_page_gives_no_chunks() -> None:
    raw, pages = _filing([LONG_A, "", LONG_B])

    chunks = _chunk(raw, pages)

    assert [chunk["page_index"] for chunk in chunks] == [0, 2]


def test_page_of_only_a_table_gives_no_prose_chunk() -> None:
    table = "<table><tr><td>1</td></tr></table>"
    raw, pages = _filing([f"\n\n{table}\n\n"], tables=(table,))

    chunks = _chunk(raw, pages)

    assert [chunk["kind"] for chunk in chunks] == ["table"]


def test_tidy_collapses_blanks() -> None:
    assert tidy("  a   b  \n    \n\n\n\n   c  ") == "a b\n\nc"


def test_count_tokens_ignores_blanks() -> None:
    assert count_tokens("one two" + " " * 500 + "\n" * 20) == count_tokens("one two")
