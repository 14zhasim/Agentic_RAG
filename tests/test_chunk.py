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
    assert table_chunk["text"] == f"{table_chunk['carried_text']} {table}"
    assert _own_text(raw, table_chunk) == table


def test_table_carries_its_title_from_just_above() -> None:
    title = "## Consolidated Balance Sheet"
    table = "<table><tr><td>Cash</td><td>2,853</td></tr></table>"
    raw, pages = _filing([f"{LONG_A}\n\n{title}\n\n{table}"], tables=(table,))

    prose, table_chunk = _chunk(raw, pages)

    # The title is carried as text; the prose chunk keeps it too, unchanged.
    assert table_chunk["carried_text"].endswith(title)
    assert table_chunk["text"].endswith(f"{title} {table}")
    assert prose["text"] == f"{LONG_A}\n\n{title}"
    # Offsets and page stay the table's own.
    assert _own_text(raw, table_chunk) == table


def test_text_above_a_table_over_the_cap_keeps_its_end() -> None:
    table = "<table><tr><td>1</td></tr></table>"
    raw, pages = _filing([f"{LONG_A}\n\n{table}"], tables=(table,))

    _, table_chunk = _chunk(raw, pages)

    carried = table_chunk["carried_text"]
    assert carried and LONG_A.endswith(carried)
    assert count_tokens(carried) <= SETTINGS["overlap_cap_tokens"]


def test_second_table_carries_only_the_text_after_the_first() -> None:
    first = "<table><tr><td>Income</td></tr></table>"
    second = "<table><tr><td>Comprehensive</td></tr></table>"
    raw, pages = _filing(
        [f"Statement of Income\n\n{first}\n\nStatement of Equity\n\n{second}"],
        tables=(first, second),
    )

    chunks = _chunk(raw, pages)
    first_chunk, second_chunk = (c for c in chunks if c["kind"] == "table")

    assert first_chunk["carried_text"] == "Statement of Income"
    assert second_chunk["carried_text"] == "Statement of Equity"


def test_table_at_the_top_of_a_page_carries_nothing() -> None:
    table = "<table><tr><td>continued</td></tr></table>"
    raw, pages = _filing([LONG_A, f"{table}\n\n{LONG_B}"], tables=(table,))

    table_chunk = next(c for c in _chunk(raw, pages) if c["kind"] == "table")

    # Nothing above it on its own page, and nothing from the previous page.
    assert table_chunk["carried_text"] == ""
    assert table_chunk["text"] == table


def test_figure_carries_nothing() -> None:
    figure = "<figure>\nUS 40%\n</figure>"
    raw, pages = _filing([f"{LONG_A}\n\n{figure}"], figures=(figure,))

    figure_chunk = next(c for c in _chunk(raw, pages) if c["kind"] == "figure")

    assert figure_chunk["carried_text"] == ""


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

    assert [_own_text(raw, chunk) for chunk in chunks] == [LONG_A, SHORT]
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


# --- Slice 2: halving (ceiling 40) and overlap (cap 6) -----------------------

SENTENCES = [LONG_A, LONG_B, LONG_C]


def test_prose_over_ceiling_is_halved_at_a_sentence() -> None:
    """Three ~15-token sentences (~47 tokens) are over the ceiling of 40: two pieces."""
    raw, pages = _filing([" ".join(SENTENCES)])

    chunks = _chunk(raw, pages)

    assert len(chunks) == 2
    for chunk in chunks:
        own = _own_text(raw, chunk)
        assert count_tokens(own) <= SETTINGS["ceiling_tokens"]
        assert any(own.startswith(sentence) for sentence in SENTENCES)


def test_halving_is_recursive() -> None:
    """Nine sentences (~140 tokens) end up as at least four pieces, none over the ceiling."""
    raw, pages = _filing([" ".join(SENTENCES * 3)])

    chunks = _chunk(raw, pages)

    assert len(chunks) >= 4
    for chunk in chunks:
        assert count_tokens(_own_text(raw, chunk)) <= SETTINGS["ceiling_tokens"]


def test_halving_splits_tokens_as_evenly_as_sentences_allow() -> None:
    """Four sentences of 15, 16, 15 and 15 tokens: the most even cut is after the second."""
    fourth = "Management expects similar growth next year if the economy holds up well."
    raw, pages = _filing([" ".join([*SENTENCES[:2], SENTENCES[2], fourth])])

    first, second = _chunk(raw, pages)

    assert _own_text(raw, first) == f"{LONG_A} {LONG_B}"
    assert _own_text(raw, second) == f"{LONG_C} {fourth}"


def test_halving_falls_back_to_a_line_start_then_a_word_start() -> None:
    """With no full stops, the cut lands at a line start; with no line breaks either, a word start."""
    words = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
    lines = "\n".join([words] * 5)  # about 60 tokens, no sentence end
    one_line = " ".join([words] * 5)  # the same words with no line breaks
    raw, pages = _filing([lines, one_line])

    chunks = _chunk(raw, pages)

    content = raw["content"]
    page0 = [chunk for chunk in chunks if chunk["page_index"] == 0]
    page1 = [chunk for chunk in chunks if chunk["page_index"] == 1]
    assert len(page0) >= 2 and len(page1) >= 2
    for chunk in page0[1:]:
        assert content[chunk["start_offset"] - 1] == "\n"
    for chunk in page1[1:]:
        assert content[chunk["start_offset"] - 1] == " "


def test_page_break_carries_last_sentence_of_previous_page() -> None:
    """Page 0 ends mid-sentence; page 1's chunk starts with the broken sentence restored."""
    raw, pages = _filing([f"{LONG_A} Sales of plan", f"assets rose. {LONG_B}"])

    _, second = _chunk(raw, pages)

    assert second["carried_text"] == "Sales of plan"
    assert second["text"].startswith("Sales of plan assets rose.")
    # Text only: the chunk's page and offsets are still page 1's own.
    assert second["page_index"] == 1
    assert _own_text(raw, second).startswith("assets rose.")


def test_no_carry_when_page_starts_with_a_heading() -> None:
    raw, pages = _filing(
        [f"{LONG_A} Short end.", f"## Outlook\n\n{LONG_B}"], headings=("## Outlook",)
    )

    _, second = _chunk(raw, pages)

    assert second["carried_text"] == ""
    assert second["text"].startswith("## Outlook")


def test_no_carry_at_a_heading_cut() -> None:
    raw, pages = _filing(
        [f"{LONG_A}\n\n## Outlook\n\n{LONG_B}"], headings=("## Outlook",)
    )

    _, second = _chunk(raw, pages)

    assert second["carried_text"] == ""


def test_halving_cut_carries_last_sentence_of_previous_piece() -> None:
    raw, pages = _filing([" ".join(SENTENCES)])

    first, second = _chunk(raw, pages)

    last_sentence_of_first = next(
        sentence for sentence in SENTENCES if _own_text(raw, first).endswith(sentence)
    )
    assert second["carried_text"]
    assert last_sentence_of_first.endswith(second["carried_text"])
    assert first["carried_text"] == ""


def test_carried_sentence_over_cap_keeps_its_end() -> None:
    """LONG_A is ~15 tokens, over the cap of 6: words are dropped from its front."""
    raw, pages = _filing([LONG_A, LONG_B])

    _, second = _chunk(raw, pages)

    carried = second["carried_text"]
    assert LONG_A.endswith(carried)
    assert carried.endswith("stayed strong.")
    assert count_tokens(carried) <= SETTINGS["overlap_cap_tokens"]


def test_carry_resets_after_a_page_with_no_prose() -> None:
    table = "<table><tr><td>1</td></tr></table>"
    raw, pages = _filing([f"{LONG_A} Tail.", table, LONG_B], tables=(table,))

    chunks = _chunk(raw, pages)

    assert chunks[-1]["page_index"] == 2
    assert chunks[-1]["carried_text"] == ""


def test_first_page_carries_nothing() -> None:
    raw, pages = _filing([LONG_A])

    [chunk] = _chunk(raw, pages)

    assert chunk["carried_text"] == ""


def test_tidy_collapses_blanks() -> None:
    assert tidy("  a   b  \n    \n\n\n\n   c  ") == "a b\n\nc"


def test_count_tokens_ignores_blanks() -> None:
    assert count_tokens("one two" + " " * 500 + "\n" * 20) == count_tokens("one two")
