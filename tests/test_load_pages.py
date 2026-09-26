"""Tests for load_pages: clean, position-preserving pages from a saved parse."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from sec_rag.ingestion.load_pages import _split_doc_name, load_pages

REAL_PARSED_DIR = Path("data/financebench/parsed")


def _config(tmp_path: Path) -> dict[str, Any]:
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()
    return {"corpus": {"parsed_dir": str(parsed_dir)}}


def _save(config: dict[str, Any], doc_name: str, raw: dict[str, Any]) -> None:
    path = Path(config["corpus"]["parsed_dir"]) / f"{doc_name}.json"
    path.write_text(json.dumps(raw), encoding="utf-8")


def _one_page_parse(page_text: str, before: str = "") -> dict[str, Any]:
    """A parse whose single page covers `page_text`, starting after `before`."""
    return {
        "content": before + page_text,
        "pages": [
            {
                "pageNumber": 1,
                "spans": [{"offset": len(before), "length": len(page_text)}],
            }
        ],
    }


def test_page_number_is_azure_page_number_minus_one(tmp_path: Path) -> None:
    """FinanceBench counts pages from 0, Azure from 1: every page metric depends on this."""
    config = _config(tmp_path)
    content = "page one page two page three"
    raw = {
        "content": content,
        "pages": [
            {"pageNumber": 1, "spans": [{"offset": 0, "length": 8}]},
            {"pageNumber": 2, "spans": [{"offset": 9, "length": 8}]},
            {"pageNumber": 3, "spans": [{"offset": 18, "length": 10}]},
        ],
    }
    _save(config, "3M_2018_10K", raw)

    pages = load_pages(config, "3M_2018_10K")

    assert [page["page_index"] for page in pages] == [0, 1, 2]


def test_markers_blanked_and_content_kept(tmp_path: Path) -> None:
    """Markers become spaces; the heading and body are untouched; length is unchanged."""
    config = _config(tmp_path)
    page_text = (
        '<!-- PageHeader="Table of Contents" -->\n\n## PART II\n\nBody text.\n\n'
        '<!-- PageNumber="13" -->\n<!-- PageBreak -->\n\n'
    )
    _save(config, "3M_2018_10K", _one_page_parse(page_text))

    [page] = load_pages(config, "3M_2018_10K")

    assert "<!--" not in page["text"]
    assert "## PART II\n\nBody text." in page["text"]
    assert len(page["text"]) == len(page_text)
    assert page["text"].startswith(" " * 39 + "\n\n## PART II")


def test_positions_still_line_up(tmp_path: Path) -> None:
    """Position i in a page's text is raw position start_offset + i."""
    config = _config(tmp_path)
    before = "x" * 18
    page_text = '<!-- PageHeader="Contents" -->' + "\n" * 12 + "## PART II\n\nBody."
    raw = _one_page_parse(page_text, before=before)
    heading_offset = raw["content"].index("## PART II")
    _save(config, "3M_2018_10K", raw)

    [page] = load_pages(config, "3M_2018_10K")

    assert page["start_offset"] == 18
    assert heading_offset == 60
    assert page["text"][heading_offset - page["start_offset"] :].startswith(
        "## PART II"
    )


def test_page_footer_is_blanked_too(tmp_path: Path) -> None:
    config = _config(tmp_path)
    page_text = 'Body.\n\n<!-- PageFooter="Confidential" -->\n'
    _save(config, "3M_2018_10K", _one_page_parse(page_text))

    [page] = load_pages(config, "3M_2018_10K")

    assert "PageFooter" not in page["text"]
    assert len(page["text"]) == len(page_text)


def test_newline_inside_a_marker_is_kept(tmp_path: Path) -> None:
    """A marker whose quoted text runs over two lines keeps its line break."""
    config = _config(tmp_path)
    page_text = '<!-- PageHeader="Annual\nReport" -->\nBody.'
    _save(config, "3M_2018_10K", _one_page_parse(page_text))

    [page] = load_pages(config, "3M_2018_10K")

    # '<!-- PageHeader="Annual' is 23 characters, 'Report" -->' is 11.
    assert page["text"] == " " * 23 + "\n" + " " * 11 + "\nBody."


def test_figure_text_is_kept(tmp_path: Path) -> None:
    config = _config(tmp_path)
    page_text = "<figure>\nRevenue by segment\n2018: 32.8bn\n</figure>\n"
    _save(config, "3M_2018_10K", _one_page_parse(page_text))

    [page] = load_pages(config, "3M_2018_10K")

    assert page["text"] == page_text


def test_each_page_gets_only_its_own_text(tmp_path: Path) -> None:
    config = _config(tmp_path)
    raw = {
        "content": "alpha words\nbeta words",
        "pages": [
            {"pageNumber": 1, "spans": [{"offset": 0, "length": 12}]},
            {"pageNumber": 2, "spans": [{"offset": 12, "length": 10}]},
        ],
    }
    _save(config, "3M_2018_10K", raw)

    first, second = load_pages(config, "3M_2018_10K")

    assert first["text"] == "alpha words\n"
    assert second["text"] == "beta words"
    assert second["start_offset"] == 12


def test_page_with_no_spans_is_blank(tmp_path: Path) -> None:
    config = _config(tmp_path)
    raw = {"content": "text", "pages": [{"pageNumber": 1, "spans": []}]}
    _save(config, "3M_2018_10K", raw)

    [page] = load_pages(config, "3M_2018_10K")

    assert page["start_offset"] is None
    assert page["text"] == ""


def test_page_with_two_spans_stops(tmp_path: Path) -> None:
    """Two spans would break the start_offset arithmetic, so fail loudly."""
    config = _config(tmp_path)
    raw = {
        "content": "one two",
        "pages": [
            {
                "pageNumber": 5,
                "spans": [{"offset": 0, "length": 3}, {"offset": 4, "length": 3}],
            }
        ],
    }
    _save(config, "3M_2018_10K", raw)

    with pytest.raises(ValueError, match="page 5 has 2 spans"):
        load_pages(config, "3M_2018_10K")


def test_metadata_comes_from_the_filename(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _save(config, "JOHNSON_JOHNSON_2022_10K", _one_page_parse("Body."))

    [page] = load_pages(config, "JOHNSON_JOHNSON_2022_10K")

    assert page["doc_name"] == "JOHNSON_JOHNSON_2022_10K"
    assert (page["company"], page["year"], page["doc_type"]) == (
        "JOHNSON_JOHNSON",
        2022,
        "10K",
    )


def test_filename_is_read_from_the_right() -> None:
    assert _split_doc_name("JOHNSON_JOHNSON_2022_10K") == (
        "JOHNSON_JOHNSON",
        2022,
        "10K",
    )
    assert _split_doc_name("3M_2018_10K") == ("3M", 2018, "10K")


@pytest.mark.parametrize("doc_name", ["NOTAFILING", "3M_10K", "3M_18_10K"])
def test_bad_filename(doc_name: str) -> None:
    with pytest.raises(ValueError, match="must end in _<year>_<type>"):
        _split_doc_name(doc_name)


def test_not_parsed_yet(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="3M_2018_10K: not parsed yet"):
        load_pages(_config(tmp_path), "3M_2018_10K")


@pytest.mark.skipif(
    not (REAL_PARSED_DIR / "3M_2018_10K.json").is_file(),
    reason="real 3M parse not on disk (data/ is Git-ignored)",
)
def test_real_3m_parse() -> None:
    """Page 13 of the real parse: raw start 54,574, PART II at text position 42."""
    config = {"corpus": {"parsed_dir": str(REAL_PARSED_DIR)}}

    pages = load_pages(config, "3M_2018_10K")

    assert len(pages) == 160
    page = pages[12]
    assert page["page_index"] == 12
    assert page["start_offset"] == 54574
    assert len(page["text"]) == 4184
    assert "<!--" not in page["text"]
    assert page["text"][42:52] == "## PART II"
    assert all("<!-- Page" not in page["text"] for page in pages)
