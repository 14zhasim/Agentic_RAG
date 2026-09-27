"""Tests for the free structure report of a saved Azure parse."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pymupdf
import pytest

from sec_rag.ingestion.headings import Heading
from sec_rag.ingestion.inspect_parse import inspect_parses
from sec_rag.ingestion.structure_report import build_page_map, build_structure_report

SECTION_TITLES = [
    "## 1. Headings",
    "## 2. Page -> heading path",
    "## 3. Figures",
]


def _write_pdf(path: Path, page_count: int = 1) -> None:
    pdf = pymupdf.open()
    for _ in range(page_count):
        pdf.new_page()
    pdf.save(path)
    pdf.close()


def _config(tmp_path: Path, parsed: list[str], unparsed: list[str]) -> dict[str, Any]:
    """Build a corpus where the `parsed` filings have a valid one-page saved JSON."""
    prepared_dir = tmp_path / "financebench"
    (prepared_dir / "pdfs").mkdir(parents=True)
    parsed_dir = prepared_dir / "parsed"
    parsed_dir.mkdir()
    raw = {
        "content": "## Item 7. Management discussion\n\nRevenue increased.",
        "pages": [{"pageNumber": 1}],
        "paragraphs": [
            {
                "content": "Item 7. Management discussion",
                "role": "sectionHeading",
                "spans": [{"offset": 0, "length": 32}],
                "boundingRegions": [{"pageNumber": 1}],
            }
        ],
        "sections": [{"elements": ["/paragraphs/0"]}],
    }
    for doc_name in parsed + unparsed:
        _write_pdf(prepared_dir / "pdfs" / f"{doc_name}.pdf")
    for doc_name in parsed:
        (parsed_dir / f"{doc_name}.json").write_text(json.dumps(raw), encoding="utf-8")
    return {
        "corpus": {"prepared_dir": str(prepared_dir), "parsed_dir": str(parsed_dir)}
    }


def test_writes_a_report_without_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The report is written next to the saved JSON, with no Azure access."""
    monkeypatch.delenv("AZURE_DOCUMENT_INTELLIGENCE_KEY", raising=False)
    monkeypatch.delenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", raising=False)
    config = _config(tmp_path, parsed=["example_2024_10K"], unparsed=[])

    result = inspect_parses(config)

    report_path = (
        Path(config["corpus"]["parsed_dir"]) / "example_2024_10K.structure.txt"
    )
    report = report_path.read_text(encoding="utf-8")
    assert result == {"inspected": 1}
    assert "Item 7. Management discussion" in report


def test_report_builder_touches_no_files(worked_example_parse: dict[str, Any]) -> None:
    """The builder turns a dictionary into text with all three sections."""
    report = build_structure_report("example_2024_10K", worked_example_parse)

    assert report.startswith("# Structure report: example_2024_10K")
    for title in SECTION_TITLES:
        assert title in report


def test_page_path(worked_example_parse: dict[str, Any]) -> None:
    """Headings carry across pages until one at the same or a shallower level starts."""
    report = build_structure_report("example", worked_example_parse)

    assert "   10    9   PART I > Item 1A. Risk Factors" in report
    assert "   13   12   PART II > Item 5. Market for ..." in report
    assert "    3    2   (before any heading)" in report


def test_heading_outside_the_tree_is_counted_but_not_placed() -> None:
    """A heading with no level appears in the count and is left out of page paths."""
    headings = [
        Heading(offset=0, page=1, level=1, text="PART I"),
        Heading(offset=10, page=1, level=None, text="Stray"),
    ]
    raw = {
        "content": "",
        "pages": [{"pageNumber": 1}],
        "paragraphs": [
            {
                "content": "PART I",
                "role": "title",
                "spans": [{"offset": 0, "length": 6}],
                "boundingRegions": [{"pageNumber": 1}],
            },
            {
                "content": "Stray",
                "role": "sectionHeading",
                "spans": [{"offset": 10, "length": 5}],
                "boundingRegions": [{"pageNumber": 1}],
            },
        ],
        "sections": [{"elements": ["/sections/1"]}, {"elements": ["/paragraphs/0"]}],
    }

    report = build_structure_report("example", raw)

    assert build_page_map(headings, 1) == {1: ["PART I"]}
    assert "outside the section tree (no level): 1" in report


def test_figure_count() -> None:
    """The Figures section counts <figure> tags in content."""
    raw = {
        "content": "<figure>Chart A</figure>\n\n<figure>Chart B</figure>",
        "pages": [{"pageNumber": 1}],
    }

    report = build_structure_report("example", raw)

    assert "<figure> tags in content: 2" in report
    assert "entries in Azure's figures list: 0" in report


def test_unparsed_filing_stops_everything(tmp_path: Path) -> None:
    """If any selected filing isn't parsed, no report is written."""
    config = _config(tmp_path, parsed=["alpha_2024_10K"], unparsed=["beta_2024_10K"])

    with pytest.raises(ValueError, match="beta_2024_10K: not parsed yet"):
        inspect_parses(config)

    assert list(Path(config["corpus"]["parsed_dir"]).glob("*.structure.txt")) == []
