"""Tests for no-spend inspection of completed Azure Layout caches."""

from __future__ import annotations

import json
from pathlib import Path

import pymupdf
import pytest

from sec_rag.ingestion.inspection import inspect_parses
from sec_rag.ingestion.structure_report import build_structure_report


def _write_pdf(path: Path) -> None:
    """Create a one-page PDF matching the fixture Azure result."""
    pdf = pymupdf.open()
    pdf.new_page()
    pdf.save(path)
    pdf.close()


def _config(tmp_path: Path) -> dict[str, object]:
    """Create a completed one-document parse cache for offline inspection."""
    prepared_dir = tmp_path / "financebench"
    pdf_dir = prepared_dir / "pdfs"
    pdf_dir.mkdir(parents=True)
    _write_pdf(pdf_dir / "example_2024_10K.pdf")
    parsed_dir = prepared_dir / "parsed"
    parsed_dir.mkdir(parents=True)
    content = "# Item 7. Management discussion\nRevenue increased."
    raw = {
        "content": content,
        "pages": [{"pageNumber": 1}],
        "paragraphs": [
            {
                "content": "Item 7. Management discussion",
                "role": "sectionHeading",
                "spans": [{"offset": 2, "length": 28}],
                "boundingRegions": [{"pageNumber": 1}],
            },
            {
                "content": "Revenue increased.",
                "spans": [{"offset": 32, "length": 18}],
                "boundingRegions": [{"pageNumber": 1}],
            },
        ],
        "sections": [
            {
                "spans": [{"offset": 0, "length": len(content)}],
                "elements": ["/paragraphs/0"],
            }
        ],
        "tables": [],
        "styles": [],
    }
    (parsed_dir / "example_2024_10K.json").write_text(json.dumps(raw), encoding="utf-8")
    return {
        "corpus": {
            "prepared_dir": str(prepared_dir),
            "parsed_dir": str(parsed_dir),
            "expected_documents": 1,
        },
        "parsing": {
            "provider": "azure-document-intelligence",
            "model_id": "prebuilt-layout",
            "output_content_format": "markdown",
        },
    }


def test_inspection_writes_structure_report_without_credentials(tmp_path: Path) -> None:
    """Inspection derives a readable report from local JSON and the local PDF."""
    config = _config(tmp_path)

    result = inspect_parses(config)

    report_path = (
        Path(config["corpus"]["parsed_dir"]) / "example_2024_10K.structure.txt"
    )
    report = report_path.read_text(encoding="utf-8")
    assert result == {"inspected": 1}
    assert "# Structure report: example_2024_10K" in report
    assert "## 1. Role census" in report
    assert "Item 7. Management discussion" in report
    assert "## 4. Page -> heading path" in report
    assert "## 5. Page-count check" in report


def test_report_builder_is_a_pure_azure_json_transformation() -> None:
    """Report rendering has no filesystem dependency after the raw cache is loaded."""
    raw = {
        "content": "# Filing title",
        "pages": [{"pageNumber": 1}],
        "paragraphs": [],
        "sections": [],
    }

    report = build_structure_report("example_2024_10K", raw)

    assert report.startswith("# Structure report: example_2024_10K")
    assert "## 4. Page -> heading path" in report
    assert "## 5. Page-count check" in report


def test_report_keeps_a_depth_zero_heading_as_the_page_map_root() -> None:
    """A real tree depth of zero must not fall through to the Markdown level."""
    raw = {
        "content": "### Root\n# Child",
        "pages": [{"pageNumber": 1}],
        "paragraphs": [
            {
                "content": "Root",
                "role": "title",
                "spans": [{"offset": 4, "length": 4}],
                "boundingRegions": [{"pageNumber": 1}],
            },
            {
                "content": "Child",
                "role": "sectionHeading",
                "spans": [{"offset": 11, "length": 5}],
                "boundingRegions": [{"pageNumber": 1}],
            },
        ],
        "sections": [
            {"elements": ["/paragraphs/0", "/sections/1"]},
            {"elements": ["/paragraphs/1"]},
        ],
    }

    report = build_structure_report("example_2024_10K", raw)

    assert "Root > Child" in report


def test_inspection_rejects_a_missing_final_cache(tmp_path: Path) -> None:
    """Inspection refuses to create a report for a filing that was not parsed."""
    config = _config(tmp_path)
    cache_path = Path(config["corpus"]["parsed_dir"]) / "example_2024_10K.json"
    cache_path.unlink()

    with pytest.raises(ValueError, match="no final Azure cache"):
        inspect_parses(config)
