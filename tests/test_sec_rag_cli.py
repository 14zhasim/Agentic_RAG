"""Tests for the SEC RAG command-line boundary."""

from __future__ import annotations

import json
from pathlib import Path

import pymupdf

from sec_rag.cli import main


def _write_config_with_pdf(tmp_path: Path) -> Path:
    """Create a minimal configured corpus for CLI output tests."""
    project = tmp_path / "project"
    pdf_dir = project / "data" / "financebench" / "pdfs"
    pdf_dir.mkdir(parents=True)
    pdf = pymupdf.open()
    pdf.new_page()
    pdf.save(pdf_dir / "example_2024_10K.pdf")
    pdf.close()
    config_dir = project / "configs"
    config_dir.mkdir()
    config_path = config_dir / "sec_rag.toml"
    config_path.write_text(
        """[corpus]
prepared_dir = "data/financebench"
parsed_dir = "data/financebench/parsed"
chunks_dir = "data/financebench/chunks"
expected_documents = 1

[parsing]
provider = "azure-document-intelligence"
model_id = "prebuilt-layout"
output_content_format = "markdown"

[chunking]
floor_tokens = 250
ceiling_tokens = 1024
overlap_cap_tokens = 100
""",
        encoding="utf-8",
    )
    return config_path


def test_parse_command_plans_without_spending(tmp_path: Path, capsys) -> None:
    """The default CLI parse operation reports missing filings without Azure."""
    config_path = _write_config_with_pdf(tmp_path)

    exit_code = main(["parse", "--config", str(config_path)])

    assert exit_code == 0
    assert capsys.readouterr().out == (
        "Selected: 1\nDone: 0\nMissing: 1\nParsed now: 0\n"
    )


def test_parse_command_reports_missing_credentials(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    """A paid run without credentials prints the error and exits 1."""
    monkeypatch.delenv("AZURE_DOCUMENT_INTELLIGENCE_KEY", raising=False)
    monkeypatch.delenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", raising=False)
    config_path = _write_config_with_pdf(tmp_path)

    exit_code = main(["parse", "--config", str(config_path), "--execute-paid"])

    assert exit_code == 1
    assert "Error: AZURE_DOCUMENT_INTELLIGENCE_KEY" in capsys.readouterr().out


def test_parse_command_returns_one_for_runtime_configuration_error(
    tmp_path: Path, capsys
) -> None:
    """Operational failures use exit code 1; argparse keeps usage errors as 2."""
    config_path = _write_config_with_pdf(tmp_path)
    config_path.write_text("[corpus]\n", encoding="utf-8")

    exit_code = main(["parse", "--config", str(config_path)])

    assert exit_code == 1
    assert "Error:" in capsys.readouterr().out


def test_inspect_parse_command_writes_report(tmp_path: Path, capsys) -> None:
    """The offline command renders a report without parsing or credentials."""
    config_path = _write_config_with_pdf(tmp_path)
    cache_dir = config_path.parent.parent / "data" / "financebench" / "parsed"
    cache_dir.mkdir(parents=True)
    raw = {
        "content": "# Filing title",
        "pages": [{"pageNumber": 1}],
        "paragraphs": [],
        "sections": [],
    }
    (cache_dir / "example_2024_10K.json").write_text(json.dumps(raw), encoding="utf-8")

    exit_code = main(["inspect-parse", "--config", str(config_path)])

    assert exit_code == 0
    assert "Inspected: 1" in capsys.readouterr().out
    assert (cache_dir / "example_2024_10K.structure.txt").is_file()


def _save_one_page_parse(config_path: Path) -> Path:
    """Save a one-page parse for example_2024_10K; return the project folder."""
    project = config_path.parent.parent
    cache_dir = project / "data" / "financebench" / "parsed"
    cache_dir.mkdir(parents=True)
    content = "Revenue grew strongly this year."
    raw = {
        "content": content,
        "pages": [{"pageNumber": 1, "spans": [{"offset": 0, "length": len(content)}]}],
        "paragraphs": [],
        "sections": [],
    }
    (cache_dir / "example_2024_10K.json").write_text(json.dumps(raw), encoding="utf-8")
    return project


def test_chunk_command_writes_chunks_and_prints_summary(tmp_path: Path, capsys) -> None:
    config_path = _write_config_with_pdf(tmp_path)
    project = _save_one_page_parse(config_path)

    exit_code = main(["chunk", "--config", str(config_path)])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Filings: 1\n" in output
    assert "Chunks: 1 (prose 1, table 0, figure 0)" in output
    assert "Prose under floor: 1" in output
    assert (project / "data/financebench/chunks/example_2024_10K.jsonl").is_file()


def test_chunk_command_reports_an_unparsed_filing(tmp_path: Path, capsys) -> None:
    config_path = _write_config_with_pdf(tmp_path)

    exit_code = main(["chunk", "--config", str(config_path)])

    assert exit_code == 1
    assert "Error: example_2024_10K: not parsed yet" in capsys.readouterr().out
