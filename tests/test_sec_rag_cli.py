"""Tests for the SEC RAG command-line boundary."""

from __future__ import annotations

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
expected_documents = 1

[parsing]
provider = "azure-document-intelligence"
model_id = "prebuilt-layout"
output_content_format = "markdown"
""",
        encoding="utf-8",
    )
    return config_path


def test_parse_command_plans_without_spending(tmp_path: Path, capsys) -> None:
    """The default CLI parse operation reports missing filings without Azure."""
    config_path = _write_config_with_pdf(tmp_path)

    exit_code = main(["parse", "--config", str(config_path)])

    assert exit_code == 0
    assert "Missing: 1" in capsys.readouterr().out


def test_parse_command_returns_one_for_runtime_configuration_error(
    tmp_path: Path, capsys
) -> None:
    """Operational failures use exit code 1; argparse keeps usage errors as 2."""
    config_path = _write_config_with_pdf(tmp_path)
    config_path.write_text("[corpus]\n", encoding="utf-8")

    exit_code = main(["parse", "--config", str(config_path)])

    assert exit_code == 1
    assert "Error:" in capsys.readouterr().out
