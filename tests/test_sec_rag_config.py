"""Tests for the RAG-system configuration separate from benchmark settings."""

from __future__ import annotations

from pathlib import Path

from sec_rag.config import load_config


def test_load_config_resolves_corpus_paths_from_project_root(tmp_path: Path) -> None:
    """The parser config is portable because paths resolve beside configs/."""
    project = tmp_path / "project"
    config_dir = project / "configs"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "sec_rag.toml"
    config_path.write_text(
        """[corpus]
prepared_dir = "data/financebench"
parsed_dir = "data/financebench/parsed"
expected_documents = 64

[parsing]
provider = "azure-document-intelligence"
model_id = "prebuilt-layout"
output_content_format = "markdown"
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config["corpus"]["prepared_dir"] == str(project / "data/financebench")
    assert config["corpus"]["parsed_dir"] == str(project / "data/financebench/parsed")
