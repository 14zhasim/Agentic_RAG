"""Tests for the RAG-system configuration separate from benchmark settings."""

from __future__ import annotations

from pathlib import Path

import pytest

from sec_rag.config import load_config

VALID_CONFIG = """[corpus]
prepared_dir = "data/financebench"
parsed_dir = "data/financebench/parsed"
chunks_dir = "data/financebench/chunks"
expected_documents = 64

[parsing]
provider = "azure-document-intelligence"
model_id = "prebuilt-layout"
output_content_format = "markdown"

[chunking]
floor_tokens = 250
ceiling_tokens = 1024
overlap_cap_tokens = 100
"""


def _write(tmp_path: Path, text: str) -> tuple[Path, Path]:
    """Write a config at <project>/configs/sec_rag.toml; return (project, config path)."""
    project = tmp_path / "project"
    config_dir = project / "configs"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "sec_rag.toml"
    config_path.write_text(text, encoding="utf-8")
    return project, config_path


def test_load_config_resolves_corpus_paths_from_project_root(tmp_path: Path) -> None:
    """The parser config is portable because paths resolve beside configs/."""
    project, config_path = _write(tmp_path, VALID_CONFIG)

    config = load_config(config_path)

    assert config["corpus"]["prepared_dir"] == str(project / "data/financebench")
    assert config["corpus"]["parsed_dir"] == str(project / "data/financebench/parsed")
    assert config["corpus"]["chunks_dir"] == str(project / "data/financebench/chunks")
    assert config["chunking"] == {
        "floor_tokens": 250,
        "ceiling_tokens": 1024,
        "overlap_cap_tokens": 100,
    }


def test_missing_chunking_section_is_refused(tmp_path: Path) -> None:
    text = VALID_CONFIG.split("[chunking]")[0]
    _, config_path = _write(tmp_path, text)

    with pytest.raises(ValueError, match=r"\[chunking\]"):
        load_config(config_path)


@pytest.mark.parametrize("bad_value", ["0", "-5", "true", '"250"'])
def test_chunking_settings_must_be_positive_whole_numbers(
    tmp_path: Path, bad_value: str
) -> None:
    text = VALID_CONFIG.replace("floor_tokens = 250", f"floor_tokens = {bad_value}")
    _, config_path = _write(tmp_path, text)

    with pytest.raises(ValueError, match="chunking.floor_tokens"):
        load_config(config_path)


def test_ceiling_under_twice_the_floor_is_refused(tmp_path: Path) -> None:
    """The halving rule can only promise both halves clear the floor if ceiling >= 2 x floor."""
    text = VALID_CONFIG.replace("ceiling_tokens = 1024", "ceiling_tokens = 400")
    _, config_path = _write(tmp_path, text)

    with pytest.raises(ValueError, match="at least twice"):
        load_config(config_path)
