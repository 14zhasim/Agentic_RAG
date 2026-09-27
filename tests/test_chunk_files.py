"""Tests for chunk_files: writing, reading back and summarising chunk files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from sec_rag.chunking.chunk_files import build_chunks, chunks_path, read_chunks

PROSE = "Revenue grew in every segment this year because demand for industrial products stayed strong."
TABLE = "<table><tr><td>Revenue</td><td>32,765</td></tr></table>"
LOGO = "<figure>\n</figure>"

RECORD_FIELDS = {
    "chunk_id",
    "doc_name",
    "company",
    "year",
    "doc_type",
    "page_index",
    "kind",
    "start_offset",
    "end_offset",
    "carried_text",
    "text",
    "token_count",
    "heading_path",
}


def _config(tmp_path: Path) -> dict[str, Any]:
    """A config pointing at empty pdfs/, parsed/ and chunks/ folders under tmp_path."""
    (tmp_path / "pdfs").mkdir()
    (tmp_path / "parsed").mkdir()
    return {
        "corpus": {
            "prepared_dir": str(tmp_path),
            "parsed_dir": str(tmp_path / "parsed"),
            "chunks_dir": str(tmp_path / "chunks"),
        },
        "chunking": {"floor_tokens": 10, "ceiling_tokens": 40, "overlap_cap_tokens": 6},
    }


def _add_filing(config: dict[str, Any], doc_name: str, parsed: bool = True) -> None:
    """Add a one-page filing: an empty stand-in PDF, and (optionally) its saved parse.

    `select_documents` only lists PDF names, so an empty file is enough.
    """
    Path(config["corpus"]["prepared_dir"], "pdfs", f"{doc_name}.pdf").write_bytes(b"")
    if not parsed:
        return
    content = f"{PROSE}\n\n{TABLE}\n\n{LOGO}"
    raw = {
        "content": content,
        "pages": [{"pageNumber": 1, "spans": [{"offset": 0, "length": len(content)}]}],
        "paragraphs": [],
        "sections": [],
        "tables": [{"spans": [{"offset": content.index(TABLE), "length": len(TABLE)}]}],
        "figures": [{"spans": [{"offset": content.index(LOGO), "length": len(LOGO)}]}],
    }
    path = Path(config["corpus"]["parsed_dir"], f"{doc_name}.json")
    path.write_text(json.dumps(raw), encoding="utf-8")


def test_build_chunks_writes_one_jsonl_per_filing(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _add_filing(config, "3M_2018_10K")
    _add_filing(config, "PEPSICO_2022_10K")

    build_chunks(config)

    for doc_name in ("3M_2018_10K", "PEPSICO_2022_10K"):
        lines = chunks_path(config, doc_name).read_text(encoding="utf-8").splitlines()
        records = [json.loads(line) for line in lines]
        assert [record["kind"] for record in records] == ["prose", "table"]
        assert all(set(record) == RECORD_FIELDS for record in records)
        assert records[0]["doc_name"] == doc_name


def test_read_chunks_round_trips(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _add_filing(config, "3M_2018_10K")
    build_chunks(config)

    records = read_chunks(config, "3M_2018_10K")

    assert [record["chunk_id"] for record in records] == [
        "3M_2018_10K:p0:c0",
        "3M_2018_10K:p0:c1",
    ]
    assert records[1]["text"] == TABLE


def test_read_chunks_names_a_filing_not_chunked_yet(tmp_path: Path) -> None:
    config = _config(tmp_path)

    with pytest.raises(ValueError, match="3M_2018_10K: not chunked yet"):
        read_chunks(config, "3M_2018_10K")


def test_unparsed_filing_writes_nothing(tmp_path: Path) -> None:
    """One unparsed filing stops the run before any chunk file is written."""
    config = _config(tmp_path)
    _add_filing(config, "3M_2018_10K")
    _add_filing(config, "PEPSICO_2022_10K", parsed=False)

    with pytest.raises(ValueError, match="PEPSICO_2022_10K: not parsed yet"):
        build_chunks(config)

    assert not Path(config["corpus"]["chunks_dir"]).exists()


def test_only_the_named_filings_are_rebuilt(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _add_filing(config, "3M_2018_10K")
    _add_filing(config, "PEPSICO_2022_10K")

    summary = build_chunks(config, ("3M_2018_10K",))

    assert summary["filings"] == 1
    assert chunks_path(config, "3M_2018_10K").is_file()
    assert not chunks_path(config, "PEPSICO_2022_10K").exists()


def test_rebuild_replaces_the_file_and_leaves_no_partial(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _add_filing(config, "3M_2018_10K")
    path = chunks_path(config, "3M_2018_10K")
    path.parent.mkdir()
    path.write_text("stale\n", encoding="utf-8")

    build_chunks(config)

    assert "stale" not in path.read_text(encoding="utf-8")
    assert list(path.parent.glob("*.partial")) == []


def test_summary_counts(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _add_filing(config, "3M_2018_10K")

    summary = build_chunks(config)

    assert summary["filings"] == 1
    assert summary["chunks"] == 2
    assert summary["kinds"] == {"prose": 1, "table": 1, "figure": 0}
    assert summary["median_prose_tokens"] == 15
    assert summary["prose_in_range_share"] == 1.0
    assert summary["prose_over_ceiling"] == 0
    assert summary["prose_under_floor"] == 0
    assert summary["empty_figures_skipped"] == 1
