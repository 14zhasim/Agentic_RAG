"""Tests for the BM25 index: built from chunk files, saved, and reloaded with our settings."""

from __future__ import annotations

from pathlib import Path

import pytest
from llama_index.retrievers.bm25 import BM25Retriever

from sec_rag.indexing.bm25_index import (
    build_bm25_index,
    load_bm25_index,
    words_only,
)

TEXTS = [
    "Capital expenditure in fiscal 2018 was 1,577 million.",
    "Revenue grew across all business segments.",
    "The board declared a quarterly dividend.",
]


def test_build_saves_the_index_and_reports_counts(index_config) -> None:
    config, add_filing = index_config
    add_filing("3M_2018_10K", TEXTS)
    add_filing("AMD_2022_10K", ["Data center revenue rose."])

    summary = build_bm25_index(config)

    assert summary == {"filings": 2, "chunks": 4}
    bm25_dir = Path(config["corpus"]["indexes_dir"]) / "bm25"
    assert (bm25_dir / "retriever.json").is_file()
    assert not bm25_dir.with_name("bm25.partial").exists()


def test_bm25_finds_fiscal_year_after_reload(index_config) -> None:
    """Finding 1: the reloaded retriever still splits FY2018 into fy + 2018."""
    config, add_filing = index_config
    add_filing("3M_2018_10K", TEXTS)
    build_bm25_index(config)

    results = load_bm25_index(config).retrieve("FY2018")

    assert results[0].node.node_id == "3M_2018_10K:p0:c0"
    assert results[0].score > 0


def test_from_persist_dir_would_lose_the_pattern(index_config) -> None:
    """Why load_bm25_index exists: the wrapper's own reload keeps fy2018 whole."""
    config, add_filing = index_config
    add_filing("3M_2018_10K", TEXTS)
    build_bm25_index(config)
    bm25_dir = Path(config["corpus"]["indexes_dir"]) / "bm25"

    results = BM25Retriever.from_persist_dir(str(bm25_dir)).retrieve("FY2018")

    assert all(result.score == 0 for result in results)


def test_bm25_does_not_match_on_metadata(index_config) -> None:
    """Finding 2: the company name is metadata, not text, so it isn't a keyword."""
    config, add_filing = index_config
    add_filing("Boeing_2022_10K", TEXTS)
    build_bm25_index(config)

    results = load_bm25_index(config).retrieve("Boeing")

    assert all(result.score == 0 for result in results)


def test_results_keep_the_node_metadata(index_config) -> None:
    config, add_filing = index_config
    add_filing("3M_2018_10K", TEXTS, page_index=59)
    build_bm25_index(config)

    [top, *_] = load_bm25_index(config).retrieve("dividend")

    assert top.node.node_id == "3M_2018_10K:p0:c2"
    assert top.node.metadata["doc_name"] == "3M_2018_10K"
    assert top.node.metadata["page_index"] == 59


def test_rebuild_replaces_the_old_index(index_config) -> None:
    config, add_filing = index_config
    add_filing("3M_2018_10K", TEXTS)
    build_bm25_index(config)
    add_filing("3M_2018_10K", ["Only one chunk now mentions dividends."])

    build_bm25_index(config)

    results = load_bm25_index(config).retrieve("dividend")
    assert [result.node.node_id for result in results] == ["3M_2018_10K:p0:c0"]


def test_words_only_removes_table_markup_and_entities() -> None:
    text = "<table><tr><th>(Millions)</th></tr><tr><td>PP&amp;E</td></tr></table>"

    assert words_only(text).split() == ["(Millions)", "PP&E"]


def test_words_only_leaves_a_less_than_sign_in_prose_alone() -> None:
    assert words_only("margins < 5% of <b>sales") == "margins < 5% of <b>sales"


def test_table_markup_is_not_a_keyword(index_config) -> None:
    """Finding 6: 'td' is not a word BM25 can match, so tables aren't lengthened by it."""
    config, add_filing = index_config
    add_filing(
        "3M_2018_10K", ["<table><tr><td>Capital</td><td>1,577</td></tr></table>"]
    )
    build_bm25_index(config)

    [result] = load_bm25_index(config).retrieve("td tr table")

    assert result.score == 0


def test_load_before_build_is_an_error(index_config) -> None:
    config, _ = index_config

    with pytest.raises(ValueError, match="BM25 index not built yet"):
        load_bm25_index(config)


def test_unchunked_filing_stops_the_build(index_config) -> None:
    config, add_filing = index_config
    add_filing("3M_2018_10K", TEXTS)
    add_filing("AMD_2022_10K", None)

    with pytest.raises(ValueError, match="AMD_2022_10K: not chunked yet"):
        build_bm25_index(config)
