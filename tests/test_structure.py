"""Tests for the structure index: heading paths, structure vectors and the build.

`heading_paths` and `structure_vector` are tested on hand-made headings and
vectors. The build runs for real in tmp_path over one tiny filing: its
Azure parse is `worked_example_parse` (PART I > Item 1, Item 1A, PART II >
Item 5), its chunks are written directly, and a fake embedder stands in
for Voyage, so no test can spend.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sec_rag.indexing import structure as structure_module
from sec_rag.indexing.embed import embed_corpus
from sec_rag.indexing.structure import (
    build_structure_index,
    heading_paths,
    open_structure_store,
    structure_vector,
)
from sec_rag.ingestion.headings import Heading


def _heading(offset: int, level: int | None, text: str) -> Heading:
    return Heading(offset=offset, page=1, level=level, text=text)


def _chunk(start: int, end: int, chunk_id: str = "D:p0:c0") -> dict:
    return {"chunk_id": chunk_id, "start_offset": start, "end_offset": end}


# --- heading_paths ------------------------------------------------------------


def test_path_is_the_open_headings_in_document_order() -> None:
    headings = [_heading(0, 1, "Item 7"), _heading(10, 2, "Liquidity")]

    paths = heading_paths(headings, [_chunk(20, 50)], max_depth=6)

    assert paths == {"D:p0:c0": ["Item 7", "Liquidity"]}


def test_same_or_higher_level_closes_a_heading() -> None:
    headings = [
        _heading(0, 1, "Item 7"),
        _heading(10, 2, "Liquidity"),
        _heading(20, 1, "Item 8"),
    ]

    paths = heading_paths(headings, [_chunk(30, 50)], max_depth=6)

    assert paths == {"D:p0:c0": ["Item 8"]}


def test_a_chunk_spanning_two_headings_gets_both() -> None:
    """The Draft's example: Part I open, Item 2 and Item 3 start inside the chunk."""
    headings = [
        _heading(0, 1, "Part I"),
        _heading(10, 2, "Item 1"),
        _heading(40, 2, "Item 2"),
        _heading(60, 2, "Item 3"),
    ]

    paths = heading_paths(headings, [_chunk(30, 80)], max_depth=6)

    # Item 1 was closed by Item 2 inside the span, but was open at the
    # chunk's start, so it is also part of the chunk's surroundings.
    assert paths == {"D:p0:c0": ["Part I", "Item 1", "Item 2", "Item 3"]}


def test_a_heading_at_the_chunks_first_character_counts_as_inside() -> None:
    headings = [_heading(0, 1, "Part I"), _heading(30, 2, "Item 2")]

    paths = heading_paths(headings, [_chunk(30, 80)], max_depth=6)

    assert paths == {"D:p0:c0": ["Part I", "Item 2"]}


def test_headings_deeper_than_max_depth_are_dropped() -> None:
    headings = [_heading(level, level, f"H{level}") for level in range(1, 9)]

    paths = heading_paths(headings, [_chunk(20, 30)], max_depth=6)

    assert paths == {"D:p0:c0": ["H1", "H2", "H3", "H4", "H5", "H6"]}


def test_a_deep_heading_opened_inside_the_chunk_is_dropped_too() -> None:
    headings = [_heading(level, level, f"H{level}") for level in range(1, 7)]
    headings.append(_heading(40, 7, "H7"))

    paths = heading_paths(headings, [_chunk(20, 50)], max_depth=6)

    assert "H7" not in paths["D:p0:c0"]


def test_repeated_text_appears_once() -> None:
    headings = [
        _heading(0, 1, "Notes"),
        _heading(40, 1, "Notes"),
    ]

    paths = heading_paths(headings, [_chunk(20, 60)], max_depth=6)

    assert paths == {"D:p0:c0": ["Notes"]}


def test_a_chunk_before_any_heading_has_no_path() -> None:
    headings = [_heading(100, 1, "Part I")]

    paths = heading_paths(headings, [_chunk(0, 50)], max_depth=6)

    assert paths == {}


def test_a_heading_without_a_level_is_refused() -> None:
    headings = [_heading(0, 1, "Part I"), _heading(10, None, "Stray heading")]

    with pytest.raises(ValueError, match="offset 10 has no level"):
        heading_paths(headings, [_chunk(20, 50)], max_depth=6)


# --- structure_vector ---------------------------------------------------------


def test_structure_vector_has_length_one_and_favours_the_closest_heading() -> None:
    chunk = np.array([1.0, 0.0, 0.0])
    headings = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    result = structure_vector(chunk, headings, divisor=0.05)

    assert np.linalg.norm(result) == pytest.approx(1.0)
    # Similarities 1 and 0: weights e^20 / (e^20 + 1) > 0.99 and the rest.
    assert result[0] > 0.99


def test_a_large_divisor_approaches_the_plain_average() -> None:
    chunk = np.array([1.0, 0.0, 0.0])
    headings = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    # Similarities 1 and 0 give weights in the ratio e^(1/1000) : 1, about
    # 1.001 : 1. (At 100 the ratio is still 1.01 : 1, off by 0.0035.)
    result = structure_vector(chunk, headings, divisor=1000.0)

    mean = headings.mean(axis=0)
    assert result == pytest.approx(mean / np.linalg.norm(mean), abs=1e-3)


def test_the_guides_worked_example_weights() -> None:
    """Similarities 0.40, 0.60, 0.55 at divisor 0.05 give weights 0.01, 0.72, 0.27.

    The three headings are the first three axes, so chunk·heading_i is the
    chunk's i-th number; a fourth number makes the chunk length 1 without
    touching those similarities. The result is then the weights themselves,
    rescaled, so dividing by its sum reads them back.
    """
    headings = np.eye(4)[:3]
    fourth = np.sqrt(1 - (0.40**2 + 0.60**2 + 0.55**2))
    chunk = np.array([0.40, 0.60, 0.55, fourth])

    result = structure_vector(chunk, headings, divisor=0.05)

    weights = result[:3] / result[:3].sum()
    assert weights == pytest.approx([0.013, 0.721, 0.265], abs=1e-3)


# --- build_structure_index ----------------------------------------------------

DOC = "AAA_2020_10K"


def _unit(vector: list[float]) -> list[float]:
    return list(np.array(vector) / np.linalg.norm(vector))


class FakeEmbedder:
    """Stands in for Voyage: a fixed vector per text, recording what it was sent."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, texts: list[str]) -> tuple[list[list[float]], int]:
        self.calls.append(list(texts))
        vectors = [_unit([1.0, float(len(text)), 0.5]) for text in texts]
        return vectors, 5 * len(texts)


@pytest.fixture
def structure_config(index_config, worked_example_parse):
    """One filing with worked_example_parse's headings and three chunks, embedded.

    Offsets in the parse: PART I 0, Item 1 10, Item 1A 60, PART II 90,
    Item 5 100. Chunk c0 sits under PART I > Item 1; c1 spans Item 1A's
    start; c2 is under PART II > Item 5.
    """
    config, add_filing = index_config
    add_filing(DOC, None)
    chunks = [
        {"chunk_id": f"{DOC}:p0:c0", "start_offset": 20, "end_offset": 50},
        {"chunk_id": f"{DOC}:p0:c1", "start_offset": 50, "end_offset": 85},
        {"chunk_id": f"{DOC}:p0:c2", "start_offset": 110, "end_offset": 140},
    ]
    company, year, doc_type = DOC.split("_")
    lines = []
    for number, chunk in enumerate(chunks):
        record = {
            **chunk,
            "doc_name": DOC,
            "company": company,
            "year": int(year),
            "doc_type": doc_type,
            "page_index": 0,
            "kind": "prose",
            "text": f"chunk text number {number}" + "!" * number,
        }
        lines.append(json.dumps(record))
    (Path(config["corpus"]["chunks_dir"]) / f"{DOC}.jsonl").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (Path(config["corpus"]["parsed_dir"]) / f"{DOC}.json").write_text(
        json.dumps(worked_example_parse), encoding="utf-8"
    )
    embed_corpus(config, execute_paid=True, embedder=FakeEmbedder())
    return config


def test_dry_run_reports_and_sends_nothing(structure_config) -> None:
    embedder = FakeEmbedder()

    report = build_structure_index(structure_config, embedder=embedder)

    assert embedder.calls == []
    assert report == {
        "filings": 1,
        "chunks": 3,
        "chunks_with_path": 3,
        # PART I, Item 1, Item 1A, PART II, Item 5
        "unique_headings": 5,
        "stored_headings": 0,
        "to_embed": 5,
        "estimated_tokens": report["estimated_tokens"],
    }
    assert report["estimated_tokens"] > 0
    with pytest.raises(ValueError, match="not built"):
        open_structure_store(structure_config)


def test_paid_build_saves_one_vector_per_chunk_with_its_path(structure_config) -> None:
    report = build_structure_index(
        structure_config, execute_paid=True, embedder=FakeEmbedder()
    )

    assert report["embedded"] == 5
    assert report["tokens_billed"] == 25
    assert report["structure_rows"] == 3
    saved = open_structure_store(structure_config).get(
        include=["embeddings", "metadatas"]
    )
    by_id = dict(zip(saved["ids"], saved["metadatas"], strict=True))
    assert by_id[f"{DOC}:p0:c0"]["heading_path"] == "PART I > Item 1. Business"
    assert by_id[f"{DOC}:p0:c1"]["heading_path"] == (
        "PART I > Item 1. Business > Item 1A. Risk Factors"
    )
    assert by_id[f"{DOC}:p0:c2"]["heading_path"] == "PART II > Item 5. Market for ..."
    assert all(metadata["doc_name"] == DOC for metadata in saved["metadatas"])
    for vector in saved["embeddings"]:
        assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-5)


def test_paid_build_embeds_only_missing_headings(structure_config) -> None:
    build_structure_index(structure_config, execute_paid=True, embedder=FakeEmbedder())
    second = FakeEmbedder()

    report = build_structure_index(structure_config, execute_paid=True, embedder=second)

    assert second.calls == []
    assert report["to_embed"] == 0
    assert report["structure_rows"] == 3


def test_a_free_rerun_rebuilds_locally_once_every_heading_is_stored(
    structure_config,
) -> None:
    """A divisor change needs no Voyage call: the free command rebuilds."""
    build_structure_index(structure_config, execute_paid=True, embedder=FakeEmbedder())
    structure_config["structure"]["softmax_divisor"] = 0.1

    report = build_structure_index(structure_config)

    assert report["structure_rows"] == 3
    assert open_structure_store(structure_config).metadata["softmax_divisor"] == 0.1


def test_build_marks_the_collection_complete_with_its_settings(
    structure_config,
) -> None:
    build_structure_index(structure_config, execute_paid=True, embedder=FakeEmbedder())

    metadata = open_structure_store(structure_config).metadata

    assert metadata == {"max_depth": 6, "softmax_divisor": 0.05, "complete": True}


@pytest.mark.parametrize(("key", "value"), [("max_depth", 5), ("softmax_divisor", 0.1)])
def test_open_structure_store_refuses_other_settings(
    structure_config, key: str, value: float
) -> None:
    build_structure_index(structure_config, execute_paid=True, embedder=FakeEmbedder())
    structure_config["structure"][key] = value

    with pytest.raises(ValueError, match="rebuild"):
        open_structure_store(structure_config)


def test_open_structure_store_refuses_an_incomplete_build(
    structure_config, monkeypatch
) -> None:
    """A crash mid-rebuild leaves the new collection marked incomplete."""
    build_structure_index(structure_config, execute_paid=True, embedder=FakeEmbedder())

    def crash(*args, **kwargs):
        raise RuntimeError("crashed mid-build")

    monkeypatch.setattr(structure_module, "_save_filing_vectors", crash)
    with pytest.raises(RuntimeError, match="mid-build"):
        build_structure_index(structure_config)

    with pytest.raises(ValueError, match="did not finish"):
        open_structure_store(structure_config)
