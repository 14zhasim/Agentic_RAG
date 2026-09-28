"""Tests for search: the ONE retrieval function, its two rankings, and RRF.

Two tiny filings are indexed for real in tmp_path: BM25 with
build_bm25_index, Chroma with embed_corpus and a fake embedder. A fake
Voyage client embeds the question, so no test can spend.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sec_rag.indexing.bm25_index import build_bm25_index
from sec_rag.indexing.embed import embed_corpus
from sec_rag.retrieval.search import (
    open_search_indexes,
    reciprocal_rank_fusion,
    search,
)

AAA_TEXTS = [
    "Capital expenditure was 100 million.",
    "Revenue grew strongly across segments.",
    "The board declared a dividend.",
]
BBB_TEXTS = [
    "Capital expenditure was 200 million.",
    "Goodwill impairment was recorded.",
]


def _vector(text: str) -> list[float]:
    """The same 3-number vector the chunks were stored with (FakeEmbedder below)."""
    return [1.0, float(len(text)), 0.5]


class FakeEmbedder:
    """Stands in for Voyage when storing chunks: one fixed vector per text."""

    def __call__(self, texts: list[str]) -> tuple[list[list[float]], int]:
        return [_vector(text) for text in texts], 10 * len(texts)


class FakeVoyage:
    """Stands in for voyageai.Client when embedding the question."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def embed(self, texts: list[str], **kwargs) -> SimpleNamespace:
        self.calls.append({"texts": texts, **kwargs})
        return SimpleNamespace(
            embeddings=[_vector(text) for text in texts], total_tokens=7
        )


@pytest.fixture
def indexes(index_config):
    """Both indexes over AAA_2020_10K and BBB_2020_10K, opened for searching."""
    config, add_filing = index_config
    config["retrieval"] = {
        "candidates_per_search": 50,
        "rrf_k": 60,
        "rerank_candidates": 50,
    }
    add_filing("AAA_2020_10K", AAA_TEXTS)
    add_filing("BBB_2020_10K", BBB_TEXTS)
    build_bm25_index(config)
    embed_corpus(config, execute_paid=True, embedder=FakeEmbedder())
    voyage = FakeVoyage()
    return open_search_indexes(config, voyage), voyage


def _scores(ranked: list[tuple[str, float]]) -> list[float]:
    return [score for _chunk_id, score in ranked]


# --- search: the ONE retrieval function -------------------------------------


def test_hybrid_returns_rrf_of_both_rankings(indexes) -> None:
    search_indexes, _ = indexes
    bm25 = search(
        "capital expenditure",
        "",
        method="bm25",
        doc_name=None,
        top_k=50,
        indexes=search_indexes,
    )["ranked"]
    semantic = search(
        "",
        "capital expenditure",
        method="semantic",
        doc_name=None,
        top_k=50,
        indexes=search_indexes,
    )["ranked"]

    hybrid = search(
        "capital expenditure",
        "capital expenditure",
        method="hybrid",
        doc_name=None,
        top_k=50,
        indexes=search_indexes,
    )

    expected = reciprocal_rank_fusion(
        [[chunk_id for chunk_id, _ in bm25], [chunk_id for chunk_id, _ in semantic]],
        k=60,
    )
    assert hybrid["ranked"] == expected
    assert hybrid["embedding_tokens"] == 7


def test_top_k_cuts_the_list(indexes) -> None:
    search_indexes, _ = indexes

    result = search(
        "capital",
        "capital",
        method="hybrid",
        doc_name=None,
        top_k=2,
        indexes=search_indexes,
    )

    assert len(result["ranked"]) == 2


def test_structure_weight_is_refused_until_exp2(indexes) -> None:
    search_indexes, _ = indexes

    with pytest.raises(NotImplementedError, match="Exp2"):
        search(
            "q",
            "q",
            method="hybrid",
            doc_name=None,
            top_k=10,
            indexes=search_indexes,
            structure_weight=0.5,
        )


def test_unknown_method_is_refused(indexes) -> None:
    search_indexes, _ = indexes

    with pytest.raises(ValueError, match="method"):
        search(
            "q", "q", method="keyword", doc_name=None, top_k=10, indexes=search_indexes
        )


def test_semantic_without_a_voyage_client_is_refused(index_config) -> None:
    """The free spot check opens the indexes without Voyage; it may only use BM25."""
    config, add_filing = index_config
    config["retrieval"] = {
        "candidates_per_search": 50,
        "rrf_k": 60,
        "rerank_candidates": 50,
    }
    add_filing("AAA_2020_10K", AAA_TEXTS)
    build_bm25_index(config)
    search_indexes = open_search_indexes(config, None)

    with pytest.raises(ValueError, match="Voyage"):
        search(
            "q", "q", method="semantic", doc_name=None, top_k=10, indexes=search_indexes
        )
    bm25_only = search(
        "dividend", "", method="bm25", doc_name=None, top_k=10, indexes=search_indexes
    )
    assert bm25_only["ranked"][0][0] == "AAA_2020_10K:p0:c2"


# --- BM25 ranking -------------------------------------------------------------


def test_bm25_filter_keeps_only_the_chosen_filing_and_no_zero_scores(indexes) -> None:
    """Only one AAA chunk mentions capital expenditure. A filtered search still
    pads to 50 with score-0 chunks (the filter is a weight mask); none survive."""
    search_indexes, _ = indexes

    result = search(
        "capital expenditure",
        "",
        method="bm25",
        doc_name="AAA_2020_10K",
        top_k=50,
        indexes=search_indexes,
    )

    assert [chunk_id for chunk_id, _ in result["ranked"]] == ["AAA_2020_10K:p0:c0"]
    assert all(score > 0 for score in _scores(result["ranked"]))
    assert result["embedding_tokens"] == 0


def test_bm25_ranking_is_best_first(indexes) -> None:
    search_indexes, _ = indexes

    ranked = search(
        "capital expenditure 200 million",
        "",
        method="bm25",
        doc_name=None,
        top_k=50,
        indexes=search_indexes,
    )["ranked"]

    assert ranked[0][0] == "BBB_2020_10K:p0:c0"  # matches every word
    assert _scores(ranked) == sorted(_scores(ranked), reverse=True)


# --- vector ranking -----------------------------------------------------------


def test_semantic_embeds_as_a_query_and_respects_the_filter(indexes) -> None:
    search_indexes, voyage = indexes

    result = search(
        "",
        "How much was capital expenditure?",
        method="semantic",
        doc_name="AAA_2020_10K",
        top_k=50,
        indexes=search_indexes,
    )

    assert sorted(chunk_id for chunk_id, _ in result["ranked"]) == [
        f"AAA_2020_10K:p0:c{n}" for n in range(3)
    ]
    assert _scores(result["ranked"]) == sorted(_scores(result["ranked"]), reverse=True)
    assert result["embedding_tokens"] == 7
    assert voyage.calls[0]["texts"] == ["How much was capital expenditure?"]
    assert voyage.calls[0]["input_type"] == "query"
    assert voyage.calls[0]["truncation"] is False


# --- reciprocal rank fusion ---------------------------------------------------


def test_rrf_worked_example_from_the_design() -> None:
    """Draft -> Retrieve -> rrf: ranks 1 + 5 (0.0318) beat rank 2 alone (0.0161)."""
    bm25 = ["A", "B"]
    chroma = ["c1", "c2", "c3", "c4", "A"]

    fused = dict(reciprocal_rank_fusion([bm25, chroma], k=60))

    assert fused["A"] == pytest.approx(1 / 61 + 1 / 65)
    assert round(fused["A"], 4) == 0.0318
    assert round(fused["B"], 4) == 0.0161
    assert fused["A"] > fused["B"]


def test_rrf_is_best_first_with_ties_broken_by_chunk_id() -> None:
    fused = reciprocal_rank_fusion([["b"], ["a"]], k=60)

    assert fused == [("a", pytest.approx(1 / 61)), ("b", pytest.approx(1 / 61))]
