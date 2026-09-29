"""Tests for search: the ONE retrieval function, its two rankings, and RRF.

Two tiny filings are indexed for real in tmp_path: BM25 with
build_bm25_index, Chroma with embed_corpus and a fake embedder. A fake
Voyage client embeds the question, so no test can spend.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import chromadb
import numpy as np
import pytest

from sec_rag.indexing.bm25_index import build_bm25_index
from sec_rag.indexing.embed import embed_corpus
from sec_rag.retrieval.search import (
    chroma_ranking,
    exact_ranking,
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


def test_search_uses_chroma_at_weight_zero_and_the_scorer_otherwise(indexes) -> None:
    """Weight 0 is Exp1 exactly (Chroma); above 0 the semantic list is exact_ranking's."""
    search_indexes, _ = indexes
    search_indexes.structure = _structure_store(search_indexes, {})
    vector = _vector("capex")

    at_zero = search(
        "", "capex", method="semantic", doc_name=None, top_k=50, indexes=search_indexes
    )
    at_one = search(
        "",
        "capex",
        method="semantic",
        doc_name=None,
        top_k=50,
        indexes=search_indexes,
        structure_weight=1.0,
    )

    assert at_zero["ranked"] == chroma_ranking(search_indexes, vector, None)
    assert at_one["ranked"] == exact_ranking(search_indexes, vector, None, 1.0)
    assert at_one["embedding_tokens"] == 7


def test_a_negative_structure_weight_is_refused(indexes) -> None:
    search_indexes, _ = indexes

    with pytest.raises(ValueError, match="at least 0"):
        search(
            "q",
            "q",
            method="hybrid",
            doc_name=None,
            top_k=10,
            indexes=search_indexes,
            structure_weight=-1.0,
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


# --- exact_ranking: Exp2's scorer -----------------------------------------------


def _structure_store(search_indexes, vectors: dict[str, list[float]]):
    """A structure collection holding the given (chunk ID -> vector) rows."""
    client = chromadb.PersistentClient(
        path=str(Path(search_indexes.config["corpus"]["indexes_dir"]) / "chroma")
    )
    collection = client.get_or_create_collection(
        "structure_test", configuration={"hnsw": {"space": "cosine"}}
    )
    if vectors:
        collection.upsert(
            ids=list(vectors),
            embeddings=np.array(list(vectors.values())),
            metadatas=[{"doc_name": chunk_id.split(":")[0]} for chunk_id in vectors],
        )
    return collection


def _cosines(search_indexes, query: list[float], doc_name=None) -> dict[str, float]:
    """numpy's question·chunk cosine for every stored chunk (optionally one filing's)."""
    where = {"doc_name": doc_name} if doc_name else None
    rows = search_indexes.store.client.get(where=where, include=["embeddings"])
    q = np.array(query) / np.linalg.norm(query)
    return {
        chunk_id: float(np.array(vector) @ q / np.linalg.norm(vector))
        for chunk_id, vector in zip(rows["ids"], rows["embeddings"], strict=True)
    }


def test_exact_ranking_at_weight_zero_ranks_by_cosine(indexes) -> None:
    search_indexes, _ = indexes
    query = _vector("How much was capital expenditure?")

    ranked = exact_ranking(search_indexes, query, None, 0.0)

    expected = sorted(
        _cosines(search_indexes, query).items(), key=lambda pair: (-pair[1], pair[0])
    )
    assert [chunk_id for chunk_id, _ in ranked] == [
        chunk_id for chunk_id, _ in expected
    ]
    assert _scores(ranked) == pytest.approx(_scores(expected))


def test_structure_lifts_a_chunk_with_a_matching_heading(indexes) -> None:
    """score = q·chunk + weight × q·structure; a structure row parallel to q lifts its chunk."""
    search_indexes, _ = indexes
    query = [0.0, 0.0, 1.0]
    aaa = [f"AAA_2020_10K:p0:c{n}" for n in range(3)]
    # Stored chunk vectors are [1, len(text), 0.5], so against q = [0, 0, 1]
    # the shortest text scores highest on content and c1 (the longest) last.
    search_indexes.structure = _structure_store(
        search_indexes,
        {
            aaa[0]: [1.0, 0.0, 0.0],  # q·structure = 0
            aaa[1]: [0.0, 0.0, 1.0],  # q·structure = 1
            aaa[2]: [1.0, 0.0, 0.0],
        },
    )
    content = _cosines(search_indexes, query, "AAA_2020_10K")

    at_zero = exact_ranking(search_indexes, query, "AAA_2020_10K", 0.0)
    at_one = exact_ranking(search_indexes, query, "AAA_2020_10K", 1.0)

    scores = dict(at_one)
    assert scores[aaa[1]] == pytest.approx(content[aaa[1]] + 1.0)
    assert scores[aaa[0]] == pytest.approx(content[aaa[0]])
    assert at_zero[-1][0] == aaa[1]  # last on content alone
    assert at_one[0][0] == aaa[1]  # first once its heading counts


def test_a_chunk_without_a_structure_row_keeps_its_content_score(indexes) -> None:
    search_indexes, _ = indexes
    query = [0.0, 0.0, 1.0]
    search_indexes.structure = _structure_store(
        search_indexes, {"AAA_2020_10K:p0:c0": [0.0, 0.0, 1.0]}
    )
    content = _cosines(search_indexes, query)

    scores = dict(exact_ranking(search_indexes, query, None, 1.0))

    assert scores["AAA_2020_10K:p0:c1"] == pytest.approx(content["AAA_2020_10K:p0:c1"])
    assert scores["AAA_2020_10K:p0:c0"] == pytest.approx(
        content["AAA_2020_10K:p0:c0"] + 1.0
    )


def test_exact_ranking_respects_the_filter(indexes) -> None:
    search_indexes, _ = indexes
    search_indexes.structure = _structure_store(
        search_indexes, {"BBB_2020_10K:p0:c0": [0.0, 0.0, 1.0]}
    )

    ranked = exact_ranking(search_indexes, [0.0, 0.0, 1.0], "AAA_2020_10K", 1.0)

    assert sorted(chunk_id for chunk_id, _ in ranked) == [
        f"AAA_2020_10K:p0:c{n}" for n in range(3)
    ]


def test_no_filter_scores_every_chunk(indexes) -> None:
    """Shared-store with no chosen filing searches all chunks, as Exp1 does."""
    search_indexes, _ = indexes

    ranked = exact_ranking(search_indexes, [1.0, 1.0, 1.0], None, 0.0)

    assert len(ranked) == len(AAA_TEXTS) + len(BBB_TEXTS)


def test_a_non_zero_weight_without_the_structure_collection_is_refused(
    indexes,
) -> None:
    search_indexes, _ = indexes

    with pytest.raises(ValueError, match="structure collection"):
        exact_ranking(search_indexes, [1.0, 1.0, 1.0], None, 1.0)


def test_open_search_indexes_opens_the_structure_collection_only_above_zero(
    index_config,
) -> None:
    """Exp1 (weight 0) never needs it built; rung B fails before any question."""
    config, add_filing = index_config
    config["retrieval"] = {"candidates_per_search": 50, "rrf_k": 60}
    add_filing("AAA_2020_10K", AAA_TEXTS)
    build_bm25_index(config)

    assert open_search_indexes(config, None).structure is None
    config["structure"]["weight"] = 1.0
    with pytest.raises(ValueError, match="index-structure"):
        open_search_indexes(config, None)


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
