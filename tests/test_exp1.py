"""Tests for retrieve_exp1: Exp1's whole retrieval path for one question.

Two tiny filings are indexed for real in tmp_path (as in test_search.py).
Fake OpenRouter and Voyage clients stand in for GLM, the query embedding
and the reranker, so no test can spend.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from sec_rag.indexing.bm25_index import build_bm25_index
from sec_rag.indexing.embed import embed_corpus
from sec_rag.retrieval.exp1 import (
    Exp1Resources,
    _choose_filter,
    load_chunk_lookup,
    load_saved_plans,
    retrieve_exp1,
)
from sec_rag.retrieval.search import open_search_indexes

TABLE = "<table><tr><td>Capital expenditure</td><td>100</td></tr></table>"
AAA_TEXTS = [
    TABLE,
    "Capital expenditure was 100 million in fiscal 2020.",
    "Revenue grew strongly across segments.",
    "The board declared a dividend.",
]
BBB_TEXTS = [
    "Capital expenditure was 200 million.",
    "Goodwill impairment was recorded.",
]
BOTH = ("AAA_2020_10K", "BBB_2020_10K")
QUESTION = "What was AAA's FY2020 capex?"


def _vector(text: str) -> list[float]:
    return [1.0, float(len(text)), 0.5]


class FakeEmbedder:
    """Stands in for Voyage when storing chunks."""

    def __call__(self, texts: list[str]) -> tuple[list[list[float]], int]:
        return [_vector(text) for text in texts], 10 * len(texts)


class FakeVoyage:
    """Stands in for voyageai.Client: embeds the query, reranks last-first."""

    def __init__(self) -> None:
        self.rerank_calls: list[dict] = []

    def embed(self, texts: list[str], **kwargs) -> SimpleNamespace:
        return SimpleNamespace(
            embeddings=[_vector(text) for text in texts], total_tokens=7
        )

    def rerank(self, query: str, documents: list[str], **kwargs) -> SimpleNamespace:
        self.rerank_calls.append({"query": query, "documents": documents, **kwargs})
        order = list(reversed(range(len(documents))))[: kwargs["top_k"]]
        results = [
            SimpleNamespace(index=index, relevance_score=1.0 - 0.1 * rank)
            for rank, index in enumerate(order)
        ]
        return SimpleNamespace(results=results, total_tokens=321)


class FakeUsage:
    def model_dump(self) -> dict:
        return {"input_tokens": 900, "output_tokens": 120, "cost": 0.0004}


class FakeOpenRouter:
    """Stands in for openai.OpenAI: always returns the given search plan."""

    def __init__(self, reply: dict | None) -> None:
        self.reply_text = json.dumps(reply) if reply is not None else "not json"
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kwargs) -> SimpleNamespace:
        return SimpleNamespace(
            output_text=self.reply_text, model="z-ai/glm-5.3-flash", usage=FakeUsage()
        )


def _reply(filename: str | None) -> dict:
    return {
        "filename": filename,
        "keyword_query": "AAA 2020 capex capital expenditure",
        "semantic_query": "How much did AAA spend on capital expenditure in 2020?",
    }


@pytest.fixture
def make_resources(index_config):
    """Build Exp1Resources over the tiny indexes with a chosen GLM reply."""
    config, add_filing = index_config
    config["retrieval"] = {
        "candidates_per_search": 50,
        "rrf_k": 60,
        "rerank_candidates": 50,
    }
    config["query_enhancement"] = {
        "model": "z-ai/glm-5.3-flash",
        "upstream_provider": "z-ai",
        "reasoning_effort": "low",
        "temperature": 0.0,
        "max_output_tokens": 4096,
    }
    config["rerank"] = {"model": "rerank-3-lite", "usd_per_million_tokens": 0.02}
    config["embedding"]["usd_per_million_tokens"] = 0.02
    add_filing("AAA_2020_10K", AAA_TEXTS)
    add_filing("BBB_2020_10K", BBB_TEXTS)
    build_bm25_index(config)
    embed_corpus(config, execute_paid=True, embedder=FakeEmbedder())

    def make(reply: dict | None) -> tuple[Exp1Resources, FakeVoyage]:
        voyage = FakeVoyage()
        resources = Exp1Resources(
            indexes=open_search_indexes(config, voyage),
            chunks=load_chunk_lookup(config),
            openrouter=FakeOpenRouter(reply),
            voyage=voyage,
            config=config,
        )
        return resources, voyage

    return make


# --- retrieve_exp1 ------------------------------------------------------------


def test_retrieve_exp1_returns_the_bundle_in_the_benchmarks_shape(
    make_resources,
) -> None:
    resources, _voyage = make_resources(_reply("AAA_2020_10K"))

    bundle = retrieve_exp1(QUESTION, BOTH, 3, resources)

    assert len(bundle["chunks"]) == 3
    for chunk in bundle["chunks"] + bundle["pre_rerank_chunks"]:
        assert set(chunk) == {"chunk_id", "doc_name", "pages", "rank", "score", "text"}
        assert chunk["pages"] == [0]
    assert [chunk["rank"] for chunk in bundle["chunks"]] == [1, 2, 3]
    assert bundle["filter_doc_name"] == "AAA_2020_10K"
    assert bundle["filter_status"] == "chosen"
    assert bundle["search_plan"]["enhancement_status"] == "ok"
    assert bundle["usage"]["embedding_tokens"] == 7
    assert bundle["usage"]["rerank_tokens"] == 321
    assert bundle["latency_seconds"] >= 0


def test_retrieve_exp1_usage_has_list_price_usd(make_resources) -> None:
    """Build Order 2.5: Voyage cost as tokens x list price ($0.02/1M here)."""
    resources, _voyage = make_resources(_reply("AAA_2020_10K"))

    usage = retrieve_exp1(QUESTION, BOTH, 3, resources)["usage"]

    assert usage["embedding_cost_usd"] == pytest.approx(7 * 0.02 / 1_000_000)
    assert usage["rerank_cost_usd"] == pytest.approx(321 * 0.02 / 1_000_000)


def test_the_filter_keeps_every_chunk_inside_the_chosen_filing(
    make_resources,
) -> None:
    resources, voyage = make_resources(_reply("AAA_2020_10K"))

    bundle = retrieve_exp1(QUESTION, BOTH, 10, resources)

    # Only AAA's chunks reach the reranker, so none of BBB's can come back.
    assert len(voyage.rerank_calls[0]["documents"]) <= len(AAA_TEXTS)
    assert {chunk["doc_name"] for chunk in bundle["chunks"]} == {"AAA_2020_10K"}


def test_chunks_carry_the_original_text_with_html_kept(make_resources) -> None:
    """BM25's copy has its HTML stripped; the answer model must see the table."""
    resources, _voyage = make_resources(_reply("AAA_2020_10K"))

    bundle = retrieve_exp1(QUESTION, BOTH, 10, resources)

    texts = [chunk["text"] for chunk in bundle["chunks"]]
    assert TABLE in texts


def test_pre_rerank_is_the_fused_order_and_the_reranker_reorders_it(
    make_resources,
) -> None:
    resources, voyage = make_resources(_reply("AAA_2020_10K"))

    bundle = retrieve_exp1(QUESTION, BOTH, 2, resources)

    sent = voyage.rerank_calls[0]["documents"]
    pre = bundle["pre_rerank_chunks"]
    assert [chunk["text"] for chunk in pre] == sent[:2]
    assert [chunk["rank"] for chunk in pre] == [1, 2]
    # FakeVoyage ranks the last text first, so the final order is reversed.
    assert bundle["chunks"][0]["text"] == sent[-1]
    assert bundle["chunks"][0]["score"] == 1.0


def test_the_reranker_is_sent_the_original_question_and_top_k(
    make_resources,
) -> None:
    """Draft -> Retrieve: rerank against the question, not GLM's rewrite."""
    resources, voyage = make_resources(_reply("AAA_2020_10K"))

    retrieve_exp1(QUESTION, BOTH, 3, resources)

    assert voyage.rerank_calls[0]["query"] == QUESTION
    assert voyage.rerank_calls[0]["top_k"] == 3


def test_an_invalid_reply_still_retrieves_over_the_whole_scope(
    make_resources,
) -> None:
    resources, _voyage = make_resources(None)

    bundle = retrieve_exp1(QUESTION, BOTH, 10, resources)

    assert bundle["search_plan"]["enhancement_status"] == "invalid_reply"
    assert bundle["filter_doc_name"] is None
    assert bundle["filter_status"] == "fallback"
    assert bundle["chunks"]


# --- reusing rung A's saved plans (Exp2 rung B) --------------------------------


class FailingOpenRouter:
    """Fails if GLM is called: rung B must reuse the saved plan instead."""

    def __init__(self) -> None:
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kwargs) -> SimpleNamespace:
        raise AssertionError("GLM was called although a saved plan exists")


def _saved_plan(filename: str | None) -> dict:
    return {
        **_reply(filename),
        "enhancement_status": "ok",
        "call": {"cost": 0.0004, "reply_text": "{...}"},
        "reused_from": "run-a",
    }


def test_saved_plan_is_used_and_glm_is_not_called(make_resources) -> None:
    resources, _voyage = make_resources(None)
    resources.openrouter = FailingOpenRouter()
    resources.saved_plans = {(QUESTION, "shared_store"): _saved_plan("AAA_2020_10K")}

    bundle = retrieve_exp1(QUESTION, BOTH, 3, resources)

    assert bundle["search_plan"] == _saved_plan("AAA_2020_10K")
    assert bundle["filter_doc_name"] == "AAA_2020_10K"


def test_the_scope_picks_the_condition_of_the_saved_plan(make_resources) -> None:
    """A one-filing scope is single-store; its plan, not shared-store's, is used."""
    resources, _voyage = make_resources(None)
    resources.openrouter = FailingOpenRouter()
    resources.saved_plans = {
        (QUESTION, "single_store"): _saved_plan("BBB_2020_10K"),
        (QUESTION, "shared_store"): _saved_plan("AAA_2020_10K"),
    }

    bundle = retrieve_exp1(QUESTION, ("BBB_2020_10K",), 3, resources)

    assert bundle["filter_doc_name"] == "BBB_2020_10K"


def test_a_question_missing_from_the_saved_plans_is_refused(make_resources) -> None:
    resources, _voyage = make_resources(None)
    resources.saved_plans = {}

    with pytest.raises(ValueError, match="no saved plan"):
        retrieve_exp1(QUESTION, BOTH, 3, resources)


def test_the_configs_structure_weight_reaches_search(make_resources) -> None:
    """Weight 1 without a structure collection fails inside exact_ranking: it was passed."""
    resources, _voyage = make_resources(_reply("AAA_2020_10K"))
    resources.config["structure"]["weight"] = 1.0

    with pytest.raises(ValueError, match="structure collection"):
        retrieve_exp1(QUESTION, BOTH, 3, resources)


def test_load_saved_plans_keys_by_question_and_condition(tmp_path) -> None:
    """The latest successful row per job wins; each plan records where it came from."""
    run_dir = tmp_path / "run-a"
    run_dir.mkdir()
    rows = [
        {"job_id": "k:q1:single_store", "status": "error"},
        {
            "job_id": "k:q1:single_store",
            "status": "success",
            "question": "Q1",
            "eval_mode": "single_store",
            "search_plan": {"filename": "AAA_2020_10K"},
        },
        {
            "job_id": "k:q1:shared_store",
            "status": "success",
            "question": "Q1",
            "eval_mode": "shared_store",
            "search_plan": {"filename": None},
        },
    ]
    (run_dir / "predictions.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )

    plans = load_saved_plans(run_dir)

    assert plans == {
        ("Q1", "single_store"): {"filename": "AAA_2020_10K", "reused_from": "run-a"},
        ("Q1", "shared_store"): {"filename": None, "reused_from": "run-a"},
    }


def test_load_saved_plans_refuses_a_missing_run(tmp_path) -> None:
    with pytest.raises(ValueError, match="no predictions"):
        load_saved_plans(tmp_path / "missing")


# --- _choose_filter -------------------------------------------------------------

CHUNKS = {
    "AAA_2020_10K:p0:c0": {"doc_name": "AAA_2020_10K"},
    "BBB_2020_10K:p0:c0": {"doc_name": "BBB_2020_10K"},
    "CCC_2020_10K:p0:c0": {"doc_name": "CCC_2020_10K"},
}


def test_a_valid_filename_is_chosen() -> None:
    assert _choose_filter("BBB_2020_10K", ("BBB_2020_10K",), CHUNKS) == (
        "BBB_2020_10K",
        "chosen",
    )


def test_no_filename_in_single_store_falls_back_to_the_scopes_filing() -> None:
    assert _choose_filter(None, ("AAA_2020_10K",), CHUNKS) == (
        "AAA_2020_10K",
        "fallback",
    )


def test_no_filename_in_shared_store_searches_every_filing() -> None:
    scope = ("AAA_2020_10K", "BBB_2020_10K", "CCC_2020_10K")
    assert _choose_filter(None, scope, CHUNKS) == (None, "fallback")


def test_a_partial_scope_is_refused() -> None:
    with pytest.raises(ValueError, match="one filing or none"):
        _choose_filter(None, ("AAA_2020_10K", "BBB_2020_10K"), CHUNKS)
