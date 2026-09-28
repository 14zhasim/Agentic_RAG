"""Tests for rerank: the Voyage call, faked so no test can spend."""

from __future__ import annotations

from types import SimpleNamespace

from sec_rag.retrieval.rerank import rerank

CONFIG = {"rerank": {"model": "rerank-3-lite"}}


class FakeVoyage:
    """Stands in for voyageai.Client.rerank: best match is the last text."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def rerank(self, query: str, documents: list[str], **kwargs) -> SimpleNamespace:
        self.calls.append({"query": query, "documents": documents, **kwargs})
        order = list(reversed(range(len(documents))))[: kwargs["top_k"]]
        results = [
            SimpleNamespace(index=index, relevance_score=1.0 - 0.1 * rank)
            for rank, index in enumerate(order)
        ]
        return SimpleNamespace(results=results, total_tokens=321)


def test_rerank_returns_positions_scores_and_tokens() -> None:
    voyage = FakeVoyage()

    order, tokens = rerank("What was capex?", ["a", "b", "c"], 2, CONFIG, voyage)

    assert order == [(2, 1.0), (1, 0.9)]
    assert tokens == 321


def test_rerank_sends_the_question_model_and_no_truncation() -> None:
    """Draft -> Retrieve: rerank-3-lite, against the original question."""
    voyage = FakeVoyage()

    rerank("What was capex?", ["a", "b", "c"], 2, CONFIG, voyage)

    call = voyage.calls[0]
    assert call["query"] == "What was capex?"
    assert call["documents"] == ["a", "b", "c"]
    assert call["model"] == "rerank-3-lite"
    assert call["top_k"] == 2
    assert call["truncation"] is False
