"""Rerank (Build Order 2.3): Voyage reorders the fused chunks against the question.

A reranker reads the question and each chunk together (a cross-encoder),
so it judges relevance more exactly than either search, which compared
separately made vectors or word counts (docs/libraries/voyage/reranker.md
lines 9-11). Called by retrieve_exp1 with the original question and the 50
fused chunks' texts; it keeps the best 10.
"""

from __future__ import annotations

from typing import Any


def rerank(
    question: str, texts: list[str], top_n: int, config: dict[str, Any], voyage: Any
) -> tuple[list[tuple[int, float]], int]:
    """Reorder texts by how well each answers the question; keep the best top_n.

    Returns ([(position in texts, relevance score), ...] best first, tokens
    billed). Positions, not chunk IDs, because Voyage only sees the texts;
    retrieve_exp1 maps positions back to chunks.

    The original question, not GLM's semantic query (Draft -> Retrieve): it
    is what the answer model answers, and it is the same in every ablation.
    """
    # reranker.md lines 190-215: rerank(query, documents, model, top_k,
    # truncation). rerank-3-lite reads 32,000 tokens and up to 1,000
    # documents, far above 50 chunks of at most 3,863 tokens.
    # truncation=False makes an oversized input an error, not a silent cut.
    result = voyage.rerank(
        question,
        texts,
        model=config["rerank"]["model"],
        top_k=top_n,
        truncation=False,
    )
    # results are sorted by descending relevance score (reranker.md line
    # 210); total_tokens is what Voyage bills (line 214).
    order = [(item.index, item.relevance_score) for item in result.results]
    return order, result.total_tokens
