"""Calculate retrieval metrics and normalize FinanceBench skill labels.

Final-answer accuracy is deliberately absent. As described in the implementation
guide, a later Azure/RAGAS pass will judge saved answers separately.
"""

from __future__ import annotations

from typing import Any


def _unique_pages_from_chunks(
    ranked_chunks: list[dict[str, Any]],
) -> set[tuple[str, int]]:
    """Convert chunk provenance into distinct document-aware page pairs."""
    unique_pages: set[tuple[str, int]] = set()
    for chunk in ranked_chunks:
        for page_index in chunk["pages"]:
            unique_pages.add((chunk["doc_name"], page_index))
    return unique_pages


def _first_relevant_chunk_rank(
    ranked_chunks: list[dict[str, Any]], gold_pages: set[tuple[str, int]]
) -> int | None:
    """Return the rank of the first chunk covering a gold page, if any."""
    for chunk in ranked_chunks:
        pages_covered_by_chunk = {
            (chunk["doc_name"], page_index) for page_index in chunk["pages"]
        }
        if gold_pages.intersection(pages_covered_by_chunk):
            return chunk["rank"]
    return None


def page_metrics(
    gold: list[tuple[str, int]], chunks: list[dict[str, Any]], top_k: int
) -> dict[str, float]:
    """Calculate the guide's document-aware page recall, precision, and MRR."""
    gold_pages = set(gold)
    if not gold_pages:
        raise ValueError("Gold pages cannot be empty")

    # Metrics apply only to the first top_k results, in retrieval-rank order.
    ranked_chunks = sorted(chunks, key=lambda chunk: chunk["rank"])[:top_k]

    # Recall and precision evaluate pages, not chunks. Convert every chunk's
    # pages into (document, page-index) pairs. The set counts a page once even
    # when several chunks overlap that same page.
    unique_retrieved_pages = _unique_pages_from_chunks(ranked_chunks)

    retrieved_gold_pages = gold_pages.intersection(unique_retrieved_pages)
    number_of_gold_pages_retrieved = len(retrieved_gold_pages)
    page_recall = number_of_gold_pages_retrieved / len(gold_pages)
    if unique_retrieved_pages:
        page_precision = (
            number_of_gold_pages_retrieved / len(unique_retrieved_pages)
        )
    else:
        page_precision = 0.0

    # MRR evaluates chunk order. Find the first chunk that covers at least one
    # gold (document, page-index) pair, then stop because later chunks cannot
    # improve the first relevant rank.
    first_relevant_chunk_rank = _first_relevant_chunk_rank(
        ranked_chunks, gold_pages
    )

    if first_relevant_chunk_rank is None:
        page_mrr = 0.0
    else:
        page_mrr = 1 / first_relevant_chunk_rank

    return {
        "page_recall": page_recall,
        "page_precision": page_precision,
        "page_mrr": page_mrr,
    }


SKILLS = (
    ("information extraction", "information_extraction"),
    ("numerical reasoning", "numerical_reasoning"),
    ("logical reasoning", "logical_reasoning"),
)


def cognitive_skills(value: str | None) -> list[str]:
    """Normalize FinanceBench's free-text reasoning labels for segmentation."""
    normalized = (value or "").casefold()
    found = [label for phrase, label in SKILLS if phrase in normalized]
    return found or ["unspecified"]
