"""Calculate retrieval metrics and normalize FinanceBench skill labels.

Final-answer accuracy is deliberately absent. The separate Azure judge reads
saved predictions later, so retrieval calculation never triggers a model call.
"""

from __future__ import annotations

import re
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


SKILL_LABELS = {
    "information extraction": "information_extraction",
    "numerical reasoning": "numerical_reasoning",
    "logical reasoning": "logical_reasoning",
    "logical reasoning (based on numerical reasoning)": "logical_reasoning",
}


def cognitive_skills(value: str | None) -> list[str]:
    """Strictly map FinanceBench reasoning labels to reporting categories."""
    if value is None:
        return ["unlabelled"]

    # FinanceBench joins multiple labels with the word OR. Splitting before
    # mapping prevents the parenthetical logical label from being mistaken for
    # a second numerical-reasoning label merely because it contains that text.
    raw_parts = re.split(r"\s+\bOR\b(?:\s+|$)", value.strip(), flags=re.IGNORECASE)
    mapped_skills: list[str] = []
    for raw_part in raw_parts:
        part = raw_part.strip()
        if not part:
            # One source row ends with a trailing OR. It represents no label.
            continue
        skill = SKILL_LABELS.get(part.casefold())
        if skill is None:
            raise ValueError(f"Unknown cognitive skill label: {part!r}")
        if skill not in mapped_skills:
            mapped_skills.append(skill)

    if not mapped_skills:
        raise ValueError("Unknown cognitive skill label: no mapped labels")
    return mapped_skills
