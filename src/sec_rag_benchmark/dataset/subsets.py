"""Select small, repeatable FinanceBench subsets for development runs."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any


def _allocate_by_question_type(
    questions: list[dict[str, Any]], subset_size: int
) -> dict[str, int]:
    """Allocate the subset proportionally, then assign leftover places fairly."""
    questions_by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for question in questions:
        questions_by_type[question["question_type"]].append(question)

    allocation: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for question_type, type_questions in questions_by_type.items():
        exact_share = subset_size * len(type_questions) / len(questions)
        allocation[question_type] = math.floor(exact_share)
        remainders.append((exact_share - math.floor(exact_share), question_type))

    places_left = subset_size - sum(allocation.values())
    # Largest-remainder allocation keeps the total exact while staying as close
    # as possible to the full dataset's question-type proportions.
    for _, question_type in sorted(remainders, reverse=True)[:places_left]:
        allocation[question_type] += 1
    return allocation


def select_development_subset(
    questions: list[dict[str, Any]],
    subset_name: str,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return a fixed, question-type-stratified smoke or pattern subset."""
    size_key = {"smoke": "smoke_size", "pattern": "pattern_size"}.get(subset_name)
    if size_key is None:
        raise ValueError(f"Unknown development subset: {subset_name}")

    subset_size = config[size_key]
    if subset_size <= 0:
        raise ValueError("Development subset size must be positive")
    if subset_size > len(questions):
        raise ValueError(
            f"Development subset requests {subset_size} questions, "
            f"but the prepared dataset contains only {len(questions)} questions"
        )

    allocation = _allocate_by_question_type(questions, subset_size)
    questions_by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for question in questions:
        questions_by_type[question["question_type"]].append(question)

    random_generator = random.Random(config["seed"])
    selected_ids: set[str] = set()
    for question_type in sorted(questions_by_type):
        type_questions = questions_by_type[question_type]
        selected_questions = random_generator.sample(
            type_questions, allocation[question_type]
        )
        for question in selected_questions:
            selected_ids.add(question["financebench_id"])

    # Preserve source dataset order so every later stage receives a stable,
    # human-readable sequence rather than a randomised execution order.
    return [
        question
        for question in questions
        if question["financebench_id"] in selected_ids
    ]
