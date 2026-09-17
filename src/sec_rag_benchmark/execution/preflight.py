"""Inspect benchmark jobs without creating an API client or results files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..conditions import CONDITIONS, build_condition
from ..data import load_run_questions
from ..development_subsets import select_development_subset
from ..generation import build_messages, count_prompt_tokens


RETRIEVAL_CONDITIONS = {"single_store", "shared_store"}


def dry_run(
    config: dict[str, Any],
    *,
    conditions: list[str] | None = None,
    limit: int | None = None,
    subset: str | None = None,
) -> dict[str, Any]:
    """Build executable condition prompts and count tokens without an API call."""
    if limit is not None and subset is not None:
        raise ValueError("--limit and --subset cannot be used together")
    questions = load_run_questions(config["dataset"], limit)
    if subset is not None:
        questions = select_development_subset(
            questions, subset, config["development_subsets"]
        )
    selected_conditions = conditions or config["run"]["conditions"]
    if not selected_conditions or set(selected_conditions) - CONDITIONS:
        raise ValueError("Unknown condition")

    pdf_dir = Path(config["dataset"]["output_dir"]) / "pdfs"
    all_doc_names = tuple(dict.fromkeys(row["doc_name"] for row in questions))
    maximum_prompt_tokens: dict[str, int | str] = {}
    oversized_jobs = 0

    for condition_name in selected_conditions:
        if condition_name in RETRIEVAL_CONDITIONS:
            maximum_prompt_tokens[condition_name] = "requires retriever"
            continue

        prompt_token_counts: list[int] = []
        for question in questions:
            condition = build_condition(
                question, condition_name, pdf_dir, all_doc_names
            )
            messages = build_messages(question["question"], condition["context"])
            prompt_token_counts.append(
                count_prompt_tokens(
                    messages, config["generation"]["reasoning_effort"]
                )
            )

        maximum_prompt_tokens[condition_name] = max(prompt_token_counts, default=0)
        reserved_tokens = (
            config["generation"]["max_output_tokens"]
            + config["generation"]["token_safety_margin"]
        )
        oversized_jobs += sum(
            token_count + reserved_tokens
            > config["generation"]["context_window_tokens"]
            for token_count in prompt_token_counts
        )

    return {
        "planned_jobs": len(questions) * len(selected_conditions),
        "conditions": selected_conditions,
        "maximum_prompt_tokens": maximum_prompt_tokens,
        "oversized_jobs": oversized_jobs,
        "api_requests": 0,
    }
