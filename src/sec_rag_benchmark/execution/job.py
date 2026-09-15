"""Turn one FinanceBench question-condition pair into one prediction row."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from ..conditions import Retriever, build_condition, gold_pages
from ..generation import build_messages, generate
from ..metrics import cognitive_skills, page_metrics


Generator = Callable[[list[dict[str, str]], dict[str, Any]], dict[str, Any]]
RETRIEVAL_CONDITIONS = {"single_store", "shared_store"}


def execute_job(
    question: dict[str, Any],
    condition_name: str,
    job_id: str,
    *,
    pdf_dir: str | Path,
    all_doc_names: tuple[str, ...],
    generation_config: dict[str, Any],
    retrieval_depth: int,
    retriever: Retriever | None = None,
    generator: Generator = generate,
) -> dict[str, Any]:
    """Build context, generate an answer, and calculate one job's metrics."""
    condition = build_condition(
        question,
        condition_name,
        pdf_dir,
        all_doc_names,
        retriever=retriever,
        top_k=retrieval_depth,
    )
    generated = generator(
        build_messages(question["question"], condition["context"]),
        generation_config,
    )

    if condition_name in RETRIEVAL_CONDITIONS:
        retrieval_metrics = page_metrics(
            gold_pages(question), condition["retrieved_chunks"], retrieval_depth
        )
    else:
        retrieval_metrics = {
            "page_recall": None,
            "page_precision": None,
            "page_mrr": None,
        }

    # The saved row contains everything needed for reporting and the future
    # judge, which can therefore run without regenerating this answer.
    return {
        "job_id": job_id,
        "status": "success",
        "financebench_id": question["financebench_id"],
        "question": question["question"],
        "gold_answer": question["answer"],
        "gold_evidence": question["evidence"],
        "human_justification": question["justification"],
        "model_answer": generated["answer"],
        "eval_mode": condition_name,
        "question_type": question["question_type"],
        "cognitive_skills": cognitive_skills(question.get("question_reasoning")),
        "gold_pages": gold_pages(question),
        "retrieved_chunks": condition["retrieved_chunks"],
        **retrieval_metrics,
        **{key: value for key, value in generated.items() if key != "answer"},
        "completed_at": datetime.now(UTC).isoformat(),
    }
