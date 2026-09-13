"""Simple checkpointed loop for FinanceBench question-condition jobs."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any, Callable

from openai import (
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    PermissionDeniedError,
    UnprocessableEntityError,
)

from .conditions import Retriever, RetrieverUnavailable, build_condition, gold_pages
from .generation import build_messages, generate
from .metrics import cognitive_skills, page_metrics


def _append(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")
        file.flush()


def _successful_jobs(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        row["job_id"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line and (row := json.loads(line)).get("status") == "success"
    }


def run(
    questions: list[dict[str, Any]],
    conditions: list[str],
    *,
    pdf_dir: str | Path,
    all_doc_names: tuple[str, ...],
    generation_config: dict[str, Any],
    run_dir: str | Path,
    retrieval_depth: int = 5,
    run_key: str = "run",
    retriever: Retriever | None = None,
    generator: Callable[[list[dict[str, str]], dict[str, Any]], dict[str, Any]] = generate,
) -> dict[str, int]:
    """Run question-condition jobs, checkpoint each attempt, and resume successes.

    This is the implementation guide's central orchestration loop. It generates
    answers and retrieval metrics only; final-answer judging is a separate,
    deliberately deferred pass over the saved predictions.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions = run_dir / "predictions.jsonl"
    errors = run_dir / "errors.jsonl"
    completed = _successful_jobs(predictions)
    counts = {"generated": 0, "skipped": 0, "failed": 0}

    for question in questions:
        for condition_name in conditions:
            # A job is one FinanceBench question under one context condition.
            job_id = f"{run_key}:{question['financebench_id']}:{condition_name}"
            if job_id in completed:
                counts["skipped"] += 1
                continue
            try:
                # Keep condition construction separate from generation so all
                # five conditions use the same answer prompt and model call.
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
                # Page metrics apply only when a retriever produced ranked
                # chunks. Closed-book, oracle, and long-context report None.
                retrieval = (
                    page_metrics(gold_pages(question), condition["retrieved_chunks"], retrieval_depth)
                    if condition_name in {"single_store", "shared_store"}
                    else {"page_recall": None, "page_precision": None, "page_mrr": None}
                )
                row = {
                    "job_id": job_id,
                    "status": "success",
                    "financebench_id": question["financebench_id"],
                    "question": question["question"],
                    "gold_answer": question["answer"],
                    # Retain the complete judging inputs with the generated
                    # answer so the future judge can run as a separate pass.
                    "gold_evidence": question["evidence"],
                    "human_justification": question["justification"],
                    "model_answer": generated["answer"],
                    "eval_mode": condition_name,
                    "question_type": question["question_type"],
                    "cognitive_skills": cognitive_skills(question.get("question_reasoning")),
                    "gold_pages": gold_pages(question),
                    "retrieved_chunks": condition["retrieved_chunks"],
                    **retrieval,
                    **{key: value for key, value in generated.items() if key != "answer"},
                    "completed_at": datetime.now(UTC).isoformat(),
                }
                _append(predictions, row)
                counts["generated"] += 1
            except Exception as error:
                _append(errors, {"job_id": job_id, "status": "error", "error": str(error)})
                counts["failed"] += 1
                # Configuration/authentication problems will not improve on the
                # next question, so stop instead of repeating a doomed request.
                if isinstance(error, (
                    AuthenticationError,
                    BadRequestError,
                    NotFoundError,
                    PermissionDeniedError,
                    RetrieverUnavailable,
                    UnprocessableEntityError,
                )):
                    raise
                if isinstance(error, RuntimeError) and "API_KEY" in str(error):
                    raise
    return counts
