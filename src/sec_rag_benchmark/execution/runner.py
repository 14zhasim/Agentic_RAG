"""Create, execute, checkpoint, and resume real benchmark runs."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

from openai import (
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    PermissionDeniedError,
    UnprocessableEntityError,
)

from sec_rag.config import load_config as load_sec_rag_config
from sec_rag.retrieval.exp1 import open_exp1, retrieve_exp1

from ..dataset.financebench import load_questions, load_run_questions
from ..dataset.subsets import select_development_subset
from ..evaluation.retrieval_metrics import cognitive_skills
from ..pipeline.conditions import (
    CONDITIONS,
    Retriever,
    RetrieverUnavailable,
    gold_pages,
)
from ..pipeline.generation import ContextLimitError, generate
from .job import RETRIEVAL_CONDITIONS, Generator, execute_job


def _append(path: Path, row: dict[str, Any]) -> None:
    """Checkpoint one attempt as one JSON line and flush it immediately."""
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")
        file.flush()


def _completed_jobs(path: Path) -> set[str]:
    """Return jobs with terminal success or did_not_fit outcomes."""
    if not path.exists():
        return set()
    completed_job_ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        if row.get("status") in {"success", "did_not_fit"}:
            completed_job_ids.add(row["job_id"])
    return completed_job_ids


def _did_not_fit_prediction(
    question: dict[str, Any],
    condition_name: str,
    job_id: str,
    error: Exception,
    generation_config: dict[str, Any],
) -> dict[str, Any]:
    """Preserve an oversized job as a terminal, non-answer prediction."""
    return {
        "job_id": job_id,
        "status": "did_not_fit",
        "financebench_id": question["financebench_id"],
        "question": question["question"],
        "gold_answer": question["answer"],
        "gold_evidence": question["evidence"],
        "human_justification": question["justification"],
        "model_answer": None,
        "eval_mode": condition_name,
        "question_type": question["question_type"],
        "cognitive_skills": cognitive_skills(question.get("question_reasoning")),
        "gold_pages": gold_pages(question),
        "retrieved_chunks": [],
        "page_recall": None,
        "page_precision": None,
        "page_mrr": None,
        "requested_model": generation_config["model"],
        "returned_model": None,
        "provider": None,
        "usage": None,
        "cost": None,
        "latency_seconds": None,
        "error": str(error),
        "completed_at": datetime.now(UTC).isoformat(),
    }


def _create_or_resume_run(
    config: dict[str, Any],
    config_path: Path,
    selected_conditions: list[str],
    selected_questions: list[dict[str, Any]],
    subset: str | None,
    requested_run_dir: Path | None,
    sec_rag_config_path: Path | None = None,
) -> dict[str, Any]:
    """Create or verify the run directory, snapshots and stable run key.

    When a retrieval condition runs, `sec_rag_config_path` names the RAG
    system's settings (RRF k, candidate counts, models, prompt version). They
    affect results as much as financebench.toml does, so they are copied into
    the run as `sec_rag.toml` and hashed into the run key (Build Order 2.5,
    decision D). Without one, the key is config.toml's hash alone, as before,
    so baseline runs keep their existing keys.
    """
    run_dir = requested_run_dir or (
        Path(config["run"]["results_dir"])
        / (
            f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}--"
            f"{config['run']['experiment']}--{config['run']['variant']}"
        )
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    selection = (
        "\n[selection]\n"
        f"conditions = [{', '.join(repr(name) for name in selected_conditions)}]\n"
        f"limit = {len(selected_questions)}\n"
        f"development_subset = {json.dumps(subset or '')}\n"
        "question_ids = ["
        + ", ".join(
            json.dumps(question["financebench_id"]) for question in selected_questions
        )
        + "]\n"
    )
    config_bytes = config_path.read_bytes() + selection.encode()
    snapshot = run_dir / "config.toml"
    if snapshot.exists() and snapshot.read_bytes() != config_bytes:
        raise ValueError("Run directory contains a different configuration")

    sec_rag_bytes = b""
    sec_rag_snapshot = run_dir / "sec_rag.toml"
    if sec_rag_config_path is not None:
        sec_rag_bytes = sec_rag_config_path.read_bytes()
        if sec_rag_snapshot.exists() and sec_rag_snapshot.read_bytes() != sec_rag_bytes:
            raise ValueError("Run directory contains a different configuration")

    # Both checks pass before either file is written, so a refused resume
    # leaves the run folder exactly as it was.
    snapshot.write_bytes(config_bytes)
    if sec_rag_config_path is not None:
        sec_rag_snapshot.write_bytes(sec_rag_bytes)
    return {
        "run_dir": run_dir,
        "run_key": hashlib.sha256(config_bytes + sec_rag_bytes).hexdigest()[:12],
    }


def _open_exp1_retriever(sec_rag_config_path: Path) -> Retriever:
    """Load Exp1 once and return it as the benchmark's retriever(question, scope, top_k).

    open_exp1 loads the BM25 index, Chroma store, chunk lookup and both paid
    clients once, so 224 jobs don't reload them; partial binds them, leaving
    the three arguments conditions.py passes (Build Order 2.0, minimal plug).
    This is the paid path: open_exp1 reads OPENROUTER_API_KEY and
    VOYAGE_API_KEY and raises RuntimeError if either is missing.
    """
    resources = open_exp1(load_sec_rag_config(sec_rag_config_path))
    return partial(retrieve_exp1, resources=resources)


def _should_stop_run(error: Exception) -> bool:
    """Identify errors that would make every following job fail too."""
    fatal_types = (
        AuthenticationError,
        BadRequestError,
        NotFoundError,
        PermissionDeniedError,
        RetrieverUnavailable,
        UnprocessableEntityError,
    )
    return isinstance(error, fatal_types) or (
        isinstance(error, RuntimeError) and "API_KEY" in str(error)
    )


def run_benchmark(
    config: dict[str, Any],
    config_path: str | Path,
    *,
    conditions: list[str] | None = None,
    limit: int | None = None,
    subset: str | None = None,
    requested_run_dir: str | Path | None = None,
    retriever: Retriever | None = None,
    generator: Generator = generate,
) -> dict[str, Any]:
    """Run every selected question-condition job and checkpoint each attempt."""
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

    retrieval_selected = bool(set(selected_conditions) & RETRIEVAL_CONDITIONS)
    sec_rag_config_path: Path | None = None
    if retrieval_selected:
        configured = config["run"].get("sec_rag_config")
        # Checked before the run folder exists, so a refused run leaves nothing.
        if configured is None and retriever is None:
            raise ValueError("Retrieval conditions need [run] sec_rag_config")
        if configured is not None:
            sec_rag_config_path = Path(configured)

    run_identity = _create_or_resume_run(
        config,
        Path(config_path),
        selected_conditions,
        questions,
        subset,
        Path(requested_run_dir) if requested_run_dir is not None else None,
        sec_rag_config_path,
    )
    run_dir = run_identity["run_dir"]
    predictions_path = run_dir / "predictions.jsonl"
    errors_path = run_dir / "errors.jsonl"
    completed_job_ids = _completed_jobs(predictions_path)
    # Build Order 2.0, decision B: shared-store searches every prepared filing
    # (all 64), whatever --subset or --limit selected, so a smoke run tests
    # the real choice and retrieve_exp1's "every filing" check holds.
    # load_run_questions above already validated the prepared dataset, so
    # read it again without repeating that check.
    every_question = load_questions(config["dataset"]["output_dir"])
    all_doc_names = tuple(dict.fromkeys(row["doc_name"] for row in every_question))
    if retrieval_selected and retriever is None and sec_rag_config_path is not None:
        retriever = _open_exp1_retriever(sec_rag_config_path)
    pdf_dir = Path(config["dataset"]["output_dir"]) / "pdfs"
    counts = {"generated": 0, "did_not_fit": 0, "skipped": 0, "failed": 0}

    for question in questions:
        for condition_name in selected_conditions:
            job_id = (
                f"{run_identity['run_key']}:"
                f"{question['financebench_id']}:{condition_name}"
            )
            if job_id in completed_job_ids:
                counts["skipped"] += 1
                continue
            try:
                prediction = execute_job(
                    question,
                    condition_name,
                    job_id,
                    pdf_dir=pdf_dir,
                    all_doc_names=all_doc_names,
                    generation_config=config["generation"],
                    retrieval_depth=config["run"]["retrieval_depth"],
                    retriever=retriever,
                    generator=generator,
                )
                _append(predictions_path, prediction)
                counts["generated"] += 1
            except ContextLimitError as error:
                # This prompt will never fit under the unchanged run config.
                # Save it with the terminal predictions so resume skips it.
                _append(
                    predictions_path,
                    _did_not_fit_prediction(
                        question,
                        condition_name,
                        job_id,
                        error,
                        config["generation"],
                    ),
                )
                counts["did_not_fit"] += 1
            except Exception as error:
                _append(
                    errors_path,
                    {"job_id": job_id, "status": "error", "error": str(error)},
                )
                counts["failed"] += 1
                if _should_stop_run(error):
                    raise

    return {"run_dir": run_dir, **counts}
