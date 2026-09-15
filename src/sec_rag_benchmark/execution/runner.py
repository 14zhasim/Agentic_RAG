"""Create, execute, checkpoint, and resume real benchmark runs."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any

from openai import (
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    PermissionDeniedError,
    UnprocessableEntityError,
)

from ..conditions import CONDITIONS, Retriever, RetrieverUnavailable
from ..data import load_run_questions
from ..generation import generate
from .job import Generator, execute_job


def _append(path: Path, row: dict[str, Any]) -> None:
    """Checkpoint one attempt as one JSON line and flush it immediately."""
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")
        file.flush()


def _successful_jobs(path: Path) -> set[str]:
    """Return job IDs whose saved prediction status is successful."""
    if not path.exists():
        return set()
    successful_job_ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        if row.get("status") == "success":
            successful_job_ids.add(row["job_id"])
    return successful_job_ids


def _create_or_resume_run(
    config: dict[str, Any],
    config_path: Path,
    selected_conditions: list[str],
    question_count: int,
    requested_run_dir: Path | None,
) -> dict[str, Any]:
    """Create or verify the run directory, snapshot and stable run key."""
    run_dir = requested_run_dir or (
        Path(config["run"]["results_dir"])
        / datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    selection = (
        "\n[selection]\n"
        f"conditions = [{', '.join(repr(name) for name in selected_conditions)}]\n"
        f"limit = {question_count}\n"
    )
    config_bytes = config_path.read_bytes() + selection.encode()
    snapshot = run_dir / "config.toml"
    if snapshot.exists() and snapshot.read_bytes() != config_bytes:
        raise ValueError("Run directory contains a different configuration")
    snapshot.write_bytes(config_bytes)
    return {
        "run_dir": run_dir,
        "run_key": hashlib.sha256(config_bytes).hexdigest()[:12],
    }


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
    requested_run_dir: str | Path | None = None,
    retriever: Retriever | None = None,
    generator: Generator = generate,
) -> dict[str, Any]:
    """Run every selected question-condition job and checkpoint each attempt."""
    questions = load_run_questions(config["dataset"], limit)
    selected_conditions = conditions or config["run"]["conditions"]
    if not selected_conditions or set(selected_conditions) - CONDITIONS:
        raise ValueError("Unknown condition")

    run_identity = _create_or_resume_run(
        config,
        Path(config_path),
        selected_conditions,
        len(questions),
        Path(requested_run_dir) if requested_run_dir is not None else None,
    )
    run_dir = run_identity["run_dir"]
    predictions_path = run_dir / "predictions.jsonl"
    errors_path = run_dir / "errors.jsonl"
    completed_job_ids = _successful_jobs(predictions_path)
    all_doc_names = tuple(dict.fromkeys(row["doc_name"] for row in questions))
    pdf_dir = Path(config["dataset"]["output_dir"]) / "pdfs"
    counts = {"generated": 0, "skipped": 0, "failed": 0}

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
            except Exception as error:
                _append(
                    errors_path,
                    {"job_id": job_id, "status": "error", "error": str(error)},
                )
                counts["failed"] += 1
                if _should_stop_run(error):
                    raise

    return {"run_dir": run_dir, **counts}
