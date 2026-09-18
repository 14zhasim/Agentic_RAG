"""Validate the production answer judge against published human decisions."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..dataset.financebench import load_questions
from ..pipeline.conditions import gold_pages
from .judge import judge_run
from .retrieval_metrics import cognitive_skills

HUMAN_LABEL_TO_ACCURACY = {
    "Correct Answer": 1,
    "Incorrect Answer": 0,
    "Refusal": 0,
}
SETUP_COMMAND = (
    "git clone https://github.com/patronus-ai/financebench.git "
    "benchmarks/financebench && git -C benchmarks/financebench checkout "
    "cc39aeb4afdf33909ee1412188bf89035950c2eb"
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read one JSON object per non-empty line."""
    try:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read {path}: {error}") from error


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write deterministic JSONL used as the validation checkpoint."""
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _source_candidates(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Join published answers to the prepared reference fields needed by the judge."""
    source_dir = Path(config["dataset"]["source_dir"])
    result_paths = sorted((source_dir / "results").glob("*.jsonl"))
    if not result_paths:
        raise ValueError(
            "FinanceBench published results are missing. Recreate them with: "
            f"{SETUP_COMMAND}"
        )

    questions = {
        row["financebench_id"]: row
        for row in load_questions(config["dataset"]["output_dir"])
    }
    candidates: list[dict[str, Any]] = []
    for result_path in result_paths:
        for published in _read_jsonl(result_path):
            question = questions.get(published.get("financebench_id"))
            human_label = published.get("label")
            candidate_answer = published.get("model_answer")
            if (
                question is None
                or human_label not in HUMAN_LABEL_TO_ACCURACY
                or not isinstance(candidate_answer, str)
                or not candidate_answer.strip()
            ):
                continue
            candidates.append(
                {
                    "financebench_id": question["financebench_id"],
                    "source_result_file": result_path.name,
                    "source_model": published.get("model_name"),
                    "source_eval_mode": published.get("eval_mode"),
                    "human_label": human_label,
                    "expected_accuracy": HUMAN_LABEL_TO_ACCURACY[human_label],
                    "candidate_answer": candidate_answer,
                    "question": question["question"],
                    "gold_answer": question["answer"],
                    "gold_evidence": question["evidence"],
                    "human_justification": question["justification"],
                    "question_type": question["question_type"],
                    "cognitive_skills": cognitive_skills(
                        question.get("question_reasoning")
                    ),
                    "gold_pages": gold_pages(question),
                }
            )
    return candidates


def _select_candidates(
    candidates: list[dict[str, Any]], validation_config: dict[str, Any]
) -> list[dict[str, Any]]:
    """Choose balanced labels while spreading selections across types and files."""
    ordered_candidates = sorted(
        candidates,
        key=lambda row: (
            row["source_result_file"],
            row["financebench_id"],
            row["candidate_answer"],
        ),
    )
    random.Random(validation_config["seed"]).shuffle(ordered_candidates)
    shuffled_position = {
        id(candidate): position for position, candidate in enumerate(ordered_candidates)
    }

    quotas = {
        "Correct Answer": validation_config["correct_examples"],
        "Incorrect Answer": validation_config["incorrect_examples"],
        "Refusal": validation_config["refusal_examples"],
    }
    selected: list[dict[str, Any]] = []
    used_question_ids: set[str] = set()
    used_answers: set[str] = set()
    question_type_counts: Counter[str] = Counter()
    source_file_counts: Counter[str] = Counter()

    for human_label, quota in quotas.items():
        for _ in range(quota):
            eligible = []
            for candidate in ordered_candidates:
                normalized_answer = " ".join(
                    candidate["candidate_answer"].casefold().split()
                )
                if candidate["human_label"] != human_label:
                    continue
                if candidate["financebench_id"] in used_question_ids:
                    continue
                if normalized_answer in used_answers:
                    continue
                eligible.append(candidate)

            if not eligible:
                raise ValueError(
                    f"Not enough unique published {human_label!r} examples "
                    f"for the configured quota of {quota}"
                )

            # Prefer the least represented question type and source file. The
            # shuffled position provides a deterministic tie-breaker.
            chosen = min(
                eligible,
                key=lambda row: (
                    question_type_counts[row["question_type"]],
                    source_file_counts[row["source_result_file"]],
                    shuffled_position[id(row)],
                ),
            )
            selected.append(chosen)
            used_question_ids.add(chosen["financebench_id"])
            used_answers.add(" ".join(chosen["candidate_answer"].casefold().split()))
            question_type_counts[chosen["question_type"]] += 1
            source_file_counts[chosen["source_result_file"]] += 1

    if len(question_type_counts) < 3:
        raise ValueError("Validation sample must cover all three question types")
    if len(source_file_counts) < 2:
        raise ValueError("Validation sample must use multiple published result files")
    return selected


def _prediction_from_sample(row: dict[str, Any]) -> dict[str, Any]:
    """Adapt one published answer to the same shape used by the production judge."""
    return {
        "job_id": row["job_id"],
        "status": "success",
        "financebench_id": row["financebench_id"],
        "question": row["question"],
        "gold_answer": row["gold_answer"],
        "gold_evidence": row["gold_evidence"],
        "human_justification": row["human_justification"],
        "model_answer": row["candidate_answer"],
        "eval_mode": row["source_eval_mode"],
        "question_type": row["question_type"],
        "cognitive_skills": row["cognitive_skills"],
        "gold_pages": row["gold_pages"],
        "retrieved_chunks": [],
        "page_recall": None,
        "page_precision": None,
        "page_mrr": None,
        "requested_model": row["source_model"],
        "returned_model": row["source_model"],
        "provider": "published-financebench",
        "request_id": None,
        "usage": {},
        "cost": None,
        "latency_seconds": None,
    }


def create_validation_sample(
    config: dict[str, Any], run_dir: str | Path
) -> list[dict[str, Any]]:
    """Create the complete fixed sample before any paid judge request is made."""
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    selected = _select_candidates(
        _source_candidates(config), config["judge_validation"]
    )

    sample_rows = []
    for candidate in selected:
        sample_rows.append(
            {
                "job_id": f"judge-validation:{candidate['financebench_id']}",
                **candidate,
            }
        )
    _write_jsonl(run_path / "validation_sample.jsonl", sample_rows)
    _write_jsonl(
        run_path / "predictions.jsonl",
        [_prediction_from_sample(row) for row in sample_rows],
    )
    return sample_rows


def _existing_validation_sample(run_path: Path) -> list[dict[str, Any]] | None:
    """Recognise a validation directory without mistaking a benchmark run for one."""
    if not run_path.exists() or not any(run_path.iterdir()):
        return None

    sample_path = run_path / "validation_sample.jsonl"
    predictions_path = run_path / "predictions.jsonl"
    if not sample_path.is_file() or not predictions_path.is_file():
        raise ValueError(
            "Refusing to initialise judge validation in a non-empty directory "
            "that is not an existing validation run"
        )

    sample = _read_jsonl(sample_path)
    predictions = _read_jsonl(predictions_path)
    sample_job_ids = {row.get("job_id") for row in sample}
    prediction_job_ids = {row.get("job_id") for row in predictions}
    if (
        not sample_job_ids
        or any(
            not isinstance(job_id, str) or not job_id.startswith("judge-validation:")
            for job_id in sample_job_ids
        )
        or prediction_job_ids != sample_job_ids
    ):
        raise ValueError(
            "Refusing to resume judge validation because its sample and "
            "predictions do not identify the same validation jobs"
        )
    return sample


def _latest_judgments(path: Path) -> dict[str, dict[str, Any]]:
    """Keep the newest append-only judgment for each validation job."""
    if not path.exists():
        return {}
    return {row["job_id"]: row for row in _read_jsonl(path)}


def summarize_validation(
    sample: list[dict[str, Any]],
    judgments: dict[str, dict[str, Any]],
    *,
    minimum_agreement: float,
    source_commit: str,
    source_file_hashes: dict[str, str],
) -> dict[str, Any]:
    """Compare the judge with humans, counting null verdicts as mismatches."""
    completed = 0
    agreements = 0
    label_counts: dict[str, dict[str, int]] = {
        label: {"total": 0, "completed": 0, "agreements": 0}
        for label in HUMAN_LABEL_TO_ACCURACY
    }
    confusion_matrix: Counter[str] = Counter()
    mismatches = []

    for sample_row in sample:
        human_label = sample_row["human_label"]
        expected_accuracy = sample_row["expected_accuracy"]
        label_counts[human_label]["total"] += 1
        judgment = judgments.get(sample_row["job_id"])
        if judgment is None:
            continue

        completed += 1
        label_counts[human_label]["completed"] += 1
        judge_accuracy = judgment.get("accuracy")
        judge_key = "null" if judge_accuracy is None else str(judge_accuracy)
        confusion_matrix[f"expected_{expected_accuracy}_judge_{judge_key}"] += 1
        if judge_accuracy == expected_accuracy:
            agreements += 1
            label_counts[human_label]["agreements"] += 1
        else:
            mismatches.append(
                {
                    "job_id": sample_row["job_id"],
                    "financebench_id": sample_row["financebench_id"],
                    "human_label": human_label,
                    "expected_accuracy": expected_accuracy,
                    "judge_accuracy": judge_accuracy,
                    "manual_review": True,
                }
            )

    by_human_label: dict[str, dict[str, Any]] = {}
    for human_label, counts in label_counts.items():
        label_completed = counts["completed"]
        by_human_label[human_label] = {
            **counts,
            "agreement_rate": (
                counts["agreements"] / label_completed if label_completed else None
            ),
        }

    total = len(sample)
    status = "complete" if completed == total else "incomplete"
    agreement_rate = agreements / completed if completed else None
    return {
        "status": status,
        "passed": (
            status == "complete"
            and agreement_rate is not None
            and agreement_rate >= minimum_agreement
        ),
        "source_commit": source_commit,
        "source_file_hashes": source_file_hashes,
        "total": total,
        "completed": completed,
        "agreements": agreements,
        "agreement_rate": agreement_rate,
        "minimum_agreement": minimum_agreement,
        "by_human_label": by_human_label,
        "confusion_matrix": dict(sorted(confusion_matrix.items())),
        "mismatches": mismatches,
    }


def _selected_source_hashes(
    config: dict[str, Any], sample: list[dict[str, Any]]
) -> dict[str, str]:
    """Record the published files that produced this exact validation sample."""
    results_dir = Path(config["dataset"]["source_dir"]) / "results"
    hashes = {}
    for filename in sorted({row["source_result_file"] for row in sample}):
        path = results_dir / filename
        hashes[f"results/{filename}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def validate_judge(
    config: dict[str, Any],
    config_path: str | Path,
    *,
    requested_run_dir: str | Path | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    """Create or resume one human-label validation and write its summary."""
    run_path = (
        Path(requested_run_dir)
        if requested_run_dir
        else (
            Path(config["run"]["results_dir"])
            / f"judge-validation-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
        )
    )
    sample = _existing_validation_sample(run_path)
    if sample is None:
        sample = create_validation_sample(config, run_path)
        shutil.copy2(config_path, run_path / "config.toml")

    # Check the selected published files before any paid request, including on
    # resume. Their hashes make the exact external validation input auditable.
    source_file_hashes = _selected_source_hashes(config, sample)
    judge_counts = judge_run(run_path, config["judge"], client=client)
    validation_config = config["judge_validation"]
    summary = summarize_validation(
        sample,
        _latest_judgments(run_path / "judgments.jsonl"),
        minimum_agreement=validation_config["minimum_agreement"],
        source_commit=validation_config["source_commit"],
        source_file_hashes=source_file_hashes,
    )
    (run_path / "judge_validation.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"run_dir": run_path, "judge_counts": judge_counts, **summary}
