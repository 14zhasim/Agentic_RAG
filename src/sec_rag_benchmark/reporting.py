"""Calculate benchmark summaries and persist the JSON and Excel reports."""

from __future__ import annotations

import json
from pathlib import Path
import tomllib
from typing import Any

import pandas as pd

from .report_workbook import write_report_workbook


REPORT_VIEWS = (
    "overall",
    "by_condition",
    "by_generation_method",
    "by_cognitive_skill",
    "cross_tab",
)
RETRIEVAL_CONDITIONS = {"single_store", "shared_store"}


def _empty_report_views() -> dict[str, list[dict[str, Any]]]:
    """Create the five named lists used by each metric family."""
    return {view: [] for view in REPORT_VIEWS}


def _latest_rows_by_job_id(path: Path) -> dict[str, dict[str, Any]]:
    """Read an append-only JSONL and retain the last row for each job."""
    latest_rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return latest_rows

    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            row = json.loads(line)
            latest_rows[row["job_id"]] = row
    return latest_rows


def _expand_by_cognitive_skill(
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Make one reporting-only row for every skill on every prediction."""
    expanded_predictions: list[dict[str, Any]] = []
    for prediction in predictions:
        for skill in prediction["cognitive_skills"]:
            expanded_predictions.append({**prediction, "cognitive_skill": skill})
    return expanded_predictions


def _average_retrieval_metrics(
    report_view: str,
    prediction_subset: pd.DataFrame,
    **subset_identity: str,
) -> dict[str, Any]:
    """Average retrieval metrics for one named subset of predictions."""
    report_row = {
        "report_view": report_view,
        **subset_identity,
        "total_predictions": len(prediction_subset),
    }
    for metric_name in ("page_recall", "page_precision", "page_mrr"):
        values = pd.to_numeric(prediction_subset[metric_name], errors="coerce").dropna()
        report_row[metric_name] = None if values.empty else float(values.mean())
        report_row[f"{metric_name}_sample_size"] = len(values)
    return report_row


def _retrieval_metric_views(
    successful_predictions: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Build five views using only jobs that actually executed retrieval."""
    views = _empty_report_views()
    retrieval_predictions = [
        row
        for row in successful_predictions
        if row["eval_mode"] in RETRIEVAL_CONDITIONS
    ]
    if not retrieval_predictions:
        return views

    predictions_table = pd.DataFrame(retrieval_predictions)
    skill_table = pd.DataFrame(_expand_by_cognitive_skill(retrieval_predictions))
    views["overall"].append(
        _average_retrieval_metrics("overall", predictions_table)
    )

    for condition, condition_rows in predictions_table.groupby("eval_mode", sort=True):
        views["by_condition"].append(
            _average_retrieval_metrics(
                "condition", condition_rows, eval_mode=condition
            )
        )

        # Detailed comparisons stay inside their condition. Otherwise an
        # oracle result and a single-store result would be averaged together.
        for question_type, rows in condition_rows.groupby("question_type", sort=True):
            views["by_generation_method"].append(
                _average_retrieval_metrics(
                    "generation_method",
                    rows,
                    eval_mode=condition,
                    question_type=question_type,
                )
            )

    for condition, condition_rows in skill_table.groupby("eval_mode", sort=True):
        for skill, rows in condition_rows.groupby("cognitive_skill", sort=True):
            views["by_cognitive_skill"].append(
                _average_retrieval_metrics(
                    "cognitive_skill",
                    rows,
                    eval_mode=condition,
                    cognitive_skill=skill,
                )
            )

        cross_tab = condition_rows.groupby(
            ["question_type", "cognitive_skill"], sort=True
        )
        for (question_type, skill), rows in cross_tab:
            views["cross_tab"].append(
                _average_retrieval_metrics(
                    "cross_tab",
                    rows,
                    eval_mode=condition,
                    question_type=question_type,
                    cognitive_skill=skill,
                )
            )
    return views


def _answer_accuracy_for_subset(
    report_view: str,
    predictions: list[dict[str, Any]],
    judgments: dict[str, dict[str, Any]],
    **subset_identity: str,
) -> dict[str, Any]:
    """Count answer outcomes and calculate accuracy for one report subset."""
    correct = 0
    agreed_judgments = 0
    disagreements = 0
    unjudged = 0
    did_not_fit = 0

    for prediction in predictions:
        if prediction.get("status") == "did_not_fit":
            did_not_fit += 1
            continue

        judgment = judgments.get(prediction["job_id"])
        if judgment is None:
            unjudged += 1
        elif judgment.get("accuracy") is None:
            disagreements += 1
        else:
            agreed_judgments += 1
            correct += judgment["accuracy"]

    including_did_not_fit_denominator = agreed_judgments + did_not_fit
    return {
        "report_view": report_view,
        **subset_identity,
        "total_predictions": len(predictions),
        "agreed_judgments": agreed_judgments,
        "disagreements": disagreements,
        "unjudged": unjudged,
        "did_not_fit": did_not_fit,
        "accuracy_excluding_did_not_fit": (
            correct / agreed_judgments if agreed_judgments else None
        ),
        "accuracy_including_did_not_fit": (
            correct / including_did_not_fit_denominator
            if including_did_not_fit_denominator
            else None
        ),
    }


def _answer_accuracy_views(
    predictions: list[dict[str, Any]], judgments: dict[str, dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    """Build the five condition-aware answer-accuracy views."""
    views = _empty_report_views()
    if not predictions:
        return views

    views["overall"].append(
        _answer_accuracy_for_subset("overall", predictions, judgments)
    )

    conditions = sorted({row["eval_mode"] for row in predictions})
    for condition in conditions:
        condition_rows = [row for row in predictions if row["eval_mode"] == condition]
        views["by_condition"].append(
            _answer_accuracy_for_subset(
                "condition", condition_rows, judgments, eval_mode=condition
            )
        )

        question_types = sorted({row["question_type"] for row in condition_rows})
        for question_type in question_types:
            selected = [
                row for row in condition_rows if row["question_type"] == question_type
            ]
            views["by_generation_method"].append(
                _answer_accuracy_for_subset(
                    "generation_method",
                    selected,
                    judgments,
                    eval_mode=condition,
                    question_type=question_type,
                )
            )

        skills = sorted(
            {skill for row in condition_rows for skill in row["cognitive_skills"]}
        )
        for skill in skills:
            selected = [row for row in condition_rows if skill in row["cognitive_skills"]]
            views["by_cognitive_skill"].append(
                _answer_accuracy_for_subset(
                    "cognitive_skill",
                    selected,
                    judgments,
                    eval_mode=condition,
                    cognitive_skill=skill,
                )
            )

        for question_type in question_types:
            for skill in skills:
                selected = [
                    row
                    for row in condition_rows
                    if row["question_type"] == question_type
                    and skill in row["cognitive_skills"]
                ]
                if selected:
                    views["cross_tab"].append(
                        _answer_accuracy_for_subset(
                            "cross_tab",
                            selected,
                            judgments,
                            eval_mode=condition,
                            question_type=question_type,
                            cognitive_skill=skill,
                        )
                    )
    return views


def write_report(run_dir: str | Path) -> dict[str, Any]:
    """Calculate one run's report data, then write its JSON and workbook."""
    run_path = Path(run_dir)
    latest_predictions = _latest_rows_by_job_id(run_path / "predictions.jsonl")
    latest_errors = _latest_rows_by_job_id(run_path / "errors.jsonl")
    latest_judgments = _latest_rows_by_job_id(run_path / "judgments.jsonl")

    terminal_predictions = list(latest_predictions.values())
    successful_predictions = [
        row for row in terminal_predictions if row.get("status") == "success"
    ]
    did_not_fit_predictions = [
        row for row in terminal_predictions if row.get("status") == "did_not_fit"
    ]

    snapshot = tomllib.loads((run_path / "config.toml").read_text(encoding="utf-8"))
    planned_jobs = snapshot["selection"]["limit"] * len(
        snapshot["selection"]["conditions"]
    )
    attempted_job_ids = set(latest_predictions) | set(latest_errors)
    failed_job_ids = set(latest_errors) - set(latest_predictions)
    run_config = snapshot.get("run", {})

    summary = {
        "run_status": {
            # Legacy smoke-test directories predate explicit run labels. New
            # configurations require both fields, while old reports stay readable.
            "experiment": run_config.get("experiment", "legacy-unlabelled"),
            "variant": run_config.get("variant", "legacy-unlabelled"),
            "planned": planned_jobs,
            "successful": len(successful_predictions),
            "did_not_fit": len(did_not_fit_predictions),
            "failed": len(failed_job_ids),
            "missing": max(0, planned_jobs - len(attempted_job_ids)),
            "complete": (
                len(successful_predictions) + len(did_not_fit_predictions)
                == planned_jobs
            ),
        },
        "answer_accuracy": _answer_accuracy_views(
            terminal_predictions, latest_judgments
        ),
        "retrieval_metrics": _retrieval_metric_views(successful_predictions),
    }

    (run_path / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    write_report_workbook(run_path / "summary.xlsx", summary)

    # Remove the previous design's CSV so one run folder cannot contain two
    # conflicting human-readable report formats after regeneration.
    old_csv = run_path / "summary.csv"
    if old_csv.exists():
        old_csv.unlink()
    return summary
