"""Aggregate checkpointed benchmark predictions into report files."""

from __future__ import annotations

import json
from pathlib import Path
import tomllib
from typing import Any

import pandas as pd


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
    """Average the retrieval metrics for one clearly identified subset."""
    report_row = {
        "report_view": report_view,
        **subset_identity,
        "total_predictions": len(prediction_subset),
    }
    for metric_name in ("page_recall", "page_precision", "page_mrr"):
        # Non-retrieval conditions store None. Excluding those values gives
        # each metric an honest denominator, which is recorded beside it.
        applicable_values = pd.to_numeric(
            prediction_subset[metric_name], errors="coerce"
        ).dropna()
        report_row[metric_name] = (
            None if applicable_values.empty else float(applicable_values.mean())
        )
        report_row[f"{metric_name}_sample_size"] = len(applicable_values)
    return report_row


def _answer_accuracy_for_subset(
    report_view: str, #type of grouping being summarised e.g. overall, condition, generation_method, cognitive_skill
    predictions: list[dict[str, Any]], #select predictions on a condition e.g. if prediction["eval_mode"] == "oracle"
    judgments: dict[str, dict[str, Any]], #contains all completed judgements
    **subset_identity: str, # ** means any name arguments not listed separately in one dictionary
                            # e.g. subset_identity = {"question_type": "calculated", "cognitive_skill": "numerical_reasoning"}
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


def _answer_accuracy_rows(
    predictions: list[dict[str, Any]], judgments: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Build the five approved answer-accuracy report views explicitly."""
    rows = [_answer_accuracy_for_subset("overall", predictions, judgments)]

    # 1. One row per context condition.
    conditions = sorted({row["eval_mode"] for row in predictions})
    for condition in conditions:
        selected = [row for row in predictions if row["eval_mode"] == condition]
        rows.append(
            _answer_accuracy_for_subset(
                "condition", selected, judgments, eval_mode=condition
            )
        )

    # 2. One row per FinanceBench generation method/question type.
    question_types = sorted({row["question_type"] for row in predictions})
    for question_type in question_types:
        selected = [
            row for row in predictions if row["question_type"] == question_type
        ]
        rows.append(
            _answer_accuracy_for_subset(
                "generation_method",
                selected,
                judgments,
                question_type=question_type,
            )
        )

    # 3. A multi-skill prediction contributes once to each of its skills.
    skills = sorted(
        {skill for row in predictions for skill in row["cognitive_skills"]}
    )
    for skill in skills:
        selected = [row for row in predictions if skill in row["cognitive_skills"]]
        rows.append(
            _answer_accuracy_for_subset(
                "cognitive_skill", selected, judgments, cognitive_skill=skill
            )
        )

    # 4. One row per generation-method/cognitive-skill combination.
    for question_type in question_types:
        for skill in skills:
            selected = [
                row
                for row in predictions
                if row["question_type"] == question_type
                and skill in row["cognitive_skills"]
            ]
            if selected:
                rows.append(
                    _answer_accuracy_for_subset(
                        "cross_tab",
                        selected,
                        judgments,
                        question_type=question_type,
                        cognitive_skill=skill,
                    )
                )
    return rows


def write_report(run_dir: str | Path) -> dict[str, Any]:
    """Write overall and segmented summaries from one run's saved attempts."""
    run_path = Path(run_dir)
    latest_predictions = _latest_rows_by_job_id(run_path / "predictions.jsonl")
    latest_errors = _latest_rows_by_job_id(run_path / "errors.jsonl")
    latest_judgments = _latest_rows_by_job_id(run_path / "judgments.jsonl")
    terminal_predictions = list(latest_predictions.values())
    successful_predictions = [
        row for row in latest_predictions.values() if row.get("status") == "success"
    ]
    did_not_fit_predictions = [
        row for row in latest_predictions.values() if row.get("status") == "did_not_fit"
    ]

    predictions_table = pd.DataFrame(successful_predictions)
    skill_predictions_table = pd.DataFrame(
        _expand_by_cognitive_skill(successful_predictions)
    )
    report_rows: list[dict[str, Any]] = []

    if not predictions_table.empty:
        # 1. Overall: all successful predictions in one group.
        report_rows.append(_average_retrieval_metrics("overall", predictions_table))

        # 2. Condition: one group for each context condition.
        for eval_mode, rows in predictions_table.groupby("eval_mode", sort=True):
            report_rows.append(
                _average_retrieval_metrics("condition", rows, eval_mode=eval_mode)
            )

        # 3. Generation method: one group for each FinanceBench question type.
        for question_type, rows in predictions_table.groupby("question_type", sort=True):
            report_rows.append(
                _average_retrieval_metrics(
                    "generation_method", rows, question_type=question_type
                )
            )

        # 4. Cognitive skill: one group per normalized skill. Multi-skill
        # predictions intentionally appear in more than one group.
        for skill, rows in skill_predictions_table.groupby("cognitive_skill", sort=True):
            report_rows.append(
                _average_retrieval_metrics("cognitive_skill", rows, cognitive_skill=skill)
            )

        # 5. Cross-tab: one group for every question-type/skill combination.
        cross_tab_groups = skill_predictions_table.groupby(
            ["question_type", "cognitive_skill"], sort=True
        )
        for (question_type, skill), rows in cross_tab_groups:
            report_rows.append(
                _average_retrieval_metrics(
                    "cross_tab",
                    rows,
                    question_type=question_type,
                    cognitive_skill=skill,
                )
            )

    snapshot = tomllib.loads((run_path / "config.toml").read_text(encoding="utf-8"))
    planned_jobs = snapshot["selection"]["limit"] * len(
        snapshot["selection"]["conditions"]
    )
    attempted_job_ids = set(latest_predictions) | set(latest_errors)
    failed_job_ids = set(latest_errors) - set(latest_predictions)
    summary = {
        "planned": planned_jobs,
        "successful": len(successful_predictions),
        "did_not_fit": len(did_not_fit_predictions),
        "failed": len(failed_job_ids),
        "missing": max(0, planned_jobs - len(attempted_job_ids)),
        "complete": len(successful_predictions) + len(did_not_fit_predictions)
        == planned_jobs,
        "report_rows": report_rows,
        "accuracy_rows": _answer_accuracy_rows(
            terminal_predictions, latest_judgments
        ),
    }
    (run_path / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    csv_rows = [
        {"metric_family": "retrieval", **row} for row in report_rows
    ] + [
        {"metric_family": "answer_accuracy", **row}
        for row in summary["accuracy_rows"]
    ]
    pd.DataFrame(csv_rows).to_csv(run_path / "summary.csv", index=False)
    return summary
