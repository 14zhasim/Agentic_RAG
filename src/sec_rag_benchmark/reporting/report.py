"""Calculate benchmark summaries and persist the JSON and Excel reports."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

import pandas as pd

from .failure_analysis import build_failure_analysis, summarize_failure_analysis
from .workbook import write_report_workbook

REPORT_VIEWS = (
    "overall",
    "by_condition",
    "by_generation_method",
    "by_cognitive_skill",
    "cross_tab",
)
RETRIEVAL_CONDITIONS = {"single_store", "shared_store"}
# The first three are measured on the reranked top-k the model saw; the
# pre-rerank three on the fused top-k before the reranker (Build Order 2.5).
RETRIEVAL_METRICS = (
    "page_recall",
    "page_precision",
    "page_mrr",
    "pre_rerank_page_recall",
    "pre_rerank_page_precision",
    "pre_rerank_page_mrr",
)


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


def _resolve_judgments(
    judgments: dict[str, dict[str, Any]],
    manual_reviews: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Fill disputed accuracies from human reviews without changing source rows."""
    resolved_judgments: dict[str, dict[str, Any]] = {}
    for job_id, judgment in judgments.items():
        resolved_judgment = dict(judgment)

        # An agreed automated verdict remains authoritative. Human review is
        # used only where the approved two-pass method produced no verdict.
        if judgment.get("accuracy") is None and job_id in manual_reviews:
            human_accuracy = manual_reviews[job_id].get("human_accuracy")
            if type(human_accuracy) is not int or human_accuracy not in {0, 1}:
                raise ValueError(
                    f"Manual review for {job_id} has invalid human_accuracy"
                )
            resolved_judgment["accuracy"] = human_accuracy

        resolved_judgments[job_id] = resolved_judgment
    return resolved_judgments


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
    """Average retrieval metrics for one named subset of predictions.

    Each metric is averaged over the rows that have it, with the count
    reported beside it. A column absent from every row (a run written
    before Exp1 added it) reports None with sample size 0, so older run
    folders stay reportable.
    """
    report_row: dict[str, Any] = {
        "report_view": report_view,
        **subset_identity,
        "total_predictions": len(prediction_subset),
    }
    for metric_name in RETRIEVAL_METRICS:
        if metric_name in prediction_subset:
            values = pd.to_numeric(
                prediction_subset[metric_name], errors="coerce"
            ).dropna()
        else:
            values = pd.Series(dtype=float)
        report_row[metric_name] = None if values.empty else float(values.mean())
        report_row[f"{metric_name}_sample_size"] = len(values)

    # filter_correct is True/False in shared_store and None elsewhere: filter
    # accuracy only means something when there is a filing to choose
    # (Build Order 2.5), so its sample is shared-store rows alone.
    # pd.api.types.is_bool also accepts numpy's bool, which pandas yields
    # when a column holds only True/False (a shared-store-only run).
    flags: list[bool] = []
    if "filter_correct" in prediction_subset:
        flags = [
            bool(value)
            for value in prediction_subset["filter_correct"]
            if pd.api.types.is_bool(value)
        ]
    report_row["filter_accuracy"] = sum(flags) / len(flags) if flags else None
    report_row["filter_accuracy_sample_size"] = len(flags)

    invalid_replies = 0
    if "enhancement_status" in prediction_subset:
        invalid_replies = int(
            (prediction_subset["enhancement_status"] == "invalid_reply").sum()
        )
    report_row["invalid_reply_count"] = invalid_replies
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
    views["overall"].append(_average_retrieval_metrics("overall", predictions_table))

    for condition, condition_rows in predictions_table.groupby("eval_mode", sort=True):
        views["by_condition"].append(
            _average_retrieval_metrics("condition", condition_rows, eval_mode=condition)
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
    correct_answers = 0
    scored_answers = 0
    unresolved = 0
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
            unresolved += 1
        else:
            scored_answers += 1
            correct_answers += judgment["accuracy"]

    including_did_not_fit_denominator = scored_answers + did_not_fit
    return {
        "report_view": report_view,
        **subset_identity,
        "total_predictions": len(predictions),
        "scored_answers": scored_answers,
        "correct_answers": correct_answers,
        "unresolved": unresolved,
        "unjudged": unjudged,
        "did_not_fit": did_not_fit,
        "review_complete": unresolved == 0 and unjudged == 0,
        "accuracy_excluding_did_not_fit": (
            correct_answers / scored_answers if scored_answers else None
        ),
        "accuracy_including_did_not_fit": (
            correct_answers / including_did_not_fit_denominator
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
            selected = [
                row for row in condition_rows if skill in row["cognitive_skills"]
            ]
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


def _generation_performance_for_subset(
    predictions: list[dict[str, Any]],
    **subset_identity: str,
) -> dict[str, Any]:
    """Summarise saved GLM cost and latency for successful answers."""
    successful_predictions = [
        row for row in predictions if row.get("status") == "success"
    ]
    costs = [
        float(row["cost"])
        for row in successful_predictions
        if isinstance(row.get("cost"), int | float)
    ]
    latencies = [
        float(row["latency_seconds"])
        for row in successful_predictions
        if isinstance(row.get("latency_seconds"), int | float)
    ]
    question_ids_with_cost = {
        row["financebench_id"]
        for row in successful_predictions
        if isinstance(row.get("cost"), int | float)
    }
    total_cost = sum(costs) if costs else None

    # Retrieval's own spend and time (query enhancement + Voyage), kept apart
    # from the answer call's `cost` so the two can be compared (Build Order
    # 2.5). Rows from runs without retrieval have neither field.
    retrieval_costs = [
        float(row["retrieval_cost_usd"])
        for row in successful_predictions
        if isinstance(row.get("retrieval_cost_usd"), int | float)
    ]
    retrieval_latencies = [
        float(row["retrieval_latency_seconds"])
        for row in successful_predictions
        if isinstance(row.get("retrieval_latency_seconds"), int | float)
    ]
    total_retrieval_cost = sum(retrieval_costs) if retrieval_costs else None

    return {
        **subset_identity,
        "successful_answers": len(successful_predictions),
        "distinct_questions": len(
            {row["financebench_id"] for row in successful_predictions}
        ),
        "cost_sample_size": len(costs),
        "total_cost_usd": total_cost,
        "average_cost_per_answer_usd": (
            total_cost / len(costs) if total_cost is not None else None
        ),
        "average_cost_per_question_usd": (
            total_cost / len(question_ids_with_cost)
            if total_cost is not None and question_ids_with_cost
            else None
        ),
        "latency_sample_size": len(latencies),
        "average_latency_per_answer_seconds": (
            sum(latencies) / len(latencies) if latencies else None
        ),
        "retrieval_cost_sample_size": len(retrieval_costs),
        "total_retrieval_cost_usd": total_retrieval_cost,
        "average_retrieval_cost_per_answer_usd": (
            total_retrieval_cost / len(retrieval_costs)
            if total_retrieval_cost is not None
            else None
        ),
        "average_retrieval_latency_seconds": (
            sum(retrieval_latencies) / len(retrieval_latencies)
            if retrieval_latencies
            else None
        ),
    }


def _generation_performance_views(
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build overall and condition-level GLM operating metrics."""
    successful_predictions = [
        row for row in predictions if row.get("status") == "success"
    ]
    by_condition = []
    for condition in sorted({row["eval_mode"] for row in successful_predictions}):
        condition_predictions = [
            row for row in successful_predictions if row["eval_mode"] == condition
        ]
        by_condition.append(
            _generation_performance_for_subset(
                condition_predictions,
                eval_mode=condition,
            )
        )
    return {
        "overall": _generation_performance_for_subset(successful_predictions),
        "by_condition": by_condition,
    }


def _execution_status(
    snapshot: dict[str, Any],
    predictions: dict[str, dict[str, Any]],
    errors: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Count current job states overall and for each selected condition."""
    conditions = snapshot["selection"]["conditions"]
    jobs_per_condition = snapshot["selection"]["limit"]
    planned_jobs = jobs_per_condition * len(conditions)
    terminal_rows = list(predictions.values())
    failed_job_ids = set(errors) - set(predictions)
    attempted_job_ids = set(predictions) | set(errors)

    successful = sum(row.get("status") == "success" for row in terminal_rows)
    did_not_fit = sum(row.get("status") == "did_not_fit" for row in terminal_rows)
    run_config = snapshot.get("run", {})
    status = {
        # Legacy smoke-test directories predate explicit run labels. New
        # configurations require both fields, while old reports stay readable.
        "experiment": run_config.get("experiment", "legacy-unlabelled"),
        "variant": run_config.get("variant", "legacy-unlabelled"),
        "planned": planned_jobs,
        "successful": successful,
        "did_not_fit": did_not_fit,
        "failed": len(failed_job_ids),
        "missing": max(0, planned_jobs - len(attempted_job_ids)),
        "complete": successful + did_not_fit == planned_jobs,
        "by_condition": [],
    }

    for condition in sorted(conditions):
        condition_predictions = [
            row for row in terminal_rows if row.get("eval_mode") == condition
        ]
        condition_prediction_ids = {row["job_id"] for row in condition_predictions}
        condition_failed_ids = {
            job_id
            for job_id in failed_job_ids
            if job_id.rsplit(":", 1)[-1] == condition
        }
        condition_successful = sum(
            row.get("status") == "success" for row in condition_predictions
        )
        condition_did_not_fit = sum(
            row.get("status") == "did_not_fit" for row in condition_predictions
        )
        condition_attempted = condition_prediction_ids | condition_failed_ids
        status["by_condition"].append(
            {
                "eval_mode": condition,
                "planned": jobs_per_condition,
                "successful": condition_successful,
                "did_not_fit": condition_did_not_fit,
                "failed": len(condition_failed_ids),
                "missing": max(0, jobs_per_condition - len(condition_attempted)),
                "complete": condition_successful + condition_did_not_fit
                == jobs_per_condition,
            }
        )
    return status


def _borrowed_oracle(
    oracle_run_dir: Path,
    snapshot: dict[str, Any],
    own_predictions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Read another run's oracle answers and their resolved judgments.

    Build Order 2.6: Exp1 runs only its two retrieval conditions, and the
    17 September baseline already answered oracle at today's settings.
    Failure diagnosis needs an oracle verdict per question, so it borrows
    that run's. The borrowed run's own manual reviews resolve its disputed
    judgments, exactly as they would in its own report.
    """
    if any(row.get("eval_mode") == "oracle" for row in own_predictions):
        raise ValueError(
            "This run has its own oracle answers; do not borrow another run's"
        )

    borrowed_snapshot = tomllib.loads(
        (oracle_run_dir / "config.toml").read_text(encoding="utf-8")
    )
    # The comparison isolates retrieval only if the same model answered
    # with the same settings (CLAUDE.md -> Invariants).
    if borrowed_snapshot.get("generation") != snapshot.get("generation"):
        raise ValueError(
            f"Oracle run {oracle_run_dir.name} used different generation settings"
        )

    oracle_rows = [
        row
        for row in _latest_rows_by_job_id(oracle_run_dir / "predictions.jsonl").values()
        if row.get("eval_mode") == "oracle"
    ]
    borrowed_judgments = _resolve_judgments(
        _latest_rows_by_job_id(oracle_run_dir / "judgments.jsonl"),
        _latest_rows_by_job_id(oracle_run_dir / "manual_reviews.jsonl"),
    )
    oracle_judgments = {
        row["job_id"]: borrowed_judgments[row["job_id"]]
        for row in oracle_rows
        if row["job_id"] in borrowed_judgments
    }
    return oracle_rows, oracle_judgments


def write_report(
    run_dir: str | Path, oracle_run_dir: str | Path | None = None
) -> dict[str, Any]:
    """Calculate one run's report data, then write its JSON and workbook.

    With `oracle_run_dir`, failure diagnosis compares this run's retrieval
    answers against that run's oracle answers. Those borrowed answers are
    used only there: accuracy, retrieval and cost views are this run's own.
    """
    run_path = Path(run_dir)
    latest_predictions = _latest_rows_by_job_id(run_path / "predictions.jsonl")
    latest_errors = _latest_rows_by_job_id(run_path / "errors.jsonl")
    latest_judgments = _latest_rows_by_job_id(run_path / "judgments.jsonl")
    latest_manual_reviews = _latest_rows_by_job_id(run_path / "manual_reviews.jsonl")
    resolved_judgments = _resolve_judgments(latest_judgments, latest_manual_reviews)

    terminal_predictions = list(latest_predictions.values())
    successful_predictions = [
        row for row in terminal_predictions if row.get("status") == "success"
    ]

    snapshot = tomllib.loads((run_path / "config.toml").read_text(encoding="utf-8"))
    if "selection" not in snapshot:
        raise ValueError(
            "Report directory is not a benchmark run: config.toml has no "
            "[selection] section"
        )
    retrieval_depth = snapshot.get("run", {}).get("retrieval_depth")
    if type(retrieval_depth) is not int or retrieval_depth <= 0:
        raise ValueError("Benchmark run config has no valid retrieval_depth")

    failure_predictions = terminal_predictions
    failure_judgments = resolved_judgments
    oracle_source = None
    if oracle_run_dir is not None:
        borrowed_rows, borrowed_judgments = _borrowed_oracle(
            Path(oracle_run_dir), snapshot, terminal_predictions
        )
        # Job IDs start with each run's own key, so the two judgment maps
        # cannot collide when merged.
        failure_predictions = terminal_predictions + borrowed_rows
        failure_judgments = {**resolved_judgments, **borrowed_judgments}
        oracle_source = Path(oracle_run_dir).name

    failure_rows = build_failure_analysis(
        failure_predictions,
        failure_judgments,
        top_k=retrieval_depth,
    )
    failure_summary = {
        **summarize_failure_analysis(failure_rows),
        "oracle_source": oracle_source,
    }

    summary = {
        "run_status": _execution_status(snapshot, latest_predictions, latest_errors),
        "answer_accuracy": _answer_accuracy_views(
            terminal_predictions, resolved_judgments
        ),
        "retrieval_metrics": _retrieval_metric_views(successful_predictions),
        "generation_performance": _generation_performance_views(terminal_predictions),
        "failure_analysis": failure_summary,
    }

    # Failure analysis is derived from the latest predictions and judgments.
    # Replace it on every report run rather than preserving stale attempts.
    (run_path / "failure_analysis.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in failure_rows),
        encoding="utf-8",
    )
    (run_path / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    write_report_workbook(run_path / "summary.xlsx", summary, failure_rows)

    # Remove the previous design's CSV so one run folder cannot contain two
    # conflicting human-readable report formats after regeneration.
    old_csv = run_path / "summary.csv"
    if old_csv.exists():
        old_csv.unlink()
    return summary
