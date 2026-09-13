"""Calculate retrieval metrics and aggregate saved predictions.

Final-answer accuracy is deliberately absent. As described in the implementation
guide, a later Azure/RAGAS pass will judge saved answers separately.
"""

from __future__ import annotations

import json
from pathlib import Path
import tomllib
from typing import Any

import pandas as pd


def page_metrics(
    gold: list[tuple[str, int]], chunks: list[dict[str, Any]], top_k: int
) -> dict[str, float]:
    """Calculate the guide's document-aware page recall, precision, and MRR."""
    gold_pages = set(gold)
    if not gold_pages:
        raise ValueError("Gold pages cannot be empty")

    # Metrics apply only to the first top_k results, in retrieval-rank order.
    ranked_chunks = sorted(chunks, key=lambda chunk: chunk["rank"])[:top_k]

    # Recall and precision evaluate pages, not chunks. Convert every chunk's
    # pages into (document, page-index) pairs. The set counts a page once even
    # when several chunks overlap that same page.
    unique_retrieved_pages: set[tuple[str, int]] = set()
    for chunk in ranked_chunks:
        for page_index in chunk["pages"]:
            unique_retrieved_pages.add((chunk["doc_name"], page_index))

    retrieved_gold_pages = gold_pages.intersection(unique_retrieved_pages)
    number_of_gold_pages_retrieved = len(retrieved_gold_pages)
    page_recall = number_of_gold_pages_retrieved / len(gold_pages)
    if unique_retrieved_pages:
        page_precision = (
            number_of_gold_pages_retrieved / len(unique_retrieved_pages)
        )
    else:
        page_precision = 0.0

    # MRR evaluates chunk order. Find the first chunk that covers at least one
    # gold (document, page-index) pair, then stop because later chunks cannot
    # improve the first relevant rank.
    first_relevant_chunk_rank: int | None = None
    for chunk in ranked_chunks:
        pages_covered_by_chunk = {
            (chunk["doc_name"], page_index) for page_index in chunk["pages"]
        }
        chunk_covers_gold_page = bool(
            gold_pages.intersection(pages_covered_by_chunk)
        )
        if chunk_covers_gold_page:
            first_relevant_chunk_rank = chunk["rank"]
            break

    if first_relevant_chunk_rank is None:
        page_mrr = 0.0
    else:
        page_mrr = 1 / first_relevant_chunk_rank

    return {
        "page_recall": page_recall,
        "page_precision": page_precision,
        "page_mrr": page_mrr,
    }


SKILLS = (
    ("information extraction", "information_extraction"),
    ("numerical reasoning", "numerical_reasoning"),
    ("logical reasoning", "logical_reasoning"),
)


def cognitive_skills(value: str | None) -> list[str]:
    """Normalize FinanceBench's free-text reasoning labels for segmentation."""
    normalized = (value or "").casefold()
    found = [label for phrase, label in SKILLS if phrase in normalized]
    return found or ["unspecified"]


def _average_retrieval_metrics(
    report_view: str,
    prediction_subset: pd.DataFrame,
    **subset_identity: str,
) -> dict[str, Any]:
    """Create one row of the benchmark's summary report.

    ``write_report`` calls this helper for five report views:

    - overall: every successful prediction;
    - condition: one subset per ``eval_mode``;
    - generation_method: one subset per ``question_type``;
    - cognitive_skill: one subset per normalized skill;
    - cross_tab: one subset per question-type/skill combination.

    For example, ``report_view="condition"``, a DataFrame containing only
    single-store predictions, and ``eval_mode="single_store"`` produce one row
    containing the average retrieval metrics for that condition.
    """
    report_row = {
        "report_view": report_view,
        **subset_identity,
        "total_predictions": len(prediction_subset),
    }
    for metric_name in ("page_recall", "page_precision", "page_mrr"):
        # Closed-book, oracle, and long-context rows contain no retrieval
        # metrics. Drop those missing values before averaging, then record how
        # many retrieval rows actually contributed to this particular average.
        applicable_values = pd.to_numeric(
            prediction_subset[metric_name], errors="coerce"
        ).dropna()
        if applicable_values.empty:
            average = None
        else:
            average = float(applicable_values.mean())
        report_row[metric_name] = average
        report_row[f"{metric_name}_sample_size"] = len(applicable_values)
    return report_row


def write_report(run_dir: str | Path) -> dict[str, Any]:
    """Turn saved predictions into the five report views defined above."""
    run_dir = Path(run_dir)

    # predictions.jsonl is append-only, so it contains attempts rather than a
    # guaranteed single final row per job. First turn each JSON line back into
    # the prediction dictionary that runner.py originally saved.
    prediction_attempts: list[dict[str, Any]] = []
    prediction_lines = (run_dir / "predictions.jsonl").read_text().splitlines()
    for line in prediction_lines:
        if line:
            prediction_attempts.append(json.loads(line))

    # A resumed run may contain several attempts. The last row for a job ID is
    # its current result, matching the runner's checkpoint/resume behavior.
    latest_prediction_by_job_id: dict[str, dict[str, Any]] = {}
    for prediction in prediction_attempts:
        latest_prediction_by_job_id[prediction["job_id"]] = prediction

    successful_predictions = []
    for prediction in latest_prediction_by_job_id.values():
        if prediction.get("status") == "success":
            successful_predictions.append(prediction)

    error_path = run_dir / "errors.jsonl"
    error_attempts: list[dict[str, Any]] = []
    if error_path.exists():
        for line in error_path.read_text().splitlines():
            if line:
                error_attempts.append(json.loads(line))
    latest_error_by_job_id: dict[str, dict[str, Any]] = {}
    for error in error_attempts:
        latest_error_by_job_id[error["job_id"]] = error

    # Multi-label questions appear once in every applicable skill group. Their
    # skill-group counts therefore are intentionally not additive. For example,
    # one prediction with two skills becomes two rows in this reporting-only
    # list, while successful_predictions still contains it only once.
    predictions_by_skill: list[dict[str, Any]] = []
    for prediction in successful_predictions:
        for skill in prediction["cognitive_skills"]:
            prediction_with_one_skill = {**prediction, "cognitive_skill": skill}
            predictions_by_skill.append(prediction_with_one_skill)

    predictions_table = pd.DataFrame(successful_predictions)
    skill_predictions_table = pd.DataFrame(predictions_by_skill)
    report_rows: list[dict[str, Any]] = []
    if not predictions_table.empty:
        # 1. One result for the entire benchmark run.
        report_rows.append(
            _average_retrieval_metrics("overall", predictions_table)
        )

        # 2. One result for each context condition, such as oracle or
        # single_store. `condition_rows` contains only that condition's answers.
        condition_groups = predictions_table.groupby("eval_mode", sort=True)
        for eval_mode, condition_rows in condition_groups:
            report_rows.append(
                _average_retrieval_metrics(
                    "condition", condition_rows, eval_mode=eval_mode
                )
            )

        # 3. One result for each FinanceBench generation method/question type.
        generation_method_groups = predictions_table.groupby(
            "question_type", sort=True
        )
        for question_type, question_rows in generation_method_groups:
            report_rows.append(
                _average_retrieval_metrics(
                    "generation_method", question_rows, question_type=question_type
                )
            )

        # 4. One result for each cognitive skill. skill_predictions_table has
        # one copy of a prediction for every skill assigned to its question.
        cognitive_skill_groups = skill_predictions_table.groupby(
            "cognitive_skill", sort=True
        )
        for skill, skill_rows in cognitive_skill_groups:
            report_rows.append(
                _average_retrieval_metrics(
                    "cognitive_skill", skill_rows, cognitive_skill=skill
                )
            )

        # 5. One result for each generation-method/skill combination, such as
        # retrieval questions that require numerical reasoning.
        generation_method_and_skill_groups = skill_predictions_table.groupby(
            ["question_type", "cognitive_skill"], sort=True
        )
        for (
            question_type,
            skill,
        ), cross_tab_rows in generation_method_and_skill_groups:
            report_rows.append(
                _average_retrieval_metrics(
                    "cross_tab",
                    cross_tab_rows,
                    question_type=question_type,
                    cognitive_skill=skill,
                )
            )
    snapshot = tomllib.loads((run_dir / "config.toml").read_text())
    planned = snapshot["selection"]["limit"] * len(snapshot["selection"]["conditions"])
    attempted_ids = set(latest_prediction_by_job_id) | set(latest_error_by_job_id)
    summary = {
        "planned": planned,
        "successful": len(successful_predictions),
        "failed": len(
            set(latest_error_by_job_id) - set(latest_prediction_by_job_id)
        ),
        "missing": max(0, planned - len(attempted_ids)),
        "complete": len(successful_predictions) == planned,
        "report_rows": report_rows,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame(report_rows).to_csv(run_dir / "summary.csv", index=False)
    return summary
