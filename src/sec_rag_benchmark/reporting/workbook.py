"""Render calculated benchmark summaries as a readable Excel workbook."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import xlsxwriter

ANSWER_COLUMNS = [
    "total_predictions",
    "scored_answers",
    "correct_answers",
    "unresolved",
    "unjudged",
    "did_not_fit",
    "review_complete",
    "accuracy_excluding_did_not_fit",
    "accuracy_including_did_not_fit",
]

RETRIEVAL_COLUMNS = [
    "total_predictions",
    "page_recall",
    "page_recall_sample_size",
    "page_precision",
    "page_precision_sample_size",
    "page_mrr",
    "page_mrr_sample_size",
]

STATUS_COLUMNS = [
    "eval_mode",
    "planned",
    "successful",
    "did_not_fit",
    "failed",
    "missing",
    "complete",
]

PERFORMANCE_CONDITION_COLUMNS = [
    "successful_answers",
    "total_cost_usd",
    "average_cost_per_answer_usd",
    "average_latency_per_answer_seconds",
]

FAILURE_DETAIL_COLUMNS = [
    "financebench_id",
    "eval_mode",
    "analysis_status",
    "oracle_accuracy",
    "condition_accuracy",
    "target_documents",
    "retrieved_documents",
    "page_recall",
    "failure_category",
    "failure_subtype",
    "manual_review",
    "classification_rule",
    "condition_judge_reasons",
    "oracle_judge_reasons",
]

LABELS = {
    "eval_mode": "Condition",
    "question_type": "Generation method",
    "cognitive_skill": "Cognitive skill",
    "total_predictions": "Predictions",
    "scored_answers": "Scored answers",
    "correct_answers": "Correct answers",
    "unresolved": "Unresolved",
    "unjudged": "Unjudged",
    "did_not_fit": "Did not fit",
    "review_complete": "Review complete",
    "accuracy_excluding_did_not_fit": "Accuracy excluding did not fit",
    "accuracy_including_did_not_fit": "Accuracy including did not fit",
    "page_recall": "Page recall",
    "page_recall_sample_size": "Recall sample size",
    "page_precision": "Page precision",
    "page_precision_sample_size": "Precision sample size",
    "page_mrr": "Page MRR",
    "page_mrr_sample_size": "MRR sample size",
    "planned": "Planned",
    "successful": "Successful",
    "failed": "Failed",
    "missing": "Missing",
    "complete": "Complete",
    "successful_answers": "Successful answers",
    "distinct_questions": "Distinct questions",
    "cost_sample_size": "Answers with cost data",
    "total_cost_usd": "Total cost (USD)",
    "average_cost_per_answer_usd": "Average cost per answer (USD)",
    "average_cost_per_question_usd": "Average cost per question (USD)",
    "latency_sample_size": "Answers with latency data",
    "average_latency_per_answer_seconds": "Average latency per answer (s)",
    "financebench_id": "FinanceBench ID",
    "analysis_status": "Analysis status",
    "oracle_accuracy": "Oracle accuracy",
    "condition_accuracy": "Condition accuracy",
    "target_documents": "Target documents",
    "retrieved_documents": "Retrieved documents",
    "failure_category": "Failure category",
    "failure_subtype": "Failure subtype",
    "manual_review": "Manual review",
    "classification_rule": "Classification rule",
    "condition_judge_reasons": "Retrieval-answer judge reasons",
    "oracle_judge_reasons": "Oracle judge reasons",
}

PERCENT_COLUMNS = {
    "accuracy_excluding_did_not_fit",
    "accuracy_including_did_not_fit",
    "page_recall",
    "page_precision",
    "page_mrr",
}

CURRENCY_COLUMNS = {
    "total_cost_usd",
    "average_cost_per_answer_usd",
    "average_cost_per_question_usd",
}

SECONDS_COLUMNS = {"average_latency_per_answer_seconds"}

TEXT_COLUMNS = {"eval_mode", "question_type", "cognitive_skill"}


def _formats(workbook: xlsxwriter.Workbook) -> dict[str, Any]:
    """Define the small, consistent visual vocabulary used by all sheets."""
    return {
        "title": workbook.add_format(
            {"bold": True, "font_size": 14, "font_color": "#1F2937"}
        ),
        "section": workbook.add_format(
            {
                "bold": True,
                "font_size": 11,
                "bottom": 1,
                "bottom_color": "#94A3B8",
            }
        ),
        "header": workbook.add_format(
            {
                "bold": True,
                "font_color": "#FFFFFF",
                "bg_color": "#334155",
                "align": "center",
                "valign": "vcenter",
            }
        ),
        "condition": workbook.add_format(
            {"bold": True, "font_color": "#FFFFFF", "bg_color": "#64748B"}
        ),
        "text": workbook.add_format({"font_color": "#1F2937"}),
        "indented": workbook.add_format({"font_color": "#1F2937", "indent": 1}),
        "integer": workbook.add_format(
            {"num_format": "#,##0", "font_color": "#1F2937"}
        ),
        "percent": workbook.add_format(
            {"num_format": "0.00%", "font_color": "#1F2937"}
        ),
        "currency": workbook.add_format(
            {"num_format": "$0.000000", "font_color": "#1F2937"}
        ),
        "seconds": workbook.add_format(
            {"num_format": '0.00 "s"', "font_color": "#1F2937"}
        ),
        "note": workbook.add_format({"italic": True, "font_color": "#475569"}),
        "wrapped": workbook.add_format(
            {"font_color": "#1F2937", "text_wrap": True, "valign": "top"}
        ),
    }


def _format_for_column(column: str, formats: dict[str, Any]) -> Any:
    """Choose a display format that makes each metric's unit explicit."""
    if column in PERCENT_COLUMNS:
        return formats["percent"]
    if column in CURRENCY_COLUMNS:
        return formats["currency"]
    if column in SECONDS_COLUMNS:
        return formats["seconds"]
    if column in TEXT_COLUMNS:
        return formats["text"]
    return formats["integer"]


def _write_table(
    worksheet: Any,
    start_row: int,
    title: str,
    rows: list[dict[str, Any]],
    columns: list[str],
    formats: dict[str, Any],
) -> int:
    """Write one ordinary table and return the next unused row."""
    worksheet.write(start_row, 0, title, formats["section"])
    header_row = start_row + 1
    for column_index, column in enumerate(columns):
        worksheet.write(header_row, column_index, LABELS[column], formats["header"])

    for row_index, row in enumerate(rows, start=header_row + 1):
        for column_index, column in enumerate(columns):
            value = row.get(column)
            cell_format = _format_for_column(column, formats)
            if isinstance(value, bool):
                value = "Yes" if value else "No"
                cell_format = formats["text"]
            worksheet.write(row_index, column_index, value, cell_format)
    return header_row + len(rows) + 3


def _write_condition_sections(
    worksheet: Any,
    start_row: int,
    title: str,
    rows: list[dict[str, Any]],
    identity_columns: list[str],
    metric_columns: list[str],
    formats: dict[str, Any],
) -> int:
    """Show each condition once, followed by its indented detail rows."""
    worksheet.write(start_row, 0, title, formats["section"])
    current_row = start_row + 1
    columns = identity_columns + metric_columns

    conditions = sorted({row["eval_mode"] for row in rows})
    for condition in conditions:
        for column_index in range(len(columns)):
            value = condition if column_index == 0 else ""
            worksheet.write(current_row, column_index, value, formats["condition"])
        current_row += 1

        for column_index, column in enumerate(columns):
            worksheet.write(
                current_row, column_index, LABELS[column], formats["header"]
            )
        current_row += 1

        condition_rows = [row for row in rows if row["eval_mode"] == condition]
        for row in condition_rows:
            for column_index, column in enumerate(columns):
                value = row.get(column)
                if isinstance(value, bool):
                    value = "Yes" if value else "No"
                    cell_format = formats["text"]
                elif column in {"question_type", "cognitive_skill"}:
                    cell_format = (
                        formats["indented"] if column_index == 0 else formats["text"]
                    )
                else:
                    cell_format = _format_for_column(column, formats)
                worksheet.write(current_row, column_index, value, cell_format)
            current_row += 1
        current_row += 1
    return current_row + 1


def _write_metric_sheet(
    workbook: xlsxwriter.Workbook,
    sheet_name: str,
    report_data: dict[str, list[dict[str, Any]]],
    metric_columns: list[str],
    empty_message: str,
    formats: dict[str, Any],
) -> None:
    """Render the five report views for one metric family."""
    worksheet = workbook.add_worksheet(sheet_name)
    worksheet.hide_gridlines(2)
    worksheet.set_column(0, 1, 24)
    worksheet.set_column(2, len(metric_columns) + 1, 18)
    worksheet.write(1, 0, sheet_name, formats["title"])

    if not any(report_data.values()):
        worksheet.write(3, 0, empty_message, formats["note"])
        return

    next_row = _write_table(
        worksheet, 3, "Overall", report_data["overall"], metric_columns, formats
    )
    next_row = _write_table(
        worksheet,
        next_row,
        "By condition",
        report_data["by_condition"],
        ["eval_mode", *metric_columns],
        formats,
    )
    next_row = _write_condition_sections(
        worksheet,
        next_row,
        "By generation method within condition",
        report_data["by_generation_method"],
        ["question_type"],
        metric_columns,
        formats,
    )
    next_row = _write_condition_sections(
        worksheet,
        next_row,
        "By cognitive skill within condition",
        report_data["by_cognitive_skill"],
        ["cognitive_skill"],
        metric_columns,
        formats,
    )
    worksheet.write(
        next_row - 1,
        0,
        "Questions with multiple cognitive skills appear in each applicable skill group; counts are not additive.",
        formats["note"],
    )
    _write_condition_sections(
        worksheet,
        next_row + 1,
        "Generation method by cognitive skill within condition",
        report_data["cross_tab"],
        ["question_type", "cognitive_skill"],
        metric_columns,
        formats,
    )


def _write_failure_analysis_sheet(
    workbook: xlsxwriter.Workbook,
    summary: dict[str, Any],
    analysis_rows: list[dict[str, Any]],
    formats: dict[str, Any],
) -> None:
    """Render the classification legend, counts and auditable detail rows."""
    worksheet = workbook.add_worksheet("Failure analysis")
    worksheet.hide_gridlines(2)
    worksheet.write(1, 0, "Failure analysis", formats["title"])

    worksheet.write(3, 0, "Classification legend", formats["section"])
    worksheet.write_row(4, 0, ["Outcome", "Meaning"], formats["header"])
    current_row = 5
    for outcome, meaning in summary["methodology"].items():
        worksheet.write(current_row, 0, outcome, formats["text"])
        worksheet.write(current_row, 1, meaning, formats["wrapped"])
        current_row += 1

    current_row += 1
    worksheet.write(current_row, 0, "Counts by condition", formats["section"])
    current_row += 1
    count_columns = [
        "eval_mode",
        "analysis_status",
        "failure_category",
        "failure_subtype",
        "count",
    ]
    count_labels = [
        "Condition",
        "Analysis status",
        "Failure category",
        "Failure subtype",
        "Count",
    ]
    worksheet.write_row(current_row, 0, count_labels, formats["header"])
    current_row += 1
    for count_row in summary["counts_by_condition"]:
        for column_index, column in enumerate(count_columns):
            cell_format = formats["integer"] if column == "count" else formats["text"]
            worksheet.write(
                current_row, column_index, count_row.get(column), cell_format
            )
        current_row += 1

    current_row += 1
    worksheet.write(current_row, 0, "Manual review", formats["text"])
    worksheet.write(current_row, 1, summary["manual_review_count"], formats["integer"])
    current_row += 1
    worksheet.write(current_row, 0, "Unclassified", formats["text"])
    worksheet.write(current_row, 1, summary["unclassified_count"], formats["integer"])

    current_row += 2
    worksheet.write(current_row, 0, "Question-level diagnoses", formats["section"])
    header_row = current_row + 1
    for column_index, column in enumerate(FAILURE_DETAIL_COLUMNS):
        worksheet.write(header_row, column_index, LABELS[column], formats["header"])

    for row_index, analysis_row in enumerate(analysis_rows, start=header_row + 1):
        for column_index, column in enumerate(FAILURE_DETAIL_COLUMNS):
            value = analysis_row.get(column)
            if isinstance(value, list):
                value = " | ".join(str(item) for item in value)
            if column == "page_recall":
                cell_format = formats["percent"]
            elif column in {"oracle_accuracy", "condition_accuracy"}:
                cell_format = formats["integer"]
            elif column in {
                "classification_rule",
                "condition_judge_reasons",
                "oracle_judge_reasons",
            }:
                cell_format = formats["wrapped"]
            else:
                cell_format = formats["text"]
            worksheet.write(row_index, column_index, value, cell_format)

    if analysis_rows:
        worksheet.autofilter(
            header_row,
            0,
            header_row + len(analysis_rows),
            len(FAILURE_DETAIL_COLUMNS) - 1,
        )
    worksheet.freeze_panes(header_row + 1, 0)
    worksheet.set_column(0, 4, 20)
    worksheet.set_column(5, 6, 28)
    worksheet.set_column(7, 10, 22)
    worksheet.set_column(11, 13, 48)


def write_report_workbook(
    path: str | Path,
    summary: dict[str, Any],
    failure_rows: list[dict[str, Any]],
) -> None:
    """Write the four approved reader-facing sheets from calculated data."""
    with xlsxwriter.Workbook(str(path)) as workbook:
        formats = _formats(workbook)

        overview = workbook.add_worksheet("Overview")
        overview.hide_gridlines(2)
        overview.set_column(0, 0, 34)
        overview.set_column(1, 1, 20)
        overview.set_column(2, 2, 18)
        overview.set_column(3, 4, 26)
        overview.write(1, 0, "FinanceBench run overview", formats["title"])
        overview.write_row(3, 0, ["Run status", "Value"], formats["header"])
        status_rows = [
            ("Experiment", "experiment"),
            ("Variant", "variant"),
            ("Planned jobs", "planned"),
            ("Successful", "successful"),
            ("Failed", "failed"),
            ("Missing", "missing"),
            ("Did not fit", "did_not_fit"),
            ("Complete", "complete"),
        ]
        for row_index, (label, key) in enumerate(status_rows, start=4):
            overview.write(row_index, 0, label, formats["text"])
            value = summary["run_status"][key]
            if isinstance(value, str):
                value_format = formats["text"]
            elif isinstance(value, bool):
                value = "Yes" if value else "No"
                value_format = formats["text"]
            else:
                value_format = formats["integer"]
            overview.write(row_index, 1, value, value_format)

        overall_accuracy = summary["answer_accuracy"]["overall"]
        if overall_accuracy:
            accuracy = overall_accuracy[0]
            overview.write(14, 0, "Overall answer accuracy", formats["section"])
            overview.write(15, 0, "Excluding did not fit", formats["text"])
            overview.write(
                15,
                1,
                accuracy["accuracy_excluding_did_not_fit"],
                formats["percent"],
            )
            overview.write(16, 0, "Including did not fit", formats["text"])
            overview.write(
                16,
                1,
                accuracy["accuracy_including_did_not_fit"],
                formats["percent"],
            )

        generation_performance = summary["generation_performance"]
        overview.write(19, 0, "Generation cost and latency", formats["section"])
        overview.write_row(20, 0, ["Metric", "Value"], formats["header"])
        overall_performance = generation_performance["overall"]
        performance_rows = [
            ("Successful answers", "successful_answers"),
            ("Distinct questions", "distinct_questions"),
            ("Answers with cost data", "cost_sample_size"),
            ("Total cost (USD)", "total_cost_usd"),
            ("Average cost per answer (USD)", "average_cost_per_answer_usd"),
            ("Average cost per question (USD)", "average_cost_per_question_usd"),
            ("Answers with latency data", "latency_sample_size"),
            ("Average latency per answer (s)", "average_latency_per_answer_seconds"),
        ]
        for row_index, (label, key) in enumerate(performance_rows, start=21):
            overview.write(row_index, 0, label, formats["text"])
            overview.write(
                row_index,
                1,
                overall_performance[key],
                _format_for_column(key, formats),
            )

        next_row = _write_table(
            overview,
            31,
            "Generation cost and latency by condition",
            generation_performance["by_condition"],
            ["eval_mode", *PERFORMANCE_CONDITION_COLUMNS],
            formats,
        )

        _write_table(
            overview,
            next_row,
            "Execution status by condition",
            summary["run_status"]["by_condition"],
            STATUS_COLUMNS,
            formats,
        )

        _write_metric_sheet(
            workbook,
            "Answer accuracy",
            summary["answer_accuracy"],
            ANSWER_COLUMNS,
            "No answer results.",
            formats,
        )
        _write_metric_sheet(
            workbook,
            "Retrieval metrics",
            summary["retrieval_metrics"],
            RETRIEVAL_COLUMNS,
            "No applicable retrieval results.",
            formats,
        )
        _write_failure_analysis_sheet(
            workbook,
            summary["failure_analysis"],
            failure_rows,
            formats,
        )
