"""Render calculated benchmark summaries as a readable Excel workbook."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import xlsxwriter


ANSWER_COLUMNS = [
    "total_predictions",
    "agreed_judgments",
    "disagreements",
    "unjudged",
    "did_not_fit",
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

LABELS = {
    "eval_mode": "Condition",
    "question_type": "Generation method",
    "cognitive_skill": "Cognitive skill",
    "total_predictions": "Predictions",
    "agreed_judgments": "Agreed judgments",
    "disagreements": "Disagreements",
    "unjudged": "Unjudged",
    "did_not_fit": "Did not fit",
    "accuracy_excluding_did_not_fit": "Accuracy excluding did not fit",
    "accuracy_including_did_not_fit": "Accuracy including did not fit",
    "page_recall": "Page recall",
    "page_recall_sample_size": "Recall sample size",
    "page_precision": "Page precision",
    "page_precision_sample_size": "Precision sample size",
    "page_mrr": "Page MRR",
    "page_mrr_sample_size": "MRR sample size",
}

PERCENT_COLUMNS = {
    "accuracy_excluding_did_not_fit",
    "accuracy_including_did_not_fit",
    "page_recall",
    "page_precision",
    "page_mrr",
}


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
        "integer": workbook.add_format({"num_format": "#,##0", "font_color": "#1F2937"}),
        "percent": workbook.add_format({"num_format": "0.0%", "font_color": "#1F2937"}),
        "note": workbook.add_format({"italic": True, "font_color": "#475569"}),
    }


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
            cell_format = (
                formats["percent"]
                if column in PERCENT_COLUMNS
                else formats["integer"]
            )
            if column in {"eval_mode", "question_type", "cognitive_skill"}:
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
            worksheet.write(current_row, column_index, LABELS[column], formats["header"])
        current_row += 1

        condition_rows = [row for row in rows if row["eval_mode"] == condition]
        for row in condition_rows:
            for column_index, column in enumerate(columns):
                value = row.get(column)
                if column in PERCENT_COLUMNS:
                    cell_format = formats["percent"]
                elif column in {"question_type", "cognitive_skill"}:
                    cell_format = (
                        formats["indented"]
                        if column_index == 0
                        else formats["text"]
                    )
                else:
                    cell_format = formats["integer"]
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


def write_report_workbook(path: str | Path, summary: dict[str, Any]) -> None:
    """Write the three approved reader-facing sheets from calculated data."""
    with xlsxwriter.Workbook(str(path)) as workbook:
        formats = _formats(workbook)

        overview = workbook.add_worksheet("Overview")
        overview.hide_gridlines(2)
        overview.set_column(0, 0, 34)
        overview.set_column(1, 1, 18)
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
            value_format = formats["text"] if isinstance(value, str) else None
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
