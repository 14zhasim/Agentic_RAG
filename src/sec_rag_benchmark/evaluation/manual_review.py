"""Export disputed judgments for review and import validated human decisions."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REVIEW_COLUMNS = [
    "job_id",
    "financebench_id",
    "eval_mode",
    "question",
    "gold_answer",
    "gold_evidence",
    "human_justification",
    "model_answer",
    "reference_first_verdict",
    "reference_first_reason",
    "candidate_first_verdict",
    "candidate_first_reason",
    "human_accuracy",
    "review_reason",
]


def _latest_rows_by_job_id(path: Path) -> dict[str, dict[str, Any]]:
    """Read an append-only JSONL and retain its latest row for each job."""
    latest_rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return latest_rows

    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            row = json.loads(line)
            latest_rows[row["job_id"]] = row
    return latest_rows


def _disagreement_passes(judgment: dict[str, Any]) -> dict[str, dict[str, Any]] | None:
    """Return both named passes only when they form a genuine disagreement."""
    if judgment.get("accuracy") is not None:
        return None

    passes: dict[str, dict[str, Any]] = {}
    for item in judgment.get("passes", []):
        if not isinstance(item, dict):
            continue
        prompt_order = item.get("prompt_order")
        if isinstance(prompt_order, str):
            passes[prompt_order] = item
    required_orders = {"reference_first", "candidate_first"}
    if set(passes) != required_orders:
        raise ValueError(f"Judgment {judgment.get('job_id')} is missing judge passes")

    verdicts = {passes[order].get("verdict") for order in required_orders}
    if verdicts != {0, 1}:
        raise ValueError(
            f"Judgment {judgment.get('job_id')} has invalid disagreement verdicts"
        )
    return passes


def _format_evidence(evidence_rows: list[dict[str, Any]]) -> str:
    """Render every evidence object as a readable, document-aware CSV block."""
    blocks: list[str] = []
    for evidence in evidence_rows:
        document = evidence.get("doc_name") or evidence.get("evidence_doc_name", "")
        page = evidence.get("evidence_page_num", "")
        text = evidence.get("evidence_text_full_page") or evidence.get(
            "evidence_text", ""
        )
        blocks.append(f"Document: {document}\nPage: {page}\n{text}")
    return "\n\n---\n\n".join(blocks)


def export_manual_review(
    run_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write one editable CSV row for every two-pass judge disagreement."""
    run_path = Path(run_dir)
    output_path = run_path / "manual_review.csv"
    if output_path.exists() and not overwrite:
        raise ValueError(f"{output_path} already exists; use --overwrite to replace it")

    predictions = _latest_rows_by_job_id(run_path / "predictions.jsonl")
    judgments = _latest_rows_by_job_id(run_path / "judgments.jsonl")
    manual_reviews = _latest_rows_by_job_id(run_path / "manual_reviews.jsonl")

    review_rows: list[dict[str, Any]] = []
    prefilled = 0
    for job_id in sorted(judgments):
        passes = _disagreement_passes(judgments[job_id])
        if passes is None:
            continue

        prediction = predictions.get(job_id)
        if prediction is None or prediction.get("status") != "success":
            raise ValueError(f"Disputed judgment {job_id} has no successful prediction")

        existing_review = manual_reviews.get(job_id, {})
        if existing_review:
            prefilled += 1
        review_rows.append(
            {
                "job_id": job_id,
                "financebench_id": prediction["financebench_id"],
                "eval_mode": prediction["eval_mode"],
                "question": prediction["question"],
                "gold_answer": prediction["gold_answer"],
                "gold_evidence": _format_evidence(prediction["gold_evidence"]),
                "human_justification": prediction["human_justification"],
                "model_answer": prediction["model_answer"],
                "reference_first_verdict": passes["reference_first"]["verdict"],
                "reference_first_reason": passes["reference_first"]["reason"],
                "candidate_first_verdict": passes["candidate_first"]["verdict"],
                "candidate_first_reason": passes["candidate_first"]["reason"],
                "human_accuracy": existing_review.get("human_accuracy", ""),
                "review_reason": existing_review.get("review_reason", ""),
            }
        )

    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(review_rows)

    return {
        "path": output_path,
        "disagreements": len(review_rows),
        "prefilled": prefilled,
    }


def _read_review_csv(path: Path) -> list[dict[str, str]]:
    """Read the review sheet and reject any changed or missing columns."""
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != REVIEW_COLUMNS:
            raise ValueError(
                "manual_review.csv columns do not match the exported template"
            )
        return list(reader)


def _append_reviews(path: Path, reviews: list[dict[str, Any]]) -> None:
    """Append a validated batch while preserving earlier review history."""
    if not reviews:
        return
    with path.open("a", encoding="utf-8") as file:
        for review in reviews:
            file.write(json.dumps(review, ensure_ascii=False) + "\n")
        file.flush()


def import_manual_review(run_dir: str | Path) -> dict[str, int]:
    """Validate the complete review CSV, then append new human decisions."""
    run_path = Path(run_dir)
    csv_rows = _read_review_csv(run_path / "manual_review.csv")
    predictions = _latest_rows_by_job_id(run_path / "predictions.jsonl")
    judgments = _latest_rows_by_job_id(run_path / "judgments.jsonl")
    latest_reviews = _latest_rows_by_job_id(run_path / "manual_reviews.jsonl")

    # Stage every decision before opening the output file. One invalid row must
    # not leave a partially imported set of human judgments.
    pending_reviews: list[dict[str, Any]] = []
    seen_job_ids: set[str] = set()
    skipped = 0
    unresolved = 0
    reviewed_at = datetime.now(UTC).isoformat()

    for row_number, row in enumerate(csv_rows, start=2):
        job_id = row["job_id"].strip()
        if job_id in seen_job_ids:
            raise ValueError(f"Row {row_number} has duplicate job_id {job_id}")
        seen_job_ids.add(job_id)

        prediction = predictions.get(job_id)
        if prediction is None:
            raise ValueError(f"Row {row_number} has unknown job_id {job_id}")
        if prediction.get("status") != "success":
            raise ValueError(
                f"Row {row_number} job_id {job_id} has no answer to review"
            )
        judgment = judgments.get(job_id)
        if judgment is None or _disagreement_passes(judgment) is None:
            raise ValueError(
                f"Row {row_number} job_id {job_id} is not a disputed judgment"
            )

        accuracy_text = row["human_accuracy"].strip()
        review_reason = row["review_reason"].strip()
        if not accuracy_text and not review_reason:
            unresolved += 1
            continue
        if accuracy_text not in {"0", "1"}:
            raise ValueError(f"Row {row_number} human_accuracy must be exactly 0 or 1")

        human_accuracy = int(accuracy_text)
        previous = latest_reviews.get(job_id)
        if previous is not None and (
            previous.get("human_accuracy") == human_accuracy
            and str(previous.get("review_reason", "")).strip() == review_reason
        ):
            skipped += 1
            continue

        pending_reviews.append(
            {
                "job_id": job_id,
                "human_accuracy": human_accuracy,
                "review_reason": review_reason,
                "reviewer": "author",
                "reviewed_at": reviewed_at,
            }
        )

    # Automated judgments remain untouched. Human corrections form their own
    # append-only audit trail, and reporting will use the latest decision.
    _append_reviews(run_path / "manual_reviews.jsonl", pending_reviews)
    return {
        "imported": len(pending_reviews),
        "skipped": skipped,
        "unresolved": unresolved,
    }
