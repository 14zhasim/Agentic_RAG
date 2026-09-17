"""Prepare, validate, and load the FinanceBench 10-K subset."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pymupdf

QUESTIONS_FILE = "financebench_open_source_10k.jsonl"
METADATA_FILE = "financebench_document_information_10k.jsonl"
MANIFEST_FILE = "manifest.json"
QUESTION_COLUMNS = {
    "financebench_id",
    "company",
    "doc_name",
    "question_type",
    "question_reasoning",
    "domain_question_num",
    "question",
    "answer",
    "justification",
    "dataset_subset_label",
    "evidence",
}
METADATA_COLUMNS = {
    "doc_name",
    "company",
    "gics_sector",
    "doc_type",
    "doc_period",
    "doc_link",
}


class DataError(ValueError):
    """Raised when source or prepared FinanceBench data is unusable."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as error:
        raise DataError(f"Cannot read {path}: {error}") from error


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pdf_path(pdf_dir: Path, doc_name: str) -> Path:
    for path in (pdf_dir / doc_name, pdf_dir / f"{doc_name}.pdf"):
        if path.is_file():
            return path
    raise DataError(f"Selected PDF is missing: {doc_name}")


def _evidence_doc(evidence: dict[str, Any]) -> str:
    return evidence.get("doc_name") or evidence.get("evidence_doc_name") or ""


def _check_rows(
    questions: list[dict[str, Any]],
    metadata: list[dict[str, Any]],
    pdf_dir: Path,
    *,
    expected_questions: int,
    expected_documents: int,
) -> None:
    """Apply the dataset checks needed before a benchmark run."""
    if any(not QUESTION_COLUMNS <= row.keys() for row in questions):
        raise DataError("Question JSONL schema is missing required fields")
    if any(not METADATA_COLUMNS <= row.keys() for row in metadata):
        raise DataError("Metadata JSONL schema is missing required fields")

    ids = [row["financebench_id"] for row in questions]
    if len(ids) != len(set(ids)):
        raise DataError("financebench_id must be unique")
    metadata_counts: dict[str, int] = {}
    for row in metadata:
        metadata_counts[row["doc_name"]] = metadata_counts.get(row["doc_name"], 0) + 1
    for question in questions:
        if metadata_counts.get(question["doc_name"]) != 1:
            raise DataError(
                f"Question has missing or ambiguous metadata: {question['doc_name']}"
            )
    if len(questions) != expected_questions or len(metadata) != expected_documents:
        raise DataError(
            f"Expected {expected_questions} questions and {expected_documents} documents; "
            f"found {len(questions)} and {len(metadata)}"
        )

    page_counts: dict[str, int] = {}
    for question in questions:
        doc_name = question["doc_name"]
        # Several questions can use the same filing. Open each PDF once, then
        # reuse its page count while checking every question's evidence.
        if doc_name not in page_counts:
            path = _pdf_path(pdf_dir, doc_name)
            with pymupdf.open(path) as pdf:
                page_counts[doc_name] = pdf.page_count
        evidence_list = question.get("evidence")
        if not isinstance(evidence_list, list) or not evidence_list:
            raise DataError(
                f"Evidence must be retained for {question['financebench_id']}"
            )
        for evidence in evidence_list:
            page = evidence.get("evidence_page_num")
            if _evidence_doc(evidence) != doc_name:
                raise DataError(f"Evidence document does not match {doc_name}")
            if type(page) is not int or page < 0 or page >= page_counts[doc_name]:
                raise DataError(f"Evidence page index is invalid for {doc_name}")


def _select_10k_subset(
    questions: list[dict[str, Any]],
    metadata: list[dict[str, Any]],
    document_type: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select questions and metadata belonging to the configured filing type."""
    selected_metadata = [
        row
        for row in metadata
        if str(row.get("doc_type", "")).strip().casefold() == document_type
    ]
    selected_document_names = {row["doc_name"] for row in selected_metadata}
    selected_questions = [
        row for row in questions if row.get("doc_name") in selected_document_names
    ]

    # Some filtered metadata rows may have no question. The prepared metadata
    # should describe exactly the PDFs actually used by the question subset.
    used_document_names = {row["doc_name"] for row in selected_questions}
    selected_metadata = [
        row for row in selected_metadata if row["doc_name"] in used_document_names
    ]
    return selected_questions, selected_metadata


def _replace_prepared_files(
    source: Path,
    output: Path,
    questions: list[dict[str, Any]],
    metadata: list[dict[str, Any]],
) -> list[Path]:
    """Rebuild the generated JSONLs and PDF subset, removing stale PDFs."""
    output.mkdir(parents=True, exist_ok=True)
    prepared_pdf_dir = output / "pdfs"
    prepared_pdf_dir.mkdir(exist_ok=True)

    for old_pdf in prepared_pdf_dir.glob("*.pdf"):
        old_pdf.unlink()
    for document_name in sorted({row["doc_name"] for row in questions}):
        source_pdf = _pdf_path(source / "pdfs", document_name)
        shutil.copy2(source_pdf, prepared_pdf_dir / source_pdf.name)

    question_path = output / QUESTIONS_FILE
    metadata_path = output / METADATA_FILE
    _write_jsonl(question_path, questions)
    _write_jsonl(metadata_path, metadata)
    return [question_path, metadata_path, *sorted(prepared_pdf_dir.glob("*.pdf"))]


def _write_manifest(
    source: Path,
    output: Path,
    config: dict[str, Any],
    questions: list[dict[str, Any]],
    metadata: list[dict[str, Any]],
    generated_files: list[Path],
) -> None:
    """Record how the ignored prepared dataset was produced and hashed."""
    manifest = {
        "source_path": str(source),
        "filter": {"doc_type": config["document_type"]},
        "expected": {
            "questions": config["expected_questions"],
            "documents": config["expected_documents"],
        },
        "observed": {
            "questions": len(questions),
            "documents": len(metadata),
        },
        "selected_filenames": sorted(
            path.name for path in (output / "pdfs").glob("*.pdf")
        ),
        "sha256": {
            str(path.relative_to(output)): _hash(path) for path in generated_files
        },
    }
    (output / MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def _validate_manifest(
    output: Path,
    config: dict[str, Any],
    actual_pdf_names: set[str],
) -> None:
    """Check that the manifest still describes the prepared files on disk."""
    try:
        manifest = json.loads((output / MANIFEST_FILE).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataError(f"Cannot read prepared manifest: {error}") from error

    if manifest.get("selected_filenames") != sorted(actual_pdf_names):
        raise DataError("Manifest filenames do not match prepared PDFs")
    expected_counts = {
        "questions": config["expected_questions"],
        "documents": config["expected_documents"],
    }
    if manifest.get("expected") != expected_counts:
        raise DataError("Manifest counts do not match configuration")
    for relative_path, expected_hash in manifest.get("sha256", {}).items():
        prepared_file = output / relative_path
        if not prepared_file.is_file() or _hash(prepared_file) != expected_hash:
            raise DataError(f"Manifest hash does not match {relative_path}")


def prepare(config: dict[str, Any]) -> dict[str, int]:
    """Filter the source to 10-Ks and copy its 112 questions and 64 PDFs."""
    source = Path(config["source_dir"])
    output = Path(config["output_dir"])
    questions = _read_jsonl(source / "data" / "financebench_open_source.jsonl")
    metadata = _read_jsonl(source / "data" / "financebench_document_information.jsonl")

    wanted_questions, wanted_metadata = _select_10k_subset(
        questions, metadata, config["document_type"]
    )

    _check_rows(
        wanted_questions,
        wanted_metadata,
        source / "pdfs",
        expected_questions=config["expected_questions"],
        expected_documents=config["expected_documents"],
    )

    generated_files = _replace_prepared_files(
        source, output, wanted_questions, wanted_metadata
    )
    _write_manifest(
        source,
        output,
        config,
        wanted_questions,
        wanted_metadata,
        generated_files,
    )
    return {"questions": len(wanted_questions), "documents": len(wanted_metadata)}


def validate(config: dict[str, Any]) -> dict[str, int]:
    """Validate the prepared files before they are used by generation."""
    output = Path(config["output_dir"])
    questions = _read_jsonl(output / QUESTIONS_FILE)
    metadata = _read_jsonl(output / METADATA_FILE)
    _check_rows(
        questions,
        metadata,
        output / "pdfs",
        expected_questions=config["expected_questions"],
        expected_documents=config["expected_documents"],
    )
    # Compare the complete directory with metadata; checking only that each
    # expected PDF exists would fail to notice stale extra files.
    expected_pdfs = {
        _pdf_path(output / "pdfs", row["doc_name"]).name for row in metadata
    }
    actual_pdfs = {path.name for path in (output / "pdfs").glob("*.pdf")}
    if actual_pdfs != expected_pdfs:
        raise DataError("Prepared PDF set does not match selected metadata")
    _validate_manifest(output, config, actual_pdfs)
    return {"questions": len(questions), "documents": len(metadata)}


def load_questions(output_dir: str | Path) -> list[dict[str, Any]]:
    """Join document metadata in memory while retaining every question field."""
    output = Path(output_dir)
    questions = _read_jsonl(output / QUESTIONS_FILE)
    metadata_rows = _read_jsonl(output / METADATA_FILE)
    metadata = {
        row["doc_name"]: row for row in metadata_rows
    }  # convert list-> dict for each doc, where doc: doc_metadata
    if len(metadata) != len(metadata_rows):
        raise DataError("Prepared metadata is ambiguous")
    # Preserve every original question/evidence field and add metadata only in
    # memory; the prepared question JSONL schema remains unchanged.
    return [
        {**question, "document_metadata": metadata[question["doc_name"]]}
        for question in questions
    ]


def load_run_questions(
    dataset_config: dict[str, Any], limit: int | None = None
) -> list[dict[str, Any]]:
    """Validate and load the questions selected for a dry or real run."""
    validate(dataset_config)
    questions = load_questions(dataset_config["output_dir"])
    if limit is None:
        return questions
    if limit <= 0:
        raise ValueError("--limit must be positive")
    return questions[:limit]
