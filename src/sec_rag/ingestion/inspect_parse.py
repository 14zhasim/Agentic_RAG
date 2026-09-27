"""Write a structure report next to each selected saved parse.

This module handles files only: it finds the saved parses, checks them, and
writes each filing's `<doc_name>.structure.txt`. The analysis is in
`structure_report.py`. It is free and offline — no Azure call, no
credentials — so it can be run as often as needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .parse import check_parse, json_path, read_json, select_documents
from .structure_report import build_structure_report


def inspect_parses(
    config: dict[str, Any],
    document_names: tuple[str, ...] | None = None,
) -> dict[str, int]:
    """Write <doc_name>.structure.txt for each selected, already-parsed filing.

    Every selected filing must be parsed and pass the same check as the parse
    run. All reports are built before any is written, so one unparsed or bad
    filing means no reports are written at all rather than some.
    """
    reports: list[tuple[Path, str]] = []
    for pdf_path in select_documents(config, document_names):
        doc_name = pdf_path.stem
        path = json_path(config, doc_name)
        if not path.is_file():
            raise ValueError(f"{doc_name}: not parsed yet")
        raw = read_json(path)
        check_parse(raw, pdf_path)
        report_path = path.with_name(f"{doc_name}.structure.txt")
        reports.append((report_path, build_structure_report(doc_name, raw)))

    for report_path, text in reports:
        report_path.write_text(text, encoding="utf-8")
    return {"inspected": len(reports)}
