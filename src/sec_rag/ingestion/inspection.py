"""Create free, repeatable structure reports from completed Azure caches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .parse import check_parse, json_path, select_documents
from .structure_report import build_structure_report


def inspect_parses(
    config: dict[str, Any],
    document_names: tuple[str, ...] | None = None,
) -> dict[str, int]:
    """Write one structure report per selected, valid raw Azure cache.

    This is intentionally offline: it loads saved JSON, validates the same
    basic cache invariants as parsing, and writes only the replaceable
    `structure.txt` inspection artefact beside that JSON.
    """
    selected_paths = select_documents(config, document_names)

    reports: list[tuple[Path, str]] = []
    for pdf_path in selected_paths:
        cache_path = json_path(config, pdf_path.stem)
        if not cache_path.is_file():
            raise ValueError(f"{pdf_path.stem}: no final Azure cache to inspect")
        try:
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"{pdf_path.stem}: cannot read Azure cache: {error}"
            ) from error
        check_parse(raw, pdf_path)
        reports.append(
            (
                cache_path.with_name(f"{pdf_path.stem}.structure.txt"),
                build_structure_report(pdf_path.stem, raw),
            )
        )

    for report_path, report in reports:
        report_path.write_text(report, encoding="utf-8")
    return {"inspected": len(reports)}
