"""Load result-affecting SEC RAG settings from TOML."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate parser settings with repository-relative paths resolved.

    The benchmark and RAG system use separate TOML files because they own
    different result-affecting behaviour. This loader has no credential or data
    dependency, so CLI planning can run without Azure access.
    """
    config_path = Path(path)
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    project_root = config_path.resolve().parent.parent
    corpus = config.get("corpus")
    parsing = config.get("parsing")
    if not isinstance(corpus, dict) or not isinstance(parsing, dict):
        raise ValueError("SEC RAG configuration needs [corpus] and [parsing] sections")

    for key in ("prepared_dir", "parsed_dir"):
        value = corpus.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"corpus.{key} must be a non-empty path")
        configured_path = Path(value)
        if not configured_path.is_absolute():
            configured_path = project_root / configured_path
        corpus[key] = str(configured_path)

    if (
        not isinstance(corpus.get("expected_documents"), int)
        or corpus["expected_documents"] <= 0
    ):
        raise ValueError("corpus.expected_documents must be positive")
    if parsing.get("provider") != "azure-document-intelligence":
        raise ValueError("Stage 1.1 requires the azure-document-intelligence provider")
    if parsing.get("model_id") != "prebuilt-layout":
        raise ValueError("Stage 1.1 requires the prebuilt-layout model")
    if parsing.get("output_content_format") != "markdown":
        raise ValueError("Stage 1.1 requires markdown output")
    return config
