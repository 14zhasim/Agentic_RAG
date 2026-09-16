"""Load and validate public benchmark settings from TOML."""

from __future__ import annotations

from pathlib import Path
import tomllib
from typing import Any

from .conditions import CONDITIONS


def load_config(path: str | Path) -> dict[str, Any]:
    """Return validated settings with project-relative paths made absolute."""
    config_path = Path(path)
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    project_root = config_path.resolve().parent.parent

    # These paths are written relative to the repository in the public TOML.
    # Resolving them here lets every other module receive ready-to-use paths.
    for section, key in (
        ("dataset", "source_dir"),
        ("dataset", "output_dir"),
        ("run", "results_dir"),
    ):
        configured_path = Path(config[section][key])
        if not configured_path.is_absolute():
            configured_path = project_root / configured_path
        config[section][key] = str(configured_path)

    configured_conditions = config["run"]["conditions"]
    if not configured_conditions or set(configured_conditions) - CONDITIONS:
        raise ValueError("Configuration contains missing or unknown conditions")

    generation = config["generation"]
    if generation["provider"] != "openrouter" or generation["allow_fallbacks"] is not False:
        raise ValueError("Golden-path generation requires OpenRouter with fallbacks disabled")
    if generation["reasoning_effort"] not in {"low", "high", "max"}:
        raise ValueError("reasoning_effort must be low, high, or max for GLM-5.3-Flash")

    token_settings = (
        generation["context_window_tokens"],
        generation["max_output_tokens"],
        generation["token_safety_margin"],
    )
    if min(token_settings) < 0:
        raise ValueError("Token limits and margins cannot be negative")
    reserved_tokens = generation["max_output_tokens"] + generation["token_safety_margin"]
    if reserved_tokens >= generation["context_window_tokens"]:
        raise ValueError("Output tokens and safety margin must leave room for the input prompt")

    judge = config["judge"]
    if judge["provider"] != "azure":
        raise ValueError("Answer judging requires the Azure provider")
    if not judge["deployment"] or not judge["prompt_version"]:
        raise ValueError("Judge deployment and prompt version cannot be empty")
    if judge["max_output_tokens"] <= 0:
        raise ValueError("Judge max_output_tokens must be positive")
    if judge["timeout_seconds"] <= 0 or judge["max_retries"] < 0:
        raise ValueError("Judge timeout must be positive and retries cannot be negative")

    return config
