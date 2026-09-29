"""Tests that Exp2's config pair is Exp1's with only the allowed settings changed.

Exp2 is Exp1 with one change (CLAUDE.md -> Invariants), so rung B's two
files may differ from Exp1's only in the run label, the RAG settings file
it points at, the structure weight and the reused plans (Implementation
Guide 3.1-3.4 -> Configuration). Anything else would be a second change.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

CONFIGS = Path(__file__).parent.parent / "configs"


def _flatten(config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """{"a": {"b": 1}} -> {"a.b": 1}, so two configs compare key by key."""
    flat: dict[str, Any] = {}
    for key, value in config.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{name}."))
        else:
            flat[name] = value
    return flat


def _differences(exp1_name: str, exp2_name: str) -> dict[str, tuple[Any, Any]]:
    exp1 = _flatten(tomllib.loads((CONFIGS / exp1_name).read_text(encoding="utf-8")))
    exp2 = _flatten(tomllib.loads((CONFIGS / exp2_name).read_text(encoding="utf-8")))
    return {
        key: (exp1.get(key), exp2.get(key))
        for key in exp1.keys() | exp2.keys()
        if exp1.get(key) != exp2.get(key)
    }


def test_the_sec_rag_pair_differs_only_in_weight_and_reused_plans() -> None:
    assert _differences("sec_rag.toml", "sec_rag-exp2.toml") == {
        "structure.weight": (0.0, 1.0),
        "query_enhancement.reuse_plans_from": (
            "",
            "results/20260928-202531--exp1--full",
        ),
    }


def test_the_benchmark_pair_differs_only_in_the_label_and_settings_file() -> None:
    assert _differences("financebench.toml", "financebench-exp2.toml") == {
        "run.experiment": ("exp1", "exp2"),
        "run.variant": ("full", "heading-path"),
        "run.sec_rag_config": ("configs/sec_rag.toml", "configs/sec_rag-exp2.toml"),
    }
