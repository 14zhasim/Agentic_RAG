"""Protect the stage-based package layout from drifting back to root clutter."""

from importlib import import_module
from pathlib import Path


NEW_MODULES = (
    "sec_rag_benchmark.dataset.financebench",
    "sec_rag_benchmark.dataset.subsets",
    "sec_rag_benchmark.pipeline.conditions",
    "sec_rag_benchmark.pipeline.generation",
    "sec_rag_benchmark.execution.job",
    "sec_rag_benchmark.execution.preflight",
    "sec_rag_benchmark.execution.runner",
    "sec_rag_benchmark.evaluation.retrieval_metrics",
    "sec_rag_benchmark.evaluation.judge",
    "sec_rag_benchmark.evaluation.judge_validation",
    "sec_rag_benchmark.evaluation.manual_review",
    "sec_rag_benchmark.reporting.report",
    "sec_rag_benchmark.reporting.failure_analysis",
    "sec_rag_benchmark.reporting.workbook",
)

OLD_ROOT_MODULES = (
    "data.py",
    "development_subsets.py",
    "conditions.py",
    "generation.py",
    "metrics.py",
    "judge.py",
    "judge_validation.py",
    "manual_review.py",
    "reporting.py",
    "failure_analysis.py",
    "report_workbook.py",
)


def test_stage_modules_are_importable_and_old_root_modules_are_removed():
    for module_name in NEW_MODULES:
        import_module(module_name)

    package_dir = Path(__file__).parents[1] / "src" / "sec_rag_benchmark"
    assert all(not (package_dir / filename).exists() for filename in OLD_ROOT_MODULES)
