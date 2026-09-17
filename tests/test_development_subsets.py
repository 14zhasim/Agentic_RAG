from collections import Counter
from pathlib import Path
import tomllib

import pytest

from sec_rag_benchmark.cli import main
from sec_rag_benchmark.dataset.subsets import select_development_subset
from sec_rag_benchmark.execution.runner import _create_or_resume_run


def _questions() -> list[dict]:
    counts = {
        "metrics-generated": 50,
        "domain-relevant": 48,
        "novel-generated": 14,
    }
    return [
        {"financebench_id": f"{question_type}-{index}", "question_type": question_type}
        for question_type, count in counts.items()
        for index in range(count)
    ]


def test_development_subsets_are_fixed_and_stratified():
    questions = _questions()
    config = {"seed": 42, "smoke_size": 10, "pattern_size": 50}

    smoke = select_development_subset(questions, "smoke", config)
    repeated_smoke = select_development_subset(questions, "smoke", config)
    pattern = select_development_subset(questions, "pattern", config)

    assert [row["financebench_id"] for row in smoke] == [
        row["financebench_id"] for row in repeated_smoke
    ]
    assert Counter(row["question_type"] for row in smoke) == {
        "metrics-generated": 5,
        "domain-relevant": 4,
        "novel-generated": 1,
    }
    assert Counter(row["question_type"] for row in pattern) == {
        "metrics-generated": 22,
        "domain-relevant": 22,
        "novel-generated": 6,
    }
    assert len({row["financebench_id"] for row in pattern}) == 50


def test_development_subset_rejects_unknown_or_oversized_selection():
    questions = _questions()
    config = {"seed": 42, "smoke_size": 10, "pattern_size": 500}

    with pytest.raises(ValueError, match="Unknown development subset"):
        select_development_subset(questions, "other", config)
    with pytest.raises(ValueError, match="only 112 questions"):
        select_development_subset(questions, "pattern", config)

    config["smoke_size"] = 0
    with pytest.raises(ValueError, match="must be positive"):
        select_development_subset(questions, "smoke", config)


def test_cli_passes_subset_to_dry_run(tmp_path, monkeypatch, capsys):
    captured = {}
    config_path = tmp_path / "financebench.toml"

    monkeypatch.setattr(
        "sec_rag_benchmark.cli.load_config", lambda path: {"loaded": str(path)}
    )

    def fake_dry_run(config, *, conditions=None, limit=None, subset=None):
        captured.update(
            config=config,
            conditions=conditions,
            limit=limit,
            subset=subset,
        )
        return {
            "planned_jobs": 30,
            "conditions": ["closed_book", "oracle", "long_context"],
            "maximum_prompt_tokens": {},
            "oversized_jobs": 0,
            "api_requests": 0,
        }

    monkeypatch.setattr("sec_rag_benchmark.cli.dry_run", fake_dry_run)
    assert main(
        ["run", "--config", str(config_path), "--subset", "smoke", "--dry-run"]
    ) == 0
    assert captured["subset"] == "smoke"
    assert captured["limit"] is None
    assert "Planned jobs: 30" in capsys.readouterr().out


def test_cli_rejects_limit_with_subset(tmp_path):
    with pytest.raises(SystemExit):
        main(
            [
                "run",
                "--config",
                str(Path(tmp_path) / "financebench.toml"),
                "--limit",
                "5",
                "--subset",
                "smoke",
                "--dry-run",
            ]
        )


def test_run_snapshot_records_subset_name_and_exact_question_ids(tmp_path):
    config_path = tmp_path / "financebench.toml"
    config_path.write_text('[run]\nexperiment = "financebench"\n')
    selected_questions = [
        {"financebench_id": "q-7"},
        {"financebench_id": "q-19"},
    ]

    _create_or_resume_run(
        {
            "run": {
                "results_dir": str(tmp_path),
                "experiment": "financebench",
                "variant": "test",
            }
        },
        config_path,
        ["oracle"],
        selected_questions,
        "smoke",
        tmp_path / "run",
    )

    snapshot = tomllib.loads((tmp_path / "run" / "config.toml").read_text())
    assert snapshot["selection"]["development_subset"] == "smoke"
    assert snapshot["selection"]["question_ids"] == ["q-7", "q-19"]
