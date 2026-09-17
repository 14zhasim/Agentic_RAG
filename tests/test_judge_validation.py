import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sec_rag_benchmark.cli import main
from sec_rag_benchmark.evaluation.judge_validation import (
    create_validation_sample,
    summarize_validation,
    validate_judge,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _validation_fixture(tmp_path: Path) -> dict:
    source = tmp_path / "financebench"
    prepared = tmp_path / "prepared"
    question_types = ["metrics-generated", "domain-relevant", "novel-generated"]
    questions = []
    metadata = []
    for index in range(1, 10):
        doc_name = f"doc-{index}.pdf"
        questions.append(
            {
                "financebench_id": f"q{index}",
                "doc_name": doc_name,
                "question": f"Question {index}?",
                "answer": f"Answer {index}",
                "justification": f"Justification {index}",
                "evidence": [
                    {
                        "doc_name": doc_name,
                        "evidence_page_num": 0,
                        "evidence_text_full_page": f"Evidence {index}",
                    }
                ],
                "question_type": question_types[(index - 1) % 3],
                "question_reasoning": "Information extraction",
            }
        )
        metadata.append({"doc_name": doc_name})

    _write_jsonl(prepared / "financebench_open_source_10k.jsonl", questions)
    _write_jsonl(prepared / "financebench_document_information_10k.jsonl", metadata)

    labels = [
        "Correct Answer",
        "Incorrect Answer",
        "Refusal",
        "Correct Answer",
        "Incorrect Answer",
        "Correct Answer",
        "Correct Answer",
        "Incorrect Answer",
        "Refusal",
    ]
    result_rows = []
    for index, label in enumerate(labels, start=1):
        result_rows.append(
            {
                "financebench_id": f"q{index}",
                "model_name": "published-model",
                "eval_mode": "oracle",
                "model_answer": f"Candidate {index}",
                "label": label,
            }
        )
    _write_jsonl(source / "results" / "model-a_oracle.jsonl", result_rows[:5])
    _write_jsonl(source / "results" / "model-b_context.jsonl", result_rows[5:])

    return {
        "dataset": {
            "source_dir": str(source),
            "output_dir": str(prepared),
        },
        "judge": {
            "provider": "azure",
            "model": "DeepSeek-V4-Flash",
            "deployment": "DeepSeek-V4-Flash",
            "prompt_version": "financebench-binary-judge-v2",
            "max_output_tokens": 512,
            "timeout_seconds": 30.0,
            "max_retries": 2,
        },
        "judge_validation": {
            "source_commit": "cc39aeb4afdf33909ee1412188bf89035950c2eb",
            "seed": 42,
            "correct_examples": 3,
            "incorrect_examples": 2,
            "refusal_examples": 1,
            "minimum_agreement": 0.90,
        },
        "run": {"results_dir": str(tmp_path / "results")},
    }


def test_create_validation_sample_is_reproducible_and_balanced(tmp_path):
    config = _validation_fixture(tmp_path)
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    first = create_validation_sample(config, first_dir)
    second = create_validation_sample(config, second_dir)

    assert first == second
    assert [row["human_label"] for row in first].count("Correct Answer") == 3
    assert [row["human_label"] for row in first].count("Incorrect Answer") == 2
    assert [row["human_label"] for row in first].count("Refusal") == 1
    assert len({row["financebench_id"] for row in first}) == 6
    assert len({row["candidate_answer"] for row in first}) == 6
    assert {row["question_type"] for row in first} == {
        "metrics-generated",
        "domain-relevant",
        "novel-generated",
    }
    assert len({row["source_result_file"] for row in first}) >= 2

    predictions = [
        json.loads(line)
        for line in (first_dir / "predictions.jsonl").read_text().splitlines()
    ]
    assert all(row["status"] == "success" for row in predictions)
    assert all(row["gold_evidence"] for row in predictions)
    assert all(row["human_justification"] for row in predictions)


def test_judge_validation_scores_matches_nulls_and_threshold():
    sample = [
        {
            "job_id": "v:q1",
            "financebench_id": "q1",
            "human_label": "Correct Answer",
            "expected_accuracy": 1,
        },
        {
            "job_id": "v:q2",
            "financebench_id": "q2",
            "human_label": "Incorrect Answer",
            "expected_accuracy": 0,
        },
        {
            "job_id": "v:q3",
            "financebench_id": "q3",
            "human_label": "Refusal",
            "expected_accuracy": 0,
        },
    ]
    judgments = {
        "v:q1": {"job_id": "v:q1", "accuracy": 1},
        "v:q2": {"job_id": "v:q2", "accuracy": None, "manual_review": True},
        "v:q3": {"job_id": "v:q3", "accuracy": 0},
    }

    summary = summarize_validation(
        sample,
        judgments,
        minimum_agreement=0.90,
        source_commit="abc",
        source_file_hashes={"results/example.jsonl": "hash"},
    )

    assert summary["status"] == "complete"
    assert summary["passed"] is False
    assert summary["agreements"] == 2
    assert summary["agreement_rate"] == pytest.approx(2 / 3)
    assert summary["by_human_label"]["Refusal"]["agreement_rate"] == 1.0
    assert summary["confusion_matrix"]["expected_0_judge_null"] == 1
    assert summary["mismatches"][0]["financebench_id"] == "q2"

    matching_judgments = {
        row["job_id"]: {
            "job_id": row["job_id"],
            "accuracy": row["expected_accuracy"],
        }
        for row in sample
    }
    passing = summarize_validation(
        sample,
        matching_judgments,
        minimum_agreement=0.90,
        source_commit="abc",
        source_file_hashes={},
    )
    assert passing["passed"] is True


def test_validate_judge_reuses_completed_results(tmp_path):
    config = _validation_fixture(tmp_path)
    config_path = tmp_path / "financebench.toml"
    config_path.write_text("# test configuration\n")
    run_dir = tmp_path / "validation-run"

    verdicts = iter([1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1])
    calls = 0

    def create(**kwargs):
        nonlocal calls
        calls += 1
        verdict = next(verdicts)
        return SimpleNamespace(
            id=f"request-{calls}",
            model="DeepSeek-V4-Flash",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps({"verdict": verdict, "reason": "checked"})
                    )
                )
            ],
            usage=None,
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    first = validate_judge(
        config, config_path, requested_run_dir=run_dir, client=client
    )
    second = validate_judge(
        config, config_path, requested_run_dir=run_dir, client=client
    )

    assert first["judge_counts"]["judged"] == 6
    assert second["judge_counts"]["skipped"] == 6
    assert calls == 12
    assert (run_dir / "judge_validation.json").is_file()


def test_missing_results_fail_before_creating_an_azure_client(tmp_path):
    config = _validation_fixture(tmp_path)
    for path in (Path(config["dataset"]["source_dir"]) / "results").glob("*.jsonl"):
        path.unlink()

    with pytest.raises(ValueError, match="git clone"):
        validate_judge(config, tmp_path / "financebench.toml")


def test_insufficient_sample_fails_before_creating_an_azure_client(tmp_path):
    config = _validation_fixture(tmp_path)
    config["judge_validation"]["correct_examples"] = 20

    with pytest.raises(ValueError, match="Not enough unique"):
        validate_judge(config, tmp_path / "financebench.toml")


def test_cli_delegates_validate_judge(tmp_path, monkeypatch, capsys):
    captured = {}

    def fake_load_config(path):
        return {"loaded_from": path}

    def fake_validate(config, config_path, requested_run_dir=None):
        captured.update(
            config=config,
            config_path=config_path,
            requested_run_dir=requested_run_dir,
        )
        return {
            "run_dir": tmp_path / "validation",
            "status": "complete",
            "agreements": 28,
            "completed": 30,
            "passed": True,
        }

    monkeypatch.setattr("sec_rag_benchmark.cli.load_config", fake_load_config)
    monkeypatch.setattr("sec_rag_benchmark.cli.validate_judge", fake_validate)
    config_path = tmp_path / "config.toml"

    assert main(["validate-judge", "--config", str(config_path)]) == 0
    assert captured["config_path"] == config_path
    assert captured["requested_run_dir"] is None
    assert "28/30" in capsys.readouterr().out
