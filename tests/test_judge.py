import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

from sec_rag_benchmark.cli import main
from sec_rag_benchmark.judge import (
    SYSTEM_PROMPT,
    _build_judge_messages,
    _combine_verdicts,
    _create_azure_client,
    _request_verdict,
    judge_run,
)
from sec_rag_benchmark.reporting import _resolve_judgments, write_report


def _prediction(job_id: str = "run:q1:oracle") -> dict:
    return {
        "job_id": job_id,
        "status": "success",
        "financebench_id": "q1",
        "question": "What was revenue?",
        "gold_answer": "$42 million",
        "gold_evidence": [
            {
                "doc_name": "a.pdf",
                "evidence_page_num": 7,
                "evidence_text_full_page": "Revenue was $42 million.",
            }
        ],
        "human_justification": "The filing directly reports the value.",
        "model_answer": "$42m",
        "eval_mode": "oracle",
        "question_type": "reported",
        "cognitive_skills": ["information_extraction"],
        "page_recall": None,
        "page_precision": None,
        "page_mrr": None,
    }


def _config() -> dict:
    return {
        "provider": "azure",
        "model": "DeepSeek-V4-Flash",
        "deployment": "DeepSeek-V4-Flash",
        "prompt_version": "financebench-binary-judge-v2",
        "max_output_tokens": 512,
        "timeout_seconds": 30.0,
        "max_retries": 2,
    }


class FakeChatCompletions:
    def __init__(self, verdicts: list[int]):
        self.verdicts = iter(verdicts)
        self.requests: list[dict] = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        verdict = next(self.verdicts)
        content = json.dumps({"verdict": verdict, "reason": f"Verdict {verdict}"})
        return SimpleNamespace(
            id=f"request-{len(self.requests)}",
            model="DeepSeek-V4-Flash",
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(model_dump=lambda: {"total_tokens": 20}),
        )


def _client(verdicts: list[int]):
    completions = FakeChatCompletions(verdicts)
    return SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    ), completions


@pytest.mark.parametrize(
    "verdicts, expected_accuracy, expected_review",
    [([1, 1], 1, False), ([0, 0], 0, False), ([1, 0], None, True)],
)
def test_judge_run_combines_two_orders_and_resumes(
    tmp_path, verdicts, expected_accuracy, expected_review
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    prediction = _prediction()
    (run_dir / "predictions.jsonl").write_text(json.dumps(prediction) + "\n")
    client, completions = _client(verdicts)

    first = judge_run(run_dir, _config(), client=client)
    second = judge_run(run_dir, _config(), client=client)

    assert first == {"judged": 1, "skipped": 0, "failed": 0}
    assert second == {"judged": 0, "skipped": 1, "failed": 0}
    assert len(completions.requests) == 2
    judgment = json.loads((run_dir / "judgments.jsonl").read_text())
    assert judgment["accuracy"] == expected_accuracy
    assert judgment["manual_review"] is expected_review
    assert [item["prompt_order"] for item in judgment["passes"]] == [
        "reference_first",
        "candidate_first",
    ]


def test_prompt_orders_include_every_required_financebench_field():
    prediction = _prediction()
    reference_first = _build_judge_messages(prediction, "reference_first")[1]["content"]
    candidate_first = _build_judge_messages(prediction, "candidate_first")[1]["content"]

    for value in (
        prediction["question"],
        prediction["gold_answer"],
        prediction["gold_evidence"][0]["evidence_text_full_page"],
        prediction["human_justification"],
        prediction["model_answer"],
    ):
        assert value in reference_first
        assert value in candidate_first
    assert reference_first.index("REFERENCE ANSWER") < reference_first.index("CANDIDATE ANSWER")
    assert candidate_first.index("CANDIDATE ANSWER") < candidate_first.index("REFERENCE ANSWER")

    # These rules prevent the judge from grading the reference/evidence instead
    # of the candidate when the two answer blocks change order.
    assert "grade only the CANDIDATE ANSWER" in SYSTEM_PROMPT
    assert "refusal" in SYSTEM_PROMPT
    assert "all material parts" in SYSTEM_PROMPT


def test_request_validation_and_missing_credentials(monkeypatch):
    client, completions = _client([1])
    result = _request_verdict(
        _build_judge_messages(_prediction(), "reference_first"), _config(), client
    )
    request = completions.requests[0]
    assert result["verdict"] == 1
    assert request["model"] == "DeepSeek-V4-Flash"
    assert request["max_completion_tokens"] == 512
    assert request["response_format"] == {"type": "json_object"}

    malformed = SimpleNamespace(
        id="bad",
        model="DeepSeek-V4-Flash",
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"verdict": 2, "reason": "bad"}'))],
        usage=None,
    )
    bad_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: malformed))
    )
    with pytest.raises(ValueError, match="verdict"):
        _request_verdict([], _config(), bad_client)

    monkeypatch.delenv("AZURE_DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_DEEPSEEK_ENDPOINT", raising=False)
    with pytest.raises(RuntimeError, match="AZURE_DEEPSEEK"):
        _create_azure_client(_config())


def test_api_failure_is_retryable_and_did_not_fit_is_not_judged(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    success = _prediction()
    did_not_fit = {**_prediction("run:q2:long_context"), "status": "did_not_fit", "model_answer": None}
    (run_dir / "predictions.jsonl").write_text(
        json.dumps(success) + "\n" + json.dumps(did_not_fit) + "\n"
    )
    calls = 0

    def fail(**kwargs):
        nonlocal calls
        calls += 1
        raise ConnectionError("temporary")

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fail))
    )
    assert judge_run(run_dir, _config(), client=client)["failed"] == 1
    assert judge_run(run_dir, _config(), client=client)["failed"] == 1
    assert calls == 2
    assert not (run_dir / "judgments.jsonl").exists()
    errors = [json.loads(line) for line in (run_dir / "errors.jsonl").read_text().splitlines()]
    assert all(row["stage"] == "judge" for row in errors)
    assert all(row["job_id"] == success["job_id"] for row in errors)


def test_accuracy_reporting_counts_agreement_disagreement_and_did_not_fit(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    predictions = [
        _prediction("run:q1:oracle"),
        {**_prediction("run:q2:oracle"), "financebench_id": "q2"},
        {
            **_prediction("run:q3:oracle"),
            "financebench_id": "q3",
            "status": "did_not_fit",
            "model_answer": None,
            "cognitive_skills": ["information_extraction", "numerical_reasoning"],
        },
        {**_prediction("run:q4:oracle"), "financebench_id": "q4"},
        {**_prediction("run:q5:oracle"), "financebench_id": "q5"},
    ]
    (run_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in predictions)
    )
    judgments = [
        {"job_id": "run:q1:oracle", "accuracy": 1},
        {"job_id": "run:q2:oracle", "accuracy": None, "manual_review": True},
        {"job_id": "run:q5:oracle", "accuracy": None, "manual_review": True},
    ]
    (run_dir / "judgments.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in judgments)
    )
    (run_dir / "manual_reviews.jsonl").write_text(
        json.dumps(
            {
                "job_id": "run:q1:oracle",
                "human_accuracy": 0,
                "review_reason": "Must not override an agreement.",
            }
        )
        + "\n"
        + json.dumps(
            {
                "job_id": "run:q2:oracle",
                "human_accuracy": 0,
                "review_reason": "Manually resolved.",
            }
        )
        + "\n"
    )
    (run_dir / "config.toml").write_text(
        '[selection]\nconditions = ["oracle"]\nlimit = 5\n'
    )

    summary = write_report(run_dir)
    overall = summary["answer_accuracy"]["overall"][0]
    assert overall == {
        "report_view": "overall",
        "total_predictions": 5,
        "scored_answers": 2,
        "correct_answers": 1,
        "unresolved": 1,
        "unjudged": 1,
        "did_not_fit": 1,
        "review_complete": False,
        "accuracy_excluding_did_not_fit": 0.5,
        "accuracy_including_did_not_fit": 1 / 3,
    }
    numerical = next(
        row for row in summary["answer_accuracy"]["by_cognitive_skill"]
        if row["cognitive_skill"] == "numerical_reasoning"
    )
    assert numerical["did_not_fit"] == 1
    assert numerical["eval_mode"] == "oracle"
    for view in ("by_generation_method", "by_cognitive_skill", "cross_tab"):
        assert all(
            row["eval_mode"] == "oracle"
            for row in summary["answer_accuracy"][view]
        )
    assert summary["retrieval_metrics"] == {
        "overall": [],
        "by_condition": [],
        "by_generation_method": [],
        "by_cognitive_skill": [],
        "cross_tab": [],
    }

    with ZipFile(run_dir / "summary.xlsx") as workbook:
        shared_strings = workbook.read("xl/sharedStrings.xml").decode()
    assert "No applicable retrieval results." in shared_strings


def test_resolve_judgments_preserves_agreements_and_fills_only_disputes():
    judgments = {
        "agreed": {"job_id": "agreed", "accuracy": 1},
        "disputed": {"job_id": "disputed", "accuracy": None},
        "unreviewed": {"job_id": "unreviewed", "accuracy": None},
    }
    manual_reviews = {
        "agreed": {"job_id": "agreed", "human_accuracy": 0},
        "disputed": {"job_id": "disputed", "human_accuracy": 0},
    }

    resolved = _resolve_judgments(judgments, manual_reviews)

    assert resolved["agreed"]["accuracy"] == 1
    assert resolved["disputed"]["accuracy"] == 0
    assert resolved["unreviewed"]["accuracy"] is None
    assert judgments["disputed"]["accuracy"] is None


def test_cli_judge_delegates_without_creating_a_real_client(
    tmp_path, monkeypatch, capsys
):
    config_path = tmp_path / "configs" / "financebench.toml"
    config_path.parent.mkdir()
    config_path.write_text(
        """[dataset]
source_dir = "source"
output_dir = "prepared"
document_type = "10k"
expected_questions = 112
expected_documents = 64
[generation]
provider = "openrouter"
model = "z-ai/glm-5.3-flash"
base_url = "https://openrouter.ai/api/v1"
upstream_provider = "z-ai"
allow_fallbacks = false
context_window_tokens = 1048576
max_output_tokens = 8192
token_safety_margin = 1024
temperature = 0.0
reasoning_effort = "high"
timeout_seconds = 30.0
max_retries = 2
[judge]
provider = "azure"
model = "DeepSeek-V4-Flash"
deployment = "DeepSeek-V4-Flash"
prompt_version = "financebench-binary-judge-v2"
max_output_tokens = 512
timeout_seconds = 30.0
max_retries = 2
[judge_validation]
source_commit = "cc39aeb4afdf33909ee1412188bf89035950c2eb"
seed = 42
correct_examples = 15
incorrect_examples = 10
refusal_examples = 5
minimum_agreement = 0.90
[development_subsets]
seed = 42
smoke_size = 10
pattern_size = 50
[run]
experiment = "financebench"
variant = "baseline-context-conditions-v1"
conditions = ["oracle"]
retrieval_depth = 10
results_dir = "results"
"""
    )
    run_dir = tmp_path / "run"
    captured = {}

    def fake_judge(selected_run_dir, judge_config):
        captured["run_dir"] = selected_run_dir
        captured["config"] = judge_config
        return {"judged": 1, "skipped": 0, "failed": 0}

    monkeypatch.setattr("sec_rag_benchmark.cli.judge_run", fake_judge)
    assert main(
        [
            "judge",
            "--config",
            str(config_path),
            "--run-dir",
            str(run_dir),
        ]
    ) == 0
    assert captured["run_dir"] == run_dir
    assert captured["config"]["deployment"] == "DeepSeek-V4-Flash"
    assert "'judged': 1" in capsys.readouterr().out
