import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

from sec_rag_benchmark.conditions import RetrieverUnavailable, build_condition, gold_pages
from sec_rag_benchmark.cli import main
from sec_rag_benchmark.config import load_config
from sec_rag_benchmark.data import DataError, load_questions, prepare, validate
from sec_rag_benchmark.execution.preflight import dry_run
from sec_rag_benchmark.execution.runner import run_benchmark
from sec_rag_benchmark.generation import (
    ContextLimitError,
    build_messages,
    count_prompt_tokens,
    generate,
)
from sec_rag_benchmark.metrics import cognitive_skills, page_metrics
from sec_rag_benchmark.reporting import write_report


def _write_config(path: Path, sample) -> None:
    """Write the small fixture configuration used by CLI and runner tests."""
    dataset, generation, run_config = sample["dataset"], sample["generation"], sample["run"]
    path.write_text(f'''[dataset]
source_dir = "{dataset['source_dir']}"
output_dir = "{dataset['output_dir']}"
document_type = "10k"
expected_questions = 2
expected_documents = 2
[generation]
provider = "openrouter"
model = "z-ai/glm-5.3-flash"
prompt_version = "financebench-answer-v1"
base_url = "https://openrouter.ai/api/v1"
upstream_provider = "z-ai"
allow_fallbacks = false
context_window_tokens = 1048576
max_output_tokens = {generation['max_output_tokens']}
token_safety_margin = 1024
temperature = 0.0
reasoning_effort = "{generation['reasoning_effort']}"
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
smoke_size = 2
pattern_size = 2
[run]
experiment = "{run_config['experiment']}"
variant = "{run_config['variant']}"
conditions = ["closed_book", "oracle", "long_context"]
retrieval_depth = {run_config['retrieval_depth']}
results_dir = "{run_config['results_dir']}"
''')


def test_prepare_validate_and_load_preserve_source_rows(sample):
    assert prepare(sample["dataset"]) == {"questions": 2, "documents": 2}
    assert validate(sample["dataset"]) == {"questions": 2, "documents": 2}
    questions = load_questions(sample["dataset"]["output_dir"])
    assert [q["financebench_id"] for q in questions] == ["q1", "q2"]
    assert len(questions[1]["evidence"]) == 2
    assert questions[0]["document_metadata"]["doc_type"] == "10K"
    source_path = Path(sample["dataset"]["source_dir"]) / "data" / "financebench_open_source.jsonl"
    prepared_path = Path(sample["dataset"]["output_dir"]) / "financebench_open_source_10k.jsonl"
    assert list(json.loads(source_path.read_text().splitlines()[0])) == list(json.loads(prepared_path.read_text().splitlines()[0]))


def test_prepare_rejects_missing_pdf_and_ambiguous_metadata(sample):
    source = Path(sample["dataset"]["source_dir"])
    (source / "pdfs" / "b.pdf").unlink()
    with pytest.raises(DataError, match="missing"):
        prepare(sample["dataset"])
    # Restore the fixture PDF by copying the still-valid A PDF, then make B's
    # metadata ambiguous; page contents do not matter for this validation.
    (source / "pdfs" / "b.pdf").write_bytes((source / "pdfs" / "a.pdf").read_bytes())
    metadata_path = source / "data" / "financebench_document_information.jsonl"
    rows = [json.loads(line) for line in metadata_path.read_text().splitlines()]
    rows.append(dict(rows[1]))
    metadata_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(DataError, match="ambiguous"):
        prepare(sample["dataset"])


@pytest.mark.parametrize("mutation, message", [
    (lambda rows: rows[1].update(financebench_id="q1"), "unique"),
    (lambda rows: rows[0]["evidence"][0].update(evidence_page_num=9), "page index"),
])
def test_validation_rejects_bad_question_data(sample, mutation, message):
    prepare(sample["dataset"])
    path = Path(sample["dataset"]["output_dir"]) / "financebench_open_source_10k.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    mutation(rows); path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(DataError, match=message): validate(sample["dataset"])


def test_all_conditions_and_retrieval_scopes(sample):
    prepare(sample["dataset"]); question = load_questions(sample["dataset"]["output_dir"])[1]
    pdf_dir = Path(sample["dataset"]["output_dir"]) / "pdfs"
    assert build_condition(question, "closed_book", pdf_dir, ("a.pdf", "b.pdf"))["context"] == ""
    oracle = build_condition(question, "oracle", pdf_dir, ("a.pdf", "b.pdf"))
    assert oracle["context_pages"] == [("b.pdf", 0), ("b.pdf", 1)]
    assert "B zero" in build_condition(question, "long_context", pdf_dir, ("a.pdf", "b.pdf"))["context"]
    calls = []
    def retrieve(query, scope, top_k):
        calls.append(scope); return [{"chunk_id": "c", "text": "x", "doc_name": "b.pdf", "pages": [1], "score": .9, "rank": 1}]
    build_condition(question, "single_store", pdf_dir, ("a.pdf", "b.pdf"), retriever=retrieve)
    build_condition(question, "shared_store", pdf_dir, ("a.pdf", "b.pdf"), retriever=retrieve)
    assert calls == [("b.pdf",), ("a.pdf", "b.pdf")]
    with pytest.raises(RetrieverUnavailable): build_condition(question, "single_store", pdf_dir, (), retriever=None)


def test_metrics_use_document_aware_unique_pages_and_chunk_rank():
    chunks = [
        {"doc_name": "wrong.pdf", "pages": [2], "rank": 1},
        {"doc_name": "a.pdf", "pages": [2, 5], "rank": 2},
        {"doc_name": "a.pdf", "pages": [2], "rank": 3},
    ]
    assert page_metrics([("a.pdf", 2), ("a.pdf", 5)], chunks, 5) == {
        "page_recall": 1.0, "page_precision": 2 / 3, "page_mrr": 0.5,
    }
    # The duplicate a.pdf page 2 counts once, while wrong.pdf page 2 is a
    # different document-aware page. Limiting retrieval to rank 1 finds no gold.
    assert page_metrics([("a.pdf", 2)], chunks, 1) == {
        "page_recall": 0.0, "page_precision": 0.0, "page_mrr": 0.0,
    }
    # No retrieved chunks means no retrieved pages and no first relevant rank.
    assert page_metrics([("a.pdf", 2)], [], 5) == {
        "page_recall": 0.0, "page_precision": 0.0, "page_mrr": 0.0,
    }
    assert cognitive_skills("Information extraction OR Numerical reasoning") == [
        "information_extraction",
        "numerical_reasoning",
    ]


@pytest.mark.parametrize(
    "raw_label, expected",
    [
        (None, ["unlabelled"]),
        ("Information extraction", ["information_extraction"]),
        ("Numerical reasoning OR information extraction", ["numerical_reasoning", "information_extraction"]),
        ("Logical reasoning (based on numerical reasoning)", ["logical_reasoning"]),
        ("Logical reasoning (based on numerical reasoning) OR Logical reasoning", ["logical_reasoning"]),
        ("Information extraction OR Logical reasoning OR", ["information_extraction", "logical_reasoning"]),
    ],
)
def test_cognitive_skills_strictly_normalize_known_labels(raw_label, expected):
    assert cognitive_skills(raw_label) == expected


def test_cognitive_skills_reject_unknown_labels():
    with pytest.raises(ValueError, match="Unknown cognitive skill"):
        cognitive_skills("New reasoning taxonomy")


def test_report_places_multi_skill_prediction_in_each_skill_view(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.toml").write_text(
        "[selection]\nconditions=['single_store', 'oracle']\nlimit=1\n"
    )
    prediction = {
        "job_id": "q1:single_store",
        "status": "success",
        "eval_mode": "single_store",
        "question_type": "calculated",
        "cognitive_skills": ["information_extraction", "numerical_reasoning"],
        "page_recall": 1.0,
        "page_precision": 0.5,
        "page_mrr": 1.0,
    }
    oracle_prediction = {
        **prediction,
        "job_id": "q1:oracle",
        "eval_mode": "oracle",
        "page_recall": None,
        "page_precision": None,
        "page_mrr": None,
    }
    (run_dir / "predictions.jsonl").write_text(
        json.dumps(prediction) + "\n" + json.dumps(oracle_prediction) + "\n"
    )
    (run_dir / "summary.csv").write_text("obsolete\n")

    report = write_report(run_dir)
    skill_rows = report["retrieval_metrics"]["by_cognitive_skill"]

    assert {row["cognitive_skill"] for row in skill_rows} == {
        "information_extraction",
        "numerical_reasoning",
    }
    assert all(row["total_predictions"] == 1 for row in skill_rows)
    assert all(row["eval_mode"] == "single_store" for row in skill_rows)
    assert {
        row["eval_mode"]
        for row in report["answer_accuracy"]["by_generation_method"]
    } == {"oracle", "single_store"}
    assert not (run_dir / "summary.csv").exists()

    with ZipFile(run_dir / "summary.xlsx") as workbook:
        workbook_xml = workbook.read("xl/workbook.xml").decode()
    assert all(
        name in workbook_xml
        for name in ("Overview", "Answer accuracy", "Retrieval metrics")
    )


def test_generation_pins_provider_without_real_api_call(sample, monkeypatch):
    captured = {}
    template_call = {}
    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            template_call.update(messages=messages, **kwargs)
            return {"input_ids": [1, 2, 3]}
    tokenizer = FakeTokenizer()
    monkeypatch.setattr("sec_rag_benchmark.generation.get_tokenizer", lambda: tokenizer)
    messages = build_messages("Q", "C")
    assert count_prompt_tokens(messages, "low") == 3
    assert template_call == {
        "messages": messages,
        "add_generation_prompt": True,
        "tokenize": True,
        "reasoning_effort": "low",
    }
    response = SimpleNamespace(id="r1", model="glm", output_text="42",
        openrouter_metadata={"endpoints": {"available": [
            {"provider": "Z.AI", "selected": True},
        ]}},
        usage=SimpleNamespace(model_dump=lambda: {"total_tokens": 10, "cost": 0.001}))
    def create(**kwargs): captured.update(kwargs); return response
    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    result = generate(messages, sample["generation"], client=client)
    assert result["answer"] == "42"
    assert result["requested_model"] == "z-ai/glm-5.3-flash"
    assert result["returned_model"] == "glm"
    assert result["provider"] == "Z.AI"
    assert result["usage"] == {"total_tokens": 10, "cost": 0.001}
    assert result["cost"] == 0.001
    assert result["latency_seconds"] >= 0
    assert captured["input"] == messages
    assert captured["max_output_tokens"] == 8192
    assert captured["reasoning"] == {"effort": "high"}
    assert captured["store"] is False
    assert captured["extra_headers"] == {"X-OpenRouter-Metadata": "enabled"}
    assert captured["extra_body"]["provider"] == {"order": ["z-ai"], "allow_fallbacks": False}


def test_generation_rejects_empty_output_and_oversized_prompt(sample, monkeypatch):
    monkeypatch.setattr(
        "sec_rag_benchmark.generation.get_tokenizer",
        lambda: SimpleNamespace(apply_chat_template=lambda *args, **kwargs: {"input_ids": [1, 2, 3]}),
    )
    response = SimpleNamespace(id="r1", model="glm", output_text="", usage=None)
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: response))
    with pytest.raises(RuntimeError, match="empty answer"):
        generate(build_messages("Q", "C"), sample["generation"], client=client)

    response.output_text = "42"
    assert generate(build_messages("Q", "C"), sample["generation"], client=client)["provider"] is None

    called = False
    def should_not_run(**kwargs):
        nonlocal called
        called = True
    client.responses.create = should_not_run
    tiny_context = {**sample["generation"], "context_window_tokens": 1}
    with pytest.raises(ContextLimitError, match="context window"):
        generate(build_messages("Q", "C"), tiny_context, client=client)
    assert called is False


def test_load_config_resolves_paths_and_rejects_unknown_condition(sample, tmp_path):
    config_path = tmp_path / "configs" / "financebench.toml"
    config_path.parent.mkdir()
    _write_config(config_path, sample)
    config = load_config(config_path)
    assert Path(config["dataset"]["source_dir"]).is_absolute()
    assert Path(config["run"]["results_dir"]).is_absolute()
    assert config["generation"]["reasoning_effort"] == "high"
    assert config["generation"]["max_output_tokens"] == 8192
    assert config["run"]["retrieval_depth"] == 10

    config_path.write_text(
        config_path.read_text().replace(
            '["closed_book", "oracle", "long_context"]', '["unknown"]'
        )
    )
    with pytest.raises(ValueError, match="unknown conditions"):
        load_config(config_path)

    _write_config(config_path, sample)
    config_path.write_text(
        config_path.read_text().replace(
            'variant = "baseline-context-conditions-v1"', 'variant = "unsafe label"'
        )
    )
    with pytest.raises(ValueError, match="experiment and variant"):
        load_config(config_path)


def test_dry_run_preflights_jobs_without_api_requests(sample, tmp_path, monkeypatch):
    prepare(sample["dataset"])
    config_path = tmp_path / "financebench.toml"
    _write_config(config_path, sample)
    config = load_config(config_path)
    monkeypatch.setattr(
        "sec_rag_benchmark.execution.preflight.count_prompt_tokens",
        lambda messages, effort: 10,
    )

    result = dry_run(config, conditions=["closed_book", "oracle"], limit=1)

    assert result == {
        "planned_jobs": 2,
        "conditions": ["closed_book", "oracle"],
        "maximum_prompt_tokens": {"closed_book": 10, "oracle": 10},
        "oversized_jobs": 0,
        "api_requests": 0,
    }


def test_runner_checkpoints_resumes_and_reports(sample):
    prepare(sample["dataset"])
    config_path = Path(sample["run"]["results_dir"]).parent / "financebench.toml"
    _write_config(config_path, sample)
    config = load_config(config_path)
    def fake(messages, config): return {"answer": "42", "requested_model": config["model"], "request_id": "r", "returned_model": "glm", "provider": "Z.AI", "usage": {}, "cost": None, "latency_seconds": 0.1}
    first_run = run_benchmark(
        config, config_path, conditions=["closed_book"], generator=fake
    )
    run_dir = first_run["run_dir"]
    assert first_run["generated"] == 2
    assert run_dir.name.endswith(
        "--financebench--baseline-context-conditions-v1"
    )
    assert run_benchmark(
        config,
        config_path,
        conditions=["closed_book"],
        requested_run_dir=run_dir,
        generator=fake,
    )["skipped"] == 2
    questions = load_questions(sample["dataset"]["output_dir"])
    prediction = json.loads((run_dir / "predictions.jsonl").read_text().splitlines()[0])
    assert "numeric_accuracy" not in prediction
    assert prediction["gold_evidence"] == questions[0]["evidence"]
    assert prediction["human_justification"] == questions[0]["justification"]
    assert prediction["requested_model"] == "z-ai/glm-5.3-flash"
    assert prediction["cost"] is None
    report = write_report(run_dir)
    assert report["run_status"]["complete"] is True
    assert report["run_status"]["experiment"] == "financebench"
    assert report["run_status"]["variant"] == "baseline-context-conditions-v1"
    assert report["retrieval_metrics"]["overall"] == []
    with ZipFile(run_dir / "summary.xlsx") as workbook:
        strings = workbook.read("xl/sharedStrings.xml").decode()
    assert "financebench" in strings
    assert "baseline-context-conditions-v1" in strings


def test_runner_resume_skips_did_not_fit(sample):
    prepare(sample["dataset"])
    config_path = Path(sample["run"]["results_dir"]).parent / "financebench.toml"
    _write_config(config_path, sample)
    config = load_config(config_path)
    run_dir = Path(sample["run"]["results_dir"]) / "did-not-fit-run"
    calls = 0

    def oversized(messages, generation_config):
        nonlocal calls
        calls += 1
        raise ContextLimitError(
            "Complete prompt and output reserve exceed the context window"
        )

    kwargs = {
        "conditions": ["long_context"],
        "limit": 1,
        "requested_run_dir": run_dir,
        "generator": oversized,
    }
    first_result = run_benchmark(config, config_path, **kwargs)
    second_result = run_benchmark(config, config_path, **kwargs)

    assert first_result["did_not_fit"] == 1
    assert second_result["skipped"] == 1
    assert calls == 1
    prediction = json.loads((run_dir / "predictions.jsonl").read_text().strip())
    assert prediction["status"] == "did_not_fit"
    assert prediction["model_answer"] is None
    assert prediction["requested_model"] == "z-ai/glm-5.3-flash"
    assert prediction["gold_pages"]
    assert not (run_dir / "errors.jsonl").exists()

    report = write_report(run_dir)
    assert report["run_status"]["did_not_fit"] == 1
    assert report["run_status"]["complete"] is True


def test_cli_prepare_validate_and_no_spend_dry_run(sample, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(
        "sec_rag_benchmark.execution.preflight.count_prompt_tokens",
        lambda messages, effort: 10,
    )
    config = tmp_path / "financebench.toml"
    _write_config(config, sample)
    assert main(["prepare", "--config", str(config)]) == 0
    assert main(["validate", "--config", str(config)]) == 0
    assert main(["run", "--config", str(config), "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "Planned jobs: 6" in output
    assert "API requests sent: 0" in output
