import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sec_rag_benchmark.conditions import RetrieverUnavailable, build_condition, gold_pages
from sec_rag_benchmark.cli import main
from sec_rag_benchmark.data import DataError, load_questions, prepare, validate
from sec_rag_benchmark.generation import build_messages, count_prompt_tokens, generate
from sec_rag_benchmark.metrics import cognitive_skills, page_metrics, write_report
from sec_rag_benchmark.runner import run


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
    assert cognitive_skills("Information extraction and numerical reasoning") == ["information_extraction", "numerical_reasoning"]


def test_report_places_multi_skill_prediction_in_each_skill_view(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.toml").write_text(
        "[selection]\nconditions=['single_store']\nlimit=1\n"
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
    (run_dir / "predictions.jsonl").write_text(json.dumps(prediction) + "\n")

    report = write_report(run_dir)
    skill_rows = [
        row
        for row in report["report_rows"]
        if row["report_view"] == "cognitive_skill"
    ]

    assert {row["cognitive_skill"] for row in skill_rows} == {
        "information_extraction",
        "numerical_reasoning",
    }
    assert all(row["total_predictions"] == 1 for row in skill_rows)


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
    assert captured["max_output_tokens"] == 2048
    assert captured["reasoning"] == {"effort": "low"}
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
    with pytest.raises(ValueError, match="context window"):
        generate(build_messages("Q", "C"), tiny_context, client=client)
    assert called is False


def test_runner_checkpoints_resumes_and_reports(sample):
    prepare(sample["dataset"]); questions = load_questions(sample["dataset"]["output_dir"])
    run_dir = Path(sample["run"]["results_dir"]); run_dir.mkdir()
    (run_dir / "config.toml").write_text("[selection]\nconditions=['closed_book']\nlimit=2\n")
    def fake(messages, config): return {"answer": "42", "requested_model": config["model"], "request_id": "r", "returned_model": "glm", "provider": "Z.AI", "usage": {}, "cost": None, "latency_seconds": 0.1}
    kwargs = dict(pdf_dir=Path(sample["dataset"]["output_dir"]) / "pdfs", all_doc_names=("a.pdf", "b.pdf"),
                  generation_config=sample["generation"], run_dir=run_dir, generator=fake)
    assert run(questions, ["closed_book"], **kwargs)["generated"] == 2
    assert run(questions, ["closed_book"], **kwargs)["skipped"] == 2
    prediction = json.loads((run_dir / "predictions.jsonl").read_text().splitlines()[0])
    assert "numeric_accuracy" not in prediction
    assert prediction["gold_evidence"] == questions[0]["evidence"]
    assert prediction["human_justification"] == questions[0]["justification"]
    assert prediction["requested_model"] == "z-ai/glm-5.3-flash"
    assert prediction["cost"] is None
    report = write_report(run_dir)
    assert report["complete"] is True
    overall = next(row for row in report["report_rows"] if row["report_view"] == "overall")
    assert overall["total_predictions"] == 2
    assert overall["page_recall"] is None
    assert overall["page_recall_sample_size"] == 0


def test_cli_prepare_validate_and_no_spend_dry_run(sample, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("sec_rag_benchmark.cli.count_prompt_tokens", lambda messages, effort: 10)
    config = tmp_path / "financebench.toml"
    dataset, generation, run_config = sample["dataset"], sample["generation"], sample["run"]
    config.write_text(f'''[dataset]
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
max_output_tokens = 2048
token_safety_margin = 1024
temperature = 0.0
reasoning_effort = "low"
timeout_seconds = 30.0
max_retries = 2
[run]
conditions = ["closed_book", "oracle", "long_context"]
retrieval_depth = 5
results_dir = "{run_config['results_dir']}"
''')
    assert main(["prepare", "--config", str(config)]) == 0
    assert main(["validate", "--config", str(config)]) == 0
    assert main(["run", "--config", str(config), "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "Planned jobs: 6" in output
    assert "API requests sent: 0" in output
