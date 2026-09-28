import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

from sec_rag_benchmark.cli import main
from sec_rag_benchmark.config import load_config
from sec_rag_benchmark.dataset.financebench import (
    DataError,
    load_questions,
    prepare,
    validate,
)
from sec_rag_benchmark.evaluation.retrieval_metrics import (
    cognitive_skills,
    page_metrics,
)
from sec_rag_benchmark.execution.job import execute_job
from sec_rag_benchmark.execution.preflight import dry_run
from sec_rag_benchmark.execution.runner import run_benchmark
from sec_rag_benchmark.pipeline.conditions import (
    RetrieverUnavailable,
    build_condition,
)
from sec_rag_benchmark.pipeline.generation import (
    ContextLimitError,
    build_messages,
    count_prompt_tokens,
    generate,
)
from sec_rag_benchmark.reporting.report import write_report


def _write_config(path: Path, sample, *, extra_run: str = "") -> None:
    """Write the small fixture configuration used by CLI and runner tests.

    `extra_run` adds lines to [run], e.g. a sec_rag_config path.
    """
    dataset, generation, run_config = (
        sample["dataset"],
        sample["generation"],
        sample["run"],
    )
    path.write_text(f'''[dataset]
source_dir = "{dataset["source_dir"]}"
output_dir = "{dataset["output_dir"]}"
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
max_output_tokens = {generation["max_output_tokens"]}
token_safety_margin = 1024
temperature = 0.0
reasoning_effort = "{generation["reasoning_effort"]}"
timeout_seconds = 30.0
max_retries = 2
[judge]
provider = "azure"
model = "DeepSeek-V4-Flash"
deployment = "DeepSeek-V4-Flash"
prompt_version = "financebench-binary-judge-v2"
temperature = 0.0
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
experiment = "{run_config["experiment"]}"
variant = "{run_config["variant"]}"
conditions = ["closed_book", "oracle", "long_context"]
retrieval_depth = {run_config["retrieval_depth"]}
results_dir = "{run_config["results_dir"]}"
{extra_run}''')


def _chunk(doc_name, page, *, rank, chunk_id=None, text="chunk text"):
    """One retrieved chunk in the shape retrieve_exp1 returns."""
    return {
        "chunk_id": chunk_id or f"{doc_name}:p{page}:c0",
        "doc_name": doc_name,
        "pages": [page],
        "rank": rank,
        "score": 1.0 / rank,
        "text": text,
    }


def _bundle(chunks, *, pre_rerank=None):
    """A fake Exp1 bundle (Implementation Guide 2.1-2.3 -> Records).

    search_plan mimics enhance_query's real shape; job.py reads only
    enhancement_status and call.cost from it, and saves the rest as-is.
    """
    return {
        "chunks": chunks,
        "pre_rerank_chunks": pre_rerank if pre_rerank is not None else chunks,
        "search_plan": {
            "filename": "b.pdf",
            "enhancement_status": "ok",
            "call": {"cost": 0.0002, "latency_seconds": 1.0},
        },
        "filter_doc_name": "b.pdf",
        "filter_status": "chosen",
        "usage": {
            "embedding_tokens": 30,
            "rerank_tokens": 1000,
            "embedding_cost_usd": 6e-07,
            "rerank_cost_usd": 2e-05,
        },
        "latency_seconds": 2.5,
    }


def _fake_generator(messages, config):
    """Stands in for generate(): no OpenRouter call."""
    return {
        "answer": "42",
        "requested_model": config["model"],
        "request_id": "r",
        "returned_model": "glm",
        "provider": "Z.AI",
        "usage": {},
        "cost": None,
        "latency_seconds": 0.1,
    }


def test_prepare_validate_and_load_preserve_source_rows(sample):
    assert prepare(sample["dataset"]) == {"questions": 2, "documents": 2}
    assert validate(sample["dataset"]) == {"questions": 2, "documents": 2}
    assert (
        Path(sample["dataset"]["output_dir"]) / "dataset-preparation-record.json"
    ).is_file()
    questions = load_questions(sample["dataset"]["output_dir"])
    assert [q["financebench_id"] for q in questions] == ["q1", "q2"]
    assert len(questions[1]["evidence"]) == 2
    assert questions[0]["document_metadata"]["doc_type"] == "10K"
    source_path = (
        Path(sample["dataset"]["source_dir"])
        / "data"
        / "financebench_open_source.jsonl"
    )
    prepared_path = (
        Path(sample["dataset"]["output_dir"]) / "financebench_open_source_10k.jsonl"
    )
    assert list(json.loads(source_path.read_text().splitlines()[0])) == list(
        json.loads(prepared_path.read_text().splitlines()[0])
    )


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


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda rows: rows[1].update(financebench_id="q1"), "unique"),
        (lambda rows: rows[0]["evidence"][0].update(evidence_page_num=9), "page index"),
    ],
)
def test_validation_rejects_bad_question_data(sample, mutation, message):
    prepare(sample["dataset"])
    path = Path(sample["dataset"]["output_dir"]) / "financebench_open_source_10k.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    mutation(rows)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(DataError, match=message):
        validate(sample["dataset"])


def test_all_conditions_and_retrieval_scopes(sample):
    prepare(sample["dataset"])
    question = load_questions(sample["dataset"]["output_dir"])[1]
    pdf_dir = Path(sample["dataset"]["output_dir"]) / "pdfs"
    assert (
        build_condition(question, "closed_book", pdf_dir, ("a.pdf", "b.pdf"))["context"]
        == ""
    )
    oracle = build_condition(question, "oracle", pdf_dir, ("a.pdf", "b.pdf"))
    assert oracle["context_pages"] == [("b.pdf", 0), ("b.pdf", 1)]
    assert (
        "B zero"
        in build_condition(question, "long_context", pdf_dir, ("a.pdf", "b.pdf"))[
            "context"
        ]
    )
    calls = []

    def retrieve(query, scope, top_k):
        calls.append(scope)
        return _bundle([_chunk("b.pdf", 1, rank=1)])

    build_condition(
        question, "single_store", pdf_dir, ("a.pdf", "b.pdf"), retriever=retrieve
    )
    build_condition(
        question, "shared_store", pdf_dir, ("a.pdf", "b.pdf"), retriever=retrieve
    )
    assert calls == [("b.pdf",), ("a.pdf", "b.pdf")]
    with pytest.raises(RetrieverUnavailable):
        build_condition(question, "single_store", pdf_dir, (), retriever=None)


def test_retrieval_condition_sorts_and_limits_chunks_before_building_context(sample):
    prepare(sample["dataset"])
    question = load_questions(sample["dataset"]["output_dir"])[1]
    pdf_dir = Path(sample["dataset"]["output_dir"]) / "pdfs"

    def retrieve(query, scope, top_k):
        return _bundle(
            [
                _chunk("b.pdf", 3, rank=3, chunk_id="rank-3", text="must not reach"),
                _chunk("b.pdf", 1, rank=1, chunk_id="rank-1", text="first"),
                _chunk("b.pdf", 2, rank=2, chunk_id="rank-2", text="second"),
            ]
        )

    condition = build_condition(
        question,
        "single_store",
        pdf_dir,
        ("a.pdf", "b.pdf"),
        retriever=retrieve,
        top_k=2,
    )

    assert [chunk["chunk_id"] for chunk in condition["retrieved_chunks"]] == [
        "rank-1",
        "rank-2",
    ]
    assert condition["context_pages"] == [("b.pdf", 1), ("b.pdf", 2)]
    assert condition["context"].index("first") < condition["context"].index("second")
    assert "must not reach" not in condition["context"]


def test_retrieval_context_uses_page_label_and_keeps_bundle(sample):
    """Build Order 2.4: chunks carry the oracle's [Document | Page] label."""
    prepare(sample["dataset"])
    question = load_questions(sample["dataset"]["output_dir"])[1]
    pdf_dir = Path(sample["dataset"]["output_dir"]) / "pdfs"
    bundle = _bundle([_chunk("b.pdf", 3, rank=1, text="Capex was 9.")])

    condition = build_condition(
        question,
        "shared_store",
        pdf_dir,
        ("a.pdf", "b.pdf"),
        retriever=lambda query, scope, top_k: bundle,
    )

    assert condition["context"] == "[Document: b.pdf | Page index: 3]\nCapex was 9."
    assert condition["retrieved_chunks"] == bundle["chunks"]
    assert condition["retrieval"] == {
        key: value for key, value in bundle.items() if key != "chunks"
    }
    oracle = build_condition(question, "oracle", pdf_dir, ("a.pdf", "b.pdf"))
    assert oracle["retrieval"] is None


def test_retrieval_context_rejects_multi_page_chunk(sample):
    prepare(sample["dataset"])
    question = load_questions(sample["dataset"]["output_dir"])[1]
    pdf_dir = Path(sample["dataset"]["output_dir"]) / "pdfs"
    two_pages = _chunk("b.pdf", 3, rank=1)
    two_pages["pages"] = [3, 4]

    with pytest.raises(ValueError, match="exactly one page"):
        build_condition(
            question,
            "single_store",
            pdf_dir,
            ("a.pdf", "b.pdf"),
            retriever=lambda query, scope, top_k: _bundle([two_pages]),
        )


def test_job_records_pre_rerank_metrics_filter_and_costs(sample):
    """Build Order 2.5: two page_metrics() calls, filter accuracy, costs."""
    prepare(sample["dataset"])
    question = load_questions(sample["dataset"]["output_dir"])[1]  # gold b.pdf p0, p1
    pdf_dir = Path(sample["dataset"]["output_dir"]) / "pdfs"

    def run(condition_name, **bundle_changes):
        bundle = _bundle(
            [_chunk("b.pdf", 0, rank=1), _chunk("b.pdf", 7, rank=2)],
            pre_rerank=[_chunk("b.pdf", 7, rank=1), _chunk("b.pdf", 0, rank=2)],
        )
        bundle.update(bundle_changes)
        return execute_job(
            question,
            condition_name,
            f"key:q2:{condition_name}",
            pdf_dir=pdf_dir,
            all_doc_names=("a.pdf", "b.pdf"),
            generation_config=sample["generation"],
            retrieval_depth=10,
            retriever=lambda query, scope, top_k: bundle,
            generator=_fake_generator,
        )

    row = run("shared_store")
    assert row["page_mrr"] == 1.0  # post-rerank: gold p0 at rank 1
    assert row["pre_rerank_page_mrr"] == 0.5  # pre-rerank: gold p0 at rank 2
    assert row["pre_rerank_page_recall"] == 0.5
    assert [chunk["rank"] for chunk in row["pre_rerank_chunks"]] == [1, 2]
    assert row["filter_correct"] is True
    assert row["filter_status"] == "chosen"
    assert row["enhancement_status"] == "ok"
    assert row["search_plan"]["filename"] == "b.pdf"
    assert row["query_enhancement_cost"] == 0.0002
    assert row["retrieval_cost_usd"] == pytest.approx(0.0002 + 6e-07 + 2e-05)
    assert row["retrieval_usage"]["rerank_tokens"] == 1000
    assert row["retrieval_latency_seconds"] == 2.5

    assert run("shared_store", filter_doc_name="a.pdf")["filter_correct"] is False
    fallback = run("shared_store", filter_doc_name=None, filter_status="fallback")
    assert fallback["filter_correct"] is False
    assert run("single_store")["filter_correct"] is None

    no_cost_plan = {"filename": "b.pdf", "enhancement_status": "ok", "call": {}}
    assert run("shared_store", search_plan=no_cost_plan)["retrieval_cost_usd"] is None


def test_non_retrieval_rows_have_null_pre_rerank_metrics(sample):
    prepare(sample["dataset"])
    question = load_questions(sample["dataset"]["output_dir"])[1]

    row = execute_job(
        question,
        "oracle",
        "key:q2:oracle",
        pdf_dir=Path(sample["dataset"]["output_dir"]) / "pdfs",
        all_doc_names=("a.pdf", "b.pdf"),
        generation_config=sample["generation"],
        retrieval_depth=10,
        generator=_fake_generator,
    )

    for name in ("page_recall", "page_precision", "page_mrr"):
        assert row[name] is None
        assert row[f"pre_rerank_{name}"] is None
    assert "filter_correct" not in row
    assert "retrieval_cost_usd" not in row


def test_metrics_use_document_aware_unique_pages_and_chunk_rank():
    chunks = [
        {"doc_name": "wrong.pdf", "pages": [2], "rank": 1},
        {"doc_name": "a.pdf", "pages": [2, 5], "rank": 2},
        {"doc_name": "a.pdf", "pages": [2], "rank": 3},
    ]
    assert page_metrics([("a.pdf", 2), ("a.pdf", 5)], chunks, 5) == {
        "page_recall": 1.0,
        "page_precision": 2 / 3,
        "page_mrr": 0.5,
    }
    # The duplicate a.pdf page 2 counts once, while wrong.pdf page 2 is a
    # different document-aware page. Limiting retrieval to rank 1 finds no gold.
    assert page_metrics([("a.pdf", 2)], chunks, 1) == {
        "page_recall": 0.0,
        "page_precision": 0.0,
        "page_mrr": 0.0,
    }
    # No retrieved chunks means no retrieved pages and no first relevant rank.
    assert page_metrics([("a.pdf", 2)], [], 5) == {
        "page_recall": 0.0,
        "page_precision": 0.0,
        "page_mrr": 0.0,
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
        (
            "Numerical reasoning OR information extraction",
            ["numerical_reasoning", "information_extraction"],
        ),
        ("Logical reasoning (based on numerical reasoning)", ["logical_reasoning"]),
        (
            "Logical reasoning (based on numerical reasoning) OR Logical reasoning",
            ["logical_reasoning"],
        ),
        (
            "Information extraction OR Logical reasoning OR",
            ["information_extraction", "logical_reasoning"],
        ),
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
        "[run]\nretrieval_depth=10\n"
        "[selection]\nconditions=['single_store', 'oracle']\nlimit=1\n"
    )
    prediction = {
        "job_id": "q1:single_store",
        "financebench_id": "q1",
        "status": "success",
        "eval_mode": "single_store",
        "question_type": "calculated",
        "cognitive_skills": ["information_extraction", "numerical_reasoning"],
        "gold_pages": [["target.pdf", 2]],
        "retrieved_chunks": [{"doc_name": "target.pdf", "pages": [2], "rank": 1}],
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
        row["eval_mode"] for row in report["answer_accuracy"]["by_generation_method"]
    } == {"oracle", "single_store"}
    assert report["run_status"]["by_condition"] == [
        {
            "eval_mode": "oracle",
            "planned": 1,
            "successful": 1,
            "did_not_fit": 0,
            "failed": 0,
            "missing": 0,
            "complete": True,
        },
        {
            "eval_mode": "single_store",
            "planned": 1,
            "successful": 1,
            "did_not_fit": 0,
            "failed": 0,
            "missing": 0,
            "complete": True,
        },
    ]
    analysis_path = run_dir / "failure_analysis.jsonl"
    first_analysis = analysis_path.read_text()
    [analysis_row] = [json.loads(line) for line in first_analysis.splitlines()]
    assert analysis_row["failure_subtype"] == "unjudged_condition"
    assert report["failure_analysis"]["unclassified_count"] == 1

    # This file is a derived report, not append-only attempt history. Running
    # the report again must replace it rather than duplicate its rows.
    write_report(run_dir)
    assert analysis_path.read_text() == first_analysis
    assert not (run_dir / "summary.csv").exists()

    with ZipFile(run_dir / "summary.xlsx") as workbook:
        workbook_xml = workbook.read("xl/workbook.xml").decode()
        shared_strings = workbook.read("xl/sharedStrings.xml").decode()
    assert all(
        name in workbook_xml
        for name in (
            "Overview",
            "Answer accuracy",
            "Retrieval metrics",
            "Failure analysis",
        )
    )
    assert "Classification legend" in shared_strings
    assert "Scored answers" in shared_strings
    assert "Correct answers" in shared_strings
    assert "Unresolved" in shared_strings
    assert "Review complete" in shared_strings


def test_report_counts_failed_and_missing_jobs_by_condition(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.toml").write_text(
        "[run]\nretrieval_depth=10\n"
        "[selection]\nconditions=['shared_store', 'oracle']\nlimit=2\n"
    )
    oracle_prediction = {
        "job_id": "run:q1:oracle",
        "financebench_id": "q1",
        "status": "success",
        "eval_mode": "oracle",
        "question_type": "calculated",
        "cognitive_skills": ["numerical_reasoning"],
        "gold_pages": [["target.pdf", 2]],
        "retrieved_chunks": [],
        "page_recall": None,
        "page_precision": None,
        "page_mrr": None,
    }
    (run_dir / "predictions.jsonl").write_text(json.dumps(oracle_prediction) + "\n")
    (run_dir / "errors.jsonl").write_text(
        json.dumps(
            {
                "job_id": "run:q1:shared_store",
                "status": "error",
                "error": "temporary failure",
            }
        )
        + "\n"
    )

    report = write_report(run_dir)

    assert report["run_status"]["successful"] == 1
    assert report["run_status"]["failed"] == 1
    assert report["run_status"]["missing"] == 2
    assert report["run_status"]["by_condition"] == [
        {
            "eval_mode": "oracle",
            "planned": 2,
            "successful": 1,
            "did_not_fit": 0,
            "failed": 0,
            "missing": 1,
            "complete": False,
        },
        {
            "eval_mode": "shared_store",
            "planned": 2,
            "successful": 0,
            "did_not_fit": 0,
            "failed": 1,
            "missing": 1,
            "complete": False,
        },
    ]


def test_generation_pins_provider_without_real_api_call(sample, monkeypatch):
    captured = {}
    template_call = {}

    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            template_call.update(messages=messages, **kwargs)
            return {"input_ids": [1, 2, 3]}

    tokenizer = FakeTokenizer()
    monkeypatch.setattr(
        "sec_rag_benchmark.pipeline.generation.get_tokenizer", lambda: tokenizer
    )
    messages = build_messages("Q", "C")
    assert count_prompt_tokens(messages, "low") == 3
    assert template_call == {
        "messages": messages,
        "add_generation_prompt": True,
        "tokenize": True,
        "reasoning_effort": "low",
    }
    response = SimpleNamespace(
        id="r1",
        model="glm",
        output_text="42",
        openrouter_metadata={
            "endpoints": {
                "available": [
                    {"provider": "Z.AI", "selected": True},
                ]
            }
        },
        usage=SimpleNamespace(model_dump=lambda: {"total_tokens": 10, "cost": 0.001}),
    )

    def create(**kwargs):
        captured.update(kwargs)
        return response

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
    assert captured["extra_body"]["provider"] == {
        "order": ["z-ai"],
        "allow_fallbacks": False,
    }


def test_generation_rejects_empty_output_and_oversized_prompt(sample, monkeypatch):
    monkeypatch.setattr(
        "sec_rag_benchmark.pipeline.generation.get_tokenizer",
        lambda: SimpleNamespace(
            apply_chat_template=lambda *args, **kwargs: {"input_ids": [1, 2, 3]}
        ),
    )
    response = SimpleNamespace(id="r1", model="glm", output_text="", usage=None)
    client = SimpleNamespace(
        responses=SimpleNamespace(create=lambda **kwargs: response)
    )
    with pytest.raises(RuntimeError, match="empty answer"):
        generate(build_messages("Q", "C"), sample["generation"], client=client)

    response.output_text = "42"
    assert (
        generate(build_messages("Q", "C"), sample["generation"], client=client)[
            "provider"
        ]
        is None
    )

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
    assert config["judge"]["temperature"] == 0.0
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

    _write_config(config_path, sample)
    config_path.write_text(
        config_path.read_text()
        .replace("[judge]\nprovider", "[judge]\ntemperature = 3.0\nprovider")
        .replace(
            "temperature = 0.0\nmax_output_tokens = 512", "max_output_tokens = 512"
        )
    )
    with pytest.raises(ValueError, match="Judge temperature"):
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

    def fake(messages, config):
        return {
            "answer": "42",
            "requested_model": config["model"],
            "request_id": "r",
            "returned_model": "glm",
            "provider": "Z.AI",
            "usage": {},
            "cost": None,
            "latency_seconds": 0.1,
        }

    first_run = run_benchmark(
        config, config_path, conditions=["closed_book"], generator=fake
    )
    run_dir = first_run["run_dir"]
    assert first_run["generated"] == 2
    assert run_dir.name.endswith("--financebench--baseline-context-conditions-v1")
    assert (
        run_benchmark(
            config,
            config_path,
            conditions=["closed_book"],
            requested_run_dir=run_dir,
            generator=fake,
        )["skipped"]
        == 2
    )
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


def test_runner_shared_store_scope_is_every_prepared_filing(sample):
    """Build Order 2.0, decision B: --limit does not shrink shared-store's scope."""
    prepare(sample["dataset"])
    config_path = Path(sample["run"]["results_dir"]).parent / "financebench.toml"
    _write_config(config_path, sample)
    config = load_config(config_path)
    scopes = []

    def retrieve(query, scope, top_k):
        scopes.append(scope)
        return _bundle([_chunk("a.pdf", 0, rank=1)])

    run_benchmark(
        config,
        config_path,
        conditions=["shared_store"],
        limit=1,
        retriever=retrieve,
        generator=_fake_generator,
    )

    assert scopes == [("a.pdf", "b.pdf")]


def test_runner_snapshots_sec_rag_config_into_run_key(sample, tmp_path):
    """Build Order 2.5, decision D: retrieval settings are part of the run."""
    prepare(sample["dataset"])
    sec_rag_path = tmp_path / "sec_rag.toml"
    sec_rag_path.write_text("[rrf]\nk = 60\n", encoding="utf-8")
    config_path = Path(sample["run"]["results_dir"]).parent / "financebench.toml"
    _write_config(config_path, sample, extra_run=f'sec_rag_config = "{sec_rag_path}"\n')
    config = load_config(config_path)
    run_dir = Path(sample["run"]["results_dir"]) / "exp1-run"
    kwargs = {
        "conditions": ["single_store"],
        "limit": 1,
        "requested_run_dir": run_dir,
        "retriever": lambda query, scope, top_k: _bundle([_chunk("a.pdf", 0, rank=1)]),
        "generator": _fake_generator,
    }

    run_benchmark(config, config_path, **kwargs)

    assert (run_dir / "sec_rag.toml").read_bytes() == sec_rag_path.read_bytes()
    job_id = json.loads((run_dir / "predictions.jsonl").read_text())["job_id"]
    expected_key = hashlib.sha256(
        (run_dir / "config.toml").read_bytes() + sec_rag_path.read_bytes()
    ).hexdigest()[:12]
    assert job_id.startswith(f"{expected_key}:")

    sec_rag_path.write_text("[rrf]\nk = 10\n", encoding="utf-8")
    with pytest.raises(ValueError, match="different configuration"):
        run_benchmark(config, config_path, **kwargs)

    baseline_dir = Path(sample["run"]["results_dir"]) / "baseline-run"
    run_benchmark(
        config,
        config_path,
        conditions=["closed_book"],
        limit=1,
        requested_run_dir=baseline_dir,
        generator=_fake_generator,
    )
    assert not (baseline_dir / "sec_rag.toml").exists()
    baseline_job = json.loads((baseline_dir / "predictions.jsonl").read_text())
    baseline_key = hashlib.sha256(
        (baseline_dir / "config.toml").read_bytes()
    ).hexdigest()[:12]
    assert baseline_job["job_id"].startswith(f"{baseline_key}:")


def test_runner_requires_sec_rag_config_for_retrieval(sample):
    prepare(sample["dataset"])
    config_path = Path(sample["run"]["results_dir"]).parent / "financebench.toml"
    _write_config(config_path, sample)
    config = load_config(config_path)

    with pytest.raises(ValueError, match="sec_rag_config"):
        run_benchmark(
            config, config_path, conditions=["shared_store"], generator=_fake_generator
        )

    assert not Path(sample["run"]["results_dir"]).exists()


def test_load_config_resolves_sec_rag_config(sample, tmp_path):
    config_path = tmp_path / "configs" / "financebench.toml"
    config_path.parent.mkdir()
    _write_config(
        config_path, sample, extra_run='sec_rag_config = "configs/sec_rag.toml"\n'
    )

    config = load_config(config_path)

    assert config["run"]["sec_rag_config"] == str(tmp_path / "configs" / "sec_rag.toml")


def test_cli_prepare_validate_and_no_spend_dry_run(
    sample, tmp_path, capsys, monkeypatch
):
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


GENERATION_SNAPSHOT = (
    "[generation]\nmodel='z-ai/glm-5.3-flash'\nreasoning_effort='high'\n"
)


def _write_report_run(
    run_dir: Path,
    conditions: list[str],
    predictions: list[dict],
    *,
    judgments: list[dict] | None = None,
    manual_reviews: list[dict] | None = None,
    generation: str = GENERATION_SNAPSHOT,
) -> Path:
    """Write a minimal run folder that write_report can read."""
    run_dir.mkdir(parents=True)
    (run_dir / "config.toml").write_text(
        "[run]\nretrieval_depth=10\n"
        f"[selection]\nconditions={conditions!r}\nlimit=1\n" + generation
    )
    for name, rows in (
        ("predictions.jsonl", predictions),
        ("judgments.jsonl", judgments or []),
        ("manual_reviews.jsonl", manual_reviews or []),
    ):
        (run_dir / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
    return run_dir


def _report_row(job_id: str, condition: str, **fields) -> dict:
    """One saved prediction row, with Exp1's retrieval fields when given."""
    return {
        "job_id": job_id,
        "financebench_id": job_id.split(":")[1],
        "status": "success",
        "eval_mode": condition,
        "question_type": "metrics-generated",
        "cognitive_skills": ["information_extraction"],
        "gold_pages": [["target.pdf", 2]],
        "retrieved_chunks": [{"doc_name": "target.pdf", "pages": [2], "rank": 1}],
        "page_recall": 1.0,
        "page_precision": 0.1,
        "page_mrr": 1.0,
        **fields,
    }


def _verdict(job_id: str, accuracy: int | None) -> dict:
    return {"job_id": job_id, "accuracy": accuracy, "passes": []}


def test_report_averages_pre_rerank_and_filter_accuracy(tmp_path):
    """Build Order 2.5: pre-rerank metrics, filter accuracy, invalid replies."""
    exp1 = {"pre_rerank_page_recall": 1.0, "pre_rerank_page_precision": 0.1}
    rows = [
        _report_row(
            "x:q1:shared_store",
            "shared_store",
            **exp1,
            pre_rerank_page_mrr=1.0,
            filter_correct=True,
            enhancement_status="ok",
        ),
        _report_row(
            "x:q2:shared_store",
            "shared_store",
            **exp1,
            pre_rerank_page_mrr=0.5,
            filter_correct=False,
            enhancement_status="invalid_reply",
        ),
        _report_row(
            "x:q1:single_store",
            "single_store",
            **exp1,
            pre_rerank_page_mrr=0.25,
            filter_correct=None,
            enhancement_status="ok",
        ),
    ]
    run_dir = _write_report_run(tmp_path / "x", ["single_store", "shared_store"], rows)

    report = write_report(run_dir)

    by_condition = {
        row["eval_mode"]: row for row in report["retrieval_metrics"]["by_condition"]
    }
    shared = by_condition["shared_store"]
    assert shared["filter_accuracy"] == 0.5
    assert shared["filter_accuracy_sample_size"] == 2
    assert shared["pre_rerank_page_mrr"] == 0.75
    assert shared["pre_rerank_page_mrr_sample_size"] == 2
    assert shared["invalid_reply_count"] == 1
    single = by_condition["single_store"]
    assert single["filter_accuracy"] is None
    assert single["filter_accuracy_sample_size"] == 0
    assert single["invalid_reply_count"] == 0

    with ZipFile(run_dir / "summary.xlsx") as workbook:
        shared_strings = workbook.read("xl/sharedStrings.xml").decode()
    for label in ("Pre-rerank page MRR", "Filter accuracy", "Retrieval cost (USD)"):
        assert label in shared_strings


def test_report_adds_retrieval_cost(tmp_path):
    rows = [
        _report_row(
            "x:q1:shared_store",
            "shared_store",
            retrieval_cost_usd=0.0002,
            retrieval_latency_seconds=4.0,
        ),
        _report_row(
            "x:q2:shared_store",
            "shared_store",
            retrieval_cost_usd=None,
            retrieval_latency_seconds=2.0,
        ),
    ]
    run_dir = _write_report_run(tmp_path / "x", ["shared_store"], rows)

    overall = write_report(run_dir)["generation_performance"]["overall"]

    assert overall["total_retrieval_cost_usd"] == 0.0002
    assert overall["retrieval_cost_sample_size"] == 1
    assert overall["average_retrieval_cost_per_answer_usd"] == 0.0002
    assert overall["average_retrieval_latency_seconds"] == 3.0


def test_report_reads_old_rows_without_new_fields(tmp_path):
    run_dir = _write_report_run(
        tmp_path / "old",
        ["shared_store"],
        [_report_row("old:q1:shared_store", "shared_store")],
    )

    report = write_report(run_dir)

    [shared] = report["retrieval_metrics"]["by_condition"]
    assert shared["pre_rerank_page_mrr"] is None
    assert shared["pre_rerank_page_mrr_sample_size"] == 0
    assert shared["filter_accuracy"] is None
    assert shared["filter_accuracy_sample_size"] == 0
    overall = report["generation_performance"]["overall"]
    assert overall["total_retrieval_cost_usd"] is None
    assert overall["retrieval_cost_sample_size"] == 0
    assert report["failure_analysis"]["oracle_source"] is None


def _wrong_filing_run(tmp_path: Path) -> Path:
    """An Exp1 run whose shared-store answer was wrong: filter chose a.pdf."""
    wrong = _report_row(
        "x:q1:shared_store",
        "shared_store",
        retrieved_chunks=[{"doc_name": "a.pdf", "pages": [2], "rank": 1}],
        enhancement_status="ok",
        filter_status="chosen",
        filter_doc_name="a.pdf",
    )
    return _write_report_run(
        tmp_path / "x",
        ["shared_store"],
        [wrong],
        judgments=[_verdict("x:q1:shared_store", 0)],
    )


def test_report_borrows_oracle_for_failure_diagnosis(tmp_path):
    """Build Order 2.6: failure diagnosis compares against the baseline oracle."""
    run_dir = _wrong_filing_run(tmp_path)
    baseline = _write_report_run(
        tmp_path / "baseline",
        ["oracle"],
        [_report_row("b:q1:oracle", "oracle")],
        judgments=[_verdict("b:q1:oracle", 1)],
    )

    without = write_report(run_dir)
    unborrowed = json.loads((run_dir / "failure_analysis.jsonl").read_text())
    assert unborrowed["failure_subtype"] == "missing_oracle"
    assert without["failure_analysis"]["oracle_source"] is None

    report = write_report(run_dir, baseline)

    [row] = [
        json.loads(line)
        for line in (run_dir / "failure_analysis.jsonl").read_text().splitlines()
    ]
    assert row["failure_subtype"] == "wrong_document_wrong_filing"
    assert row["oracle_job_id"] == "b:q1:oracle"
    assert report["failure_analysis"]["oracle_source"] == "baseline"
    # Borrowed answers are compared against, never counted as this run's.
    assert {row["eval_mode"] for row in report["answer_accuracy"]["by_condition"]} == {
        "shared_store"
    }


def test_report_borrowed_oracle_uses_its_manual_review(tmp_path):
    run_dir = _wrong_filing_run(tmp_path)
    baseline = _write_report_run(
        tmp_path / "baseline",
        ["oracle"],
        [_report_row("b:q1:oracle", "oracle")],
        judgments=[_verdict("b:q1:oracle", None)],
        manual_reviews=[{"job_id": "b:q1:oracle", "human_accuracy": 1}],
    )

    write_report(run_dir, baseline)

    row = json.loads((run_dir / "failure_analysis.jsonl").read_text())
    assert row["failure_subtype"] == "wrong_document_wrong_filing"


def test_report_refuses_borrow_on_own_oracle_or_generation_mismatch(tmp_path):
    baseline = _write_report_run(
        tmp_path / "baseline",
        ["oracle"],
        [_report_row("b:q1:oracle", "oracle")],
    )
    with_oracle = _write_report_run(
        tmp_path / "own",
        ["shared_store", "oracle"],
        [
            _report_row("x:q1:shared_store", "shared_store"),
            _report_row("x:q1:oracle", "oracle"),
        ],
    )
    with pytest.raises(ValueError, match="its own oracle"):
        write_report(with_oracle, baseline)
    assert not (with_oracle / "summary.json").exists()

    low_effort = _write_report_run(
        tmp_path / "low",
        ["oracle"],
        [_report_row("l:q1:oracle", "oracle")],
        generation=GENERATION_SNAPSHOT.replace("high", "low"),
    )
    run_dir = _wrong_filing_run(tmp_path)
    with pytest.raises(ValueError, match="different generation settings"):
        write_report(run_dir, low_effort)
    assert not (run_dir / "summary.json").exists()


def test_cli_report_passes_oracle_run_dir(monkeypatch, tmp_path):
    calls = []

    def fake_write_report(run_dir, oracle_run_dir=None):
        calls.append((run_dir, oracle_run_dir))
        return {"run_status": {"successful": 0}}

    monkeypatch.setattr("sec_rag_benchmark.cli.write_report", fake_write_report)

    assert main(["report", "--run-dir", str(tmp_path / "x")]) == 0
    assert (
        main(
            [
                "report",
                "--run-dir",
                str(tmp_path / "x"),
                "--oracle-run-dir",
                str(tmp_path / "b"),
            ]
        )
        == 0
    )

    assert calls == [(tmp_path / "x", None), (tmp_path / "x", tmp_path / "b")]
