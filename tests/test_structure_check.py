"""Tests for the weight-zero check (sec-rag-benchmark check-structure).

The benchmark config is the repository's Exp2 file, copied into tmp_path
with its results folder redirected. The RAG side is the tiny two-filing
corpus from conftest's index_config, indexed for real; the loaded sec_rag
config and the Voyage client are swapped for it and a fake, so no test
can spend.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sec_rag.indexing.bm25_index import build_bm25_index
from sec_rag.indexing.embed import embed_corpus
from sec_rag.retrieval.exp1 import load_chunk_lookup
from sec_rag.retrieval.search import open_search_indexes, search
from sec_rag_benchmark.cli import main
from sec_rag_benchmark.evaluation import structure_check
from sec_rag_benchmark.evaluation.retrieval_metrics import page_metrics
from sec_rag_benchmark.evaluation.structure_check import check_structure

REPO_CONFIGS = Path(__file__).parent.parent / "configs"


def _vector(text: str) -> list[float]:
    return [1.0, float(len(text)), 0.5]


class FakeEmbedder:
    def __call__(self, texts: list[str]) -> tuple[list[list[float]], int]:
        return [_vector(text) for text in texts], 10 * len(texts)


class FakeVoyage:
    def embed(self, texts: list[str], **kwargs) -> SimpleNamespace:
        return SimpleNamespace(
            embeddings=[_vector(text) for text in texts], total_tokens=7
        )


def _row(condition: str, doc_name: str | None, gold_doc: str) -> dict:
    """One of rung A's saved rows, with only the fields the check reads."""
    return {
        "job_id": f"k:q1:{condition}",
        "status": "success",
        "eval_mode": condition,
        "filter_doc_name": doc_name,
        "gold_pages": [[gold_doc, 0]],
        "pre_rerank_page_recall": 1.0,
        "search_plan": {
            "keyword_query": "capital expenditure",
            "semantic_query": "How much was capital expenditure?",
        },
    }


@pytest.fixture
def check_setup(index_config, tmp_path, monkeypatch):
    """Tiny indexes, rung A's run folder, and an Exp2 benchmark config beside them."""
    config, add_filing = index_config
    config["retrieval"] = {"candidates_per_search": 50, "rrf_k": 60}
    add_filing("AAA_2020_10K", ["Capital expenditure was 100.", "Revenue grew."])
    add_filing("BBB_2020_10K", ["Capital expenditure was 200.", "Goodwill."])
    build_bm25_index(config)
    embed_corpus(config, execute_paid=True, embedder=FakeEmbedder())

    run_a = tmp_path / "run-a"
    run_a.mkdir()
    rows = [
        _row("single_store", "AAA_2020_10K", "AAA_2020_10K"),
        _row("shared_store", None, "BBB_2020_10K"),
    ]
    (run_a / "predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    project = tmp_path / "project"
    (project / "configs").mkdir(parents=True)
    benchmark_toml = (REPO_CONFIGS / "financebench-exp2.toml").read_text(
        encoding="utf-8"
    )
    config_path = project / "configs" / "financebench-exp2.toml"
    config_path.write_text(benchmark_toml, encoding="utf-8")
    (project / "configs" / "sec_rag-exp2.toml").write_text("# stand-in\n")

    monkeypatch.setattr(structure_check, "load_sec_rag_config", lambda path: config)
    monkeypatch.setattr(structure_check, "voyage_client", FakeVoyage)
    return config_path, run_a, config, project


def test_dry_run_reads_no_key_and_counts_rows(check_setup, monkeypatch) -> None:
    config_path, run_a, _config, project = check_setup

    def no_key():
        raise AssertionError("the free report must not create a Voyage client")

    monkeypatch.setattr(structure_check, "voyage_client", no_key)

    result = check_structure(config_path, run_a)

    assert result["rows"] == 2
    assert result["estimated_query_tokens"] > 0
    assert not (project / "results").exists()


def test_identical_rankings_give_full_overlap(check_setup) -> None:
    config_path, run_a, _config, project = check_setup

    result = check_structure(config_path, run_a, execute_paid=True)

    # On this tiny corpus Chroma's search is exact, so the lists agree in
    # full: 2 of AAA's 2 chunks (single-store), 4 of all 4 (shared-store,
    # no filter). Mean (2 + 4) / 2.
    assert result["mean_top50_overlap"] == 3.0
    assert result["min_top50_overlap"] == 2
    assert result["embedding_tokens"] == 14
    out_dir = Path(result["out_dir"])
    assert out_dir.parent == project / "results"
    assert out_dir.name.endswith("--exp2--weight-zero-check")
    assert (out_dir / "sec_rag.toml").read_text() == "# stand-in\n"
    saved = json.loads((out_dir / "structure_check.json").read_text())
    assert saved["rows"] == 2


def test_pre_rerank_metrics_match_page_metrics_on_the_fused_list(check_setup) -> None:
    """At weight 0 the rebuilt list equals Exp1's hybrid search on this corpus."""
    config_path, run_a, config, _project = check_setup

    result = check_structure(config_path, run_a, execute_paid=True)

    lines = (Path(result["out_dir"]) / "structure_check_rows.jsonl").read_text()
    rows = [json.loads(line) for line in lines.splitlines()]
    single = next(row for row in rows if row["condition"] == "single_store")
    indexes = open_search_indexes(config, FakeVoyage())
    chunks = load_chunk_lookup(config)
    found = search(
        "capital expenditure",
        "How much was capital expenditure?",
        method="hybrid",
        doc_name="AAA_2020_10K",
        top_k=10,
        indexes=indexes,
    )["ranked"]
    expected = page_metrics(
        [("AAA_2020_10K", 0)],
        [
            {"doc_name": chunks[chunk_id]["doc_name"], "pages": [0], "rank": rank}
            for rank, (chunk_id, _score) in enumerate(found, start=1)
        ],
        10,
    )
    assert single["pre_rerank_page_recall"] == expected["page_recall"]
    assert single["pre_rerank_page_precision"] == expected["page_precision"]
    assert single["pre_rerank_page_mrr"] == expected["page_mrr"]


def test_summary_averages_per_condition_and_passes(check_setup) -> None:
    config_path, run_a, _config, _project = check_setup

    result = check_structure(config_path, run_a, execute_paid=True)

    assert set(result["pre_rerank"]) == {"single_store", "shared_store"}
    for metrics in result["pre_rerank"].values():
        assert metrics["page_recall"] == 1.0  # every chunk is on page 0
        assert metrics["reported_page_recall"] == 1.0
    # Overlap 3 < 49.5 only because the toy filings hold 2-4 chunks.
    assert result["passed"] is False


def test_the_command_reports_without_spending(check_setup, capsys) -> None:
    config_path, run_a, _config, _project = check_setup

    exit_code = main(
        [
            "check-structure",
            "--config",
            str(config_path),
            "--baseline-run-dir",
            str(run_a),
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Rows: 2\n" in output
    assert "Nothing sent" in output


def test_a_baseline_without_predictions_is_refused(check_setup, tmp_path) -> None:
    config_path, _run_a, _config, _project = check_setup

    with pytest.raises(ValueError, match="no predictions"):
        check_structure(config_path, tmp_path / "missing")
