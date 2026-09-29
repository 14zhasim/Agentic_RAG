"""The weight-zero check: does Exp2's exact scorer reproduce Exp1 (Build Order 3.4)?

Rung A (Exp1) found its semantic candidates with Chroma's approximate
nearest-neighbour search; rung B scores every chunk exactly, adding the
structure term. So A and B differ in two ways unless the scorer itself is
shown to match Chroma. This check re-runs each of A's saved searches with
the exact scorer at weight 0 - where its score is question·chunk alone,
what Chroma computes - and reports:

- the overlap of Chroma's top 50 with the scorer's (expected ~49.98 of 50:
  only Chroma's approximation can differ), and
- A's pre-rerank page metrics rebuilt with the scorer, so A vs B before
  reranking compares identical code with one number changed.

It searches exactly as A did: A's saved queries and A's filter, wrong
filings included. The only paid step is re-embedding each semantic query
(A did not save query vectors): ~224 short texts, inside Voyage's free
allowance. Without `execute_paid`, nothing is sent and no key is read.
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from sec_rag.chunking.chunk import count_tokens
from sec_rag.config import load_config as load_sec_rag_config
from sec_rag.retrieval.exp1 import load_chunk_lookup, voyage_client
from sec_rag.retrieval.search import (
    bm25_ranking,
    chroma_ranking,
    embed_query,
    exact_ranking,
    open_search_indexes,
    reciprocal_rank_fusion,
)

from ..config import load_config
from .retrieval_metrics import page_metrics

# Pass rule (Implementation Guide 3.1-3.4 -> Slice 3): the scorer agrees
# with Chroma, and the rebuilt pre-rerank recall matches A's reported one.
MINIMUM_MEAN_OVERLAP = 49.5
MAXIMUM_RECALL_GAP = 0.01


def check_structure(
    benchmark_config_path: Path, baseline_run_dir: Path, execute_paid: bool = False
) -> dict[str, Any]:
    """Re-run rung A's searches with the exact scorer at weight 0 and compare.

    Steps: 1) read A's latest successful rows; 2) without execute_paid,
    report the row count and estimated query tokens and stop; 3) for each
    row, embed its semantic query once, rank with Chroma and with the
    scorer, count the top-50 overlap, fuse the scorer's list with BM25 by
    RRF as Exp1 does, and score the top retrieval_depth against the gold
    pages; 4) summarise per condition and write the results folder.

    Returns the summary written to structure_check.json (or, without
    execute_paid, {"rows", "estimated_query_tokens"}).
    """
    config = load_config(benchmark_config_path)
    sec_rag_config_path = Path(config["run"]["sec_rag_config"])
    sec_rag_config = load_sec_rag_config(sec_rag_config_path)
    top_k = config["run"]["retrieval_depth"]

    # Step 1: one row per job, as report.py reads a run.
    rows = _latest_successful_rows(baseline_run_dir)

    # Step 2: the free report. count_tokens is cl100k, an estimate of
    # Voyage's count, as in sec_rag's embed.
    estimated = sum(count_tokens(row["search_plan"]["semantic_query"]) for row in rows)
    if not execute_paid:
        return {"rows": len(rows), "estimated_query_tokens": estimated}

    # Step 3. The Exp2 config has weight 1, so open_search_indexes also
    # opens the structure collection: the check refuses to run before
    # `sec-rag index-structure` has built it, the order the guide sets.
    indexes = open_search_indexes(sec_rag_config, voyage_client())
    chunks = load_chunk_lookup(sec_rag_config)
    rrf_k = sec_rag_config["retrieval"]["rrf_k"]

    out_dir = Path(config["run"]["results_dir"]) / (
        f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}--exp2--weight-zero-check"
    )
    out_dir.mkdir(parents=True)
    # The settings behind the numbers, as a benchmark run keeps its own.
    shutil.copyfile(sec_rag_config_path, out_dir / "sec_rag.toml")

    results: list[dict[str, Any]] = []
    tokens = 0
    with (out_dir / "structure_check_rows.jsonl").open("w", encoding="utf-8") as out:
        for row in rows:
            result, row_tokens = _check_one(row, indexes, chunks, rrf_k, top_k)
            tokens += row_tokens
            results.append(result)
            # One line per search as it finishes, so a crash keeps the rest.
            out.write(json.dumps(result) + "\n")
            out.flush()

    # Step 4.
    summary = _summarise(results) | {"embedding_tokens": tokens}
    (out_dir / "structure_check.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary | {"out_dir": str(out_dir)}


def _check_one(
    row: dict[str, Any],
    indexes: Any,
    chunks: dict[str, dict[str, Any]],
    rrf_k: int,
    top_k: int,
) -> tuple[dict[str, Any], int]:
    """One of A's searches, redone: its overlap and rebuilt pre-rerank metrics."""
    plan = row["search_plan"]
    doc_name = row["filter_doc_name"]  # A's filter, exactly as it searched
    vector, tokens = embed_query(indexes, plan["semantic_query"])

    chroma_ids = [chunk_id for chunk_id, _ in chroma_ranking(indexes, vector, doc_name)]
    exact_ids = [
        chunk_id for chunk_id, _ in exact_ranking(indexes, vector, doc_name, 0.0)
    ]
    bm25_ids = [
        chunk_id
        for chunk_id, _ in bm25_ranking(indexes, plan["keyword_query"], doc_name)
    ]

    # Exp1's hybrid step with the scorer's list in Chroma's place.
    fused = reciprocal_rank_fusion([bm25_ids, exact_ids], k=rrf_k)[:top_k]
    pre_rerank = [
        {
            "doc_name": chunks[chunk_id]["doc_name"],
            "pages": [chunks[chunk_id]["page_index"]],
            "rank": rank,
        }
        for rank, (chunk_id, _score) in enumerate(fused, start=1)
    ]
    # Stored as [doc_name, page] lists in JSON; page_metrics compares tuples.
    gold = [(doc, page) for doc, page in row["gold_pages"]]
    metrics = page_metrics(gold, pre_rerank, top_k)
    result = {
        "job_id": row["job_id"],
        "condition": row["eval_mode"],
        "filter_doc_name": doc_name,
        "top50_overlap": len(set(chroma_ids) & set(exact_ids)),
        "compared": len(chroma_ids),
        **{f"pre_rerank_{key}": value for key, value in metrics.items()},
        "reported_pre_rerank_page_recall": row["pre_rerank_page_recall"],
    }
    return result, tokens


def _summarise(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Overlap over every search; page metrics per condition; the pass verdict."""
    overlaps = [result["top50_overlap"] for result in results]
    by_condition: dict[str, dict[str, float]] = {}
    for condition in sorted({result["condition"] for result in results}):
        group = [result for result in results if result["condition"] == condition]
        by_condition[condition] = {
            "page_recall": mean(r["pre_rerank_page_recall"] for r in group),
            "page_precision": mean(r["pre_rerank_page_precision"] for r in group),
            "page_mrr": mean(r["pre_rerank_page_mrr"] for r in group),
            "reported_page_recall": mean(
                r["reported_pre_rerank_page_recall"] for r in group
            ),
        }
    recall_gaps = [
        abs(metrics["page_recall"] - metrics["reported_page_recall"])
        for metrics in by_condition.values()
    ]
    return {
        "rows": len(results),
        "mean_top50_overlap": mean(overlaps),
        "min_top50_overlap": min(overlaps),
        "pre_rerank": by_condition,
        "passed": mean(overlaps) >= MINIMUM_MEAN_OVERLAP
        and max(recall_gaps) <= MAXIMUM_RECALL_GAP,
    }


def _latest_successful_rows(run_dir: Path) -> list[dict[str, Any]]:
    """Each job's latest successful row: a resumed run appends, never rewrites."""
    path = run_dir / "predictions.jsonl"
    if not path.is_file():
        raise ValueError(f"no predictions in the baseline run: {path}")
    latest: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        if row["status"] == "success":
            latest[row["job_id"]] = row
    if not latest:
        raise ValueError(f"no successful rows in {path}")
    return list(latest.values())
