"""Turn one FinanceBench question-condition pair into one prediction row."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..evaluation.retrieval_metrics import cognitive_skills, page_metrics
from ..pipeline.conditions import Retriever, build_condition, gold_pages
from ..pipeline.generation import build_messages, generate

Generator = Callable[[list[dict[str, str]], dict[str, Any]], dict[str, Any]]
RETRIEVAL_CONDITIONS = {"single_store", "shared_store"}


def execute_job(
    question: dict[str, Any],
    condition_name: str,
    job_id: str,
    *,
    pdf_dir: str | Path,
    all_doc_names: tuple[str, ...],
    generation_config: dict[str, Any],
    retrieval_depth: int,
    retriever: Retriever | None = None,
    generator: Generator = generate,
) -> dict[str, Any]:
    """Build context, generate an answer, and calculate one job's metrics.

    Retrieval conditions get page metrics twice (Build Order 2.5): once on
    the reranked chunks the model read (page_*), once on the fused chunks
    before reranking (pre_rerank_page_*), plus the filter and cost fields.
    """
    condition = build_condition(
        question,
        condition_name,
        pdf_dir,
        all_doc_names,
        retriever=retriever,
        top_k=retrieval_depth,
    )
    generated = generator(
        build_messages(question["question"], condition["context"]),
        generation_config,
    )

    retrieval_metrics: dict[str, Any] = {}
    if condition_name in RETRIEVAL_CONDITIONS:
        # 1st page_metrics() call: the reranked top 10 the model read
        # (retrieved_chunks is bundle["chunks"]), i.e. the post-rerank metrics.
        retrieval_metrics.update(
            page_metrics(
                gold_pages(question), condition["retrieved_chunks"], retrieval_depth
            )
        )
        # 2nd call, inside _retrieval_fields: the fused top 10 before reranking.
        retrieval_metrics.update(
            _retrieval_fields(
                question, condition_name, condition["retrieval"], retrieval_depth
            )
        )
    else:
        retrieval_metrics.update(
            page_recall=None,
            page_precision=None,
            page_mrr=None,
            pre_rerank_page_recall=None,
            pre_rerank_page_precision=None,
            pre_rerank_page_mrr=None,
        )

    # The saved row contains everything needed for reporting and the future
    # judge, which can therefore run without regenerating this answer.
    return {
        "job_id": job_id,
        "status": "success",
        "financebench_id": question["financebench_id"],
        "question": question["question"],
        "gold_answer": question["answer"],
        "gold_evidence": question["evidence"],
        "human_justification": question["justification"],
        "model_answer": generated["answer"],
        "eval_mode": condition_name,
        "question_type": question["question_type"],
        "cognitive_skills": cognitive_skills(question.get("question_reasoning")),
        "gold_pages": gold_pages(question),
        "retrieved_chunks": condition["retrieved_chunks"],
        **retrieval_metrics,
        **{key: value for key, value in generated.items() if key != "answer"},
        "completed_at": datetime.now(UTC).isoformat(),
    }


def _retrieval_fields(
    question: dict[str, Any],
    condition_name: str,
    retrieval: dict[str, Any],
    retrieval_depth: int,
) -> dict[str, Any]:
    """The Build Order 2.5 row fields beyond the post-rerank page metrics.

    `retrieval` is the bundle minus its final chunks (conditions.py).

    - pre-rerank metrics score the fused top 10 at the same depth as the
      post-rerank ones, so the pair shows the reranker's own effect and
      doubles as the no-reranker ablation's retrieval numbers (Draft ->
      Retrieve, DECIDED 27 Sep).
    - filter_correct is scored only in shared_store. Single-store's scope is
      already the gold filing, so its "choice" is not a real one. A fallback
      (no filter, whole corpus searched) counts as wrong: Draft -> Retrieve,
      "null -> recorded as a filter miss".
    - retrieval_cost_usd adds query enhancement's OpenRouter cost to Voyage's
      list-price embedding and rerank costs; if OpenRouter returned no cost,
      the total is unknown (None) rather than understated.
    """
    pre_rerank = page_metrics(
        gold_pages(question), retrieval["pre_rerank_chunks"], retrieval_depth
    )
    search_plan = retrieval["search_plan"]
    usage = retrieval["usage"]

    if condition_name == "shared_store":
        filter_correct: bool | None = (
            retrieval["filter_doc_name"] == question["doc_name"]
        )
    else:
        filter_correct = None

    query_enhancement_cost = search_plan["call"].get("cost")
    cost_parts = [
        query_enhancement_cost,
        usage["embedding_cost_usd"],
        usage["rerank_cost_usd"],
    ]
    if any(part is None for part in cost_parts):
        retrieval_cost_usd = None
    else:
        retrieval_cost_usd = sum(cost_parts)

    return {
        "pre_rerank_chunks": retrieval["pre_rerank_chunks"],
        "pre_rerank_page_recall": pre_rerank["page_recall"],
        "pre_rerank_page_precision": pre_rerank["page_precision"],
        "pre_rerank_page_mrr": pre_rerank["page_mrr"],
        "search_plan": search_plan,
        # Copied up from search_plan so reporting and failure diagnosis read
        # one flat field (Benchmark.md -> failure diagnosis).
        "enhancement_status": search_plan["enhancement_status"],
        "filter_doc_name": retrieval["filter_doc_name"],
        "filter_status": retrieval["filter_status"],
        "filter_correct": filter_correct,
        "retrieval_usage": usage,
        "query_enhancement_cost": query_enhancement_cost,
        "retrieval_cost_usd": retrieval_cost_usd,
        "retrieval_latency_seconds": retrieval["latency_seconds"],
    }
