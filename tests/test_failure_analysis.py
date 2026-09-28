"""Focused tests for the derived FinanceBench failure diagnoses."""

from __future__ import annotations

import pytest

from sec_rag_benchmark.reporting.failure_analysis import (
    METHODOLOGY,
    build_failure_analysis,
    summarize_failure_analysis,
)
from sec_rag_benchmark.reporting.report import _resolve_judgments


def _prediction(
    condition: str,
    *,
    status: str = "success",
    chunks: list[dict] | None = None,
    page_recall: float | None = 0.0,
) -> dict:
    """Create one small saved prediction with the real reporting fields."""
    return {
        "job_id": f"run:q1:{condition}",
        "financebench_id": "q1",
        "status": status,
        "eval_mode": condition,
        "question_type": "metrics-generated",
        "cognitive_skills": ["numerical_reasoning"],
        "gold_pages": [["target.pdf", 2], ["target.pdf", 3]],
        "retrieved_chunks": chunks or [],
        "page_recall": page_recall,
        "page_precision": 0.5 if page_recall else 0.0,
        "page_mrr": 1.0 if page_recall else 0.0,
    }


def _judgment(job_id: str, accuracy: int | None) -> dict:
    """Create the combined two-pass judgment consumed by reporting."""
    return {
        "job_id": job_id,
        "accuracy": accuracy,
        "passes": [
            {"reason": "reference-first reason"},
            {"reason": "candidate-first reason"},
        ],
    }


@pytest.mark.parametrize(
    ("chunks", "page_recall", "expected_subtype", "manual_review"),
    [
        ([], 0.0, "no_chunks_retrieved", False),
        (
            [{"doc_name": "other.pdf", "pages": [2], "rank": 1}],
            0.0,
            "wrong_document",
            False,
        ),
        (
            [{"doc_name": "target.pdf", "pages": [9], "rank": 1}],
            0.0,
            "wrong_section_or_chunk",
            False,
        ),
        (
            [{"doc_name": "target.pdf", "pages": [2], "rank": 1}],
            0.5,
            "partial_gold_page_recall",
            False,
        ),
        (
            [{"doc_name": "target.pdf", "pages": [2, 3], "rank": 1}],
            1.0,
            "retrieved_gold_pages_but_answer_failed",
            True,
        ),
    ],
)
def test_classifies_retrieval_failures_from_documents_and_page_recall(
    chunks, page_recall, expected_subtype, manual_review
):
    oracle = _prediction("oracle", page_recall=None)
    retrieval = _prediction("shared_store", chunks=chunks, page_recall=page_recall)
    judgments = {
        oracle["job_id"]: _judgment(oracle["job_id"], 1),
        retrieval["job_id"]: _judgment(retrieval["job_id"], 0),
    }

    [result] = build_failure_analysis([oracle, retrieval], judgments)

    assert result["analysis_status"] == "classified"
    assert result["failure_category"] == "retrieval_context_failure"
    assert result["failure_subtype"] == expected_subtype
    assert result["manual_review"] is manual_review
    assert result["target_documents"] == ["target.pdf"]
    assert result["retrieved_target_document"] is (
        expected_subtype not in {"no_chunks_retrieved", "wrong_document"}
    )


def test_document_name_is_part_of_page_identity():
    oracle = _prediction("oracle", page_recall=None)
    retrieval = _prediction(
        "shared_store",
        chunks=[{"doc_name": "other.pdf", "pages": [2, 3], "rank": 1}],
        page_recall=0.0,
    )
    judgments = {
        oracle["job_id"]: _judgment(oracle["job_id"], 1),
        retrieval["job_id"]: _judgment(retrieval["job_id"], 0),
    }

    [result] = build_failure_analysis([oracle, retrieval], judgments)

    assert result["failure_subtype"] == "wrong_document"


def test_failure_analysis_ignores_gold_pages_outside_retrieval_depth():
    oracle = _prediction("oracle", page_recall=None)
    retrieval = _prediction(
        "shared_store",
        chunks=[
            {"doc_name": "target.pdf", "pages": [20 + rank], "rank": rank}
            for rank in range(1, 11)
        ]
        + [{"doc_name": "target.pdf", "pages": [2], "rank": 11}],
    )
    judgments = {
        oracle["job_id"]: _judgment(oracle["job_id"], 1),
        retrieval["job_id"]: _judgment(retrieval["job_id"], 0),
    }

    [result] = build_failure_analysis([oracle, retrieval], judgments, top_k=10)

    assert result["page_recall"] == 0.0
    assert result["failure_subtype"] == "wrong_section_or_chunk"


@pytest.mark.parametrize(
    (
        "retrieval_status",
        "retrieval_accuracy",
        "oracle_accuracy",
        "expected_status",
        "expected_category",
    ),
    [
        ("did_not_fit", None, 1, "classified", "context_limit"),
        ("success", None, 1, "unclassified", None),
        ("success", 1, 0, "classified", "success"),
        ("success", 0, 0, "classified", "oracle_baseline_failed"),
    ],
)
def test_preserves_nonclassifiable_and_context_limit_outcomes(
    retrieval_status,
    retrieval_accuracy,
    oracle_accuracy,
    expected_status,
    expected_category,
):
    oracle = _prediction("oracle", page_recall=None)
    retrieval = _prediction("single_store", status=retrieval_status)
    judgments = {oracle["job_id"]: _judgment(oracle["job_id"], oracle_accuracy)}
    if retrieval_status == "success" and retrieval_accuracy is not None:
        judgments[retrieval["job_id"]] = _judgment(
            retrieval["job_id"], retrieval_accuracy
        )

    [result] = build_failure_analysis([oracle, retrieval], judgments)

    assert result["analysis_status"] == expected_status
    assert result["failure_category"] == expected_category


def test_missing_or_disputed_oracle_is_not_given_a_causal_failure_category():
    retrieval = _prediction(
        "shared_store",
        chunks=[{"doc_name": "other.pdf", "pages": [2], "rank": 1}],
    )
    retrieval_judgment = _judgment(retrieval["job_id"], 0)

    [missing_oracle] = build_failure_analysis(
        [retrieval], {retrieval["job_id"]: retrieval_judgment}
    )
    assert missing_oracle["analysis_status"] == "unclassified"
    assert missing_oracle["failure_subtype"] == "missing_oracle"

    oracle = _prediction("oracle", page_recall=None)
    judgments = {
        retrieval["job_id"]: retrieval_judgment,
        oracle["job_id"]: _judgment(oracle["job_id"], None),
    }
    [disputed_oracle] = build_failure_analysis([oracle, retrieval], judgments)
    assert disputed_oracle["analysis_status"] == "unclassified"
    assert disputed_oracle["failure_subtype"] == "oracle_judge_disagreement"


@pytest.mark.parametrize(
    ("oracle_status", "expected_subtype"),
    [
        ("did_not_fit", "oracle_did_not_fit"),
        ("success", "unjudged_oracle"),
    ],
)
def test_unavailable_oracle_states_remain_explicit(oracle_status, expected_subtype):
    oracle = _prediction("oracle", status=oracle_status, page_recall=None)
    retrieval = _prediction("shared_store")
    judgments = {
        retrieval["job_id"]: _judgment(retrieval["job_id"], 0),
    }
    [result] = build_failure_analysis([oracle, retrieval], judgments)

    assert result["analysis_status"] == "unclassified"
    assert result["failure_subtype"] == expected_subtype


def test_disputed_condition_and_missing_provenance_remain_unclassified():
    oracle = _prediction("oracle", page_recall=None)
    retrieval = _prediction("shared_store")
    disputed_judgments = {
        oracle["job_id"]: _judgment(oracle["job_id"], 1),
        retrieval["job_id"]: _judgment(retrieval["job_id"], None),
    }

    [disputed] = build_failure_analysis([oracle, retrieval], disputed_judgments)
    assert disputed["failure_subtype"] == "condition_judge_disagreement"

    retrieval["retrieved_chunks"] = [{"doc_name": "target.pdf"}]
    incorrect_judgments = {
        oracle["job_id"]: _judgment(oracle["job_id"], 1),
        retrieval["job_id"]: _judgment(retrieval["job_id"], 0),
    }
    [missing_provenance] = build_failure_analysis(
        [oracle, retrieval], incorrect_judgments
    )
    assert missing_provenance["analysis_status"] == "unclassified"
    assert missing_provenance["failure_subtype"] == "insufficient_analysis_data"


def test_manual_verdict_allows_disputed_answer_to_be_classified():
    oracle = _prediction("oracle", page_recall=None)
    retrieval = _prediction(
        "shared_store",
        chunks=[{"doc_name": "other.pdf", "pages": [2], "rank": 1}],
    )
    disputed_judgments = {
        oracle["job_id"]: _judgment(oracle["job_id"], None),
        retrieval["job_id"]: _judgment(retrieval["job_id"], None),
    }
    manual_reviews = {
        oracle["job_id"]: {"human_accuracy": 1},
        retrieval["job_id"]: {"human_accuracy": 0},
    }

    resolved = _resolve_judgments(disputed_judgments, manual_reviews)
    [result] = build_failure_analysis([oracle, retrieval], resolved)

    assert result["analysis_status"] == "classified"
    assert result["failure_category"] == "retrieval_context_failure"
    assert result["failure_subtype"] == "wrong_document"


def test_summarizes_failure_rows_by_condition_and_outcome():
    rows = [
        {
            "eval_mode": "shared_store",
            "analysis_status": "classified",
            "failure_category": "retrieval_context_failure",
            "failure_subtype": "wrong_document",
            "manual_review": False,
        },
        {
            "eval_mode": "shared_store",
            "analysis_status": "classified",
            "failure_category": "retrieval_context_failure",
            "failure_subtype": "wrong_document",
            "manual_review": False,
        },
        {
            "eval_mode": "single_store",
            "analysis_status": "unclassified",
            "failure_category": None,
            "failure_subtype": "unjudged_condition",
            "manual_review": True,
        },
    ]

    summary = summarize_failure_analysis(rows)

    assert summary["manual_review_count"] == 1
    assert summary["unclassified_count"] == 1
    assert {
        (row["eval_mode"], row["failure_subtype"], row["count"])
        for row in summary["counts_by_condition"]
    } == {
        ("shared_store", "wrong_document", 2),
        ("single_store", "unjudged_condition", 1),
    }


@pytest.mark.parametrize(
    ("exp1_fields", "expected_subtype"),
    [
        # Rule order: a broken reply also falls back, so it wins; a fallback
        # has no chosen filing to be wrong, so it beats wrong filing.
        (
            {
                "enhancement_status": "invalid_reply",
                "filter_status": "fallback",
                "filter_doc_name": None,
            },
            "wrong_document_invalid_reply",
        ),
        (
            {
                "enhancement_status": "ok",
                "filter_status": "fallback",
                "filter_doc_name": None,
            },
            "wrong_document_declined",
        ),
        (
            {
                "enhancement_status": "ok",
                "filter_status": "chosen",
                "filter_doc_name": "other.pdf",
            },
            "wrong_document_wrong_filing",
        ),
        (
            {
                "enhancement_status": "ok",
                "filter_status": "chosen",
                "filter_doc_name": "target.pdf",
            },
            "wrong_document_filter_bug",
        ),
        ({}, "wrong_document"),
    ],
)
def test_wrong_document_failure_is_split_by_cause(exp1_fields, expected_subtype):
    """Build Order 2.5: an Exp1 wrong-document failure records its cause."""
    oracle = _prediction("oracle", page_recall=None)
    retrieval = {
        **_prediction(
            "shared_store",
            chunks=[{"doc_name": "other.pdf", "pages": [2], "rank": 1}],
        ),
        **exp1_fields,
    }
    judgments = {
        oracle["job_id"]: _judgment(oracle["job_id"], 1),
        retrieval["job_id"]: _judgment(retrieval["job_id"], 0),
    }

    [result] = build_failure_analysis([oracle, retrieval], judgments)

    assert result["failure_category"] == "retrieval_context_failure"
    assert result["failure_subtype"] == expected_subtype
    assert expected_subtype in METHODOLOGY
    assert result["filter_status"] == exp1_fields.get("filter_status")
    assert result["filter_doc_name"] == exp1_fields.get("filter_doc_name")


def test_methodology_explains_every_emitted_failure_value():
    expected_values = {
        "retrieval_context_failure",
        "unjudged_condition",
        "condition_judge_disagreement",
        "missing_oracle",
        "oracle_did_not_fit",
        "unjudged_oracle",
        "oracle_judge_disagreement",
        "insufficient_analysis_data",
    }

    assert expected_values <= METHODOLOGY.keys()
