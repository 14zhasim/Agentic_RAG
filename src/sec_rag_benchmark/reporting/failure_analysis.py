"""Derive auditable failure diagnoses from saved benchmark results."""

from __future__ import annotations

from typing import Any

RETRIEVAL_CONDITIONS = {"single_store", "shared_store"}

# This legend is written to summary.json and the workbook. Keeping it beside
# the classifier prevents the report from describing different rules than the
# code actually applies.
METHODOLOGY = {
    "context_limit": "The complete prompt did not fit the model context window.",
    "success": "The retrieval-condition answer was judged correct.",
    "oracle_baseline_failed": (
        "The retrieval answer and oracle answer were both incorrect, so "
        "retrieval cannot be isolated as the cause."
    ),
    "retrieval_context_failure": (
        "The oracle answer was correct but the retrieval-condition answer was "
        "incorrect; the subtype records what the saved retrieval evidence supports."
    ),
    "no_chunks_retrieved": "The retriever returned no chunks.",
    "wrong_document": ("Chunks were returned, but none came from a target filing."),
    "wrong_document_invalid_reply": (
        "No chunk came from a target filing; query enhancement's reply was "
        "unusable, so all filings were searched."
    ),
    "wrong_document_declined": (
        "No chunk came from a target filing; query enhancement chose no listed "
        "filing, so all filings were searched."
    ),
    "wrong_document_wrong_filing": (
        "No chunk came from a target filing; query enhancement chose a different "
        "filing."
    ),
    "wrong_document_filter_bug": (
        "The gold filing was chosen as the filter, yet no chunk came from it."
    ),
    "wrong_section_or_chunk": (
        "A target filing was retrieved, but no retrieved page was a gold page."
    ),
    "partial_gold_page_recall": (
        "Some but not all annotated gold pages were retrieved; this does not "
        "prove that the missing pages caused the incorrect answer."
    ),
    "retrieved_gold_pages_but_answer_failed": (
        "All gold pages were represented, but the answer was incorrect; "
        "manual review is required."
    ),
    "unclassified": (
        "Missing or disputed evidence prevents an automatic causal diagnosis."
    ),
    "unjudged_condition": "The retrieval-condition answer has no judgment.",
    "condition_judge_disagreement": (
        "The two judge passes disagreed on the retrieval-condition answer."
    ),
    "missing_oracle": "No oracle result exists for the question.",
    "oracle_did_not_fit": "The oracle prompt did not fit the context window.",
    "unjudged_oracle": "The oracle answer has no judgment.",
    "oracle_judge_disagreement": (
        "The two judge passes disagreed on the oracle answer."
    ),
    "insufficient_analysis_data": (
        "Gold-page or retrieved-chunk provenance is missing or invalid."
    ),
}


def _judge_reasons(judgment: dict[str, Any] | None) -> list[str]:
    """Return the two saved judge explanations when they are available."""
    if judgment is None:
        return []
    return [
        one_pass["reason"]
        for one_pass in judgment.get("passes", [])
        if one_pass.get("reason")
    ]


def _mark(
    row: dict[str, Any],
    *,
    analysis_status: str,
    category: str | None,
    subtype: str | None,
    rule: str,
    manual_review: bool = False,
) -> dict[str, Any]:
    """Finish one row with the outcome supported by its saved evidence."""
    row.update(
        analysis_status=analysis_status,
        failure_category=category,
        failure_subtype=subtype,
        manual_review=manual_review,
        classification_rule=rule,
    )
    return row


def _wrong_document_cause(
    prediction: dict[str, Any], target_documents: list[str]
) -> tuple[str, str]:
    """Say why no retrieved chunk came from the target filing.

    Build Order 2.5 splits Exp1's wrong-document failures by where the filing
    was lost. The checks run in order, so each failure gets its earliest
    cause: an unusable reply also produces a fallback, and a fallback has no
    chosen filing that could be wrong. Rows written before Exp1 carry no
    `filter_status` and keep the plain subtype.
    """
    if "filter_status" not in prediction:
        return "wrong_document", "No retrieved chunk came from a target filing."
    if prediction.get("enhancement_status") == "invalid_reply":
        return (
            "wrong_document_invalid_reply",
            "Query enhancement's reply was unusable; all filings were searched.",
        )
    if prediction["filter_status"] == "fallback":
        return (
            "wrong_document_declined",
            "Query enhancement chose no listed filing; all filings were searched.",
        )
    if prediction.get("filter_doc_name") not in target_documents:
        return (
            "wrong_document_wrong_filing",
            "Query enhancement chose a different filing.",
        )
    return (
        "wrong_document_filter_bug",
        "The gold filing was chosen, yet no chunk came from it.",
    )


def build_failure_analysis(
    predictions: list[dict[str, Any]],
    judgments: dict[str, dict[str, Any]],
    *,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Classify each saved retrieval-condition result against its oracle."""
    predictions_by_question_and_condition = {
        (row["financebench_id"], row["eval_mode"]): row for row in predictions
    }
    analysis_rows: list[dict[str, Any]] = []

    retrieval_predictions = [
        row for row in predictions if row.get("eval_mode") in RETRIEVAL_CONDITIONS
    ]
    for prediction in retrieval_predictions:
        question_id = prediction["financebench_id"]
        oracle = predictions_by_question_and_condition.get((question_id, "oracle"))
        condition_judgment = judgments.get(prediction["job_id"])
        oracle_judgment = judgments.get(oracle["job_id"]) if oracle else None

        gold_page_pairs = {
            (doc_name, page_index)
            for doc_name, page_index in prediction.get("gold_pages", [])
        }
        target_documents = sorted({doc_name for doc_name, _ in gold_page_pairs})
        chunks = prediction.get("retrieved_chunks")

        ranked_chunks: list[dict[str, Any]] = []
        chunk_provenance_is_valid = isinstance(chunks, list)
        if isinstance(chunks, list):
            if any(type(chunk.get("rank")) is not int for chunk in chunks):
                chunk_provenance_is_valid = False
            else:
                ranked_chunks = sorted(chunks, key=lambda chunk: chunk["rank"])[:top_k]

        retrieved_page_pairs: set[tuple[str, int]] = set()
        retrieved_documents: set[str] = set()
        if chunk_provenance_is_valid:
            for chunk in ranked_chunks:
                doc_name = chunk.get("doc_name")
                pages = chunk.get("pages")
                if not isinstance(doc_name, str) or not isinstance(pages, list):
                    chunk_provenance_is_valid = False
                    break
                retrieved_documents.add(doc_name)
                for page_index in pages:
                    if type(page_index) is not int:
                        chunk_provenance_is_valid = False
                        break
                    retrieved_page_pairs.add((doc_name, page_index))
                if not chunk_provenance_is_valid:
                    break

        retrieved_gold_pages = gold_page_pairs & retrieved_page_pairs
        calculated_page_recall = (
            len(retrieved_gold_pages) / len(gold_page_pairs)
            if gold_page_pairs
            else None
        )
        retrieved_target_document = bool(set(target_documents) & retrieved_documents)

        row = {
            "job_id": prediction["job_id"],
            "oracle_job_id": oracle.get("job_id") if oracle else None,
            "financebench_id": question_id,
            "eval_mode": prediction["eval_mode"],
            "question_type": prediction.get("question_type"),
            "cognitive_skills": prediction.get("cognitive_skills", []),
            "oracle_accuracy": (
                oracle_judgment.get("accuracy") if oracle_judgment else None
            ),
            "condition_accuracy": (
                condition_judgment.get("accuracy") if condition_judgment else None
            ),
            "target_documents": target_documents,
            "retrieved_documents": sorted(retrieved_documents),
            "retrieved_target_document": retrieved_target_document,
            "page_recall": calculated_page_recall,
            "page_precision": prediction.get("page_precision"),
            "page_mrr": prediction.get("page_mrr"),
            # Exp1's filter choice, shown beside the diagnosis so a
            # wrong-document row can be checked by eye; None for older rows.
            "enhancement_status": prediction.get("enhancement_status"),
            "filter_status": prediction.get("filter_status"),
            "filter_doc_name": prediction.get("filter_doc_name"),
            "condition_judge_reasons": _judge_reasons(condition_judgment),
            "oracle_judge_reasons": _judge_reasons(oracle_judgment),
        }

        if prediction.get("status") == "did_not_fit":
            analysis_rows.append(
                _mark(
                    row,
                    analysis_status="classified",
                    category="context_limit",
                    subtype=None,
                    rule="The retrieval-condition prompt did not fit.",
                )
            )
            continue

        if condition_judgment is None:
            analysis_rows.append(
                _mark(
                    row,
                    analysis_status="unclassified",
                    category=None,
                    subtype="unjudged_condition",
                    rule="The retrieval-condition answer has no completed judgment.",
                    manual_review=True,
                )
            )
            continue
        if condition_judgment.get("accuracy") is None:
            analysis_rows.append(
                _mark(
                    row,
                    analysis_status="unclassified",
                    category=None,
                    subtype="condition_judge_disagreement",
                    rule="The two judge passes disagreed on the retrieval answer.",
                    manual_review=True,
                )
            )
            continue
        if condition_judgment["accuracy"] == 1:
            analysis_rows.append(
                _mark(
                    row,
                    analysis_status="classified",
                    category="success",
                    subtype=None,
                    rule="The retrieval-condition answer was judged correct.",
                )
            )
            continue

        if oracle is None:
            unavailable_subtype = "missing_oracle"
            unavailable_rule = "No oracle result exists for this question."
        elif oracle.get("status") == "did_not_fit":
            unavailable_subtype = "oracle_did_not_fit"
            unavailable_rule = "The oracle prompt did not fit."
        elif oracle_judgment is None:
            unavailable_subtype = "unjudged_oracle"
            unavailable_rule = "The oracle answer has no completed judgment."
        elif oracle_judgment.get("accuracy") is None:
            unavailable_subtype = "oracle_judge_disagreement"
            unavailable_rule = "The two judge passes disagreed on the oracle answer."
        else:
            unavailable_subtype = None
            unavailable_rule = None

        if unavailable_subtype is not None and unavailable_rule is not None:
            analysis_rows.append(
                _mark(
                    row,
                    analysis_status="unclassified",
                    category=None,
                    subtype=unavailable_subtype,
                    rule=unavailable_rule,
                    manual_review=True,
                )
            )
            continue

        # Reaching this point proves that both the oracle row and its agreed
        # judgment exist; the earlier branches handle every missing case.
        assert oracle_judgment is not None
        if oracle_judgment["accuracy"] == 0:
            analysis_rows.append(
                _mark(
                    row,
                    analysis_status="classified",
                    category="oracle_baseline_failed",
                    subtype=None,
                    rule=(
                        "Both the oracle and retrieval-condition answers were "
                        "incorrect, so retrieval cannot be isolated as the cause."
                    ),
                )
            )
            continue

        # Document identity is checked before page recall. A numerically equal
        # page in another filing is not relevant to the FinanceBench question.
        if not gold_page_pairs or not chunk_provenance_is_valid:
            analysis_rows.append(
                _mark(
                    row,
                    analysis_status="unclassified",
                    category=None,
                    subtype="insufficient_analysis_data",
                    rule="Gold pages or retrieved chunk provenance are missing.",
                    manual_review=True,
                )
            )
        elif not ranked_chunks:
            analysis_rows.append(
                _mark(
                    row,
                    analysis_status="classified",
                    category="retrieval_context_failure",
                    subtype="no_chunks_retrieved",
                    rule="The retriever returned no chunks.",
                )
            )
        elif not retrieved_target_document:
            subtype, rule = _wrong_document_cause(prediction, target_documents)
            analysis_rows.append(
                _mark(
                    row,
                    analysis_status="classified",
                    category="retrieval_context_failure",
                    subtype=subtype,
                    rule=rule,
                )
            )
        elif calculated_page_recall == 0:
            analysis_rows.append(
                _mark(
                    row,
                    analysis_status="classified",
                    category="retrieval_context_failure",
                    subtype="wrong_section_or_chunk",
                    rule="A target filing was retrieved, but no gold page was.",
                )
            )
        elif calculated_page_recall is not None and calculated_page_recall < 1:
            analysis_rows.append(
                _mark(
                    row,
                    analysis_status="classified",
                    category="retrieval_context_failure",
                    subtype="partial_gold_page_recall",
                    rule=(
                        "Some but not all annotated gold pages were retrieved; "
                        "causation is not proven."
                    ),
                )
            )
        elif calculated_page_recall == 1:
            # A page-level match does not prove that chunk boundaries retained
            # the answer passage, so this case cannot be diagnosed more finely.
            analysis_rows.append(
                _mark(
                    row,
                    analysis_status="classified",
                    category="retrieval_context_failure",
                    subtype="retrieved_gold_pages_but_answer_failed",
                    rule="All gold pages were retrieved, but the answer was incorrect.",
                    manual_review=True,
                )
            )
        else:
            analysis_rows.append(
                _mark(
                    row,
                    analysis_status="unclassified",
                    category=None,
                    subtype="insufficient_analysis_data",
                    rule="The saved page information cannot support a diagnosis.",
                    manual_review=True,
                )
            )

    return sorted(
        analysis_rows,
        key=lambda row: (row["financebench_id"], row["eval_mode"]),
    )


def summarize_failure_analysis(
    analysis_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Count diagnoses by condition while preserving the shared legend."""
    grouped_counts: dict[tuple[Any, ...], int] = {}
    for row in analysis_rows:
        identity = (
            row["eval_mode"],
            row["analysis_status"],
            row.get("failure_category"),
            row.get("failure_subtype"),
        )
        grouped_counts[identity] = grouped_counts.get(identity, 0) + 1

    count_rows = []
    for identity, count in sorted(
        grouped_counts.items(),
        key=lambda item: tuple(
            "" if value is None else str(value) for value in item[0]
        ),
    ):
        condition, status, category, subtype = identity
        count_rows.append(
            {
                "eval_mode": condition,
                "analysis_status": status,
                "failure_category": category,
                "failure_subtype": subtype,
                "count": count,
            }
        )

    return {
        "methodology": METHODOLOGY,
        "counts_by_condition": count_rows,
        "manual_review_count": sum(
            1 for row in analysis_rows if row.get("manual_review")
        ),
        "unclassified_count": sum(
            1 for row in analysis_rows if row["analysis_status"] == "unclassified"
        ),
    }
