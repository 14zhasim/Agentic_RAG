"""Build the context supplied by each FinanceBench test condition."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pymupdf

CONDITIONS = {"closed_book", "oracle", "long_context", "single_store", "shared_store"}
Retriever = Callable[[str, tuple[str, ...], int], list[dict[str, Any]]]


class RetrieverUnavailable(RuntimeError):
    pass


def gold_pages(question: dict[str, Any]) -> list[tuple[str, int]]:
    """Return unique (document, zero-indexed page) evidence references."""
    pages: list[tuple[str, int]] = []
    for evidence in question["evidence"]:
        doc = evidence.get("doc_name") or evidence.get("evidence_doc_name")
        page = (doc, evidence["evidence_page_num"])
        if page not in pages:
            pages.append(page)
    return pages


def _pdf_pages(pdf_dir: Path, doc_name: str) -> list[str]:
    path = pdf_dir / doc_name
    if not path.is_file():
        path = pdf_dir / f"{doc_name}.pdf"
    with pymupdf.open(path) as pdf:
        return [page.get_text() for page in pdf]


def _page_block(doc: str, page: int, text: str) -> str:
    return f"[Document: {doc} | Page index: {page}]\n{text}"


def _build_closed_book() -> dict[str, Any]:
    """Supply no filing context."""
    return {"context": "", "context_pages": [], "retrieved_chunks": []}


def _build_oracle(question: dict[str, Any]) -> dict[str, Any]:
    """Supply each unique gold evidence page in dataset order."""
    context_pages: list[tuple[str, int]] = []
    page_blocks: list[str] = []
    for evidence in question["evidence"]:
        document_name = evidence.get("doc_name") or evidence.get("evidence_doc_name")
        page = (document_name, evidence["evidence_page_num"])
        if page not in context_pages:
            context_pages.append(page)
            page_blocks.append(
                _page_block(document_name, page[1], evidence["evidence_text_full_page"])
            )
    return {
        "context": "\n\n".join(page_blocks),
        "context_pages": context_pages,
        "retrieved_chunks": [],
    }


def _build_long_context(question: dict[str, Any], pdf_dir: Path) -> dict[str, Any]:
    """Supply the complete question filing in physical page order."""
    pages = _pdf_pages(pdf_dir, question["doc_name"])
    context_pages = [(question["doc_name"], index) for index in range(len(pages))]
    page_blocks = [
        _page_block(question["doc_name"], index, text)
        for index, text in enumerate(pages)
    ]
    return {
        "context": "\n\n".join(page_blocks),
        "context_pages": context_pages,
        "retrieved_chunks": [],
    }


def _build_retrieval_context(
    question: dict[str, Any],
    scope: tuple[str, ...],
    retriever: Retriever | None,
    top_k: int,
) -> dict[str, Any]:
    """Retrieve ranked chunks and convert them into context plus provenance."""
    if retriever is None:
        raise RetrieverUnavailable("Retrieval condition needs a retriever")

    retrieved_chunks = sorted(
        retriever(question["question"], scope, top_k),
        key=lambda item: item["rank"],
    )[:top_k]
    context_blocks: list[str] = []
    context_pages: list[tuple[str, int]] = []
    for chunk in retrieved_chunks:
        context_blocks.append(
            f"[Chunk: {chunk['chunk_id']} | Document: {chunk['doc_name']} | "
            f"Pages: {','.join(map(str, chunk['pages']))} | Rank: {chunk['rank']}]\n"
            f"{chunk['text']}"
        )
        for page_index in chunk["pages"]:
            page = (chunk["doc_name"], page_index)
            if page not in context_pages:
                context_pages.append(page)
    return {
        "context": "\n\n".join(context_blocks),
        "context_pages": context_pages,
        "retrieved_chunks": retrieved_chunks,
    }


def _build_single_store(
    question: dict[str, Any], retriever: Retriever | None, top_k: int
) -> dict[str, Any]:
    """Retrieve only from the filing associated with the question."""
    return _build_retrieval_context(question, (question["doc_name"],), retriever, top_k)


def _build_shared_store(
    question: dict[str, Any],
    all_doc_names: tuple[str, ...],
    retriever: Retriever | None,
    top_k: int,
) -> dict[str, Any]:
    """Retrieve from the complete prepared FinanceBench document set."""
    return _build_retrieval_context(question, all_doc_names, retriever, top_k)


def build_condition(
    question: dict[str, Any],
    condition: str,
    pdf_dir: str | Path,
    all_doc_names: tuple[str, ...],
    *,
    retriever: Retriever | None = None,
    top_k: int = 10,
) -> dict[str, Any]:
    """Build context/provenance without calling the generation model.

    The five branches vary only what information accompanies the question. The
    runner later sends every returned context through the same shared prompt.
    """
    match condition:
        case "closed_book":
            condition_data = _build_closed_book()
        case "oracle":
            condition_data = _build_oracle(question)
        case "long_context":
            condition_data = _build_long_context(question, Path(pdf_dir))
        case "single_store":
            condition_data = _build_single_store(question, retriever, top_k)
        case "shared_store":
            condition_data = _build_shared_store(
                question, all_doc_names, retriever, top_k
            )
        case _:
            raise ValueError(f"Unknown condition: {condition}")

    return {
        "financebench_id": question["financebench_id"],
        "condition": condition,
        "question": question["question"],
        **condition_data,
    }
