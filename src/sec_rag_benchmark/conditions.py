"""Build the context supplied by each FinanceBench test condition."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

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


def build_condition(
    question: dict[str, Any],
    condition: str,
    pdf_dir: str | Path,
    all_doc_names: tuple[str, ...],
    *,
    retriever: Retriever | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Build context/provenance without calling the generation model.

    The five branches vary only what information accompanies the question. The
    runner later sends every returned context through the same shared prompt.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    context = ""
    context_pages: list[tuple[str, int]] = []
    chunks: list[dict[str, Any]] = []

    # closed_book deliberately keeps these defaults: the model sees the
    # question and shared instructions, but no filing evidence.
    if condition == "oracle":
        # Supply every unique gold evidence page in dataset order. This is the
        # reasoning ceiling when retrieval is assumed perfect.
        for evidence in question["evidence"]:
            doc = evidence.get("doc_name") or evidence.get("evidence_doc_name")
            page = (doc, evidence["evidence_page_num"])
            if page not in context_pages:
                context_pages.append(page)
                context += ("\n\n" if context else "") + _page_block(
                    doc, page[1], evidence["evidence_text_full_page"]
                )
    elif condition == "long_context":
        # Preserve the complete filing in physical page order. Context fitting
        # is checked later against the complete assembled prompt.
        pages = _pdf_pages(Path(pdf_dir), question["doc_name"])
        context_pages = [(question["doc_name"], index) for index in range(len(pages))]
        context = "\n\n".join(
            _page_block(question["doc_name"], index, text)
            for index, text in enumerate(pages)
        )
    elif condition in {"single_store", "shared_store"}:
        if retriever is None:
            raise RetrieverUnavailable(f"{condition} needs a retriever")
        # single_store searches only the answer filing; shared_store searches
        # the full prepared 64-document collection through the same contract.
        scope = (question["doc_name"],) if condition == "single_store" else all_doc_names
        chunks = retriever(question["question"], scope, top_k)
        blocks = []
        #retrieves chunks + combine with metadata; turns them into one text context for LLM
        #separately records unique (doc, page) pairs represented by those chunks
        for chunk in sorted(chunks, key=lambda item: item["rank"]):
            blocks.append(
                f"[Chunk: {chunk['chunk_id']} | Document: {chunk['doc_name']} | "
                f"Pages: {','.join(map(str, chunk['pages']))} | Rank: {chunk['rank']}]\n{chunk['text']}"
            )
            for page_index in chunk["pages"]:
                page = (chunk["doc_name"], page_index)
                if page not in context_pages:
                    context_pages.append(page)
        context = "\n\n".join(blocks)

    return {
        "financebench_id": question["financebench_id"],
        "condition": condition,
        "question": question["question"],
        "context": context,
        "context_pages": context_pages, #has unique pages retrieved, for metrics
        "retrieved_chunks": chunks, #context has combined text LLM will need
    }
