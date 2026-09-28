"""Exp1's fixed retrieval path for one question (Build Order 2.1-2.3).

`retrieve_exp1` is the orchestrator: query enhancement (Slice 2) chooses a
filing and writes two queries, `search` (Slice 1) runs BM25 and Chroma
inside that filing and fuses them with RRF, and the reranker (Slice 2)
keeps the best 10 of the fused 50. It returns one dictionary, the bundle,
holding the final chunks plus everything the benchmark's metrics need.

The benchmark calls it as `retriever(question, scope, top_k)`
(sec_rag_benchmark/pipeline/conditions.py); Guide 2 binds `resources` with
functools.partial, so everything slow is loaded once by `open_exp1`, not
per question (Implementation Guide 2.1-2.3 -> How this plugs into the
benchmark).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import voyageai

from ..indexing.nodes import load_corpus_nodes
from ..ingestion.parse import select_documents
from .query_enhancement import enhance_query, openrouter_client
from .rerank import rerank
from .search import SearchIndexes, open_search_indexes, search


@dataclass
class Exp1Resources:
    """Everything loaded once by open_exp1 and reused for every question."""

    indexes: SearchIndexes  # the two searchable indexes (search.py)
    chunks: dict[str, dict[str, Any]]  # the chunk lookup: chunk ID -> record
    openrouter: Any  # openai.OpenAI client for the query-enhancement call
    voyage: Any  # voyageai.Client for reranking (search embeds with the same one)
    config: dict[str, Any]  # the loaded configs/sec_rag.toml


def retrieve_exp1(
    question: str, scope: tuple[str, ...], top_k: int, resources: Exp1Resources
) -> dict[str, Any]:
    """Plan, filter, hybrid search, rerank; return the bundle for one question.

    `scope` is the filings the benchmark condition allows: one filename in
    single-store, all 64 in shared-store. `top_k` is the benchmark's
    retrieval depth (10). Steps:

    1. `enhance_query` asks GLM for a filing and two queries. It is called
       in both conditions, so single-store also gets the rewritten queries
       (Draft -> Retrieve, "same call in both conditions").
    2. `_choose_filter` turns GLM's choice and the scope into the one filing
       both searches look inside, or none.
    3. `search(method="hybrid")` returns the fused top `rerank_candidates`
       (50) chunk IDs; `_to_chunks` swaps each ID for its original text.
    4. The first `top_k` fused chunks are kept as pre-rerank, so the
       reranker's own effect can be measured (Build Order 2.5).
    5. `rerank` reorders the 50 against the original question and keeps
       `top_k`: these are the chunks the answer model reads.
    """
    started = time.perf_counter()
    config = resources.config

    plan = enhance_query(question, list(scope), config, resources.openrouter)
    filter_doc_name, filter_status = _choose_filter(
        plan["filename"], scope, resources.chunks
    )

    found = search(
        plan["keyword_query"],
        plan["semantic_query"],
        method="hybrid",
        doc_name=filter_doc_name,
        top_k=config["retrieval"]["rerank_candidates"],
        indexes=resources.indexes,
    )
    fused = _to_chunks(found["ranked"], resources.chunks)

    # Voyage sees only texts and answers with positions in that list, so
    # position i in `order` means fused[i].
    order, rerank_tokens = rerank(
        question,
        [chunk["text"] for chunk in fused],
        top_k,
        config,
        resources.voyage,
    )
    reranked = []
    for new_rank, (position, relevance) in enumerate(order, start=1):
        reranked.append({**fused[position], "rank": new_rank, "score": relevance})

    # Build Order 2.5: Voyage cost at list price. Our usage sits inside the
    # free allowance, so this is what the run would cost, not what was billed.
    embedding_cost_usd = (
        found["embedding_tokens"]
        * config["embedding"]["usd_per_million_tokens"]
        / 1_000_000
    )
    rerank_cost_usd = (
        rerank_tokens * config["rerank"]["usd_per_million_tokens"] / 1_000_000
    )

    return {
        "chunks": reranked,
        "pre_rerank_chunks": fused[:top_k],
        "search_plan": plan,
        "filter_doc_name": filter_doc_name,
        "filter_status": filter_status,
        "usage": {
            "embedding_tokens": found["embedding_tokens"],
            "rerank_tokens": rerank_tokens,
            "embedding_cost_usd": embedding_cost_usd,
            "rerank_cost_usd": rerank_cost_usd,
        },
        "latency_seconds": time.perf_counter() - started,
    }


def _choose_filter(
    filename: str | None, scope: tuple[str, ...], chunks: dict[str, dict[str, Any]]
) -> tuple[str | None, str]:
    """Return (the filing to search inside or None, "chosen" or "fallback").

    `_read_plan` has already turned null, unlisted names and broken replies
    into None, so a filename here is always one of the scope's filings.

    - a filename -> search that filing ("chosen", scored for filter accuracy)
    - None, one-filing scope (single-store) -> that filing, since search may
      never go wider than the benchmark allows
    - None, every filing in scope (shared-store) -> no filter: search
      everything rather than nothing (Draft -> Retrieve, fallback rule)
    - any other scope -> ValueError; FinanceBench never produces one, and
      Exp1's filter holds one filing or none
    """
    if filename is not None:
        return filename, "chosen"
    if len(scope) == 1:
        return scope[0], "fallback"
    every_filing = {chunk["doc_name"] for chunk in chunks.values()}
    if set(scope) == every_filing:
        return None, "fallback"
    raise ValueError(
        f"Exp1 filters to one filing or none; got a scope of {len(scope)} of "
        f"{len(every_filing)} filings"
    )


def _to_chunks(
    ranked: list[tuple[str, float]], chunks: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Turn search's (chunk_id, score) pairs into chunks in the benchmark's shape.

    The shape is what conditions.py and page_metrics already read:
    chunk_id, doc_name, pages, rank, score, text. `pages` is a list holding
    the chunk's one page index (0-based, as FinanceBench's evidence pages
    are), because no chunk crosses a page (Build Order 1.3). The text comes
    from the chunk lookup, not from BM25, whose copy has its HTML stripped
    (Stage 1.5 finding 6): the answer model must read tables as tables.
    """
    result = []
    for rank, (chunk_id, score) in enumerate(ranked, start=1):
        record = chunks[chunk_id]
        result.append(
            {
                "chunk_id": chunk_id,
                "doc_name": record["doc_name"],
                "pages": [record["page_index"]],
                "rank": rank,
                "score": score,
                "text": record["text"],
            }
        )
    return result


def preview_bm25(
    question: str, scope: tuple[str, ...], top_k: int, config: dict[str, Any]
) -> list[dict[str, Any]]:
    """The free spot check: BM25 over the raw question, inside the scope.

    No GLM call, no embedding, no rerank, so no key is needed and nothing is
    spent. It shows whether the indexes load and the scope filter holds
    before a paid run. The filter follows `_choose_filter`'s no-choice rule:
    a one-filing scope searches that filing, the whole corpus searches all.
    """
    chunks = load_chunk_lookup(config)
    doc_name, _status = _choose_filter(None, scope, chunks)
    found = search(
        question,
        question,
        method="bm25",
        doc_name=doc_name,
        top_k=top_k,
        indexes=open_search_indexes(config, None),
    )
    return _to_chunks(found["ranked"], chunks)


def all_filings(config: dict[str, Any]) -> tuple[str, ...]:
    """Every prepared filing's name, sorted: the shared-store scope.

    `select_documents(config, None)` lists the PDFs in the prepared folder
    at run time (nothing hardcoded); a PDF's stem is its doc_name. It is
    the same list `load_corpus_nodes` reads chunk files for, so it always
    matches the chunk lookup `_choose_filter` checks against.
    """
    return tuple(pdf_path.stem for pdf_path in select_documents(config, None))


def open_exp1(config: dict[str, Any]) -> Exp1Resources:
    """Load everything retrieve_exp1 needs, once: indexes, chunk lookup, clients.

    This is the paid path: it reads OPENROUTER_API_KEY and VOYAGE_API_KEY
    and raises RuntimeError if either is missing. Nothing is spent until
    retrieve_exp1 is called.
    """
    voyage = voyage_client()
    return Exp1Resources(
        indexes=open_search_indexes(config, voyage),
        chunks=load_chunk_lookup(config),
        openrouter=openrouter_client(config),
        voyage=voyage,
        config=config,
    )


def load_chunk_lookup(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Map every chunk ID to {doc_name, page_index, text} from the chunk files.

    Built from the same nodes both indexes were built from
    (indexing/nodes.py), so every ID search returns is here. `node.text`
    is the chunk's original text, HTML kept.
    """
    lookup = {}
    for node in load_corpus_nodes(config):
        lookup[node.id_] = {
            "doc_name": node.metadata["doc_name"],
            "page_index": node.metadata["page_index"],
            "text": node.text,
        }
    return lookup


def voyage_client() -> Any:
    """Create the Voyage client from VOYAGE_API_KEY, read only when needed.

    max_retries=3 retries rate-limit and server errors with backoff
    (docs/libraries/voyage/rate-limits.md lines 95-140; same setting as
    indexing/embed.py).
    """
    key = os.environ.get("VOYAGE_API_KEY")
    if not key:
        raise RuntimeError("VOYAGE_API_KEY is not set")
    return voyageai.Client(api_key=key, max_retries=3)
