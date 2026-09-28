"""The ONE retrieval function (Build Order 2.2): BM25, semantic, or both fused.

`search` ranks chunks for a keyword query and a semantic query. It never
decides its own arguments: Exp1's `retrieve_exp1` passes fixed ones, the
ablations pass other methods, and Exp3's agent will pass its own choices.

Every ranking here has the same shape: a list of (chunk_id, score) pairs,
best first, higher score = better match. Only chunk IDs leave this module;
turning an ID into its text, filing and page is `exp1`'s job, because BM25's
stored text has its HTML stripped (Stage 1.5 finding 6).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bm25s
import Stemmer
from llama_index.core.vector_stores import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
)
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore

from ..indexing.embed import open_chunk_store

METHODS = ("bm25", "semantic", "hybrid")


@dataclass
class SearchIndexes:
    """The two searchable indexes, loaded once, plus what searching them needs."""

    bm25: bm25s.BM25  # the Stage 1.5 keyword index
    store: ChromaVectorStore  # the Stage 1.6 vector store
    voyage: Any  # voyageai.Client to embed the semantic query; None = BM25 only
    config: dict[str, Any]  # the loaded configs/sec_rag.toml


def search(
    keyword_query: str,
    semantic_query: str,
    *,
    method: str,
    doc_name: str | None,
    top_k: int,
    indexes: SearchIndexes,
    structure_weight: float = 0.0,
) -> dict[str, Any]:
    """Rank chunks for one query pair; return {"ranked": [...], "embedding_tokens": n}.

    `method` picks which searches run: "bm25" returns BM25's list,
    "semantic" returns Chroma's, "hybrid" runs both and fuses them with RRF.
    `doc_name` limits both searches to one filing (None = every chunk).
    `top_k` is how many (chunk_id, score) pairs to return, best first.

    Two queries rather than one, because hybrid sends different text to each
    search (Draft -> Retrieve: keyword query -> BM25, semantic -> Chroma).
    `structure_weight` is Exp2's hook (Build Order 2.2, "Exp2 hook"): at 0
    this is exactly Exp1; Stage 3.3 gives it meaning.
    """
    if structure_weight != 0.0:
        raise NotImplementedError("structure_weight is Exp2 (Build Order 3.3)")
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}: expected one of {METHODS}")

    embedding_tokens = 0
    rankings: list[list[tuple[str, float]]] = []  # one list per search that ran

    # BM25 runs for "bm25" and "hybrid".
    if method in ("bm25", "hybrid"):
        rankings.append(_bm25_ranking(indexes, keyword_query, doc_name))

    # Chroma runs for "semantic" and "hybrid".
    if method in ("semantic", "hybrid"):
        vector_ranking, embedding_tokens = _vector_ranking(
            indexes, semantic_query, doc_name
        )
        rankings.append(vector_ranking)

    # Only "hybrid" has two lists to merge; otherwise return the one list.
    if method == "hybrid":
        id_lists = [[chunk_id for chunk_id, _score in ranking] for ranking in rankings]
        ranked = reciprocal_rank_fusion(
            id_lists, k=indexes.config["retrieval"]["rrf_k"]
        )
    else:
        ranked = rankings[0]
    return {"ranked": ranked[:top_k], "embedding_tokens": embedding_tokens}


def _bm25_ranking(
    indexes: SearchIndexes, query: str, doc_name: str | None
) -> list[tuple[str, float]]:
    """BM25's best chunks for the keyword query, inside the filter.

    Returns [(chunk_id, bm25_score), ...], best first, at most
    candidates_per_search, every score > 0. Sorted because bm25s's retrieve
    defaults to sorted=True (bm25s/__init__.py line 681) and the wrapper
    keeps that order; test_search.py pins it.
    """
    settings = indexes.config["bm25"]
    # Step 1 - build a retriever for THIS filter. Stage 1.5 finding 5: the
    # retriever reads its filter only when created (bm25/base.py lines
    # 128-146), so one is made per search from the already-loaded index
    # (~0.01 s). The pattern and stemmer are the build-time ones, as in
    # load_bm25_index (finding 1).
    retriever = BM25Retriever(
        existing_bm25=indexes.bm25,
        filters=_doc_filter(doc_name),
        similarity_top_k=indexes.config["retrieval"]["candidates_per_search"],
        token_pattern=settings["token_pattern"],
        stemmer=Stemmer.Stemmer(settings["stemmer"]),
    )
    # Step 2 - search: this is where the keyword query is used. retrieve()
    # splits it into words, scores every chunk the filter allows, and returns
    # NodeWithScore objects best first.
    results = retriever.retrieve(query)
    ranking: list[tuple[str, float]] = []
    for result in results:
        # A filtered search still returns candidates_per_search results,
        # padding with score-0 chunks from outside the filter (the filter is
        # a 0/1 weight mask), and an unfiltered one pads with non-matching
        # chunks (finding 3). A score of 0 means no query word matched.
        if result.score is not None and result.score > 0:
            ranking.append((result.node.node_id, result.score))
    return ranking


def _vector_ranking(
    indexes: SearchIndexes, query: str, doc_name: str | None
) -> tuple[list[tuple[str, float]], int]:
    """Chroma's nearest chunks to the semantic query, inside the filter.

    Returns the ranking - [(chunk_id, similarity), ...], best first - and
    the tokens Voyage billed for embedding the query (the question's
    embedding cost, recorded per Build Order 2.5).

    Sorted because Chroma returns nearest first and ChromaVectorStore.query
    keeps that order (chroma/base.py lines 445-480); test_search.py pins it.
    The similarity is exp(-distance), not the cosine (chroma/base.py line
    472) - fine here, since only the order is used (Draft -> Tech stack ->
    Chroma). Exp2 must compute real cosines.
    """
    if indexes.voyage is None:
        raise ValueError("semantic search needs a Voyage client to embed the query")
    settings = indexes.config["embedding"]
    # Step 1 - turn the semantic query into a vector (one paid Voyage call).
    # input_type="query" pairs with the chunks' "document" embeddings
    # (docs/libraries/voyage/embeddings.md lines 60-82); model and length
    # must match what the chunks were stored with.
    embedded = indexes.voyage.embed(
        [query],
        model=settings["model"],
        input_type="query",
        truncation=False,
        output_dimension=settings["output_dimension"],
        output_dtype=settings["output_dtype"],
    )
    # Step 2 - ask Chroma for the nearest chunk vectors, inside the filter.
    # Chroma applies the where-filter before finding neighbours.
    found = indexes.store.query(
        VectorStoreQuery(
            query_embedding=embedded.embeddings[0],
            similarity_top_k=indexes.config["retrieval"]["candidates_per_search"],
            filters=_doc_filter(doc_name),
        )
    )
    ids = found.ids or []
    similarities = found.similarities or []
    ranking = list(zip(ids, similarities, strict=True))
    return ranking, embedded.total_tokens


def _doc_filter(doc_name: str | None) -> MetadataFilters | None:
    """The "only this filing" filter, in the one format both searches accept.

    _doc_filter("3M_2018_10K") keeps chunks whose doc_name metadata equals
    "3M_2018_10K"; _doc_filter(None) is no filter. BM25Retriever turns it
    into a 0/1 mask over every chunk (bm25/base.py lines 128-146);
    ChromaVectorStore translates it to {"doc_name": {"$eq": ...}}
    (chroma/base.py, _to_chroma_filter line 64, "==" -> "$eq" line 47).
    Read: docs/libraries/llamaindex/metadata_filtering.md lines 140-190.
    """
    if doc_name is None:
        return None
    return MetadataFilters(
        filters=[
            MetadataFilter(key="doc_name", operator=FilterOperator.EQ, value=doc_name)
        ]
    )


def reciprocal_rank_fusion(
    rankings: list[list[str]], k: int
) -> list[tuple[str, float]]:
    """Merge ranked ID lists: each chunk scores the sum of 1/(k + rank) over lists.

    Positions only, not scores, because BM25 scores (~14) and similarities
    (~0.7) are on different scales (Draft -> Retrieve -> rrf). Written here,
    not with LlamaIndex's QueryFusionRetriever, which sends one query to
    every retriever (fusion_retriever.py lines 270-271) and fixes k=60 (line
    122). Example with k=60: 1st in one list and 5th in the other scores
    1/61 + 1/65 = 0.0318.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    # Best first; ties broken by chunk ID so the order never depends on
    # which list happened to be read first.
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def open_search_indexes(config: dict[str, Any], voyage: Any) -> SearchIndexes:
    """Load BM25 and open Chroma once, for every search that follows.

    `voyage` is a voyageai.Client, or None for BM25-only use (the free spot
    check), where no key is needed. BM25 is loaded with bm25s directly, as
    load_bm25_index does, because each search builds its own filtered
    BM25Retriever from the raw index (finding 5).
    """
    bm25_dir = Path(config["corpus"]["indexes_dir"]) / "bm25"
    if not bm25_dir.is_dir():
        raise ValueError("BM25 index not built yet: run sec-rag index-bm25")
    # bm25s README, "save/load": load_corpus=True brings back the node
    # records (with doc_name metadata), which the filter and results need.
    bm25 = bm25s.BM25.load(str(bm25_dir), load_corpus=True)
    return SearchIndexes(
        bm25=bm25, store=open_chunk_store(config), voyage=voyage, config=config
    )
