"""Stage 0.1 spike, kept as a regression test.

`BM25Retriever`'s `filters` argument is a recent addition to
`llama-index-retrievers-bm25`. Stage 2's retrieval design assumes filter-then-rank:
only chunks from in-scope filings are scored, so ranks have no gaps and RRF fuses
comparable lists.

Finding (llama-index-retrievers-bm25 0.8.0): the filter IS applied before scoring
-- internally it becomes a `corpus_weight_mask` passed to `bm25s` -- but filtered
chunks are NOT removed from the result list. They come back with `score == 0.0`
whenever `similarity_top_k` exceeds the number of surviving chunks. Callers must
therefore drop zero-score hits; see `nonzero_hits` below.

Fallback if this ever breaks: build the retriever from an already-filtered node
list instead of passing `filters`.
"""

import pytest
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.retrievers.bm25 import BM25Retriever

# Deliberately overlapping wording: every chunk is a plausible BM25 match for
# "total revenue", and the out-of-scope filings repeat the query terms most, so
# an unfiltered search ranks them top. That is what makes the filter's effect
# visible rather than coincidental.
CHUNKS = [
    ("3M_2018_10K", "Total revenue for the year was 32,765 million dollars."),
    ("3M_2018_10K", "Operating income grew while total revenue stayed flat."),
    ("AMD_2022_10K", "Total revenue total revenue total revenue rose sharply."),
    ("AMD_2022_10K", "Revenue, total revenue, and total revenue again."),
    ("PEPSICO_2021_10K", "Total revenue total revenue reached 79,474 million."),
]

IN_SCOPE = "3M_2018_10K"
QUERY = "total revenue"


def names(hits: list[NodeWithScore]) -> set[str]:
    return {hit.metadata["doc_name"] for hit in hits}


def nonzero_hits(hits: list[NodeWithScore]) -> list[NodeWithScore]:
    """Drop the zero-score padding the retriever returns alongside real matches."""
    return [hit for hit in hits if hit.score]


@pytest.fixture
def docstore() -> SimpleDocumentStore:
    store = SimpleDocumentStore()
    store.add_documents(
        [
            TextNode(text=text, metadata={"doc_name": doc_name})
            for doc_name, text in CHUNKS
        ]
    )
    return store


def _retrieve(
    docstore: SimpleDocumentStore,
    filters: MetadataFilters | None,
    top_k: int = 5,
) -> list[NodeWithScore]:
    return BM25Retriever.from_defaults(
        docstore=docstore,
        similarity_top_k=top_k,
        filters=filters,
    ).retrieve(QUERY)


def _equals(doc_name: str) -> MetadataFilters:
    return MetadataFilters(
        filters=[
            MetadataFilter(key="doc_name", value=doc_name, operator=FilterOperator.EQ)
        ],
        condition=FilterCondition.AND,
    )


def test_unfiltered_search_returns_out_of_scope_filings(
    docstore: SimpleDocumentStore,
) -> None:
    """Baseline: without a filter the wrong filings dominate the ranking."""
    hits = nonzero_hits(_retrieve(docstore, None))

    assert names(hits) != {IN_SCOPE}, (
        "fixtures are not discriminating: the unfiltered search already returns "
        "only the in-scope filing, so a passing filter test would prove nothing"
    )


def test_filter_restricts_scored_results_to_the_named_filing(
    docstore: SimpleDocumentStore,
) -> None:
    """The design's filter-then-rank rule, once padding is discarded."""
    hits = nonzero_hits(_retrieve(docstore, _equals(IN_SCOPE)))

    assert len(hits) == 2, "both in-scope chunks should be scored"
    assert names(hits) == {IN_SCOPE}


def test_out_of_scope_chunks_come_back_as_zero_score_padding(
    docstore: SimpleDocumentStore,
) -> None:
    """Pin the surprise: filtered chunks are demoted to 0.0, not removed.

    This is why retrieval must discard zero-score hits rather than trusting the
    retriever to honour the filter by itself.
    """
    hits = _retrieve(docstore, _equals(IN_SCOPE), top_k=5)

    assert len(hits) == 5, "the retriever pads up to similarity_top_k"
    assert all(hit.score == 0.0 for hit in hits if hit.metadata["doc_name"] != IN_SCOPE)


def test_no_padding_when_top_k_fits_the_filtered_set(
    docstore: SimpleDocumentStore,
) -> None:
    """With top_k at or below the surviving count, the raw results are clean."""
    hits = _retrieve(docstore, _equals(IN_SCOPE), top_k=2)

    assert names(hits) == {IN_SCOPE}


def test_filter_accepts_several_filings(docstore: SimpleDocumentStore) -> None:
    """Questions can name more than one filing, so IN must work too."""
    wanted = [IN_SCOPE, "PEPSICO_2021_10K"]
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="doc_name", value=wanted, operator=FilterOperator.IN)
        ],
        condition=FilterCondition.AND,
    )

    hits = nonzero_hits(_retrieve(docstore, filters))

    assert hits
    assert names(hits) <= set(wanted)
    assert "AMD_2022_10K" not in names(hits)


def test_filtering_everything_out_raises(docstore: SimpleDocumentStore) -> None:
    """A filter matching nothing fails loudly rather than returning the corpus."""
    with pytest.raises(ValueError, match="filtered out"):
        _retrieve(docstore, _equals("NOT_A_FILING"))
