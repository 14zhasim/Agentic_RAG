"""Tests for embed_corpus: Chroma kept in line with the chunk files, paying only for changes.

A fake embedder stands in for Voyage (no test can spend) and Chroma lives in
tmp_path, so every test starts from an empty store.
"""

from __future__ import annotations

import pytest
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
)

from sec_rag.indexing.embed import embed_corpus, open_chunk_store


class FakeEmbedder:
    """Returns a fixed 3-number vector per text and 10 billed tokens per text."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, texts: list[str]) -> tuple[list[list[float]], int]:
        self.calls.append(texts)
        vectors = [[1.0, float(len(text)), 0.5] for text in texts]
        return vectors, 10 * len(texts)


def _saved(config) -> dict[str, dict]:
    """Everything in the collection, as {chunk_id: metadata}."""
    saved = open_chunk_store(config).client.get(include=["metadatas"])
    return dict(zip(saved["ids"], saved["metadatas"], strict=True))


def test_report_mode_embeds_nothing_and_needs_no_key(index_config, monkeypatch) -> None:
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    config, add_filing = index_config
    add_filing("3M_2018_10K", ["one", "two", "three"])
    fake = FakeEmbedder()

    report = embed_corpus(config, embedder=fake)

    assert report["chunks"] == 3
    assert report["stored"] == 0
    assert report["to_embed"] == 3
    assert report["to_delete"] == 0
    assert report["batches"] == 2  # batch_size 2 in the fixture: 2 + 1
    assert report["estimated_tokens"] > 0
    assert fake.calls == []
    assert _saved(config) == {}


def test_execute_paid_stores_every_chunk_with_metadata(index_config) -> None:
    config, add_filing = index_config
    add_filing("3M_2018_10K", ["one", "two", "three"], page_index=59)
    fake = FakeEmbedder()

    report = embed_corpus(config, execute_paid=True, embedder=fake)

    assert report["embedded"] == 3
    assert report["tokens_billed"] == 30
    assert [len(call) for call in fake.calls] == [2, 1]
    saved = _saved(config)
    assert sorted(saved) == [f"3M_2018_10K:p0:c{n}" for n in range(3)]
    metadata = saved["3M_2018_10K:p0:c0"]
    assert metadata["doc_name"] == "3M_2018_10K"
    assert metadata["page_index"] == 59
    assert len(metadata["text_hash"]) == 64


def test_second_run_embeds_nothing(index_config) -> None:
    config, add_filing = index_config
    add_filing("3M_2018_10K", ["one", "two", "three"])
    embed_corpus(config, execute_paid=True, embedder=FakeEmbedder())
    fake = FakeEmbedder()

    report = embed_corpus(config, execute_paid=True, embedder=fake)

    assert report["stored"] == 3
    assert report["to_embed"] == 0
    assert report["to_delete"] == 0
    assert fake.calls == []


def test_changed_chunk_is_re_embedded_and_removed_chunk_deleted(index_config) -> None:
    """The guide's tiny example: c0 same, c1 changed, c2 new, the old c2 gone."""
    config, add_filing = index_config
    add_filing("3M_2018_10K", ["A", "B", "Z"])
    embed_corpus(config, execute_paid=True, embedder=FakeEmbedder())
    old_hashes = {cid: meta["text_hash"] for cid, meta in _saved(config).items()}
    add_filing("3M_2018_10K", ["A", "B2"])
    fake = FakeEmbedder()

    report = embed_corpus(config, execute_paid=True, embedder=fake)

    assert report["to_delete"] == 2  # c1 changed, c2 gone
    assert report["to_embed"] == 1  # c1 again
    assert fake.calls == [["B2"]]
    saved = _saved(config)
    assert sorted(saved) == ["3M_2018_10K:p0:c0", "3M_2018_10K:p0:c1"]
    assert saved["3M_2018_10K:p0:c0"]["text_hash"] == old_hashes["3M_2018_10K:p0:c0"]
    assert saved["3M_2018_10K:p0:c1"]["text_hash"] != old_hashes["3M_2018_10K:p0:c1"]


def test_collection_uses_cosine(index_config) -> None:
    config, _ = index_config

    store = open_chunk_store(config)

    assert store.client.name == "chunks_voyage-4-lite_3"
    # The raw JSON: `.configuration` warns about any collection without one of
    # Chroma's own embedding functions, which ours deliberately has not.
    assert store.client.configuration_json["hnsw"]["space"] == "cosine"


def test_filtered_search_returns_only_that_filing(index_config) -> None:
    config, add_filing = index_config
    add_filing("3M_2018_10K", ["one", "two"])
    add_filing("AMD_2022_10K", ["one", "two"])
    embed_corpus(config, execute_paid=True, embedder=FakeEmbedder())
    only_amd = MetadataFilters(
        filters=[MetadataFilter(key="doc_name", value="AMD_2022_10K")]
    )

    result = open_chunk_store(config).query(
        VectorStoreQuery(
            query_embedding=[1.0, 3.0, 0.5], similarity_top_k=10, filters=only_amd
        )
    )

    assert result.ids is not None
    assert sorted(result.ids) == ["AMD_2022_10K:p0:c0", "AMD_2022_10K:p0:c1"]


def test_missing_key_is_an_error_only_when_paying(index_config, monkeypatch) -> None:
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    config, add_filing = index_config
    add_filing("3M_2018_10K", ["one"])

    embed_corpus(config)  # report mode: fine without a key

    with pytest.raises(ValueError, match="VOYAGE_API_KEY"):
        embed_corpus(config, execute_paid=True)


def test_unchunked_filing_stops_before_anything_is_sent(index_config) -> None:
    config, add_filing = index_config
    add_filing("3M_2018_10K", ["one"])
    add_filing("AMD_2022_10K", None)
    fake = FakeEmbedder()

    with pytest.raises(ValueError, match="AMD_2022_10K: not chunked yet"):
        embed_corpus(config, execute_paid=True, embedder=fake)
    assert fake.calls == []
