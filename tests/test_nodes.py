"""Tests for chunks_to_nodes: chunk records become the nodes both indexes share."""

from __future__ import annotations

from hashlib import sha256

from conftest import chunk_record
from llama_index.core.schema import MetadataMode

from sec_rag.indexing.nodes import METADATA_KEYS, chunks_to_nodes

TEXT = "Purchases of property, plant and equipment (PP&E) were 1,577 in 2018."


def test_node_has_chunk_id_text_and_metadata() -> None:
    chunk = chunk_record("3M_2018_10K", 1, TEXT, page_index=59, kind="table")

    [node] = chunks_to_nodes([chunk])

    assert node.node_id == "3M_2018_10K:p0:c1"
    assert node.text == TEXT
    assert node.metadata == {
        "chunk_id": "3M_2018_10K:p0:c1",
        "doc_name": "3M_2018_10K",
        "company": "3M",
        "year": 2018,
        "doc_type": "10K",
        "page_index": 59,
        "kind": "table",
        "text_hash": sha256(TEXT.encode("utf-8")).hexdigest(),
    }
    assert set(node.metadata) == set(METADATA_KEYS)


def test_node_metadata_is_hidden_from_indexed_text() -> None:
    """Finding 2: BM25Retriever indexes get_content(MetadataMode.EMBED)."""
    [node] = chunks_to_nodes([chunk_record("3M_2018_10K", 0, TEXT)])

    assert node.get_content(metadata_mode=MetadataMode.EMBED) == TEXT
    assert node.get_content(metadata_mode=MetadataMode.LLM) == TEXT


def test_same_text_gives_same_hash_and_different_text_a_different_one() -> None:
    first, same, changed = chunks_to_nodes(
        [
            chunk_record("3M_2018_10K", 0, TEXT),
            chunk_record("3M_2018_10K", 1, TEXT),
            chunk_record("3M_2018_10K", 2, TEXT + " Restated."),
        ]
    )

    assert first.metadata["text_hash"] == same.metadata["text_hash"]
    assert first.metadata["text_hash"] != changed.metadata["text_hash"]
