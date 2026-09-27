"""Turn chunk records into LlamaIndex nodes, the one form both indexes are built from.

A node is a chunk's text plus an ID and metadata (LlamaIndex, "Using
Nodes": docs/libraries/llamaindex/usage_nodes.md). Building both indexes
from the same nodes gives a chunk one ID and one metadata set everywhere,
so BM25 and Chroma results can be fused by ID in Stage 2.2.
"""

from __future__ import annotations

from hashlib import sha256
from typing import Any

from llama_index.core.schema import TextNode

from ..chunking.chunk_files import read_chunks
from ..ingestion.parse import select_documents

# The fields a search result needs: filtering (company, year, doc_type,
# doc_name), page metrics (doc_name, page_index), and text_hash, Chroma's
# cache key. Offsets, carried_text (already inside text), token_count and
# heading_path are left out (guide 1.5-1.6, "The node").
METADATA_KEYS = (
    "chunk_id",
    "doc_name",
    "company",
    "year",
    "doc_type",
    "page_index",
    "kind",
    "text_hash",
)


def load_corpus_nodes(config: dict[str, Any]) -> list[TextNode]:
    """Read every filing's chunk file and return all chunks as nodes.

    Both indexes start here. `select_documents` lists the prepared PDFs in
    sorted order (None means all of them); a PDF's stem is its doc_name.
    `read_chunks` raises "<doc>: not chunked yet" if a chunk file is missing,
    so nothing is built from a partial corpus.
    """
    chunks: list[dict[str, Any]] = []
    for pdf_path in select_documents(config, None):
        chunks.extend(read_chunks(config, pdf_path.stem))
    return chunks_to_nodes(chunks)


def chunks_to_nodes(chunks: list[dict[str, Any]]) -> list[TextNode]:
    """One TextNode per chunk record: id = chunk_id, text = the chunk's text.

    Metadata is attached for filtering and scoring, but hidden from the text
    the indexes see. BM25Retriever indexes
    `node.get_content(metadata_mode=MetadataMode.EMBED)`
    (llama_index/retrievers/bm25/base.py lines 99-112), which by default
    prepends every metadata field as "key: value" lines, so "3M" or "2018"
    would match every chunk of a filing (guide 1.5-1.6, BM25 finding 2).
    Excluded keys are left out of that text (usage_documents.md,
    "excluded_embed_metadata_keys"); pinned by test_nodes.py.
    """
    nodes: list[TextNode] = []
    for chunk in chunks:
        metadata = {key: chunk[key] for key in METADATA_KEYS if key != "text_hash"}
        # Chroma is the embedding cache (Build Order 1.6): a saved vector is
        # reused only while the chunk's text still has the same hash.
        metadata["text_hash"] = sha256(chunk["text"].encode("utf-8")).hexdigest()
        nodes.append(
            TextNode(
                id_=chunk["chunk_id"],
                text=chunk["text"],
                metadata=metadata,
                excluded_embed_metadata_keys=list(METADATA_KEYS),
                excluded_llm_metadata_keys=list(METADATA_KEYS),
            )
        )
    return nodes
