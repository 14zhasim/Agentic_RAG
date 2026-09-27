"""Embed every chunk with Voyage and keep the vectors in Chroma (Build Order 1.6).

Chroma is both the vector store and the embedding cache. Each saved chunk
carries the SHA-256 of the text its vector was made from (`text_hash`, set
in `nodes.chunks_to_nodes`), so a rerun compares hashes and pays Voyage only
for chunks that are new or whose text changed.

Voyage is called directly with its own client, not through LlamaIndex
(Stage 0.4 decision), and nodes are written with LlamaIndex's
`ChromaVectorStore`, so Stage 2.2 filters BM25 and Chroma with the same
`MetadataFilters`. Without `execute_paid` nothing is sent and no key is
needed (CLAUDE.md: every paid command has a no-spend equivalent).
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import chromadb
import voyageai
from llama_index.core.schema import BaseNode, TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore

from ..chunking.chunk import count_tokens
from .nodes import load_corpus_nodes

# Texts in -> (one vector per text, tokens billed). The real one calls
# Voyage; tests pass a fake, so no test can spend.
Embedder = Callable[[list[str]], tuple[list[list[float]], int]]


def embed_corpus(
    config: dict[str, Any],
    execute_paid: bool = False,
    embedder: Embedder | None = None,
) -> dict[str, int]:
    """Bring Chroma in line with the chunk files, embedding only what's missing or changed.

    Four steps (guide 1.5-1.6, `embed_corpus` diagram): 1) read the chunks
    and what Chroma holds; 2) compare them; 3) report; 4) only with
    execute_paid, delete stale entries and embed the rest in batches.

    Returns {"chunks", "stored", "to_embed", "to_delete", "estimated_tokens",
    "batches"}; with execute_paid also "embedded" and "tokens_billed".
    """
    # Step 1a: every chunk as a node, each already carrying its text_hash.
    nodes = load_corpus_nodes(config)
    # Step 1b: what Chroma already holds, as {chunk_id: text_hash}.
    store = open_chunk_store(config)
    hashes_in_chroma = _hashes_in_chroma(store)

    # Step 2: compare the two, chunk by chunk.
    to_embed, to_delete = _compare(nodes, hashes_in_chroma)

    # Step 3: report. Without execute_paid this is all that happens: no key
    # is read and nothing is sent.
    batch_size = config["embedding"]["batch_size"]
    report = {
        "chunks": len(nodes),
        "stored": len(hashes_in_chroma),
        "to_embed": len(to_embed),
        "to_delete": len(to_delete),
        # count_tokens is cl100k (Stage 1.3's counter): an estimate, since
        # Voyage counts tokens with its own tokeniser.
        "estimated_tokens": sum(count_tokens(node.text) for node in to_embed),
        "batches": math.ceil(len(to_embed) / batch_size),
    }
    if not execute_paid or (not to_embed and not to_delete):
        return report

    # Step 4a: delete. Chroma's add ignores an ID it already holds
    # (chroma/base.py lines 284-324 call collection.add), so a changed
    # chunk's old entry must go before its new vector is added.
    if to_delete:
        store.delete_nodes(node_ids=to_delete)
    # Step 4b: embed and save one batch at a time. The key is read only now
    # (CLAUDE.md: load credentials lazily).
    if embedder is None:
        embedder = _voyage_embedder(config["embedding"])
    tokens_billed = 0
    for start in range(0, len(to_embed), batch_size):
        batch = to_embed[start : start + batch_size]
        # One Voyage call: batch_size texts in, as many vectors out, in order.
        vectors, tokens = embedder([node.text for node in batch])
        for node, vector in zip(batch, vectors, strict=True):
            node.embedding = vector
        # Saved now, so a crash loses at most the batch in flight; the next
        # run sees everything before it as "same hash" and skips it.
        batch_nodes: list[BaseNode] = list(batch)
        store.add(batch_nodes)
        tokens_billed += tokens
        number = start // batch_size + 1
        print(f"batch {number}/{report['batches']}, {tokens_billed:,} tokens")
    return report | {"embedded": len(to_embed), "tokens_billed": tokens_billed}


def open_chunk_store(config: dict[str, Any]) -> ChromaVectorStore:
    """Open (or create) the on-disk collection of chunk vectors.

    Used here to write vectors, by Stage 2.2 to search, and by Stage 3 to
    read vectors back. One collection per model and vector length, so a
    change of either never mixes vectors that can't be compared.
    """
    indexes_dir = Path(config["corpus"]["indexes_dir"])
    model = config["embedding"]["model"]
    dimension = config["embedding"]["output_dimension"]
    # chromadb/__init__.py line 199: an on-disk database at this path,
    # created on first use and reopened after.
    client = chromadb.PersistentClient(path=str(indexes_dir / "chroma"))
    collection = client.get_or_create_collection(
        f"chunks_{model}_{dimension}",
        # Chroma defaults to squared L2 distance (chromadb/api/
        # collection_configuration.py line 442). Cosine is what the design
        # scores with, and Exp 2 adds a cosine structure term to it. The space
        # is fixed when the collection is created (Chroma docs, "configure").
        configuration={"hnsw": {"space": "cosine"}},
        # Otherwise Chroma attaches its own default embedding model and uses it
        # for any text that arrives without a vector. We always supply
        # Voyage's vectors, so a slip should be an error, not a silent second
        # model.
        embedding_function=None,
    )
    # metadata_filtering.md lines 120-130: wrap the collection for LlamaIndex.
    return ChromaVectorStore(chroma_collection=collection)


def _hashes_in_chroma(store: ChromaVectorStore) -> dict[str, str]:
    """Step 1b: {chunk_id: text_hash} for every chunk already saved.

    `store.client` is the raw Chroma collection (chroma/base.py lines
    366-369). Its get(include=["metadatas"]) returns every saved item as
    two lists in the same order, ids and metadatas, without the vectors
    (Chroma docs, "get"). The metadata is what `store.add` saved, so it
    includes the text_hash each vector was made from.
    """
    saved = store.client.get(include=["metadatas"])
    hashes: dict[str, str] = {}
    for chunk_id, metadata in zip(saved["ids"], saved["metadatas"], strict=True):
        hashes[chunk_id] = metadata["text_hash"]
    return hashes


def _compare(
    nodes: list[TextNode], hashes_in_chroma: dict[str, str]
) -> tuple[list[TextNode], list[str]]:
    """Step 2: which chunks to embed, and which saved chunk IDs to delete.

    Delete: saved chunks no longer in the files, or whose text changed.
    Embed: chunks never saved, or whose text changed (deleted first).
    """
    node_by_id = {node.node_id: node for node in nodes}
    to_delete: list[str] = []
    for chunk_id, saved_hash in hashes_in_chroma.items():
        if chunk_id not in node_by_id:
            # Chunk gone from the files.
            to_delete.append(chunk_id)
        elif node_by_id[chunk_id].metadata["text_hash"] != saved_hash:
            # Text changed: the saved vector is stale.
            to_delete.append(chunk_id)
    deleting = set(to_delete)
    to_embed: list[TextNode] = []
    for node in nodes:
        if node.node_id not in hashes_in_chroma:
            to_embed.append(node)  # new chunk
        elif node.node_id in deleting:
            to_embed.append(node)  # changed chunk, deleted in step 4a
    return to_embed, to_delete


def _voyage_embedder(settings: dict[str, Any]) -> Embedder:
    """Return an Embedder that sends chunk texts to Voyage.

    Reads VOYAGE_API_KEY here, so report mode and tests never need it.
    """
    key = os.environ.get("VOYAGE_API_KEY")
    if not key:
        raise ValueError("VOYAGE_API_KEY is not set")
    # The installed client's default is max_retries=0 (checked with
    # inspect.signature on voyageai 0.5.0); rate-limits.md lines 95-140
    # recommend retrying rate-limit and network errors with back-off.
    client = voyageai.Client(api_key=key, max_retries=3)

    def embed(texts: list[str]) -> tuple[list[list[float]], int]:
        # embeddings.md lines 60-82: input_type="document" for chunks
        # ("query" for questions, Stage 2.2); truncation=False makes an
        # over-long text an error instead of a silently cut vector. The
        # length and type are Voyage's defaults, stated so the config is the
        # single record of them.
        result = client.embed(
            texts,
            model=settings["model"],
            input_type="document",
            truncation=False,
            output_dimension=settings["output_dimension"],
            output_dtype=settings["output_dtype"],
        )
        # The client types embeddings as floats or ints (ints for the
        # quantised output_dtypes); load_config allows only "float", so these
        # are floats. total_tokens is what Voyage bills for this call.
        vectors = cast(list[list[float]], result.embeddings)
        return vectors, result.total_tokens

    return embed
