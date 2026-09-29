"""Give every chunk a structure vector from its heading path (Build Order 3.1-3.3).

Exp2's one change is a second term in the dense score: question · structure,
where a chunk's structure vector blends the embeddings of the headings it
sits under. This module builds those vectors once, at ingestion, so a run
only reads them:

1. `heading_paths` attributes Azure's raw headings to chunks by character
   offset (Draft -> chunking rule 6).
2. Each unique heading text is embedded once, into the **headings**
   collection, whose IDs are text hashes: it is also the embedding cache,
   so a rebuild pays only for new headings.
3. `structure_vector` blends each chunk's heading vectors, weighted by how
   close each is to the chunk, into the **structure** collection, one row
   per chunk, keyed by chunk ID.

Both collections sit beside the chunk collection in the same Chroma folder,
so the chunk files and chunk vectors stay byte-identical to Exp1's. Without
`execute_paid`, nothing is sent to Voyage and no key is read (CLAUDE.md:
every paid command has a no-spend equivalent).
"""

from __future__ import annotations

import math
from hashlib import sha256
from pathlib import Path
from typing import Any

import chromadb
import numpy as np

from ..chunking.chunk import count_tokens
from ..chunking.chunk_files import read_chunks
from ..ingestion.headings import Heading, extract_headings
from ..ingestion.parse import json_path, read_json, select_documents
from .embed import Embedder, _voyage_embedder, open_chunk_store


def build_structure_index(
    config: dict[str, Any],
    execute_paid: bool = False,
    embedder: Embedder | None = None,
) -> dict[str, int]:
    """Attribute heading paths, embed missing headings, and rebuild every structure vector.

    Five steps (Implementation Guide 3.1-3.4 -> Slice 1):

    1. every filing's heading paths, from its saved Azure parse and chunk file
    2. compare the heading texts needed with those already embedded
    3. report; without execute_paid, stop here if any heading is missing
    4. embed the missing headings, saving each batch before the next
    5. rebuild the structure collection from scratch, then mark it complete

    Step 5 is local (no Voyage call), so it also runs without execute_paid
    once every heading is stored, e.g. after a change of softmax_divisor.

    Returns {"filings", "chunks", "chunks_with_path", "unique_headings",
    "stored_headings", "to_embed", "estimated_tokens"}, plus "embedded",
    "tokens_billed" and "structure_rows" when it built.
    """
    settings = config["structure"]

    # Step 1: paths per filing. The chunk files are Exp1's, unchanged; only
    # their offsets are read here.
    paths_by_filing: dict[str, dict[str, list[str]]] = {}
    chunk_count = 0
    for pdf_path in select_documents(config, None):
        doc_name = pdf_path.stem
        headings = extract_headings(read_json(json_path(config, doc_name)))
        chunks = read_chunks(config, doc_name)
        chunk_count += len(chunks)
        paths_by_filing[doc_name] = heading_paths(
            headings, chunks, settings["max_depth"]
        )

    # Step 2: which heading texts are needed, and which are already stored.
    # A dict keeps first-seen order, so batches are the same on every run.
    needed: dict[str, str] = {}  # heading ID (hash of its text) -> text
    for paths in paths_by_filing.values():
        for path in paths.values():
            for text in path:
                needed[_heading_id(text)] = text
    headings_store = _open_headings_store(config)
    stored_ids = set(headings_store.get(ids=list(needed), include=[])["ids"])
    missing = [
        (heading_id, text)
        for heading_id, text in needed.items()
        if heading_id not in stored_ids
    ]

    # Step 3: report. count_tokens is cl100k, an estimate, as in embed.py.
    report = {
        "filings": len(paths_by_filing),
        "chunks": chunk_count,
        "chunks_with_path": sum(len(paths) for paths in paths_by_filing.values()),
        "unique_headings": len(needed),
        "stored_headings": len(stored_ids),
        "to_embed": len(missing),
        "estimated_tokens": sum(count_tokens(text) for _id, text in missing),
    }
    if missing and not execute_paid:
        return report

    # Step 4: embed the missing headings. The key is read only here.
    tokens_billed = 0
    if missing:
        if embedder is None:
            # Headings are stored text, so Voyage's input_type="document",
            # as for chunks (_voyage_embedder; docs/libraries/voyage/
            # embeddings.md lines 60-82).
            embedder = _voyage_embedder(config["embedding"])
        batch_size = config["embedding"]["batch_size"]
        batches = math.ceil(len(missing) / batch_size)
        for start in range(0, len(missing), batch_size):
            batch = missing[start : start + batch_size]
            vectors, tokens = embedder([text for _id, text in batch])
            # Saved now, so a crash loses at most the batch in flight; the
            # next run finds everything before it already stored.
            headings_store.upsert(
                ids=[heading_id for heading_id, _text in batch],
                embeddings=np.array(vectors, dtype=np.float32),
                metadatas=[{"text": text} for _id, text in batch],
            )
            tokens_billed += tokens
            print(
                f"batch {start // batch_size + 1}/{batches}, {tokens_billed:,} tokens"
            )

    # Step 5: rebuild the structure collection. From scratch, because a
    # change of max_depth or softmax_divisor changes every vector.
    structure_store = _recreate_structure_store(config)
    chunk_collection = open_chunk_store(config).client
    rows = 0
    for doc_name, paths in paths_by_filing.items():
        rows += _save_filing_vectors(
            doc_name, paths, chunk_collection, headings_store, structure_store, settings
        )
    # Only now is the collection searchable: open_structure_store refuses
    # one whose build did not reach this line.
    structure_store.modify(metadata=_build_metadata(settings, complete=True))
    return report | {
        "embedded": len(missing),
        "tokens_billed": tokens_billed,
        "structure_rows": rows,
    }


def heading_paths(
    headings: list[Heading], chunks: list[dict[str, Any]], max_depth: int
) -> dict[str, list[str]]:
    """Chunk ID -> the heading texts open anywhere in that chunk (Draft -> chunking rule 6).

    Stack rule, as the Stage 0 structure report's `build_page_map`: a new
    heading closes every open heading at the same or a deeper level, then
    opens. A heading's chain depth is its position in the stack once it
    opens (1 = outermost); only depth <= max_depth is kept, which drops the
    deepest (Build Order 3.1, depth <= 6).

    The path is the headings open at the chunk's start_offset, then those
    that start inside its span, each text once, in document order. So a
    chunk under Part I that spans Item 2 and Item 3 gets all three. A chunk
    before the first heading gets no entry.

    The rule is replayed from the top for each chunk, so each chunk's path
    depends only on its own offsets: ~330 chunks x ~325 headings per filing
    is fast enough not to need a single shared sweep.
    """
    # extract_headings gives level=None to a heading that opens no section
    # in Azure's tree. It can't be placed on the stack, and none of the
    # corpus's 20,789 headings has one (checked 29 Sep 2026), so refuse
    # rather than guess where it belongs.
    levelled: list[tuple[int, Heading]] = []  # (level, heading), in offset order
    for heading in headings:
        if heading.level is None:
            raise ValueError(
                f"heading at offset {heading.offset} has no level: {heading.text!r}"
            )
        levelled.append((heading.level, heading))

    paths: dict[str, list[str]] = {}
    for chunk in chunks:
        stack: list[tuple[int, Heading]] = []  # open headings, outermost first
        open_at_start: list[tuple[int, Heading]] = []  # the stack when the chunk began
        opened_inside: list[tuple[int, Heading]] = []  # started inside the span
        for level, heading in levelled:
            if heading.offset >= chunk["end_offset"]:
                break
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading))
            if heading.offset < chunk["start_offset"]:
                open_at_start = list(stack)
            elif len(stack) <= max_depth:
                opened_inside.append((level, heading))
        kept = open_at_start[:max_depth] + opened_inside
        # dict.fromkeys drops repeats and keeps first-seen order.
        texts = list(dict.fromkeys(heading.text for _level, heading in kept))
        if texts:
            paths[chunk["chunk_id"]] = texts
    return paths


def structure_vector(
    chunk_vector: np.ndarray, heading_vectors: np.ndarray, divisor: float
) -> np.ndarray:
    """Softmax-weighted average of a chunk's heading vectors, at length 1 (Build Order 3.3).

    weights = softmax(chunk·heading / divisor), so the heading closest to
    the chunk's own text takes most of the weight. Chunk-to-heading, not
    query-to-heading, so the vector is fixed at build time. The divisor
    (0.05) widens the gaps between similarities 20-fold; a large divisor
    would make this a plain average. The worked example is in the guide,
    under Slice 1's pseudocode.
    """
    # Length 1 first, so each dot product is a cosine. Voyage already
    # returns length-1 vectors; this is a safeguard.
    chunk_unit = chunk_vector / np.linalg.norm(chunk_vector)
    heading_units = heading_vectors / np.linalg.norm(
        heading_vectors, axis=1, keepdims=True
    )
    similarities = heading_units @ chunk_unit
    scaled = similarities / divisor
    # Subtracting the max leaves the softmax unchanged (it cancels top and
    # bottom) and stops exp overflowing.
    weights = np.exp(scaled - scaled.max())
    weights = weights / weights.sum()
    blended = weights @ heading_units
    # An average of unit vectors pointing different ways is shorter than 1,
    # by a different amount per chunk, so it is scaled back to length 1.
    result: np.ndarray = blended / np.linalg.norm(blended)
    return result


def open_structure_store(config: dict[str, Any]) -> Any:
    """Open the structure collection for searching, refusing a stale or partial build.

    Raises ValueError when the collection is missing, its build did not
    finish, or it was built with a max_depth or softmax_divisor other than
    the config's: those two settings shape every saved vector, so a run
    must not score with vectors made under different ones.
    """
    client = _chroma_client(config)
    name = _structure_collection_name(config)
    rebuild = "rebuild with `sec-rag index-structure`"
    if name not in {collection.name for collection in client.list_collections()}:
        raise ValueError(f"structure vectors not built: {rebuild}")
    collection = client.get_collection(name)
    metadata = collection.metadata or {}
    settings = config["structure"]
    if metadata.get("complete") is not True:
        raise ValueError(f"structure build did not finish: {rebuild}")
    if (
        metadata.get("max_depth") != settings["max_depth"]
        or metadata.get("softmax_divisor") != settings["softmax_divisor"]
    ):
        raise ValueError(
            f"structure vectors built with max_depth={metadata.get('max_depth')}, "
            f"softmax_divisor={metadata.get('softmax_divisor')}; the config has "
            f"{settings['max_depth']} and {settings['softmax_divisor']}: {rebuild}"
        )
    return collection


def _save_filing_vectors(
    doc_name: str,
    paths: dict[str, list[str]],
    chunk_collection: Any,
    headings_store: Any,
    structure_store: Any,
    settings: dict[str, Any],
) -> int:
    """Compute and save one filing's structure vectors; return how many were saved.

    One filing per write keeps each write (~330 rows) far below Chroma's
    5,461-row limit (client.get_max_batch_size(), checked on chromadb 1.5.9).
    """
    if not paths:
        return 0
    # get(where=..., include=["embeddings"]) returns the vectors as one
    # numpy array, rows in the same order as "ids" (checked on chromadb
    # 1.5.9; .claude/skills/chroma-local/querying/python.md, "include").
    chunk_rows = chunk_collection.get(
        where={"doc_name": doc_name}, include=["embeddings"]
    )
    chunk_vectors = dict(zip(chunk_rows["ids"], chunk_rows["embeddings"], strict=True))
    heading_ids = list({_heading_id(text) for path in paths.values() for text in path})
    # get(ids=...) does not promise the requested order, so vectors are
    # matched back to their headings by ID.
    heading_rows = headings_store.get(ids=heading_ids, include=["embeddings"])
    heading_vectors = dict(
        zip(heading_rows["ids"], heading_rows["embeddings"], strict=True)
    )

    ids: list[str] = []
    vectors: list[np.ndarray] = []
    metadatas: list[dict[str, str]] = []
    for chunk_id, path in paths.items():
        if chunk_id not in chunk_vectors:
            raise ValueError(f"{chunk_id}: no chunk vector: run sec-rag embed first")
        path_vectors = np.array([heading_vectors[_heading_id(text)] for text in path])
        ids.append(chunk_id)
        vectors.append(
            structure_vector(
                np.asarray(chunk_vectors[chunk_id]),
                path_vectors,
                settings["softmax_divisor"],
            )
        )
        metadatas.append(
            # doc_name for the run's filter; the path is for inspection only
            # (Chroma metadata values must be strings or numbers, not lists).
            {"doc_name": doc_name, "heading_path": " > ".join(path)}
        )
    structure_store.upsert(
        ids=ids, embeddings=np.array(vectors, dtype=np.float32), metadatas=metadatas
    )
    return len(ids)


def _open_headings_store(config: dict[str, Any]) -> Any:
    """Open (or create) the headings collection: one vector per unique heading text."""
    model = config["embedding"]["model"]
    dimension = config["embedding"]["output_dimension"]
    # Cosine space and no embedding function, as the chunk collection
    # (embed.open_chunk_store): Voyage's vectors are always supplied.
    return _chroma_client(config).get_or_create_collection(
        f"headings_{model}_{dimension}",
        configuration={"hnsw": {"space": "cosine"}},
        embedding_function=None,
    )


def _recreate_structure_store(config: dict[str, Any]) -> Any:
    """Delete the structure collection if present, and create it empty, marked incomplete."""
    client = _chroma_client(config)
    name = _structure_collection_name(config)
    if name in {collection.name for collection in client.list_collections()}:
        client.delete_collection(name)
    return client.create_collection(
        name,
        configuration={"hnsw": {"space": "cosine"}},
        # How the vectors were built, so open_structure_store can refuse a
        # run whose config disagrees (Build Order 3.2).
        metadata=_build_metadata(config["structure"], complete=False),
        embedding_function=None,
    )


def _build_metadata(settings: dict[str, Any], complete: bool) -> dict[str, Any]:
    """The structure collection's own metadata: its build settings and whether it finished."""
    return {
        "max_depth": settings["max_depth"],
        "softmax_divisor": settings["softmax_divisor"],
        "complete": complete,
    }


def _structure_collection_name(config: dict[str, Any]) -> str:
    """structure_<model>_<length>, beside chunks_<model>_<length>."""
    model = config["embedding"]["model"]
    dimension = config["embedding"]["output_dimension"]
    return f"structure_{model}_{dimension}"


def _chroma_client(config: dict[str, Any]) -> Any:
    """The on-disk Chroma database the chunk collection lives in."""
    return chromadb.PersistentClient(
        path=str(Path(config["corpus"]["indexes_dir"]) / "chroma")
    )


def _heading_id(text: str) -> str:
    """A heading's ID: the SHA-256 of its text, so each unique text is stored once."""
    return sha256(text.encode("utf-8")).hexdigest()
