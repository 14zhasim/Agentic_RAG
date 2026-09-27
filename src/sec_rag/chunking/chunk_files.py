"""Build, save and read back the per-filing chunk files.

This module handles files only: it picks the filings, reads each saved parse,
hands it to `chunk.chunk_filing` for the rules, and writes one JSON Lines file
per filing to `data/financebench/chunks/<doc_name>.jsonl`, one chunk per line
in reading order. Stage 1.5 (keyword index) and 1.6 (embeddings) read those
files. Free and offline: no API is called, so there is no dry-run mode;
re-running just rebuilds.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from ..ingestion.load_pages import load_pages
from ..ingestion.parse import json_path, read_json, select_documents
from .chunk import chunk_filing


def build_chunks(
    config: dict[str, Any],
    document_names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Chunk each selected parsed filing, write its chunk file, and return a summary.

    Every filing is chunked before any file is written, so one unparsed
    filing stops the run with nothing written, as `inspect-parse` does.
    """
    built: list[tuple[str, int, list[dict[str, Any]]]] = []
    for pdf_path in select_documents(config, document_names):
        doc_name = pdf_path.stem
        parsed_path = json_path(config, doc_name)
        if not parsed_path.is_file():
            raise ValueError(f"{doc_name}: not parsed yet")
        raw = read_json(parsed_path)
        pages = load_pages(config, doc_name)
        records = chunk_filing(raw, pages, config["chunking"])
        figures_in_parse = len(raw.get("figures") or [])
        built.append((doc_name, figures_in_parse, records))

    for doc_name, _, records in built:
        _write_chunks(chunks_path(config, doc_name), records)
    return _summary(built, config["chunking"])


def _write_chunks(path: Path, records: list[dict[str, Any]]) -> None:
    """Save one filing's chunks as JSON Lines, so the file is either whole or absent.

    Written to `<doc_name>.jsonl.partial` and then renamed, as the parse run
    does (`parse._write_json`): the rename is instant, so an interrupted run
    never leaves half a chunk file for Stage 1.5 to index.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = path.with_suffix(".jsonl.partial")
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    partial_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    partial_path.replace(path)


def _summary(
    built: list[tuple[str, int, list[dict[str, Any]]]], chunking: dict[str, int]
) -> dict[str, Any]:
    """Count what `sec-rag chunk` prints: how many chunks, of what kind and size.

    This is Build Order 1.3's "re-measure on all 64 filings" with a real
    tokenizer rather than characters ÷ 4. Prose over the ceiling can only
    come from carried text; prose under the floor is mostly small pages,
    which the rules keep whole on purpose.
    """
    floor = chunking["floor_tokens"]
    ceiling = chunking["ceiling_tokens"]
    kinds = {"prose": 0, "table": 0, "figure": 0}
    prose_sizes: list[int] = []
    figures_in_parses = 0
    for _, figures_in_parse, records in built:
        figures_in_parses += figures_in_parse
        for record in records:
            kinds[record["kind"]] += 1
            if record["kind"] == "prose":
                prose_sizes.append(record["token_count"])

    in_range = sum(1 for size in prose_sizes if floor <= size <= ceiling)
    return {
        "filings": len(built),
        "chunks": sum(kinds.values()),
        "kinds": kinds,
        "median_prose_tokens": statistics.median(prose_sizes) if prose_sizes else 0,
        "prose_in_range_share": in_range / len(prose_sizes) if prose_sizes else 0.0,
        "prose_over_ceiling": sum(1 for size in prose_sizes if size > ceiling),
        "prose_under_floor": sum(1 for size in prose_sizes if size < floor),
        "empty_figures_skipped": figures_in_parses - kinds["figure"],
    }


def read_chunks(config: dict[str, Any], doc_name: str) -> list[dict[str, Any]]:
    """Read one filing's chunk file back into records, in reading order."""
    path = chunks_path(config, doc_name)
    if not path.is_file():
        raise ValueError(f"{doc_name}: not chunked yet")
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line]


def chunks_path(config: dict[str, Any], doc_name: str) -> Path:
    """Return where a filing's chunk file lives: chunks/<doc_name>.jsonl."""
    return Path(config["corpus"]["chunks_dir"]) / f"{doc_name}.jsonl"


def write_chunk_report(config: dict[str, Any], doc_name: str) -> Path:
    """Write every chunk of one filing, page by page, to chunks/<doc_name>.chunks.txt.

    Stage 1.4's "view chunks manually": one readable file to hold beside the
    PDF. Nothing downstream reads it.
    """
    chunks = read_chunks(config, doc_name)
    page_indexes = sorted({chunk["page_index"] for chunk in chunks})
    parts = [
        f"# Chunks: {doc_name} ({len(chunks)} chunks on {len(page_indexes)} pages)"
    ]
    for page_index in page_indexes:
        parts.append(format_page_chunks(chunks, page_index))
    path = chunks_path(config, doc_name).with_suffix(".chunks.txt")
    path.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    return path


def format_page_chunks(chunks: list[dict[str, Any]], page_index: int) -> str:
    """Show every chunk on one page as readable text, for Stage 1.4's eyeballing.

    The header gives both page numbers: the page index (counted from 0, the
    same as FinanceBench's evidence_page_num) and the PDF page (counted from
    1, what a PDF viewer shows). Each chunk shows its ID, kind, size and raw
    offsets, then any carried sentence marked `[carried: ...]`, then its own
    text, so you can see exactly where each cut fell.
    """
    on_page = [chunk for chunk in chunks if chunk["page_index"] == page_index]
    header = f"=== Page index {page_index} (PDF page {page_index + 1}) ==="
    if not on_page:
        return f"{header}\nNo chunks on page {page_index}"

    lines = [header]
    for chunk in on_page:
        lines.append("")
        lines.append(
            f"--- {chunk['chunk_id']} | {chunk['kind']} | "
            f"{chunk['token_count']} tokens | "
            f"offsets {chunk['start_offset']}-{chunk['end_offset']} ---"
        )
        carried = chunk["carried_text"]
        own_text = chunk["text"]
        if carried:
            lines.append(f"[carried: {carried}]")
            # `text` is the carried sentence, one space, then the chunk's own text.
            own_text = own_text[len(carried) + 1 :]
        lines.append(own_text)
    return "\n".join(lines)
