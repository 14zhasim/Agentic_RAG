"""Translate SEC RAG terminal commands into ingestion, chunking and indexing operations."""

from __future__ import annotations

import argparse
from pathlib import Path

from .chunking.chunk_files import (
    build_chunks,
    format_page_chunks,
    read_chunks,
    write_chunk_report,
)
from .config import load_config
from .indexing.bm25_index import build_bm25_index
from .indexing.embed import embed_corpus
from .ingestion.inspect_parse import inspect_parses
from .ingestion.parse import ParseStateError, parse_corpus


def main(argv: list[str] | None = None) -> int:
    """Parse one SEC RAG command, delegate it, and return its exit status."""
    parser = argparse.ArgumentParser(prog="sec-rag")
    commands = parser.add_subparsers(dest="command", required=True)
    parse_parser = commands.add_parser("parse")
    parse_parser.add_argument("--config", type=Path, required=True)
    parse_parser.add_argument("--documents", nargs="+")
    parse_parser.add_argument("--execute-paid", action="store_true")
    inspect_parser = commands.add_parser("inspect-parse")
    inspect_parser.add_argument("--config", type=Path, required=True)
    inspect_parser.add_argument("--documents", nargs="+")
    chunk_parser = commands.add_parser("chunk")
    chunk_parser.add_argument("--config", type=Path, required=True)
    chunk_parser.add_argument("--documents", nargs="+")
    inspect_chunks_parser = commands.add_parser("inspect-chunks")
    inspect_chunks_parser.add_argument("--config", type=Path, required=True)
    inspect_chunks_parser.add_argument("--document", required=True)
    inspect_chunks_parser.add_argument("--page", type=int)
    index_bm25_parser = commands.add_parser("index-bm25")
    index_bm25_parser.add_argument("--config", type=Path, required=True)
    embed_parser = commands.add_parser("embed")
    embed_parser.add_argument("--config", type=Path, required=True)
    embed_parser.add_argument("--execute-paid", action="store_true")
    args = parser.parse_args(argv)

    try:
        match args.command:
            case "parse":
                return _run_parse(args)
            case "inspect-parse":
                return _run_inspect_parse(args)
            case "chunk":
                return _run_chunk(args)
            case "inspect-chunks":
                return _run_inspect_chunks(args)
            case "index-bm25":
                return _run_index_bm25(args)
            case "embed":
                return _run_embed(args)
            case _:
                raise ValueError(f"Unknown SEC RAG command: {args.command}")
    except (OSError, RuntimeError, ValueError, ParseStateError) as error:
        print(f"Error: {error}")
        return 1


def _run_parse(args: argparse.Namespace) -> int:
    """Load parser settings, run the parse report (or paid parse), and print counts."""
    config = load_config(args.config)
    names = tuple(args.documents) if args.documents else None
    result = parse_corpus(
        config,
        names,
        execute_paid=args.execute_paid,
    )
    print(f"Selected: {result['selected']}")
    print(f"Done: {len(result['done'])}")
    print(f"Missing: {len(result['missing'])}")
    print(f"Parsed now: {len(result['parsed'])}")
    return 0


def _run_inspect_parse(args: argparse.Namespace) -> int:
    """Load settings, render selected cache reports, and print their count."""
    config = load_config(args.config)
    names = tuple(args.documents) if args.documents else None
    result = inspect_parses(config, names)
    print(f"Inspected: {result['inspected']}")
    return 0


def _run_chunk(args: argparse.Namespace) -> int:
    """Load settings, rebuild the selected chunk files, and print the summary."""
    config = load_config(args.config)
    names = tuple(args.documents) if args.documents else None
    summary = build_chunks(config, names)
    kinds = summary["kinds"]
    print(f"Filings: {summary['filings']}")
    print(
        f"Chunks: {summary['chunks']} (prose {kinds['prose']}, "
        f"table {kinds['table']}, figure {kinds['figure']})"
    )
    print(f"Median prose tokens: {summary['median_prose_tokens']}")
    print(f"Prose between floor and ceiling: {summary['prose_in_range_share']:.1%}")
    print(f"Prose over ceiling (carried text): {summary['prose_over_ceiling']}")
    print(f"Prose under floor: {summary['prose_under_floor']}")
    print(f"Empty figures skipped: {summary['empty_figures_skipped']}")
    return 0


def _run_inspect_chunks(args: argparse.Namespace) -> int:
    """Print one page's chunks, or write the whole filing's chunks to a report file.

    `--page` is the page index counted from 0, so a question's
    evidence_page_num can be pasted straight in.
    """
    config = load_config(args.config)
    doc_name = args.document.removesuffix(".pdf")
    if args.page is None:
        path = write_chunk_report(config, doc_name)
        print(f"Wrote: {path}")
    else:
        print(format_page_chunks(read_chunks(config, doc_name), args.page))
    return 0


def _run_index_bm25(args: argparse.Namespace) -> int:
    """Load settings, rebuild the BM25 index over every chunk file, and print counts."""
    config = load_config(args.config)
    summary = build_bm25_index(config)
    print(f"Filings: {summary['filings']}")
    print(f"Chunks indexed: {summary['chunks']}")
    return 0


def _run_embed(args: argparse.Namespace) -> int:
    """Load settings, compare chunks with Chroma, and print the plan (or embed, if paid).

    Without --execute-paid this only reports what a paid run would send.
    """
    config = load_config(args.config)
    result = embed_corpus(config, execute_paid=args.execute_paid)
    print(f"Chunks: {result['chunks']}")
    print(f"Already stored: {result['stored']}")
    print(f"To embed: {result['to_embed']}")
    print(f"To delete: {result['to_delete']}")
    print(f"Estimated tokens: {result['estimated_tokens']:,}")
    print(f"Batches: {result['batches']}")
    if "embedded" in result:
        print(f"Embedded now: {result['embedded']}")
        print(f"Tokens billed: {result['tokens_billed']:,}")
    return 0
