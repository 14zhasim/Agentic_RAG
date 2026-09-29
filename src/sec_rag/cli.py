"""Translate SEC RAG terminal commands into ingestion, indexing and retrieval operations."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .chunking.chunk_files import (
    build_chunks,
    format_page_chunks,
    read_chunks,
    write_chunk_report,
)
from .config import load_config
from .indexing.bm25_index import build_bm25_index
from .indexing.embed import embed_corpus
from .indexing.structure import build_structure_index
from .ingestion.inspect_parse import inspect_parses
from .ingestion.parse import ParseStateError, parse_corpus
from .retrieval.exp1 import all_filings, open_exp1, preview_bm25, retrieve_exp1


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
    structure_parser = commands.add_parser("index-structure")
    structure_parser.add_argument("--config", type=Path, required=True)
    structure_parser.add_argument("--execute-paid", action="store_true")
    retrieve_parser = commands.add_parser("retrieve")
    retrieve_parser.add_argument("--config", type=Path, required=True)
    retrieve_parser.add_argument("--question", required=True)
    scope_group = retrieve_parser.add_mutually_exclusive_group(required=True)
    scope_group.add_argument("--scope", nargs="+")
    scope_group.add_argument("--all-filings", action="store_true")
    retrieve_parser.add_argument("--top-k", type=int, default=10)
    retrieve_parser.add_argument("--execute-paid", action="store_true")
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
            case "index-structure":
                return _run_index_structure(args)
            case "retrieve":
                return _run_retrieve(args)
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


def _run_index_structure(args: argparse.Namespace) -> int:
    """Load settings, attribute heading paths, and print the plan (or build, if paid).

    Without --execute-paid nothing is sent to Voyage; if every heading is
    already embedded, the structure vectors are still rebuilt, locally.
    """
    config = load_config(args.config)
    result = build_structure_index(config, execute_paid=args.execute_paid)
    print(f"Filings: {result['filings']}")
    print(
        f"Chunks: {result['chunks']} ({result['chunks_with_path']} with a heading path)"
    )
    print(f"Unique headings: {result['unique_headings']}")
    print(f"Already embedded: {result['stored_headings']}")
    print(f"To embed: {result['to_embed']}")
    print(f"Estimated tokens: {result['estimated_tokens']:,}")
    if "structure_rows" in result:
        print(f"Embedded now: {result['embedded']}")
        print(f"Tokens billed: {result['tokens_billed']:,}")
        print(f"Structure vectors saved: {result['structure_rows']}")
    else:
        print("Nothing sent: rerun with --execute-paid to embed the missing headings")
    return 0


def _run_retrieve(args: argparse.Namespace) -> int:
    """Retrieve for one question: a free BM25 preview, or Exp1's paid path.

    `--scope` names the filings the question may search (single-store: one);
    `--all-filings` is the shared-store scope. Without --execute-paid only
    BM25 runs over the raw question, so no key is read and nothing is spent.
    """
    config = load_config(args.config)
    if args.all_filings:
        scope = all_filings(config)
    else:
        scope = tuple(name.removesuffix(".pdf") for name in args.scope)

    if not args.execute_paid:
        print("BM25 preview over the raw question (no paid calls):")
        _print_chunks(preview_bm25(args.question, scope, args.top_k, config))
        return 0

    bundle = retrieve_exp1(args.question, scope, args.top_k, open_exp1(config))
    plan = bundle["search_plan"]
    print(f"Enhancement: {plan['enhancement_status']}")
    print(f"Filter: {bundle['filter_doc_name']} ({bundle['filter_status']})")
    print(f"Keyword query: {plan['keyword_query']}")
    print(f"Semantic query: {plan['semantic_query']}")
    print("Fused, before rerank:")
    _print_chunks(bundle["pre_rerank_chunks"])
    print("Reranked:")
    _print_chunks(bundle["chunks"])
    usage = bundle["usage"]
    print(f"Embedding tokens: {usage['embedding_tokens']:,}")
    print(f"Rerank tokens: {usage['rerank_tokens']:,}")
    print(f"Query-enhancement cost: {plan['call']['cost']}")
    print(f"Latency: {bundle['latency_seconds']:.1f}s")
    return 0


def _print_chunks(chunks: list[dict[str, Any]]) -> None:
    """One line per chunk: rank, filing, page index (0-based) and score."""
    for chunk in chunks:
        print(
            f"  {chunk['rank']:>2}. {chunk['doc_name']} page {chunk['pages'][0]}"
            f"  score {chunk['score']:.4f}  {chunk['chunk_id']}"
        )
