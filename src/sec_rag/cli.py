"""Translate SEC RAG terminal commands into ingestion operations."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .ingestion.azure import ParseStateError, parse_corpus


def main(argv: list[str] | None = None) -> int:
    """Parse one SEC RAG command, delegate it, and return its exit status."""
    parser = argparse.ArgumentParser(prog="sec-rag")
    commands = parser.add_subparsers(dest="command", required=True)
    parse_parser = commands.add_parser("parse")
    parse_parser.add_argument("--config", type=Path, required=True)
    parse_parser.add_argument("--documents", nargs="+")
    parse_parser.add_argument("--execute-paid", action="store_true")
    args = parser.parse_args(argv)

    try:
        match args.command:
            case "parse":
                return _run_parse(args)
            case _:
                raise ValueError(f"Unknown SEC RAG command: {args.command}")
    except (OSError, RuntimeError, ValueError, ParseStateError) as error:
        print(f"Error: {error}")
        return 1


def _run_parse(args: argparse.Namespace) -> int:
    """Load parser settings, run the selected parse action, and print counts."""
    config = load_config(args.config)
    names = tuple(args.documents) if args.documents else None
    result = parse_corpus(
        config,
        names,
        execute_paid=args.execute_paid,
    )
    counts = result["counts"]
    print(f"Selected: {result['selected']}")
    print(f"Complete: {counts['complete']}")
    print(f"Missing: {counts['missing']}")
    print(f"Invalid: {counts['invalid']}")
    print(f"Parsed: {counts['parsed']}")
    return 0
