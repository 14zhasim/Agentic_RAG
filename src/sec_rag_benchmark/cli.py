"""Small command-line interface for the FinanceBench golden path."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
from pathlib import Path
import tomllib
from typing import Any

from .conditions import CONDITIONS, build_condition
from .data import DataError, load_questions, prepare, validate
from .generation import build_messages, count_prompt_tokens
from .metrics import write_report
from .runner import run


def load_config(path: Path) -> dict[str, Any]:
    """Read TOML and resolve relative paths against the repository root."""
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    root = path.resolve().parent.parent
    for section, key in (("dataset", "source_dir"), ("dataset", "output_dir"), ("run", "results_dir")):
        value = Path(raw[section][key])
        raw[section][key] = str(value if value.is_absolute() else root / value)
    conditions = raw["run"]["conditions"]
    if not conditions or set(conditions) - CONDITIONS:
        raise ValueError("Configuration contains missing or unknown conditions")
    generation = raw["generation"]
    if generation["provider"] != "openrouter" or generation["allow_fallbacks"] is not False:
        raise ValueError("Golden-path generation requires OpenRouter with fallbacks disabled")
    if generation["reasoning_effort"] not in {"low", "high", "max"}:
        raise ValueError("reasoning_effort must be low, high, or max for GLM-5.3-Flash")
    if min(generation["context_window_tokens"], generation["max_output_tokens"], generation["token_safety_margin"]) < 0:
        raise ValueError("Token limits and margins cannot be negative")
    if generation["max_output_tokens"] + generation["token_safety_margin"] >= generation["context_window_tokens"]:
        raise ValueError("Output tokens and safety margin must leave room for the input prompt")
    return raw


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sec-rag-benchmark")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "validate"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument("--conditions", nargs="+")
    run_parser.add_argument("--run-dir", type=Path)
    run_parser.add_argument("--limit", type=int)
    run_parser.add_argument("--dry-run", action="store_true")
    report = commands.add_parser("report")
    report.add_argument("--run-dir", type=Path, required=True)
    judge = commands.add_parser("judge")
    judge.add_argument("--run-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Expose each stage of the guide's benchmark flow as a separate command."""
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        # Reporting reads already checkpointed jobs; it never regenerates them.
        if args.command == "report":
            summary = write_report(args.run_dir)
            print(f"Reported {summary['successful']} successful jobs")
            return 0
        if args.command == "judge":
            parser.error("Azure/RAGAS judge is not implemented yet")

        config = load_config(args.config)
        if args.command == "prepare":
            counts = prepare(config["dataset"])
            print(f"Prepared {counts['questions']} questions and {counts['documents']} documents")
            return 0
        if args.command == "validate":
            counts = validate(config["dataset"])
            print(f"Validated {counts['questions']} questions and {counts['documents']} documents")
            return 0

        validate(config["dataset"])
        questions = load_questions(config["dataset"]["output_dir"])
        if args.limit is not None:
            if args.limit <= 0:
                parser.error("--limit must be positive")
            questions = questions[:args.limit]
        selected_conditions = args.conditions or config["run"]["conditions"]
        if set(selected_conditions) - CONDITIONS:
            parser.error("Unknown condition")
        docs = tuple(dict.fromkeys(question["doc_name"] for question in questions))
        pdf_dir = Path(config["dataset"]["output_dir"]) / "pdfs"

        if args.dry_run:
            # Exercise dataset and condition assembly, including context-size
            # preflight, without constructing a client or spending API credit.
            prompt_tokens: dict[str, int | str] = {}
            oversized = 0
            for name in selected_conditions:
                if name in {"single_store", "shared_store"}:
                    prompt_tokens[name] = "requires retriever"
                    continue
                condition_token_counts = [
                    count_prompt_tokens(
                        build_messages(item["question"], build_condition(item, name, pdf_dir, docs)["context"]),
                        config["generation"]["reasoning_effort"],
                    )
                    for item in questions
                ]
                prompt_tokens[name] = max(condition_token_counts)
                reserved = config["generation"]["max_output_tokens"] + config["generation"]["token_safety_margin"]
                oversized += sum(
                    count + reserved > config["generation"]["context_window_tokens"]
                    for count in condition_token_counts
                )
            print(f"Planned jobs: {len(questions) * len(selected_conditions)}")
            print(f"Conditions: {', '.join(selected_conditions)}")
            print(f"Maximum prompt tokens: {prompt_tokens}")
            print(f"Jobs that would fail the context preflight: {oversized}")
            print("API requests sent: 0")
            return 0

        run_dir = args.run_dir or Path(config["run"]["results_dir"]) / datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)
        # Append the runtime selection to the copied TOML so a resume cannot
        # silently change --conditions or --limit while reusing old answers.
        selection = (
            "\n[selection]\n"
            f"conditions = [{', '.join(repr(name) for name in selected_conditions)}]\n"
            f"limit = {args.limit if args.limit is not None else config['dataset']['expected_questions']}\n"
        )
        config_bytes = args.config.read_bytes() + selection.encode()
        snapshot = run_dir / "config.toml"
        if snapshot.exists() and snapshot.read_bytes() != config_bytes:
            raise ValueError("Run directory contains a different configuration")
        snapshot.write_bytes(config_bytes)
        run_key = hashlib.sha256(config_bytes).hexdigest()[:12]
        counts = run(
            questions,
            selected_conditions,
            pdf_dir=pdf_dir,
            all_doc_names=docs,
            generation_config=config["generation"],
            run_dir=run_dir,
            retrieval_depth=config["run"]["retrieval_depth"],
            run_key=run_key,
        )
        print(f"Run {run_dir}: {counts}")
        return 0
    except (OSError, ValueError, RuntimeError, DataError) as error:
        print(f"Error: {error}")
        return 2
