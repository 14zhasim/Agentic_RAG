"""Translate terminal commands into calls to the benchmark modules."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .data import DataError, prepare, validate
from .execution.preflight import dry_run
from .execution.runner import run_benchmark
from .judge import judge_run
from .reporting import write_report


def main(argv: list[str] | None = None) -> int:
    """Parse one command, delegate its work, and print a short result."""
    parser = argparse.ArgumentParser(prog="sec-rag-benchmark")
    commands = parser.add_subparsers(dest="command", required=True)

    for command_name in ("prepare", "validate"):
        command = commands.add_parser(command_name)
        command.add_argument("--config", type=Path, required=True)

    run_parser = commands.add_parser("run")
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument("--conditions", nargs="+")
    run_parser.add_argument("--run-dir", type=Path)
    run_parser.add_argument("--limit", type=int)
    run_parser.add_argument("--dry-run", action="store_true")

    report_parser = commands.add_parser("report")
    report_parser.add_argument("--run-dir", type=Path, required=True)
    judge_parser = commands.add_parser("judge")
    judge_parser.add_argument("--config", type=Path, required=True)
    judge_parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        match args.command:
            case "prepare":
                config = load_config(args.config)
                counts = prepare(config["dataset"])
                print(
                    f"Prepared {counts['questions']} questions and "
                    f"{counts['documents']} documents"
                )

            case "validate":
                config = load_config(args.config)
                counts = validate(config["dataset"])
                print(
                    f"Validated {counts['questions']} questions and "
                    f"{counts['documents']} documents"
                )

            case "run" if args.dry_run:
                config = load_config(args.config)
                result = dry_run(
                    config, conditions=args.conditions, limit=args.limit
                )
                print(f"Planned jobs: {result['planned_jobs']}")
                print(f"Conditions: {', '.join(result['conditions'])}")
                print(f"Maximum prompt tokens: {result['maximum_prompt_tokens']}")
                print(
                    "Jobs that would fail the context preflight: "
                    f"{result['oversized_jobs']}"
                )
                print(f"API requests sent: {result['api_requests']}")

            case "run":
                config = load_config(args.config)
                result = run_benchmark(
                    config,
                    args.config,
                    conditions=args.conditions,
                    limit=args.limit,
                    requested_run_dir=args.run_dir,
                )
                counts = {
                    key: result[key]
                    for key in ("generated", "did_not_fit", "skipped", "failed")
                }
                print(f"Run {result['run_dir']}: {counts}")

            case "report":
                summary = write_report(args.run_dir)
                print(
                    f"Reported {summary['run_status']['successful']} successful jobs"
                )

            case "judge":
                config = load_config(args.config)
                counts = judge_run(args.run_dir, config["judge"])
                print(f"Judged {args.run_dir}: {counts}")

        return 0
    except (OSError, ValueError, RuntimeError, DataError) as error:
        print(f"Error: {error}")
        return 2
