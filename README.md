# SEC RAG benchmark

This repository contains a plain-Python benchmark harness for evaluating RAG
over financial filings. The current implementation prepares and validates the
open-source FinanceBench 10-K subset: **112 questions across 64 PDFs**.

The harness currently supports:

- `closed_book`: question only;
- `oracle`: FinanceBench gold evidence pages;
- `long_context`: the complete relevant filing;
- no-spend dry runs;
- fixed 10-question smoke and 50-question pattern development subsets;
- append-only answer checkpointing and resumption;
- page recall, page precision and page MRR;
- reports segmented by condition, question type and cognitive skill.

`single_store` and `shared_store` are defined but require the future retriever.
The Azure DeepSeek binary answer judge is implemented, paid-smoke tested and
validated against published labels. HiREC/LOFin support remains deferred.

## Project structure

```text
configs/financebench.toml          editable dataset, generation and run settings
src/sec_rag_benchmark/
├── config.py                      load and validate benchmark settings
├── dataset/                       preparation and development subsets
├── pipeline/                      context construction and generation
├── execution/                     preflight, one-job execution and run loop
├── evaluation/                    metrics, judging and manual review
├── reporting/                     aggregation, diagnosis and workbook output
└── cli.py                         terminal commands and orchestration
tests/                             no-spend automated tests
data/financebench/                 generated 10-K subset; ignored by Git
results/                           generated benchmark runs; ignored by Git
benchmarks/financebench/           read-only original FinanceBench clone
```

## Prerequisites

- Python 3.12;
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/);
- Git;
- an OpenRouter API key only when running paid generation.

Install `uv` if it is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
```

## One-time setup

From the repository root, clone FinanceBench if
`benchmarks/financebench/` does not already exist:

```bash
git clone https://github.com/patronus-ai/financebench.git benchmarks/financebench
git -C benchmarks/financebench checkout cc39aeb4afdf33909ee1412188bf89035950c2eb
```

The pinned clone supplies the source questions, PDFs and published
human-labelled model results used to validate the answer judge. It remains
ignored by this repository and can be recreated on another machine.

Create the local Python environment and install the exact locked dependencies:

```bash
uv sync --locked
```

Before claiming a change is complete, run the full local quality suite:

```bash
uv run ruff format --check src tests
uv run ruff check src tests
uv run mypy src
uv run pytest -q
uv lock --check
git diff --check
```

If the formatting check fails, apply the formatter with
`uv run ruff format src tests`, then rerun the suite. GitHub Actions repeats
these no-spend checks on every push and pull request.

`pyproject.toml` contains the direct dependency pins and defines the
`sec-rag-benchmark` command. `uv.lock` pins the complete dependency graph;
`uv sync` recreates it inside the ignored `.venv/` directory.

## Prepare and validate FinanceBench

These commands make no model requests and spend no API credit:

```bash
uv run sec-rag-benchmark prepare --config configs/financebench.toml
uv run sec-rag-benchmark validate --config configs/financebench.toml
```

Preparation filters the original source files to 10-Ks and creates the local
112-question/64-PDF subset under `data/financebench/`. Validation checks the
questions, metadata, evidence pages, PDFs and manifest.

## Run a no-spend preflight

Test the three currently executable conditions without creating an API client
or spending credit:

```bash
uv run sec-rag-benchmark run --config configs/financebench.toml --dry-run
```

For a quick five-question check:

```bash
uv run sec-rag-benchmark run \
  --config configs/financebench.toml \
  --conditions closed_book oracle long_context \
  --limit 5 \
  --dry-run
```

For repeatable development checks, use the named subsets. These preserve the
FinanceBench `question_type` proportions and use the same question IDs on every
machine:

```bash
uv run sec-rag-benchmark run \
  --config configs/financebench.toml \
  --subset smoke \
  --dry-run

uv run sec-rag-benchmark run \
  --config configs/financebench.toml \
  --subset pattern \
  --dry-run
```

Remove `--dry-run` only when you intend to make paid generation requests.

## Run paid generation

Create an ignored `.env` file in the repository root:

```text
OPENROUTER_API_KEY=your-key-here
```

Load the file into the current terminal shell:

```bash
set -a
source .env
set +a
```

The following command **sends paid OpenRouter requests**:

```bash
uv run sec-rag-benchmark run \
  --config configs/financebench.toml \
  --conditions closed_book oracle long_context
```

The CLI creates a labelled directory such as
`results/20260913-143052--financebench--baseline-context-conditions-v1/` and
prints its path. Keep the path if you need to resume or report the run.

## Resume and report a run

Resume an interrupted run by passing its existing directory and the same
conditions. Successful job IDs are skipped:

```bash
uv run sec-rag-benchmark run \
  --config configs/financebench.toml \
  --conditions closed_book oracle long_context \
  --run-dir results/20260913-143052--financebench--baseline-context-conditions-v1
```

Create or refresh the reports without calling a model:

```bash
uv run sec-rag-benchmark report \
  --run-dir results/20260913-143052--financebench--baseline-context-conditions-v1
```

If the two judge passes disagree, follow the manual-adjudication workflow in
`Benchmark.md`. The run ID is the name of its subfolder under `results/`.
Export the review CSV once:

```bash
uv run sec-rag-benchmark export-manual-review --run-dir results/<run-id>
```

Open `results/<run-id>/manual_review.csv`, enter `1` for a correct answer or
`0` for an incorrect answer in `human_accuracy`, and leave `review_reason`
blank unless a note is useful. Then import the completed decisions:

```bash
uv run sec-rag-benchmark import-manual-review --run-dir results/<run-id>
```

After importing completed reviews, regenerate `summary.json` and `summary.xlsx`:

```bash
uv run sec-rag-benchmark report --run-dir results/<run-id>
```

Each run directory contains:

```text
results/<run-id>/
├── config.toml        effective configuration for this run
├── predictions.jsonl successful answers and terminal did_not_fit outcomes
├── judgments.jsonl   completed two-pass answer judgments, when present
├── manual_review.csv editable disagreements exported for human review
├── manual_reviews.jsonl validated append-only human decisions
├── errors.jsonl      retryable generation or judge failures, when present
├── summary.json      machine-readable metrics and completion status
└── summary.xlsx      formatted overview, answer and retrieval sheets
```

To see all available options:

```bash
uv run sec-rag-benchmark --help
uv run sec-rag-benchmark run --help
```

Judge one successful answer first and inspect `judgments.jsonl` before a full
run:

```bash
uv run sec-rag-benchmark judge \
  --config configs/financebench.toml \
  --run-dir results/<one-answer-run>
```

This sends two paid Azure requests for each previously unjudged answer. Azure
setup and troubleshooting are documented in
[`Azure Judge Setup.md`](docs%20sys%20design/benchmark/Azure%20Judge%20Setup.md).

Before judging the full benchmark, validate the judge against the fixed
published human-labelled sample. This makes at most 60 paid Azure requests:

```bash
uv run sec-rag-benchmark validate-judge \
  --config configs/financebench.toml
```

The command prints its resumable directory. Inspect `judge_validation.json`
for the overall result, per-label agreement and mismatches requiring review.
It exits with status 1 when the completed sample misses the required agreement
threshold, so scripted workflows stop at the validation gate.
When resuming, pass only that validation directory to `--run-dir`; the command
refuses to initialise validation inside a non-empty benchmark run directory.

## Detailed documentation

- [`docs sys design/Benchmark.md`](docs%20sys%20design/Benchmark.md) contains the
  benchmark requirements and methodology.
- [`FinanceBench Implementation Guide.md`](docs%20sys%20design/benchmark/FinanceBench%20Implementation%20Guide.md)
  explains the architecture, data shapes, pseudocode, code-reading order and
  complete terminal workflow.
- [`docs sys design/Benchmark Progress.md`](docs%20sys%20design/Benchmark%20Progress.md)
  records implemented and deferred work.
