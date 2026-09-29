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

- `single_store` and `shared_store`: Exp1's hybrid retriever
  (`sec-rag retrieve`) over the filing itself or all 64 filings.
The Azure DeepSeek binary answer judge is implemented, paid-smoke tested and
validated against published labels. HiREC/LOFin support remains deferred.

## Project structure

```text
configs/financebench.toml          editable dataset, generation and run settings
configs/sec_rag.toml               parser, chunker and index settings for the system being evaluated
configs/*-exp2.toml                Exp2 rung B: the two files above with only the structure change
src/sec_rag/                       the system being evaluated
├── ingestion/                     Azure parse run, heading list, inspection report
├── chunking/                      page-bounded chunks and chunk reports
├── indexing/                      BM25 keyword index, Voyage vectors in Chroma, Exp2 structure vectors
├── retrieval/                     query enhancement, hybrid search with RRF, rerank, Exp1 path
└── cli.py                         sec-rag terminal commands
src/sec_rag_benchmark/
├── config.py                      load and validate benchmark settings
├── dataset/                       preparation and development subsets
├── pipeline/                      context construction and generation
├── execution/                     preflight, one-job execution and run loop
├── evaluation/                    metrics, judging, manual review, Exp2 weight-zero check
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
- an OpenRouter API key only when running paid generation;
- Azure Document Intelligence and Voyage keys only for paid parsing and
  embedding.

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
questions, metadata, evidence pages, PDFs and the dataset preparation record.

## Parse and inspect filings

These commands belong to the SEC RAG system, not the benchmark harness. Each
filing is parsed once and saved as `data/financebench/parsed/<doc_name>.json`;
a filing already saved is never sent again.

Start with the no-spend plan. It reads local files only and prints how many
filings are selected, already parsed, missing, and parsed by this run:

```bash
uv run sec-rag parse --config configs/sec_rag.toml
uv run sec-rag parse --config configs/sec_rag.toml --documents PEPSICO_2022_10K JPMORGAN_2022_10K
```

```text
Selected: 64
Done: 1
Missing: 63
Parsed now: 0
```

Paid parsing (~$1.70 and a few minutes per filing) needs the two Azure
Document Intelligence variables in `.env`, loaded into the shell:

```text
AZURE_DOCUMENT_INTELLIGENCE_KEY=your-key-here
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
```

```bash
set -a
source .env
set +a
```

Parse a small named batch first. `--execute-paid` is what makes a command
spend; without it nothing is sent:

```bash
uv run sec-rag parse --config configs/sec_rag.toml \
  --documents PEPSICO_2022_10K JPMORGAN_2022_10K --execute-paid
```

Filings are sent one at a time, and the run stops at the first error; anything
saved before it stays saved, so rerunning the same command continues from
there. A saved file that fails the page-count check also stops the run, before
anything is spent.

Inspect the saved results before authorising the rest of the corpus. This is
free and offline: it writes `data/financebench/parsed/<doc_name>.structure.txt`
with each filing's headings, each page's heading path, and a figure count.
Leave out `--documents` to inspect every parsed filing.

```bash
uv run sec-rag inspect-parse --config configs/sec_rag.toml \
  --documents 3M_2018_10K PEPSICO_2022_10K JPMORGAN_2022_10K
```

Only after reviewing the reports, parse the remaining filings (~$104 for 61):

```bash
uv run sec-rag parse --config configs/sec_rag.toml --execute-paid
```

Azure parsing and benchmark generation are separate paid operations. The
benchmark dry runs below neither parse documents nor call a generation model.

## Chunk and index filings

Chunking and the keyword index are free and offline. Chunking reads the saved
parses and writes `data/financebench/chunks/<doc_name>.jsonl`; leave out
`--documents` to chunk every parsed filing:

```bash
uv run sec-rag chunk --config configs/sec_rag.toml
uv run sec-rag chunk --config configs/sec_rag.toml --documents 3M_2018_10K
```

Check chunks by eye. `--page` is the page index counted from 0, so a
question's `evidence_page_num` can be pasted in; without it, the whole filing
is written to `data/financebench/chunks/<doc_name>.chunks.txt`:

```bash
uv run sec-rag inspect-chunks --config configs/sec_rag.toml --document 3M_2018_10K --page 59
uv run sec-rag inspect-chunks --config configs/sec_rag.toml --document 3M_2018_10K
```

Build the BM25 keyword index over every chunk (a few seconds; always a full
rebuild) into `data/financebench/indexes/bm25/`:

```bash
uv run sec-rag index-bm25 --config configs/sec_rag.toml
```

Embeddings are made by Voyage and stored in Chroma under
`data/financebench/indexes/chroma/`. Start with the no-spend report, which
needs no key and prints how many chunks would be embedded, the estimated
tokens and the number of Voyage calls:

```bash
uv run sec-rag embed --config configs/sec_rag.toml
```

The paid run needs `VOYAGE_API_KEY` in `.env`, loaded into the shell as above.
The Voyage account needs a payment method: without one its limits are 3
requests and 10,000 tokens a minute, too low for a single batch. The first
200M tokens are free, and the whole corpus is about 10M.

```bash
set -a
source .env
set +a
uv run sec-rag embed --config configs/sec_rag.toml --execute-paid
```

Only new or changed chunks are sent, and each batch is saved as it returns,
so an interrupted run continues where it stopped when rerun. Afterwards the
no-spend report should show `To embed: 0`.

## Build structure vectors (Exp2)

Exp2 scores each chunk by its text and by the headings it sits under. This
gives every chunk a heading path, embeds each unique heading once with
Voyage, and saves one structure vector per chunk beside the chunk vectors in
Chroma. The chunk vectors must be embedded first (above). Start with the
no-spend report: it needs no key and prints how many headings would be
embedded and the estimated tokens:

```bash
uv run sec-rag index-structure --config configs/sec_rag.toml
```

The paid run (about 9,500 headings, ~99k tokens, inside Voyage's free
allowance) needs `VOYAGE_API_KEY` loaded from `.env` as above:

```bash
uv run sec-rag index-structure --config configs/sec_rag.toml --execute-paid
```

As with `embed`, only headings not yet stored are sent, and an interrupted
run continues when rerun. Once every heading is stored, the command without
`--execute-paid` is still free but not read-only: it rebuilds the structure
vectors locally, which is how a change to `[structure]` in
`configs/sec_rag.toml` is applied. What each output line should read is in
`docs sys design/Exp 2/Implementation Guide 3.1-3.4 Structure.md` → Slice 1.

## Retrieve for one question (Exp1)

Check retrieval for one question before a benchmark run. `--scope` names the
filing(s) the question may search (single-store: its own filing);
`--all-filings` searches the whole corpus (shared-store). Without
`--execute-paid`, only BM25 runs over the raw question: no key, no spend.
`--top-k` sets how many chunks are shown (default 10, the benchmark's
retrieval depth). Page numbers are 0-based, as in FinanceBench's evidence.

```bash
uv run sec-rag retrieve --config configs/sec_rag.toml \
  --question "What is the FY2018 capital expenditure amount (in USD millions) for 3M?" \
  --all-filings

uv run sec-rag retrieve --config configs/sec_rag.toml \
  --question "What is the FY2018 capital expenditure amount (in USD millions) for 3M?" \
  --scope 3M_2018_10K --top-k 20
```

`--execute-paid` runs Exp1's whole path: GLM's query enhancement
(OpenRouter), the hybrid search (one Voyage query embedding) and the Voyage
reranker, a fraction of a penny per question. It prints the chosen filing,
both queries, the fused and reranked top 10 and the token counts, and needs
`OPENROUTER_API_KEY` and `VOYAGE_API_KEY` loaded from `.env` as above.

```bash
uv run sec-rag retrieve --config configs/sec_rag.toml \
  --question "What is the FY2018 capital expenditure amount (in USD millions) for 3M?" \
  --all-filings --execute-paid
```

## Run a no-spend preflight

Plan a run without creating an API client or spending credit. The default
`[run]` in `configs/financebench.toml` is Exp1 (`experiment = "exp1"`,
`variant = "full"`, conditions `single_store` and `shared_store`, retrieval
settings from `configs/sec_rag.toml`):

```bash
uv run sec-rag-benchmark run --config configs/financebench.toml --dry-run
```

The retrieval conditions' prompt sizes show as `requires retriever`, because
the chunks are unknown until the paid search runs.

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
VOYAGE_API_KEY=your-key-here
```

(`VOYAGE_API_KEY` is needed only for `single_store` and `shared_store`.)

Load the file into the current terminal shell:

```bash
set -a
source .env
set +a
```

The following commands **send paid OpenRouter and Voyage requests**. They run
Exp1 on the 10-question smoke subset, then on all 112 questions:

```bash
uv run sec-rag-benchmark run --config configs/financebench.toml --subset smoke
uv run sec-rag-benchmark run --config configs/financebench.toml
```

The three baseline conditions (paid OpenRouter only) are run by naming them:

```bash
uv run sec-rag-benchmark run \
  --config configs/financebench.toml \
  --conditions closed_book oracle long_context
```

The CLI creates a labelled directory such as
`results/20260928-143052--exp1--full/` (`--experiment--variant` from
`[run]`) and
prints its path. Keep the path if you need to resume or report the run.

## Run Exp2 (structure-aware retrieval)

Exp2's rung B is Exp1 with one change: the semantic score adds each chunk's
structure vector. Its settings are in `configs/financebench-exp2.toml` and
`configs/sec_rag-exp2.toml`, copies of Exp1's with only the run label, the
structure weight (1) and the reused query plans changed. Rung B reuses the
filing and queries Exp1's full run chose, so it makes no query-enhancement
calls. Rung A is that Exp1 run, not re-run. Build the structure vectors
first (above).

Before running B, check that the exact scorer reproduces Exp1 when the
structure weight is 0. The free report counts the searches; the paid run
re-embeds Exp1's 224 semantic queries (~7k Voyage tokens, inside the free
allowance) and exits with status 1 if the check fails:

```bash
uv run sec-rag-benchmark check-structure --config configs/financebench-exp2.toml \
  --baseline-run-dir results/20260928-202531--exp1--full
uv run sec-rag-benchmark check-structure --config configs/financebench-exp2.toml \
  --baseline-run-dir results/20260928-202531--exp1--full --execute-paid
```

It writes `results/<timestamp>--exp2--weight-zero-check/`. Then run, judge
and report B as for Exp1, with the Exp2 config (paid). The smoke run gets
its own folder, since a folder's snapshot records its question selection:

```bash
uv run sec-rag-benchmark run --config configs/financebench-exp2.toml --dry-run
uv run sec-rag-benchmark run --config configs/financebench-exp2.toml --limit 5
uv run sec-rag-benchmark run --config configs/financebench-exp2.toml
```

What each command should print, and the pass rule for the check, are in
`docs sys design/Exp 2/Implementation Guide 3.1-3.4 Structure.md` → Slices
3 and 4.

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

An Exp1 run holds only `single_store` and `shared_store`, so its failure
diagnosis borrows the oracle answers and judgments from the 17 Sep baseline
run. `--oracle-run-dir` names that folder; it is only read, never written.
Without it, every incorrect retrieval answer is reported as `missing_oracle`.
The report refuses the borrow if the run has its own oracle answers or if the
two runs' `[generation]` settings differ.

```bash
uv run sec-rag-benchmark report \
  --run-dir results/<exp1-run-id> \
  --oracle-run-dir results/20260917-012959--financebench--baseline-context-conditions-v1
```

If the two judge passes disagree, follow the manual-adjudication workflow in
`Benchmark.md`. The run ID is the name of its subfolder under `results/`.
Export the review CSV once:

```bash
uv run sec-rag-benchmark export-manual-review --run-dir results/<run-id>
```

It refuses to replace an existing `manual_review.csv`, so typed decisions
are never lost by accident. Add `--overwrite` only to start the CSV again.

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
