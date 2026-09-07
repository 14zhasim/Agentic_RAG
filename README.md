# Agentic RAG for Financial Statement Analysis

This repository contains the benchmark harness and experimental retrieval systems for evaluating agentic RAG over financial filings. The first implementation target is the open-source FinanceBench 10-K subset: 112 questions across 64 PDFs.

The benchmark harness is plain Python rather than a notebook. This keeps configuration explicit, makes runs resumable, and allows the retrieval implementation to change without rewriting data preparation, generation, judging, metrics, or reporting.

## Planned project structure

The Python code lives inside the named `finrag_benchmark` package. A named package avoids ambiguous imports such as `from data import ...` and keeps project code separate from generated data and cloned reference repositories.

```text
pyproject.toml
uv.lock
.python-version
.env.example

configs/
└── financebench.toml

src/
└── finrag_benchmark/
    ├── cli.py
    ├── runner.py
    ├── data/
    │   └── financebench.py
    ├── generation/
    │   ├── models.py
    │   └── prompts.py
    ├── conditions/
    │   └── financebench.py
    ├── retrieval/
    │   ├── base.py
    │   └── placeholder.py
    └── evaluation/
        ├── retrieval_metrics.py
        ├── numeric_scorer.py
        ├── judge.py
        └── reporting.py

data/
└── financebench/
    ├── financebench_open_source_10k.jsonl
    ├── financebench_document_information_10k.jsonl
    ├── manifest.json
    └── pdfs/

vectorstores/
results/
benchmarks/
```

The major boundaries are:

- `data`: prepares, validates, and loads FinanceBench while preserving the schemas of its two source JSONL files.
- `conditions`: constructs the context for each of the five FinanceBench conditions.
- `retrieval`: defines the interface that baseline and future dissertation RAG implementations must satisfy.
- `generation`: handles OpenRouter/GLM model calls and shared answer prompts.
- `evaluation`: calculates retrieval metrics, numeric and LLM-judged answer scores, and summaries.
- `runner`: coordinates jobs, retries, checkpointing, and resumption.
- `cli`: exposes the commands run through `uv`.
- `configs`: contains public run configuration such as model identifiers, conditions, retrieval depth, chunk settings, and prompt versions.
- `data/financebench`: contains the deterministically prepared local dataset and selected PDFs. These generated files are not committed.
- `vectorstores`: contains rebuildable retrieval indexes and is not committed.
- `results`: contains separate, reproducible benchmark run directories.
- `benchmarks`: contains read-only clones of the original benchmark repositories. New project code does not go here.

The structure above describes the approved target architecture. See `docs sys design/Benchmark Progress.md` for what has and has not been implemented.

## How the benchmark works

```text
prepare data
    ↓
validate 112 questions and 64 PDFs
    ↓
construct condition inputs
    ↓
retrieve where applicable
    ↓
generate and checkpoint answers
    ↓
judge saved answers separately
    ↓
calculate and aggregate metrics
    ↓
write reports
```

The five FinanceBench conditions are:

- `closed_book`: the question without filing context.
- `oracle`: the complete gold evidence pages supplied by FinanceBench.
- `single_store`: retrieval restricted to the question's filing.
- `shared_store`: retrieval over all 64 selected 10-Ks.
- `long_context`: the complete relevant filing in page order.

All conditions use the same answer-generation stage after constructing their different inputs. The retrieval implementation is replaceable, so future RAG systems can be compared without changing the surrounding benchmark.

Generation and judging are deliberately separate:

- A judge outage cannot discard generated answers.
- The judge can be replaced or rerun without paying to regenerate answers.
- Azure judge setup can remain deferred while generation is developed.
- Closed-book, oracle, and long-context generation can run before the RAG retriever exists.

The generation target is GLM-5.3-Flash through OpenRouter. The planned final-answer judge is GPT-5.6 Luna through a Direct-from-Azure Microsoft Foundry deployment. Both integrations use separate clients built from the pinned OpenAI Python SDK.

## CLI responsibilities

The command-line interface will expose separate operations for:

- Preparing FinanceBench data
- Validating the prepared questions, metadata, evidence, and PDFs
- Running selected context conditions
- Judging previously generated answers
- Calculating and writing reports
- Performing a no-spend dry run

Commands that do not call a model will not require an API key. Until a retriever or judge is configured, their corresponding paid operations will fail explicitly rather than creating placeholder results that look real.

## Run artifacts

Every benchmark run receives a unique directory:

```text
results/<run-id>/
├── config.toml
├── predictions.jsonl
├── judgments.jsonl
├── summary.json
├── summary.csv
└── errors.jsonl
```

The copied `config.toml` makes the run self-describing. It records model identifiers, prompts, provider selection, retrieval depth, chunk settings, context and output limits, and other hyperparameters.

Predictions are saved after each question. Interrupted runs can resume without repeating successful calls. Judgments are saved separately from predictions. Transient throttling, timeout, and server errors are retried; permanent configuration and authentication failures stop clearly and are recorded.

## Dependency management

The project uses `uv`:

- `.python-version` pins Python.
- `pyproject.toml` records exact direct dependency versions.
- `uv.lock` pins the complete transitive dependency graph.
- Project commands run through `uv run`.
- LangChain and Chroma are not installed merely because the original notebook used them. They are added only if the selected retrieval implementation requires them.
- The pinned `openai` SDK can serve both the OpenRouter generator and Azure Foundry judge clients.

## Credentials and local configuration

Credentials live in an ignored `.env` file. The committed `.env.example` contains variable names with blank values and is safe to copy locally.

Expected credentials include:

```dotenv
OPENROUTER_API_KEY=
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_ENDPOINT=
```

VS Code can inject these values into newly created integrated terminals with:

```json
{
  "python.terminal.useEnvFile": true,
  "python.envFile": "${workspaceFolder}/.env"
}
```

Never commit `.env`, print secret values into logs, or store credentials in run artifacts.
