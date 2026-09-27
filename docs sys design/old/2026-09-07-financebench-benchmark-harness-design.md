# FinanceBench Benchmark Harness Design

> **Historical input:** This document records the first approved design but is
> no longer the current source of truth. Use `docs sys design/Benchmark.md` and
> `docs sys design/benchmark/FinanceBench Implementation Guide.md`.

Status: approved on 7 September 2026

## Purpose and scope

Build a reproducible, plain-Python evaluation harness for the open-source FinanceBench questions whose documents are classified as 10-Ks. The harness will reproduce FinanceBench's five context conditions, support replacement retrieval pipelines, calculate deterministic page-retrieval metrics, generate answers with GLM-5.3-Flash, and later judge those answers with GPT-5.6 Luna.

The initial dataset contains 112 questions across 64 PDFs. FinanceBench 10-Qs, 8-Ks, earnings documents, LOFin/HiREC, the final dissertation retriever, optional RAGAS metrics, and a comprehensive development test suite are intentionally deferred.

## 1. Provider architecture

Answer generation uses `z-ai/glm-5.3-flash` through OpenRouter. The Python integration uses the pinned OpenAI SDK with the OpenRouter base URL. The client is constructed lazily from `OPENROUTER_API_KEY`, so data, validation, metric, and report commands do not require generation credentials.

Final-answer judging will use `gpt-5.6-luna` through a Direct-from-Azure Microsoft Foundry deployment associated with the sponsored startup subscription. This route is preferred over Azure Databricks ADI because Direct-from-Azure models are eligible for Microsoft for Startups sponsorship credits. Azure deployment, quota, credentials, and a small billing-verification request are deferred until the judge is implemented.

Generation and judging are separate stages and use separate clients. The judge operates on saved predictions, so it can be replaced or rerun without regenerating answers.

For final experiment runs, select one upstream OpenRouter provider rather than allowing provider choice to vary silently. Record the requested model, returned model, serving provider, token usage, latency, and cost for each response.

## 2. FinanceBench data preparation

Treat `benchmarks/financebench/` as a read-only reference clone. A deterministic preparation command reads:

- `data/financebench_open_source.jsonl`
- `data/financebench_document_information.jsonl`
- the upstream `pdfs/` directory

It resolves document metadata by `doc_name`, normalizes `doc_type`, and selects exactly `doc_type == "10k"`. The result must contain 112 unique questions and 64 distinct PDFs.

Project-owned prepared data has this shape:

```text
data/
└── financebench/
    ├── financebench_open_source_10k.jsonl
    ├── financebench_document_information_10k.jsonl
    ├── manifest.json
    └── pdfs/
```

The two filtered JSONL files preserve the original files' keys and schemas. They are not flattened into one stored file; the project loader joins them in memory. In particular, the source question key remains `answer`, not `gold_answer`. Results may use `gold_answer` through an explicit mapping.

The manifest records source provenance, filter rules, expected and observed counts, selected filenames, and file hashes. Preparation is idempotent and must not produce duplicate records on repeated execution. Prepared data and PDFs are generated local artifacts and remain out of Git.

Validation checks:

- `financebench_id` is unique.
- Every selected question resolves to unambiguous document metadata.
- Every selected PDF exists.
- Evidence document names and zero-indexed page numbers are usable.
- The complete evidence list is retained for every question.
- Final counts are 112 questions and 64 PDFs.

The upstream metadata contains a duplicate `FOOTLOCKER_2023_annualreport` entry with conflicting periods. That document is not referenced by the open-source questions, but the loader must detect metadata conflicts rather than trust an unrestricted merge.

## 3. Evaluation conditions

Each of the 112 questions creates one job for each condition, for up to 560 generated answers:

- `closed_book`: the question without filing context.
- `oracle`: all unique gold evidence pages in dataset order.
- `single_store`: ranked chunks retrieved only from the question's filing.
- `shared_store`: ranked chunks retrieved from all 64 selected 10-Ks.
- `long_context`: the entire relevant filing in page order.

Condition builders construct context and provenance but do not call the model. A shared generator and answer prompt consume the resulting condition input. This holds the generation behavior constant while changing only the information made available to the model.

`single_store` and `shared_store` depend on a replaceable retriever contract. Until a retriever is configured, dry runs may enumerate these jobs, but paid execution fails explicitly rather than fabricating retrieval output. Closed-book, oracle, and long-context can run without a retriever.

Every retrieved chunk includes a stable chunk ID, text, `doc_name`, all zero-indexed pages covered by the chunk, retrieval score, and rank.

Long-context assembly measures the complete prompt using model-appropriate accounting, reserves output space, and checks the selected model's context limit. It never silently keeps only the beginning of a filing. An input that cannot fit produces an explicit context-limit failure.

## 4. Metrics and segmentation

For each question, the gold retrieval target is the set of unique `(doc_name, evidence_page_num)` pairs from its evidence. Upstream evidence objects use `doc_name`; the in-memory loader may also accept the legacy alias `evidence_doc_name`, but preparation must not add or rename nested fields. Convert ranked chunks into unique retrieved pages by retaining the earliest occurrence of each page. A page in the wrong document is not relevant even if its numeric page index matches. Multiple chunks from the same page therefore count as retrieving that page once for page recall and precision.

At the configured retrieval depth:

- Page recall is the number of retrieved gold pages divided by the number of gold pages.
- Page precision is the number of retrieved gold pages divided by the number of retrieved unique pages.
- Page MRR is the reciprocal rank of the first retrieved chunk whose document-aware page set intersects the gold page set, or zero when none is retrieved. Chunk rank is used because a multi-page chunk has one retrieval rank and imposing a second arbitrary order on its pages would distort MRR.

Store sufficient ranking information to calculate additional cut-offs without rerunning retrieval. These metrics apply only to `single_store` and `shared_store`; the remaining conditions report them as not applicable because they do not execute retrieval.

Preserve the raw `question_type` and `question_reasoning` values. `question_type` supplies the generation-method categories. Convert the inconsistent `question_reasoning` alternatives into a transparent multi-label `cognitive_skills` field containing any applicable values from:

- `information_extraction`
- `numerical_reasoning`
- `logical_reasoning`
- `unspecified`

A multi-label question contributes to each named skill summary, so skill counts are not additive. Reports state this and include the sample size for every aggregation.

Report macro-averaged results overall and by condition, generation method, cognitive skill, and the generation-method/cognitive-skill cross-tab.

## 5. Answer scoring

Numeric answers use a deterministic scorer that documents accepted parsing, percentage/comma handling, significant-figure rounding, truncation tolerance, and comparison rules.

The pointwise GPT judge receives the question, reference answer, candidate answer, and reference evidence. It does not receive the generation provider, model identity, context condition, or retrieval score. It returns structured output containing a binary correctness verdict, concise reason, and manual-review flag. Store the judge model/deployment, prompt version, reasoning configuration, parsed decision, and raw response.

Validate the judge against a human-reviewed sample before using its output as benchmark ground truth. Conflicts between deterministic numeric scoring and the LLM judge are flagged for review rather than silently reconciled.

## 6. Package and artifact structure

Use a named package to avoid ambiguous imports:

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

data/financebench/
vectorstores/
results/
```

`.python-version` pins Python, `pyproject.toml` records exact direct dependency versions, and `uv.lock` locks the complete dependency graph. LangChain, Chroma, and RAGAS are not installed unless an approved implementation actually uses them. The same pinned OpenAI SDK supports the OpenRouter generator and Azure Foundry judge clients.

Secrets live only in an ignored `.env`. The committed `.env.example` contains variable names without values.

## 7. Execution and artifacts

The staged flow is:

```text
prepare → validate → construct conditions → retrieve where applicable
        → generate → judge saved answers → aggregate → report
```

The CLI exposes separate operations for data preparation, validation, selected-condition generation, judging existing predictions, reporting, and a no-spend dry run.

Each run receives a unique directory:

```text
results/<run-id>/
├── config.toml
├── predictions.jsonl
├── judgments.jsonl
├── summary.json
├── summary.csv
└── errors.jsonl
```

The immutable configuration snapshot records dataset provenance, selected conditions, prompts, models, provider selection, output/context limits, retrieval depth, chunk settings, and other hyperparameters.

Every job has a stable identity derived from the run configuration, question ID, and condition. Persist each successful prediction immediately. Resume skips successful matching jobs and retries incomplete jobs without mixing incompatible configurations.

Retry transient throttling, timeouts, and server failures with bounded exponential backoff and jitter. Missing credentials, invalid configuration, unsupported placeholder execution, and authentication failures fail immediately. Retain failed-job records and report completion counts. Official comparisons require complete runs or an explicit incomplete-run label.

No artifact may contain an API key or other secret.

## 8. Testing scope

The first implementation includes two minimum correctness test groups:

- Data preparation produces 112 questions and 64 PDFs while preserving the two source schemas and multi-page evidence lists.
- Page recall, precision, and MRR handle multiple gold pages, duplicated retrieved pages, wrong-document pages, and no-hit cases.

The broader 5-10-question smoke suite, fixed-seed stratified pattern-check suite, provider integration tests, and retrieval-system tests are deferred until the harness runs end to end.

## Acceptance criteria

- A no-spend command prepares and validates the exact 10-K subset deterministically.
- Dry-run output enumerates all selected question/condition jobs without requiring API keys.
- Closed-book, oracle, and long-context runs can generate and resume independently of retrieval and judging.
- Unconfigured retrieval and judge stages fail clearly without creating plausible-looking placeholder scores.
- Retrieval outputs support deterministic page recall, precision, and MRR.
- Reports include row-level provenance, failures, usage/cost, and the approved segmentations.
- All dependencies and Python are reproducibly pinned with `uv`.
