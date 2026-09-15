# Benchmark Harness Implementation Guide

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` or `superpowers:executing-plans` to
> implement one approved vertical slice at a time. Steps use checkbox (`- [ ]`)
> syntax for tracking.

**Goal:** A reproducible, plain-Python harness that runs the FinanceBench 10-K
subset (112 questions, 64 PDFs) through the context conditions that need no
retriever, records exactly what each call cost, and reports segmented
page-retrieval metrics.

**Architecture:** Four commands communicating through files on disk —
`prepare` → `validate` → `run` → `report`. Generation is the only stage that
spends money, so every downstream stage is free to re-run.

**Tech Stack:** Python 3.12, `uv`, `openai`, `transformers` + `jinja2`,
`PyMuPDF`, `pytest`. Section 4 explains each.

**Requirements:** [`../Benchmark.md`](../Benchmark.md) — authoritative for *what*
the system must do. This guide is authoritative for *how* it is built.

**Status: not yet implemented.** Sections 1–13 are the approved design. Section 14
holds the implementation slices. An as-built record is added to each slice's
section as it lands.

## Global Constraints

- Python 3.12; `uv` for environment and locking. Direct dependencies pinned in
  `pyproject.toml`; complete `uv.lock` committed.
- `benchmarks/financebench/` is **read-only source material** and is never written to.
- `data/` and `results/` are generated, Git-ignored, and recreatable by command.
- Prompts are **never truncated**. A prompt that does not fit is recorded as
  `did_not_fit` (D3).
- `predictions.jsonl` is **append-only** and never rewritten.
- Every tunable lives in `configs/financebench.toml` or `config.py`. No call site
  hardcodes a number or builds a path.
- Deferred and out of scope: `single_store`/`shared_store`, the LLM judge, the
  reasoning/generation half of failure classification, LOFin/HiREC.

---

## 1. Scope

Build a reproducible, plain-Python evaluation harness for the FinanceBench questions
whose documents are 10-Ks: **112 questions across 64 PDFs**.

In scope now:

- prepare and validate the 10-K subset;
- construct the three context conditions that need no retriever;
- generate answers with checkpointing, resumption and per-answer telemetry;
- deterministic page recall, page precision and page MRR;
- segmented reporting with per-segment sample counts;
- development subsets for a smoke test and a pattern check.

Deferred, with the interface defined now so it drops in without harness changes:

- `single_store` and `shared_store`, which require the retrieval pipeline;
- the LLM-as-judge and its answer-accuracy metric;
- the reasoning/generation half of the failure classification, which depends on the
  judge's correctness signal;
- LOFin/HiREC, which introduces multi-document evidence.

---

## 2. Verified dataset facts

Every fact below was read from `benchmarks/financebench/data/`, not from secondary
notes. Several contradict assumptions that were reasonable beforehand, so each one
that constrains the design is marked.

| Fact | Value |
| --- | --- |
| Question rows | 150 |
| Document metadata rows | 361, covering 360 **unique** `doc_name`s |
| `doc_type` values | `10k` (269), `8k` (30), `Earnings` (29), `10q` (27), `10k_annualreport` (6) |
| Questions after `doc_type == "10k"` | **112**, across **64** unique PDFs |
| `financebench_id` | unique across all 150 |
| `evidence_page_num` | `int`, zero-indexed, observed range 0–303 |
| Evidence documents | every question's evidence is inside its own `doc_name`; zero cross-document cases |
| PDFs | all 64 required files present in `benchmarks/financebench/pdfs/` |

**Constraint A — the `doc_type` filter must be exact and lowercase.** The value is
`10k`, and a separate `10k_annualreport` type exists. A `startswith` or
case-insensitive match silently admits annual reports and breaks the count of 112.

**Constraint B — `doc_name` is not unique in the metadata file.**
`FOOTLOCKER_2023_annualreport` appears twice with conflicting `doc_period` (2023 and
2022) and an identical `doc_link`. It is an annual report, so no selected question
resolves to it. The naive join `{d["doc_name"]: d for d in docs}` silently keeps
whichever row is read last. The requirement is that every **selected** question
resolves to unambiguous metadata, so the ambiguity check belongs at resolution time,
not at indexing time — see §8.3. Raising during indexing would fail `prepare` on a
document nothing uses.

**Constraint C — evidence is frequently multi-page.** 78 questions have one gold page,
32 have two, and 2 have three. Page recall therefore has a denominator greater than
one for 34 of 112 questions, and the oracle condition concatenates several pages.
`Benchmark.md` states the questions are answered "on ONE page"; that half of the
sentence is not supported by the data, though the single-document half is exact.

**Constraint D — `question_reasoning` is not a clean three-way label.** Ten distinct
raw values appear across the 112, including `None` (14 questions, all
`novel-generated`), inconsistent casing (`information extraction`), `OR`-compounded
multi-labels, one malformed trailing `OR`, and a parenthetical variant
`Logical reasoning (based on numerical reasoning)` that the paper never defines.
Section 10 gives the normalisation.

**Constraint E — `evidence_text` and `evidence_text_full_page` always differ.** The
former is the targeted snippet, the latter the whole page (median ~2.1k characters,
max ~7.8k). Oracle is specified as "exact correct pages", so oracle uses
`evidence_text_full_page`; the snippet is judge and RAGAS material.

**Consequence — oracle needs no PDF parsing.** Because the dataset ships the full
page text, oracle assembles context straight from `questions.jsonl`. This removes
the most dangerous failure mode available to this harness: an extractor whose page
59 is not FinanceBench's page 59 would silently hand the model the wrong page.
PDF extraction is needed for long-context only, where the whole filing is passed and
alignment does not matter. Alignment becomes critical again when the retriever
lands, and these 149 known-good pages are then a ready-made test fixture.

Free segmentation axes also present in the metadata: `gics_sector` (9 sectors) and
`doc_period` (2015–2023).

---

## 3. Design decisions

Each decision records what it satisfies and why the alternative was rejected.

### D1 — The retriever is an interface now, not an implementation

`single_store` and `shared_store` cannot run until the retrieval pipeline exists, but
the metrics that consume retrieval output can be written and tested immediately
against synthetic chunks. The protocol is fixed now; the two conditions raise
`NotImplementedError` until it is satisfied.

Rejected: shipping a throwaway BM25 retriever so all five conditions execute today.
It would exercise the full path earlier, but it costs days that the schedule does not
have, and it risks the harness being tuned around a placeholder's behaviour.

### D2 — Failure classification is derived from signals, not assigned by a model

The retrieval axis is computable now: page recall of zero is a retrieval failure.
Splitting reasoning from generation failures requires knowing the answer was wrong,
which is the judge's job, so that half waits. An LLM classifier reaching the leaf
categories is a later extension once the judge exists.

This suits the stated purpose of the FinanceBench subset: at n=112 it supports
qualitative failure-mode diagnosis, not confident quantitative claims, and a derived
top-level label is defensible in a way a model's self-assessment is not.

### D3 — A prompt that does not fit is a third outcome, never a truncation

Prompts are never truncated. A prompt that exceeds the model's context window after
reserving space for output is recorded as `did_not_fit`: not correct, not incorrect,
and not an error.

Rejected: counting it incorrect (hides why the answer is missing) and excluding it
(changes *n* between conditions, which breaks the paired comparison that McNemar's
test requires). Reporting it as its own category keeps *n* constant across conditions
**and** preserves the evidence for the central claim that long-context is limited by
context window size. Accuracy is reported both including and excluding these rows.

The token count is taken over the **assembled prompt** — system message, instructions,
question and document text, rendered through the model's own chat template — with an
explicit configured reserve for the completion.

### D4 — Cognitive skills normalise to the paper's three labels, multi-label

`question_reasoning` is normalised into `Information extraction`, `Numerical
reasoning` and `Logical reasoning`, with unlabelled questions kept visible as their
own segment rather than dropped or guessed. Questions carrying more than one skill
appear in more than one segment, so segment counts sum to more than 112 — which is why
every segment is reported with its own sample count.

Rejected: treating the parenthetical variant as a fourth category, which would invent
taxonomy the paper does not define and break comparability with published results; and
single-labelling by precedence, which discards labels annotators deliberately gave.

### D5 — Stages communicate through files on disk

Each stage is a separate command that reads the previous stage's artifact and writes
its own. Generation is the only stage that costs money, so every downstream stage can
be re-run indefinitely for free. Resumption falls out of an append-only results file,
and every intermediate state named in the failure-inspection requirement exists as an
openable artifact rather than a local variable.

Rejected: a single orchestrated command (metrics changes would force paid re-runs, and
a crash loses the run unless checkpointing is added back, which reconstructs this
design badly); and building on an eval framework, which would fight both distinctive
requirements — the no-truncation policy and page-level metrics against zero-indexed
evidence. RAGAS is still used later as a library inside the judge stage.

The cost of this choice is staleness: a report computed against an older results file.
Every artifact therefore carries the run id and a dataset hash, and a stage refuses to
proceed on mismatch.

### D6 — Store what was observed; recompute what was derived; join what is regenerable

| Kind | Examples | Where it lives |
| --- | --- | --- |
| Observed once, unrecoverable | latency, cost, token usage, serving provider | in the prediction record |
| Pure function of stored data | page recall, precision, MRR | computed by `report` |
| Present in an immutable input | evidence text, justification | joined on demand |

Retrieval metrics are deliberately **not** written into the results file. Storing them
creates a second source of truth that silently disagrees with the recomputed value the
first time a metric is corrected, over a file that is otherwise immutable. Per-question
metrics are instead emitted to `per_question.csv`, which is free to regenerate.

Evidence text and the human justification are not copied into results either: they are
identical across all three conditions for a given question, so copying triples them,
and the judge prompt will be iterated more than once. The dataset hash on every row is
what proves the join is valid.

### D7 — Gold evidence is keyed by `(doc_name, page_num)` pairs

A bare page number is sufficient for FinanceBench, where all evidence is
single-document, and would break silently on the multi-document benchmark planned
later, where page 47 of two different filings would collide. The pair costs nothing now
and makes that benchmark a data-loading change rather than a metrics rewrite.

### D8 — `src/` layout with a console entry point

A `src/` layout cannot be imported accidentally from the working directory, so tests
exercise the same import path a user gets. An editable install exposes a real
`sec-rag-benchmark` command. This matters more as retrieval libraries, a vector store
client and RAGAS are added.

---

## 4. Libraries

Four direct dependencies. Everything else is standard library.

| Library | Job | Why this one |
| --- | --- | --- |
| `openai` | OpenRouter calls now, Azure judge later | OpenRouter is OpenAI-compatible; the SDK gives retries, backoff, timeouts and typed errors, and its response models allow extra fields so OpenRouter's `provider` and `usage.cost` are readable as attributes. `extra_body` passes OpenRouter-specific parameters. The judge will use `AzureOpenAI` from the same package, so there is one client library rather than two HTTP paths. |
| `transformers` + `jinja2` | exact prompt token counts | The count must cover the prompt **as the model receives it** — role markers, special tokens, generation prompt — which is what `chat_template.jinja` defines. `apply_chat_template()` is the reference implementation of that rendering. Reimplementing it against raw `tokenizers` to save install size risks a template mismatch, which is precisely the bug that would make `did_not_fit` fire at the wrong threshold. `AutoTokenizer` downloads only the tokenizer files, not the model weights. |
| `PyMuPDF` | PDF → page text, long-context only | 10–50× faster than `pdfplumber`; across 64 filings that is minutes rather than hours, which matters while iterating. Current scope needs linear text only, where its reading order is at least as good. |
| `pytest` | tests | — |

Stdlib: `tomllib` (config), `json`, `csv`, `hashlib`, `collections`, `re`, `pathlib`.

**Alternatives rejected.** `httpx` with hand-rolled retries — more code for less
reliability than the SDK, and a second HTTP path once Azure lands. `pdfplumber` —
MIT-licensed and better at table structure, but far slower; that trade would matter
if current scope needed table structure, and it does not. `pypdf` — lighter still,
but mangles table layout, and degraded long-context text would unfairly weaken the
baseline this project claims to beat. `pandas` — aggregating 336 rows is stdlib work;
`per_question.csv` exists so pandas can be used for analysis outside the harness.
`tenacity` — the SDK already retries. `python-dotenv` — `set -a; source .env` works.

**Licence note.** `PyMuPDF` is AGPL-3.0. That is fine for running experiments and for
a dissertation, but its copyleft would extend to a published repository that imports
it. `pdfplumber` (MIT) is the swap if the repository is ever published permissively;
because extracted text is cached, switching costs a cache delete and one re-extract.

**Reproducibility.** `tokenizer_revision` in the config pins the tokenizer to a
commit SHA. An unpinned `from_pretrained` silently changes token counts if the
upstream repository updates, which would make results unreproducible.

---

## 5. Architecture

```text
prepare  ──► data/financebench/{questions.jsonl, manifest.json}
                │
validate ──►   (exit 0/1; writes nothing)
                │
preview  ──►   (terminal only; no artifacts, no spend)
                │
run      ──► results/<run-id>/{predictions.jsonl, errors.jsonl, config.toml}
                │
report   ──► results/<run-id>/{per_question.csv, summary.json, summary.csv}

  deferred:
judge    ──► results/<run-id>/judgements.jsonl
classify ──► results/<run-id>/failures.csv
```

`metrics` is not a separate command. Computing and aggregating are both free and
deterministic, so splitting them would add a staleness failure mode for no benefit.

### Artifacts

| Path | Written by | Contents |
| --- | --- | --- |
| `data/financebench/questions.jsonl` | `prepare` | 112 joined question records, complete evidence retained |
| `data/financebench/manifest.json` | `prepare` | source, filter, expected vs observed counts, PDF filenames, file hashes |
| `data/financebench/text/<doc_name>.json` | `documents` (lazily) | extracted page text, one list entry per PDF page |
| `results/<run-id>/config.toml` | `run` | effective configuration for this run |
| `results/<run-id>/predictions.jsonl` | `run` | append-only; one row per question × condition |
| `results/<run-id>/errors.jsonl` | `run` | transient failures, retryable on resume |
| `results/<run-id>/per_question.csv` | `report` | one row per prediction, with derived metrics |
| `results/<run-id>/summary.{json,csv}` | `report` | segmented aggregates with sample counts |

---

## 6. Module structure

```text
src/sec_rag_benchmark/
├── cli.py          argparse dispatch and terminal formatting. No logic, no I/O.
├── config.py       TOML to typed settings; every constant and derived path
├── records.py      the Prediction record and JSONL read/write
├── data.py         prepare, validate, load, dev-subset selection
├── documents.py    PDF to cached page text (long-context only)
├── conditions.py   condition inputs and the Retriever protocol
├── budget.py       chat-template token counting and the fit / did_not_fit decision
├── generation.py   OpenAI client and telemetry capture
├── runner.py       job loop, checkpointing, resumption
├── metrics.py      page recall, precision, MRR — pure functions
└── reporting.py    segmentation and summary writers
```

Dependencies run strictly downward, with no cycles:

```text
cli         → data, runner, reporting, conditions
runner      → data, conditions, generation, records
conditions  → documents, budget, records
reporting   → metrics, records
all         → config
```

Cognitive-skill normalisation lives in `data.py`, not `reporting.py`: it is dataset
normalisation, applied once at load, so `Question.cognitive_skills` is already
canonical everywhere downstream. Putting it in `reporting.py` would force `runner`
to import `reporting` in order to stamp `segments` onto each prediction, which would
invert the dependency direction.

`config.py` imports nothing local. `records.py` imports only `config`.

Four modules earn their separation. `budget.py` holds the subtlest logic in the
system and the highest cost of being wrong, and is fully testable offline.
`documents.py` keeps PDF handling out of `conditions.py`, which would otherwise be
untestable without PDF fixtures. `records.py` is depended on by producers and
consumers alike, so placing it in either would create a cycle. `metrics.py` stays
free of file handling so it can be tested against synthetic chunks with no retriever
present, which is what D1 requires.

`cli.py` is a pure dispatcher: build subparsers, match the command, call one stage
function, format its return. Stages return raw values; only the CLI knows about
rounding and display.

---

## 7. Configuration

`configs/financebench.toml`:

```toml
[model]
id                 = "z-ai/glm-5.3-flash"
tokenizer_repo     = "zai-org/GLM-5.3-Flash"
tokenizer_revision = ""          # pinned commit SHA; resolved in Slice 2
context_window     = 200000
completion_reserve = 2048
temperature        = 0.0

[dataset]
source_dir         = "benchmarks/financebench"
doc_type           = "10k"
expected_questions = 112
expected_pdfs      = 64

[run]
conditions   = ["closed_book", "oracle", "long_context"]
top_k        = 10
seed         = 42
smoke_size   = 5
pattern_size = 50
```

`config.py` additionally owns `PROJECT_ROOT` and every path derived from it
(`DATA_DIR`, `RESULTS_DIR`, `TEXT_CACHE_DIR`). Paths are `pathlib` objects derived
once, never string-joined at a call site.

---

## 8. Interfaces

### 8.1 Retriever

`Protocol` is Python's way of describing a shape that other code must match without
requiring inheritance: any object with a `retrieve` method of this signature satisfies
it. That is what lets the metrics and runner be written and tested now, against a fake
retriever, before the real one exists.

```python
class Scope(StrEnum):
    SINGLE_DOCUMENT = "single_document"   # only the filing containing the answer
    SHARED_CORPUS   = "shared_corpus"     # chunks from all companies' filings

class Chunk(Protocol):
    text: str
    doc_name: str
    page_num: int          # zero-indexed, matching evidence_page_num
    score: float

class Retriever(Protocol):
    def retrieve(self, question: str, scope: Scope, top_k: int) -> list[Chunk]: ...
```

One implementation serves both deferred conditions; the scope argument is the only
difference between them. Every chunk must carry `doc_name` and `page_num`, because
those are what the metrics compare against gold evidence.

### 8.2 The prediction record

One row per question × condition, appended to `predictions.jsonl`. With 112 questions
and three runnable conditions, a full run produces 336 rows.

```jsonc
{
  "job_id": "financebench_id_03029::oracle",
  "financebench_id": "financebench_id_03029",
  "condition": "oracle",

  "question": "What is the FY2018 capital expenditure amount ...",
  "gold_answer": "$1577.00",
  "gold_evidence": [{"doc_name": "3M_2018_10K", "page_num": 59}],

  "retrieval": null,
  "context_ref": {
    "doc_name": "3M_2018_10K",
    "pages": [59],
    "token_count": 1820,
    "content_hash": "sha256:..."
  },

  "outcome": "ok",
  "generated_answer": "$1,577 million",
  "telemetry": {
    "model_requested": "z-ai/glm-5.3-flash",
    "model_returned":  "z-ai/glm-5.3-flash",
    "provider": "...",
    "prompt_tokens": 1974,
    "completion_tokens": 87,
    "latency_ms": 3412,
    "cost_usd": 0.00214
  },

  "run_id": "20260913-143052",
  "dataset_hash": "sha256:...",
  "segments": {
    "question_type": "metrics-generated",
    "cognitive_skills": ["Information extraction"],
    "gics_sector": "Industrials",
    "doc_period": 2018
  }
}
```

**Identity.** `job_id` is `financebench_id` and `condition` joined. Resumption reads
existing `job_id`s and skips them, which is what allows the file to be append-only: a
row is added, never rewritten.

**`retrieval` is one nested object, not sibling fields.** `null` means the condition
never retrieves — closed-book, oracle and long-context. An object whose chunk list is
empty means retrieval ran and returned nothing, which is a genuine recall failure. If
those collapse into the same value, averaged recall silently includes three conditions
that never retrieved. Nesting makes the distinction a single check rather than four
correlated ones.

**`outcome` is an enum whose values behave differently on resume.**

| Value | Meaning | Written to | On resume |
| --- | --- | --- | --- |
| `ok` | answer generated | `predictions.jsonl` | skip |
| `did_not_fit` | prompt exceeded the window; **terminal** | `predictions.jsonl` | skip — it will never fit |
| `api_error` | transient failure | `errors.jsonl` only | retry |

Because `api_error` rows are never written to `predictions.jsonl`, that file contains
only terminal outcomes, and resumption is simply "skip every `job_id` already
present". The file layout does the work; no outcome filtering is needed.

`telemetry` is `null` on a `did_not_fit` row: no call was made, so nothing was
observed. Zero would misreport cost.

**`context_ref`, not raw context.** Retrieved chunk text is stored, because it cannot
be recreated without re-running the retriever. Oracle and long-context context is
referenced by document, pages, token count and hash, because it is regenerable
deterministically and inlining 112 complete 10-Ks would make the file too large to
open in the tools used to inspect failures.

**Judge verdicts and failure classifications are never written here.** They land in
separate files keyed by `job_id` and are joined at report time, so the artifact that
cost money stays immutable and the judge can be re-run with a different prompt without
regenerating answers.

### 8.3 Public signatures

Read in this order; helpers sit beneath the public function that calls them.

```python
# records.py — the shared vocabulary; read first
class Condition(StrEnum):  CLOSED_BOOK; ORACLE; LONG_CONTEXT; SINGLE_STORE; SHARED_STORE
class Outcome(StrEnum):    OK; DID_NOT_FIT; API_ERROR

@dataclass(frozen=True)
class GoldEvidence:   doc_name: str; page_num: int
class RetrievedChunk: doc_name: str; page_num: int; score: float; text: str
class Retrieval:      chunks: list[RetrievedChunk]
class ContextRef:     doc_name: str | None; pages: list[int]
                      token_count: int; content_hash: str
class Telemetry:      model_requested: str; model_returned: str; provider: str
                      prompt_tokens: int; completion_tokens: int
                      latency_ms: int; cost_usd: float
class Prediction:     job_id; financebench_id; condition; question; gold_answer
                      gold_evidence; retrieval: Retrieval | None; context_ref
                      outcome; generated_answer: str | None
                      telemetry: Telemetry | None; run_id; dataset_hash; segments

def make_job_id(financebench_id: str, condition: Condition) -> str
def append_jsonl(path: Path, row: dict) -> None
def read_jsonl(path: Path) -> Iterator[dict]

# data.py
@dataclass(frozen=True)
class Question:  financebench_id: str; company: str; doc_name: str
                 question: str; gold_answer: str; justification: str
                 question_type: str                    # used as-is
                 question_reasoning_raw: str | None    # kept for traceability
                 cognitive_skills: list[str]           # normalised at load
                 evidence: list[Evidence]              # complete, never trimmed
                 gics_sector: str; doc_period: int

def prepare(cfg: Config) -> Manifest
def validate(cfg: Config) -> list[str]            # empty list == valid
def load_questions(cfg: Config) -> list[Question]
def select_subset(qs: list[Question],
                  kind: Literal["smoke", "pattern"], seed: int) -> list[Question]
def normalise_skills(raw: str | None) -> list[str]
  # helpers
  def _index_documents(rows: list[dict]) -> dict[str, list[dict]]
  def _resolve(by_name: dict[str, list[dict]], doc_name: str) -> dict

# documents.py
def page_texts(doc_name: str, cfg: Config) -> list[str]   # cached; index == PDF page

# conditions.py
def build_context(q: Question, cond: Condition, cfg: Config,
                  retriever: Retriever | None = None) -> ConditionInput
@dataclass
class ConditionInput: text: str | None; context_ref: ContextRef
                      retrieval: Retrieval | None

# budget.py
def load_tokenizer(cfg: Config)
def assemble_messages(question: str, context: str | None) -> list[dict]
def count_prompt_tokens(tokenizer, messages: list[dict]) -> int
def check_fit(n_tokens: int, cfg: Config) -> bool

# generation.py
def make_client(cfg: Config) -> OpenAI
def generate(client, messages: list[dict], cfg: Config) -> tuple[str, Telemetry]

# runner.py
def run(cfg: Config, conditions: list[Condition], run_dir: Path,
        limit: int | None = None, dry_run: bool = False) -> Path
  def completed_job_ids(run_dir: Path) -> set[str]

# metrics.py — pure, no I/O
def page_metrics(chunks: list[RetrievedChunk],
                 gold: list[GoldEvidence]) -> PageMetrics | None

# reporting.py
def write_per_question(run_dir: Path, predictions: Iterable[dict]) -> Path
def write_summary(run_dir: Path, predictions: Iterable[dict]) -> tuple[Path, Path]
```

---

## 9. Metrics

Computed by `report` from each prediction row, and `None` — never `0` — when
`retrieval` is `null`, so conditions that never retrieved are excluded from averages
rather than dragging them down.

```text
gold_pages      = set of (doc_name, page_num) from gold_evidence
retrieved_pages = (doc_name, page_num) of retrieved chunks, deduplicated,
                  order preserved

page_recall    = |gold_pages ∩ retrieved_pages| / |gold_pages|
page_precision = |gold_pages ∩ retrieved_pages| / |retrieved_pages|
page_mrr       = 1 / (1-based rank of the first chunk whose page is in gold_pages)
                 0.0 if no retrieved chunk hits a gold page
```

Recall and precision operate over the **deduplicated page set**, while MRR ranks over
the **ordered chunk list** — two granularities over the same result, and the easiest
part of this design to get wrong. Precision is `0.0` when retrieval returned no
chunks at all.

---

## 10. Segmentation

Results are reported by condition, by generation method, and by cognitive skill, each
with its sample count.

`question_type` is used as-is: `metrics-generated` (50), `domain-relevant` (48),
`novel-generated` (14).

`question_reasoning` is normalised per D4. Verified output over the 112 questions,
which serves as a test fixture:

| Segment | n |
| --- | --- |
| Numerical reasoning | 57 |
| Information extraction | 36 |
| Logical reasoning | 21 |
| unlabelled | 14 |

Counts sum to 128 because 16 questions carry more than one skill.

---

## 11. Development subsets

Selected by `data.py` with a fixed seed, stratified by `question_type` so all three
generation methods appear:

- **smoke** — 5 questions (configurable 5–10), enough to prove the pipeline runs end
  to end;
- **pattern** — 50 questions, run as a development test suite to see which segments
  break.

Both are deterministic: the same seed yields the same subset, so results are
comparable across runs.

---

## 12. One question through the system

`financebench_id_03029` under `oracle`:

```text
data.load_questions()       → Question(gold_evidence=[("3M_2018_10K", 59)],
                                       cognitive_skills=["Information extraction"])
conditions.build_context()  → reads evidence_text_full_page  (no PDF touched)
                              ContextRef(doc_name="3M_2018_10K", pages=[59],
                                         token_count=1820, content_hash="sha256:…")
budget.check_fit(1974)      → 1974 + 2048 ≤ 200000  → True
generation.generate()       → "$1,577 million", Telemetry(cost_usd=0.00214, …)
runner                      → append_jsonl(predictions.jsonl,
                                           Prediction(outcome=OK))
report                      → per_question.csv row; metrics blank (retrieval is None)
```

The same question under `long_context` differs at exactly one step: `build_context`
calls `documents.page_texts("3M_2018_10K")` and joins every page.

---

## 13. Reading order

`cli.py` → `records.py` → `config.py`, then per command:

| Command | Modules, in call order |
| --- | --- |
| `prepare` / `validate` | `data.py` |
| `preview` | `conditions.py` → `documents.py` → `budget.py` |
| `run` | `runner.py` → `conditions.py` → `generation.py` |
| `report` | `reporting.py` → `metrics.py` |

---

## 14. Implementation slices

Four slices. Each ends in behaviour you can run and reject independently.

| # | Runnable outcome | Spend |
| --- | --- | --- |
| 1 | `prepare` + `validate` produce and check the 112/64 subset | none |
| 2 | `preview` prints a prompt's token count and fit verdict | none |
| 3 | `run` generates answers with checkpointing and resume | **paid** |
| 4 | `report` writes `per_question.csv` and `summary.*` | none |

---

### Slice 1: `prepare` and `validate`

**Purpose:** A project-owned 112-question / 64-PDF subset exists on disk, with a
manifest, and six validation checks pass against it.

**Files and responsibilities:**

- Create: `pyproject.toml` — dependencies, `[project.scripts]` entry point
- Create: `configs/financebench.toml` — §7
- Create: `src/sec_rag_benchmark/config.py` — TOML to typed settings, paths
- Create: `src/sec_rag_benchmark/records.py` — enums, dataclasses, JSONL helpers
- Create: `src/sec_rag_benchmark/data.py` — prepare, validate, load, subsets
- Create: `src/sec_rag_benchmark/cli.py` — `prepare`, `validate` subcommands
- Test: `tests/test_config.py`, `tests/test_records.py`, `tests/test_data.py`

**Interfaces produced:** `Config`, `Condition`, `Outcome`, `Question`,
`GoldEvidence`, `Manifest`, `prepare`, `validate`, `load_questions`,
`select_subset`, `read_jsonl`, `append_jsonl`, `make_job_id`.

**Reading path:** `cli.main` → `data.prepare` → `_index_documents` → `_resolve`
→ `data.validate`.

- [ ] **Step 1: Write the failing tests**

```python
def test_prepare_selects_112_questions_and_64_pdfs(cfg):
    manifest = prepare(cfg)
    assert manifest.observed_questions == 112
    assert len(manifest.pdf_filenames) == 64

def test_annual_reports_are_excluded(cfg):
    # doc_type "10k_annualreport" must not match the "10k" filter
    names = {q.doc_name for q in load_questions(cfg)}
    assert not any(n.endswith("_annualreport") for n in names)

def test_conflicting_metadata_for_a_selected_doc_raises():
    by_name = {"ACME_2020_10K": [{"doc_period": 2020}, {"doc_period": 2019}]}
    with pytest.raises(ValueError, match="conflicting metadata"):
        _resolve(by_name, "ACME_2020_10K")

def test_conflicting_metadata_for_an_unselected_doc_is_tolerated(cfg):
    # FOOTLOCKER_2023_annualreport is duplicated upstream but never selected
    prepare(cfg)                      # must not raise

def test_prepare_is_idempotent(cfg):
    first = (DATA_DIR / "questions.jsonl").read_bytes() if prepare(cfg) else None
    prepare(cfg)
    assert (DATA_DIR / "questions.jsonl").read_bytes() == first

@pytest.mark.parametrize("mutation", [
    "duplicate_financebench_id", "missing_pdf", "negative_page_num",
    "empty_evidence", "unknown_doc_name", "wrong_question_count",
])
def test_each_validation_check_fails_on_its_mutation(cfg, mutation):
    apply_mutation(cfg, mutation)
    assert validate(cfg) != []

@pytest.mark.parametrize("raw,expected", [
    (None, ["unlabelled"]),
    ("Numerical reasoning", ["Numerical reasoning"]),
    ("information extraction", ["Information extraction"]),          # casing
    ("Logical reasoning (based on numerical reasoning)", ["Logical reasoning"]),
    ("Information extraction OR Logical reasoning OR",               # trailing OR
     ["Information extraction", "Logical reasoning"]),
    ("Logical reasoning (based on numerical reasoning) OR Logical reasoning",
     ["Logical reasoning"]),                                         # deduplicated
])
def test_normalise_skills(raw, expected):
    assert normalise_skills(raw) == expected

def test_unmapped_skill_raises():
    with pytest.raises(ValueError, match="unmapped"):
        normalise_skills("Vibes-based reasoning")

def test_real_dataset_segment_counts(cfg):
    counts = Counter(s for q in load_questions(cfg) for s in q.cognitive_skills)
    assert counts == {"Numerical reasoning": 57, "Information extraction": 36,
                      "Logical reasoning": 21, "unlabelled": 14}
```

- [ ] **Step 2: Verify the expected failure**

Run: `uv run pytest tests/test_data.py -v`
Expected: all FAIL with `ModuleNotFoundError: sec_rag_benchmark.data`.

- [ ] **Step 3: Implement**

`_index_documents` collects rows per `doc_name` into lists rather than
overwriting. `_resolve` raises only when a name a question actually needs has
conflicting rows — see Constraint B for why this cannot be an indexing-time check:

```python
def _index_documents(rows: list[dict]) -> dict[str, list[dict]]:
    by_name = defaultdict(list)
    for row in rows:
        by_name[row["doc_name"]].append(row)
    return by_name

def _resolve(by_name: dict[str, list[dict]], doc_name: str) -> dict:
    candidates = by_name.get(doc_name, [])
    if not candidates:
        raise KeyError(f"no metadata for {doc_name!r}")
    # Duplicates are tolerated only when every copy agrees; a genuine conflict
    # means the question cannot be resolved to one filing.
    if any(c != candidates[0] for c in candidates[1:]):
        raise ValueError(f"conflicting metadata for {doc_name!r}: {candidates}")
    return candidates[0]
```

`prepare` then: read both JSONL files → index documents → for each of the 150
questions resolve its metadata and keep it when `doc_type == cfg.dataset.doc_type`
(exact string equality, Constraint A) → write `questions.jsonl` with keys sorted and
a trailing newline so byte-identical reruns are guaranteed → write `manifest.json`
recording source, filter, expected vs observed counts, PDF filenames and SHA-256
hashes.

`validate` returns a list of human-readable problems, empty when valid, covering the
six checks in `Benchmark.md`. It writes nothing.

`select_subset` groups by `question_type`, sorts each group by `financebench_id` for
determinism, and takes a `random.Random(seed)` sample proportional to group size.

- [ ] **Step 4: Verify the slice**

```bash
uv sync && uv lock --check
uv run pytest -q
uv run sec-rag-benchmark prepare --config configs/financebench.toml
uv run sec-rag-benchmark validate --config configs/financebench.toml
```

Expected: tests pass; `prepare` reports 112 questions / 64 PDFs; `validate` exits 0.

- [ ] **Step 5: Reconcile guide and implementation**
- [ ] **Step 6: User reviews the uncommitted diff**
- [ ] **Step 7: Commit after approval**

```bash
git add pyproject.toml uv.lock configs/ src/ tests/ "docs sys design/benchmark/Benchmark Harness Implementation Guide.md"
git commit -m "feat: prepare and validate the FinanceBench 10-K subset"
```

---

### Slice 2: `preview` — context assembly, tokenisation, fit

**Purpose:** For any question and condition, print the assembled prompt's token
count and whether it fits. This is how the completion reserve is calibrated **before
any spend**, and how the `did_not_fit` rate is known in advance.

**Files and responsibilities:**

- Create: `src/sec_rag_benchmark/documents.py` — PyMuPDF extraction, disk cache
- Create: `src/sec_rag_benchmark/conditions.py` — `build_context`, `Retriever`
- Create: `src/sec_rag_benchmark/budget.py` — chat template, token count, fit
- Modify: `src/sec_rag_benchmark/cli.py` — add `preview`
- Modify: `configs/financebench.toml` — fill `tokenizer_revision`
- Test: `tests/test_documents.py`, `tests/test_conditions.py`, `tests/test_budget.py`

**Interfaces produced:** `page_texts`, `build_context`, `ConditionInput`,
`ContextRef`, `Scope`, `Chunk`, `Retriever`, `assemble_messages`,
`count_prompt_tokens`, `check_fit`.

**Reading path:** `cli.preview` → `conditions.build_context` →
`documents.page_texts` → `budget.assemble_messages` → `count_prompt_tokens` →
`check_fit`.

- [ ] **Step 1: Write the failing tests**

```python
def test_oracle_context_uses_dataset_page_text_and_touches_no_pdf(cfg, monkeypatch):
    monkeypatch.setattr(documents, "page_texts", _boom)   # must not be called
    ci = build_context(question_03029, Condition.ORACLE, cfg)
    assert ci.text == question_03029.evidence[0].evidence_text_full_page
    assert ci.context_ref.pages == [59]

def test_multi_page_oracle_concatenates_in_page_order(cfg):
    ci = build_context(two_page_question, Condition.ORACLE, cfg)
    assert ci.context_ref.pages == sorted(ci.context_ref.pages)

def test_closed_book_has_no_context(cfg):
    ci = build_context(question_03029, Condition.CLOSED_BOOK, cfg)
    assert ci.text is None and ci.retrieval is None

def test_extracted_page_index_aligns_with_evidence_page_num(cfg):
    # anchors PyMuPDF's paging to FinanceBench's zero-indexed page numbers
    pages = page_texts("3M_2018_10K", cfg)
    assert "Purchases of property, plant and equipment" in pages[59]

def test_second_call_reads_cache(cfg, monkeypatch):
    page_texts("3M_2018_10K", cfg)
    monkeypatch.setattr(fitz, "open", _boom)
    page_texts("3M_2018_10K", cfg)                        # must not re-extract

def test_chat_template_adds_tokens_beyond_naive_concatenation(cfg):
    msgs = assemble_messages("Q?", "CTX")
    assert count_prompt_tokens(tok, msgs) > len(tok.encode("Q?CTX"))

def test_one_token_over_the_budget_does_not_fit(cfg):
    limit = cfg.model.context_window - cfg.model.completion_reserve
    assert check_fit(limit, cfg) is True
    assert check_fit(limit + 1, cfg) is False

def test_retriever_conditions_are_not_implemented(cfg):
    with pytest.raises(NotImplementedError):
        build_context(question_03029, Condition.SINGLE_STORE, cfg)
```

- [ ] **Step 2: Verify the expected failure**

Run: `uv run pytest tests/test_conditions.py tests/test_budget.py -v`
Expected: FAIL with `ModuleNotFoundError: sec_rag_benchmark.conditions`.

- [ ] **Step 3: Implement**

First resolve and pin the tokenizer revision, then write it into the config:

```bash
uv run python -c "from huggingface_hub import HfApi; \
print(HfApi().model_info('zai-org/GLM-5.3-Flash').sha)"
```

`documents.page_texts` checks `data/financebench/text/<doc_name>.json`; on a miss it
opens the PDF with PyMuPDF, takes `page.get_text()` for every page **in document
order so the list index equals the zero-indexed page number**, writes the cache, and
returns it.

`conditions.build_context` dispatches on condition:

```text
CLOSED_BOOK   → text=None, context_ref with token_count=0, retrieval=None
ORACLE        → evidence_text_full_page for each gold page, sorted by page_num,
                joined with a blank line; no PDF access
LONG_CONTEXT  → documents.page_texts(doc_name) joined in page order
SINGLE_STORE,
SHARED_STORE  → raise NotImplementedError   (D1)
```

`budget.assemble_messages` builds the system and user messages;
`count_prompt_tokens` runs
`tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)`
and returns the length — the template is what makes this the count the model
actually sees (D3). `check_fit` is
`n_tokens + cfg.model.completion_reserve <= cfg.model.context_window`.

- [ ] **Step 4: Verify the slice**

```bash
uv run pytest -q
uv run sec-rag-benchmark preview --id financebench_id_03029 --condition oracle
uv run sec-rag-benchmark preview --id financebench_id_03029 --condition long_context
```

Expected: tests pass; each prints token count, reserve, window and a FITS /
DID_NOT_FIT verdict.

**Calibration (do this before Slice 3):** run `preview` across all 112 under
`long_context` and record how many do not fit. That number is the expected
`did_not_fit` count, and it sets `completion_reserve` with evidence rather than by
guess. Record the result in this section.

- [ ] **Step 5: Reconcile guide and implementation**
- [ ] **Step 6: User reviews the uncommitted diff**
- [ ] **Step 7: Commit after approval**

```bash
git commit -m "feat: assemble condition contexts and enforce the token budget"
```

---

### Slice 3: `run` — generation, checkpointing, resumption

**Purpose:** Answers are generated for every selected question × condition, appended
as they arrive, with full telemetry, and an interrupted run resumes without
re-spending.

**This is the only slice that costs money.**

**Files and responsibilities:**

- Create: `src/sec_rag_benchmark/generation.py` — client, request, telemetry
- Create: `src/sec_rag_benchmark/runner.py` — job loop, checkpoint, resume
- Modify: `src/sec_rag_benchmark/cli.py` — add `run`
- Test: `tests/test_generation.py`, `tests/test_runner.py`

**Interfaces consumed:** `build_context`, `check_fit`, `count_prompt_tokens`,
`append_jsonl`, `read_jsonl`, `make_job_id`.
**Interfaces produced:** `make_client`, `generate`, `run`, `completed_job_ids`.

**Reading path:** `cli.run` → `runner.run` → `completed_job_ids` →
`conditions.build_context` → `budget.check_fit` → `generation.generate`.

- [ ] **Step 1: Write the failing tests**

```python
def test_dry_run_creates_no_client(cfg, monkeypatch):
    monkeypatch.setattr(generation, "make_client", _boom)
    run(cfg, [Condition.CLOSED_BOOK], run_dir, dry_run=True)

def test_oversized_prompt_records_did_not_fit_without_calling(cfg, monkeypatch):
    monkeypatch.setattr(budget, "check_fit", lambda *_: False)
    monkeypatch.setattr(generation, "generate", _boom)
    run(cfg, [Condition.LONG_CONTEXT], run_dir, limit=1)
    row = next(read_jsonl(run_dir / "predictions.jsonl"))
    assert row["outcome"] == "did_not_fit"
    assert row["telemetry"] is None and row["generated_answer"] is None

def test_resume_skips_terminal_outcomes(run_dir):
    append_jsonl(run_dir / "predictions.jsonl", {"job_id": "a::oracle"})
    append_jsonl(run_dir / "predictions.jsonl", {"job_id": "b::oracle"})
    assert completed_job_ids(run_dir) == {"a::oracle", "b::oracle"}

def test_api_errors_go_only_to_errors_file(cfg, monkeypatch):
    monkeypatch.setattr(generation, "generate", _raise_api_error)
    run(cfg, [Condition.CLOSED_BOOK], run_dir, limit=1)
    assert not (run_dir / "predictions.jsonl").exists()
    assert sum(1 for _ in read_jsonl(run_dir / "errors.jsonl")) == 1

def test_telemetry_is_parsed_from_a_recorded_openrouter_response():
    t = telemetry_from(RECORDED_RESPONSE, requested="z-ai/glm-5.3-flash", latency_ms=3412)
    assert t.provider and t.cost_usd > 0
    assert t.model_returned == RECORDED_RESPONSE["model"]
```

- [ ] **Step 2: Verify the expected failure**

Run: `uv run pytest tests/test_runner.py -v`
Expected: FAIL with `ModuleNotFoundError: sec_rag_benchmark.runner`.

- [ ] **Step 3: Implement**

`make_client` returns `OpenAI(base_url="https://openrouter.ai/api/v1",
api_key=os.environ["OPENROUTER_API_KEY"])`, raising a clear message naming the
variable if it is absent.

`generate` sends `extra_body={"usage": {"include": True}}` so OpenRouter returns
cost, times the call with `time.perf_counter()`, and reads `provider`, `model`,
`usage.prompt_tokens`, `usage.completion_tokens` and `usage.cost` from the response.
`provider` and `cost` are OpenRouter extensions rather than SDK guarantees, so the
first live call must be checked against a real response and the recorded fixture
updated from it.

`run` creates `results/<timestamp>/`, writes the effective `config.toml`, computes
`dataset_hash`, then for each (question, condition):

```text
job_id = make_job_id(q.financebench_id, condition)
if job_id in completed_job_ids(run_dir):        continue
context = build_context(q, condition, cfg)
n       = count_prompt_tokens(tokenizer, messages)
if not check_fit(n, cfg):
    append_jsonl(predictions.jsonl, Prediction(outcome=DID_NOT_FIT,
                                               telemetry=None,
                                               generated_answer=None))
    continue                                     # terminal — never retried
if dry_run:                                      continue
try:
    answer, telemetry = generate(client, messages, cfg)
    append_jsonl(predictions.jsonl, Prediction(outcome=OK, ...))
except APIError as exc:
    append_jsonl(errors.jsonl, {...})            # absent from predictions → retried
```

- [ ] **Step 4: Verify the slice**

```bash
uv run pytest -q
uv run sec-rag-benchmark run --config configs/financebench.toml --dry-run
set -a; source .env; set +a
uv run sec-rag-benchmark run --config configs/financebench.toml --subset smoke
```

Expected: dry run covers 336 jobs and spends nothing; the smoke run writes 5×
conditions rows with populated telemetry. **Inspect the first row's telemetry before
running the full set.**

- [ ] **Step 5: Reconcile guide and implementation** — record the observed
      `did_not_fit` count and whether computed token counts matched the API's
      `prompt_tokens`.
- [ ] **Step 6: User reviews the uncommitted diff**
- [ ] **Step 7: Commit after approval**

```bash
git commit -m "feat: generate answers with checkpointing, resumption and telemetry"
```

---

### Slice 4: `report` — metrics and segmented summaries

**Purpose:** A run directory yields per-question metrics and segmented aggregates,
recomputable at any time for free.

**Files and responsibilities:**

- Create: `src/sec_rag_benchmark/metrics.py` — pure metric functions
- Create: `src/sec_rag_benchmark/reporting.py` — normalisation, writers
- Modify: `src/sec_rag_benchmark/cli.py` — add `report`
- Test: `tests/test_metrics.py`, `tests/test_reporting.py`

**Interfaces consumed:** `normalise_skills` (Slice 1), `read_jsonl`.
**Interfaces produced:** `page_metrics`, `PageMetrics`, `write_per_question`,
`write_summary`.

**Reading path:** `cli.report` → `reporting.write_per_question` →
`metrics.page_metrics` → `reporting.write_summary` → `normalise_skills`.

- [ ] **Step 1: Write the failing tests**

```python
def test_worked_example_metrics():
    # 5 chunks over 4 unique pages; gold is page 59; first hit at rank 2
    chunks = [chunk(12), chunk(59), chunk(59), chunk(87), chunk(103)]
    m = page_metrics(chunks, [GoldEvidence("3M_2018_10K", 59)])
    assert (m.recall, m.precision, m.mrr) == (1.0, 0.25, 0.5)

def test_multi_page_gold_partial_recall():
    m = page_metrics([chunk(59)], [GoldEvidence(D, 59), GoldEvidence(D, 60)])
    assert m.recall == 0.5

def test_no_retrieval_returns_none():
    assert page_metrics(None, gold) is None       # distinct from a miss

def test_empty_retrieval_scores_zero():
    m = page_metrics([], gold)
    assert (m.recall, m.precision, m.mrr) == (0.0, 0.0, 0.0)

def test_skill_segments_sum_to_128_over_112_questions(run_dir):
    # normalisation itself is tested in Slice 1; this asserts grouping, not parsing
    rows = json.loads((run_dir / "summary.json").read_text())["segments"]
    skill_rows = [r for r in rows if r["dimension"] == "cognitive_skill"]
    assert sum(r["n"] for r in skill_rows) == 128

def test_every_summary_row_carries_its_n(run_dir):
    rows = json.loads((run_dir / "summary.json").read_text())["segments"]
    assert all("n" in r for r in rows)
```

- [ ] **Step 2: Verify the expected failure**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: sec_rag_benchmark.metrics`.

- [ ] **Step 3: Implement**

```python
def page_metrics(chunks, gold) -> PageMetrics | None:
    if chunks is None:                       # condition never retrieves (D6)
        return None
    gold_pages = {(g.doc_name, g.page_num) for g in gold}
    ranked     = [(c.doc_name, c.page_num) for c in chunks]
    unique     = dict.fromkeys(ranked)       # dedup, rank order preserved
    hits       = gold_pages & unique.keys()
    return PageMetrics(
        recall    = len(hits) / len(gold_pages),
        precision = len(hits) / len(unique) if unique else 0.0,
        # MRR ranks over the raw chunk list, not the deduplicated pages
        mrr = next((1 / rank for rank, page in enumerate(ranked, 1)
                    if page in gold_pages), 0.0),
    )

`normalise_skills` lives in `data.py` and is implemented in Slice 1.

`write_per_question` emits one CSV row per prediction: `job_id`, `condition`,
`outcome`, the three metrics (blank when `None`), `cost_usd`, `latency_ms`, and the
segment columns.

`write_summary` groups by condition, by `question_type`, and by each cognitive
skill, reporting for every segment: `n`, the `did_not_fit` count, mean page recall /
precision / MRR over rows that retrieved, and total cost. **Answer accuracy is not
reported yet** — it requires the judge, which is deferred; the column is added when
the judge lands rather than stubbed now. Both writers refuse to run if the run's
`dataset_hash` does not match the current `questions.jsonl` (D5).

- [ ] **Step 4: Verify the slice**

```bash
uv run pytest -q
uv run sec-rag-benchmark report --run-dir results/<run-id>
```

Expected: tests pass; `per_question.csv` and `summary.{json,csv}` appear; blank
metric cells on non-retrieval conditions; segment counts sum to 128 for skills.

- [ ] **Step 5: Reconcile guide and implementation**
- [ ] **Step 6: User reviews the uncommitted diff**
- [ ] **Step 7: Commit after approval**

```bash
git commit -m "feat: compute retrieval metrics and segmented reports"
```

---

## 15. Open questions

- Whether the 14 `unlabelled` questions are reported as a segment or excluded from
  skill-segmented tables. Current decision: reported.
- Which evidence field the judge receives. `evidence_text` is the focused snippet and
  the likely choice; `evidence_text_full_page` remains available. Settled when the
  judge is designed.
- `completion_reserve`, to be set from the Slice 2 calibration rather than guessed.
- Whether OpenRouter's `provider` and `usage.cost` arrive as expected — confirmed by
  the first live call in Slice 3.
