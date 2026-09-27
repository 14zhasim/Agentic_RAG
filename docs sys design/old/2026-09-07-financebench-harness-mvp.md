# FinanceBench Harness MVP Implementation Plan

> **Historical input:** This agent-oriented plan produced the earlier engineered
> branch and is not the current implementation guide. Use `docs sys
> design/Benchmark.md` and `docs sys design/benchmark/FinanceBench Implementation
> Guide.md`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible FinanceBench 10-K harness that prepares and validates 112 questions and 64 PDFs, runs the closed-book/oracle/long-context conditions through GLM-5.3-Flash on OpenRouter, resumes interrupted runs, and produces deterministic retrieval/numeric metrics and segmented reports.

**Architecture:** A named `finrag_benchmark` package separates configuration, FinanceBench data, condition construction, model access, execution, and evaluation. All conditions produce a common `ConditionInput`, generation writes append-only row records, and reporting consumes saved records independently. The real retriever and Azure GPT judge are separate follow-up subsystems; this MVP exposes strict boundaries for them and refuses unsupported paid execution instead of emitting synthetic results.

**Tech Stack:** Python 3.12, uv, pandas, PyMuPDF, OpenAI Python SDK pointed at OpenRouter, pytest, standard-library `argparse`, `dataclasses`, `tomllib`, and `json`.

---

## Scope boundaries

This plan implements:

- Exact-version dependency management with `uv`.
- Deterministic FinanceBench 10-K preparation and validation.
- Closed-book, oracle, and long-context condition construction.
- Contracts and explicit unavailable errors for single-store/shared-store retrieval.
- GLM-5.3-Flash generation through OpenRouter.
- Context-size preflight, per-question persistence, failures, and resumption.
- Deterministic page retrieval metrics, numeric scoring, cognitive-skill normalization, and reporting.
- CLI commands for preparation, validation, dry-run, generation, reporting, and an explicit unavailable judge command.

Separate follow-up plans will cover:

- The dissertation retriever used by `single_store` and `shared_store`.
- Microsoft Foundry deployment and GPT-5.6 Luna judging.
- Optional RAGAS evaluation.
- LOFin/HiREC and the broader development test suite.

## Required rationale documentation in the Python source

The implementation must explain the important benchmark decisions at the point where they affect behavior. These comments are part of the deliverable, not optional cleanup. Prefer module and function docstrings for stable contracts, and short inline comments immediately above non-obvious decisions. Do not narrate ordinary Python syntax.

Use the following rationale in the corresponding files:

- `config.py`: explain that public, versioned TOML captures experiment choices while credentials stay in environment variables; paths resolve from the project root so commands behave consistently from different terminals; rejecting unknown/duplicate conditions prevents accidental mixing of incomparable runs.
- `records.py`: explain that `(doc_name, zero-indexed page_index)` is the page identity because the same page number in another filing is not relevant; retain the complete source row/evidence list so multi-page gold answers and future analyses are not lost.
- `data/financebench.py`: explain why the upstream clone is read-only, why `doc_type` is normalized only in a temporary series, why both original JSONL schemas and source order are preserved, why conflicting metadata is rejected before filtering, and why the manifest hashes every copied PDF. State explicitly that FinanceBench evidence page numbers are zero-indexed and are checked against the physical PDF.
- `conditions/financebench.py`: document that condition builders change only available context while the prompt/model remain constant; oracle deduplicates repeated evidence pages without losing dataset order; `single_store` limits scope to one filing while `shared_store` uses all 64; long-context keeps full page order and never performs beginning-only truncation.
- `generation/prompts.py`: explain that one shared prompt prevents prompt wording from becoming a confound across the five conditions.
- `generation/openrouter.py`: explain lazy client creation, provider pinning, why fallbacks are disabled for official comparisons, why the complete prompt plus output reserve is checked, and why the UTF-8 byte calculation is deliberately a safe upper bound rather than pretending to be an exact GLM tokenizer. Explain that returned usage/cost and measured latency are stored for auditability.
- `evaluation/retrieval_metrics.py`: explain ranked-page deduplication, document-aware relevance, and why metrics are not computed for non-retrieval conditions.
- `evaluation/numeric_scorer.py`: document the deliberately narrow one-number parser, percentage semantics, and significant-figure comparison; ambiguous answers remain for the later LLM judge instead of receiving a misleading deterministic score.
- `evaluation/segmentation.py`: explain that FinanceBench reasoning labels are inconsistent composites, so normalization is multi-label and group counts are intentionally non-additive.
- `runner.py`: explain stable job IDs, immutable config snapshots, append-and-fsync persistence, resume behavior, and why generation and judging are separate stages.
- `evaluation/reporting.py`: explain that reports consume saved rows without rerunning paid models, use the latest record per job, preserve incomplete-run counts, and macro-average within each requested segment.
- `cli.py`: explain why data/validation/dry-run commands do not construct an API client, why unavailable retrieval/judging fails explicitly, and why `--limit` is intended for bounded smoke calls.

Each module must begin with a concise domain-level docstring. Critical functions must include `Args`, `Returns`, and `Raises` only where those sections add information not already obvious from the signature. Tests should validate behavior, not exact comment wording.

## Final file map

```text
.env.example                              # secret names only
.gitignore                                # generated data/index/run exclusions
.python-version                           # Python 3.12 pin
pyproject.toml                            # exact direct dependencies and CLI entry point
uv.lock                                   # complete resolved dependency graph
configs/financebench.toml                 # public dataset/model/run settings
src/finrag_benchmark/__init__.py          # package version
src/finrag_benchmark/cli.py               # working bootstrap, then full argparse commands
src/finrag_benchmark/config.py            # TOML loading and validation
src/finrag_benchmark/records.py           # shared immutable records
src/finrag_benchmark/data/financebench.py # prepare, validate, load, PDF-page cache
src/finrag_benchmark/conditions/base.py   # retriever contract and unavailable error
src/finrag_benchmark/conditions/financebench.py # five condition builders
src/finrag_benchmark/generation/prompts.py      # shared message construction
src/finrag_benchmark/generation/openrouter.py   # lazy OpenRouter client and generation
src/finrag_benchmark/evaluation/retrieval_metrics.py # page metrics
src/finrag_benchmark/evaluation/numeric_scorer.py     # strict numeric scoring
src/finrag_benchmark/evaluation/segmentation.py       # cognitive skills
src/finrag_benchmark/evaluation/reporting.py          # CSV/JSON aggregates
src/finrag_benchmark/runner.py               # jobs, run store, resumption
tests/conftest.py                             # synthetic FinanceBench fixture
tests/test_config.py
tests/data/test_financebench.py
tests/conditions/test_financebench.py
tests/generation/test_openrouter.py
tests/evaluation/test_retrieval_metrics.py
tests/evaluation/test_numeric_scorer.py
tests/evaluation/test_segmentation.py
tests/evaluation/test_reporting.py
tests/test_runner.py
tests/test_cli.py
```

### Task 1: Scaffold the pinned uv package

**Files:**
- Create: `.python-version`
- Create: `pyproject.toml`
- Create: `uv.lock`
- Create: `.env.example`
- Create: `configs/financebench.toml`
- Create: `src/finrag_benchmark/__init__.py`
- Create: `src/finrag_benchmark/cli.py`
- Modify: `.gitignore`

- [ ] **Step 1: Initialize the Python 3.12 package**

Run:

```bash
uv init --package --python 3.12 --name finrag-benchmark
```

Expected: `.python-version`, `pyproject.toml`, and `src/finrag_benchmark/__init__.py` exist.

- [ ] **Step 2: Add exact direct and development dependencies**

Run:

```bash
uv add --bounds exact pandas pymupdf openai
uv add --dev --bounds exact pytest
```

Expected: every direct dependency in `pyproject.toml` uses `==`, and `uv.lock` exists. Do not manually choose versions; commit the exact compatible versions resolved together by uv.

- [ ] **Step 3: Configure a working bootstrap CLI entry point**

Set the project script in `pyproject.toml` to:

```toml
[project.scripts]
finrag-benchmark = "finrag_benchmark.cli:main"
```

Keep `requires-python = ">=3.12"`; `.python-version` supplies the exact interpreter pin while `uv.lock` supplies exact packages.

Create `src/finrag_benchmark/cli.py` so the installed entry point works before the full commands arrive in Task 11:

```python
"""Command-line entry point for the staged FinanceBench harness."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    """Parse the currently available CLI surface and print its help."""
    parser = argparse.ArgumentParser(
        prog="finrag-benchmark",
        description="Prepare, run, and evaluate the FinanceBench 10-K benchmark.",
    )
    parser.parse_args(argv)
    parser.print_help()
    return 0
```

Run `uv run finrag-benchmark` and expect help text beginning with `usage: finrag-benchmark`.

- [ ] **Step 4: Add generated-artifact exclusions and safe secret names**

Append these exact entries to `.gitignore` without changing the existing `/benchmarks/` and `/docs/` rules:

```gitignore
/data/financebench/
/vectorstores/
/results/
```

Create `.env.example`:

```dotenv
OPENROUTER_API_KEY=
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_ENDPOINT=
```

- [ ] **Step 5: Create the public MVP configuration**

Create `configs/financebench.toml`:

```toml
[dataset]
source_dir = "benchmarks/financebench"
output_dir = "data/financebench"
document_type = "10k"
expected_questions = 112
expected_documents = 64

[generation]
provider = "openrouter"
model = "z-ai/glm-5.3-flash"
base_url = "https://openrouter.ai/api/v1"
upstream_provider = "z-ai"
allow_fallbacks = false
context_window_tokens = 1310720
max_output_tokens = 1024
token_safety_margin = 4096
temperature = 0.0
reasoning_effort = "medium"
timeout_seconds = 180.0
max_retries = 5

[run]
conditions = ["closed_book", "oracle", "long_context"]
retrieval_depth = 5
results_dir = "results"
```

- [ ] **Step 6: Verify the environment resolves**

Run:

```bash
uv lock --check
uv run python -c "import pandas, pymupdf, openai; print('dependencies ok')"
```

Expected:

```text
dependencies ok
```

- [ ] **Step 7: Commit the scaffold**

```bash
git add .gitignore .env.example .python-version pyproject.toml uv.lock configs/financebench.toml src/finrag_benchmark/__init__.py src/finrag_benchmark/cli.py
git commit -m "build: scaffold pinned FinanceBench package"
```

### Task 2: Load and validate benchmark configuration

**Files:**
- Create: `src/finrag_benchmark/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing configuration tests**

Create `tests/test_config.py`:

```python
from pathlib import Path

import pytest

from finrag_benchmark.config import ConfigError, load_config


def write_config(path: Path, *, conditions: str = '["closed_book", "oracle"]') -> None:
    path.write_text(
        f"""
[dataset]
source_dir = "source"
output_dir = "prepared"
document_type = "10k"
expected_questions = 2
expected_documents = 2

[generation]
provider = "openrouter"
model = "z-ai/glm-5.3-flash"
base_url = "https://openrouter.ai/api/v1"
upstream_provider = "z-ai"
allow_fallbacks = false
context_window_tokens = 10000
max_output_tokens = 500
token_safety_margin = 100
temperature = 0.0
reasoning_effort = "medium"
timeout_seconds = 30.0
max_retries = 2

[run]
conditions = {conditions}
retrieval_depth = 5
results_dir = "results"
""".strip()
        + "\n",
        encoding="utf-8",
    )


def test_load_config_resolves_paths_from_project_root(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    write_config(config_path)

    config = load_config(config_path, project_root=tmp_path)

    assert config.dataset.source_dir == tmp_path / "source"
    assert config.dataset.output_dir == tmp_path / "prepared"
    assert config.run.results_dir == tmp_path / "results"
    assert config.run.conditions == ("closed_book", "oracle")
    assert config.generation.model == "z-ai/glm-5.3-flash"


def test_default_project_root_is_independent_of_shell_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    config_path = project / "configs/financebench.toml"
    config_path.parent.mkdir(parents=True)
    write_config(config_path)
    unrelated_cwd = tmp_path / "somewhere-else"
    unrelated_cwd.mkdir()
    monkeypatch.chdir(unrelated_cwd)

    config = load_config(config_path)

    assert config.dataset.source_dir == project / "source"
    assert config.dataset.output_dir == project / "prepared"
    assert config.run.results_dir == project / "results"


def test_load_config_rejects_unknown_condition(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    write_config(config_path, conditions='["closed_book", "invented"]')

    with pytest.raises(ConfigError, match="Unsupported conditions: invented"):
        load_config(config_path, project_root=tmp_path)


def test_load_config_rejects_impossible_token_budget(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    write_config(config_path)
    text = config_path.read_text(encoding="utf-8").replace(
        "context_window_tokens = 10000", "context_window_tokens = 550"
    )
    config_path.write_text(text, encoding="utf-8")

    with pytest.raises(ConfigError, match="Token budget"):
        load_config(config_path, project_root=tmp_path)


def test_load_config_rejects_empty_conditions(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    write_config(config_path, conditions="[]")
    with pytest.raises(ConfigError, match="at least one condition"):
        load_config(config_path, project_root=tmp_path)


@pytest.mark.parametrize(
    ("field", "old", "new"),
    [
        ("max_output_tokens", "500", "-1"),
        ("token_safety_margin", "100", "-1"),
        ("timeout_seconds", "30.0", "0.0"),
        ("max_retries", "2", "-1"),
        ("temperature", "0.0", "2.1"),
    ],
)
def test_load_config_rejects_invalid_generation_limits(
    tmp_path: Path, field: str, old: str, new: str
) -> None:
    config_path = tmp_path / "config.toml"
    write_config(config_path)
    text = config_path.read_text(encoding="utf-8").replace(
        f"{field} = {old}", f"{field} = {new}"
    )
    config_path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match=field):
        load_config(config_path, project_root=tmp_path)


def test_malformed_structure_is_reported_as_config_error(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    write_config(config_path)
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace("[generation]", "[wrong_section]"),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="Invalid configuration"):
        load_config(config_path, project_root=tmp_path)


@pytest.mark.parametrize(
    ("old", "new", "field"),
    [
        ('model = "z-ai/glm-5.3-flash"', "model = 123", "model"),
        ('allow_fallbacks = false', 'allow_fallbacks = "false"', "allow_fallbacks"),
        ('context_window_tokens = 10000', 'context_window_tokens = 10000.5', "context_window_tokens"),
        ('max_retries = 2', 'max_retries = 0.5', "max_retries"),
    ],
)
def test_load_config_rejects_wrong_generation_types(
    tmp_path: Path, old: str, new: str, field: str
) -> None:
    config_path = tmp_path / "config.toml"
    write_config(config_path)
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(old, new), encoding="utf-8"
    )
    with pytest.raises(ConfigError, match=field):
        load_config(config_path, project_root=tmp_path)
```

- [ ] **Step 2: Run the configuration tests and confirm failure**

Run:

```bash
uv run pytest tests/test_config.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'finrag_benchmark.config'`.

- [ ] **Step 3: Implement typed configuration**

Create `src/finrag_benchmark/config.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


ALL_CONDITIONS = frozenset(
    {"closed_book", "oracle", "single_store", "shared_store", "long_context"}
)


class ConfigError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    source_dir: Path
    output_dir: Path
    document_type: str
    expected_questions: int
    expected_documents: int


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    provider: str
    model: str
    base_url: str
    upstream_provider: str
    allow_fallbacks: bool
    context_window_tokens: int
    max_output_tokens: int
    token_safety_margin: int
    temperature: float
    reasoning_effort: str
    timeout_seconds: float
    max_retries: int


@dataclass(frozen=True, slots=True)
class RunConfig:
    conditions: tuple[str, ...]
    retrieval_depth: int
    results_dir: Path


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    dataset: DatasetConfig
    generation: GenerationConfig
    run: RunConfig


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_config(path: Path, *, project_root: Path | None = None) -> BenchmarkConfig:
    # Public configs live in <project>/configs. Anchoring their relative paths to
    # that project makes an installed command behave the same from every shell CWD.
    root = project_root.resolve() if project_root else path.resolve().parent.parent
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    dataset_raw = raw["dataset"]
    generation_raw = raw["generation"]
    run_raw = raw["run"]

    raw_conditions = run_raw["conditions"]
    if (
        not isinstance(raw_conditions, list)
        or not raw_conditions
        or not all(isinstance(item, str) for item in raw_conditions)
    ):
        raise ConfigError("Run conditions must be a non-empty list of strings")
    conditions = tuple(raw_conditions)
    unknown = sorted(set(conditions) - ALL_CONDITIONS)
    if unknown:
        raise ConfigError(f"Unsupported conditions: {', '.join(unknown)}")
    if len(conditions) != len(set(conditions)):
        raise ConfigError("Run conditions must be unique")

    for field in ("provider", "model", "base_url", "upstream_provider", "reasoning_effort"):
        if type(generation_raw[field]) is not str or not generation_raw[field].strip():
            raise ConfigError(f"{field} must be a non-empty string")
    if type(generation_raw["allow_fallbacks"]) is not bool:
        raise ConfigError("allow_fallbacks must be a boolean")
    for field in (
        "context_window_tokens", "max_output_tokens", "token_safety_margin", "max_retries"
    ):
        if type(generation_raw[field]) is not int:
            raise ConfigError(f"{field} must be an integer")
    for field in ("temperature", "timeout_seconds"):
        if type(generation_raw[field]) not in (int, float):
            raise ConfigError(f"{field} must be numeric")
    generation = GenerationConfig(
        provider=generation_raw["provider"],
        model=generation_raw["model"],
        base_url=generation_raw["base_url"],
        upstream_provider=generation_raw["upstream_provider"],
        allow_fallbacks=generation_raw["allow_fallbacks"],
        context_window_tokens=generation_raw["context_window_tokens"],
        max_output_tokens=generation_raw["max_output_tokens"],
        token_safety_margin=generation_raw["token_safety_margin"],
        temperature=float(generation_raw["temperature"]),
        reasoning_effort=generation_raw["reasoning_effort"],
        timeout_seconds=float(generation_raw["timeout_seconds"]),
        max_retries=generation_raw["max_retries"],
    )
    if generation.context_window_tokens <= 0:
        raise ConfigError("context_window_tokens must be positive")
    if generation.max_output_tokens <= 0:
        raise ConfigError("max_output_tokens must be positive")
    if generation.token_safety_margin < 0:
        raise ConfigError("token_safety_margin must be non-negative")
    if generation.timeout_seconds <= 0:
        raise ConfigError("timeout_seconds must be positive")
    if generation.max_retries < 0:
        raise ConfigError("max_retries must be non-negative")
    if not 0.0 <= generation.temperature <= 2.0:
        raise ConfigError("temperature must be between 0 and 2")
    reserved = generation.max_output_tokens + generation.token_safety_margin
    if reserved >= generation.context_window_tokens:
        raise ConfigError("Token budget must leave positive room for the input prompt")
    if generation.provider != "openrouter":
        raise ConfigError("This MVP supports generation.provider = 'openrouter' only")

    dataset = DatasetConfig(
        source_dir=_resolve(root, dataset_raw["source_dir"]),
        output_dir=_resolve(root, dataset_raw["output_dir"]),
        document_type=str(dataset_raw["document_type"]).strip().casefold(),
        expected_questions=int(dataset_raw["expected_questions"]),
        expected_documents=int(dataset_raw["expected_documents"]),
    )
    run = RunConfig(
        conditions=conditions,
        retrieval_depth=int(run_raw["retrieval_depth"]),
        results_dir=_resolve(root, run_raw["results_dir"]),
    )
    if dataset.expected_questions <= 0 or dataset.expected_documents <= 0:
        raise ConfigError("Expected dataset counts must be positive")
    if run.retrieval_depth <= 0:
        raise ConfigError("retrieval_depth must be positive")
    return BenchmarkConfig(dataset=dataset, generation=generation, run=run)


def load_config(path: Path, *, project_root: Path | None = None) -> BenchmarkConfig:
    """Return a validated config or one stable, CLI-safe `ConfigError`."""
    try:
        return _load_config(path, project_root=project_root)
    except ConfigError:
        raise
    except (KeyError, TypeError, ValueError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(f"Invalid configuration: {error}") from error
```

- [ ] **Step 4: Run the configuration tests**

Run:

```bash
uv run pytest tests/test_config.py -q
```

Expected: `18 passed`.

- [ ] **Step 5: Commit typed configuration**

```bash
git add src/finrag_benchmark/config.py tests/test_config.py
git commit -m "feat: load validated benchmark configuration"
```

### Task 3: Define shared benchmark records

**Files:**
- Create: `src/finrag_benchmark/records.py`
- Create: `tests/test_records.py`

- [ ] **Step 1: Write failing record tests**

Create `tests/test_records.py`:

```python
from finrag_benchmark.records import PageRef, QuestionRecord, RetrievedChunk


def test_question_record_deduplicates_gold_pages_in_dataset_order() -> None:
    row = {
        "financebench_id": "q1",
        "doc_name": "ACME_2024_10K",
        "question": "What was revenue?",
        "answer": "$10",
        "question_type": "metrics-generated",
        "question_reasoning": "Information extraction",
        "evidence": [
            {"doc_name": "ACME_2024_10K", "evidence_page_num": 2,
             "evidence_text_full_page": "first"},
            {"doc_name": "ACME_2024_10K", "evidence_page_num": 2,
             "evidence_text_full_page": "duplicate"},
            {"doc_name": "ACME_2024_10K", "evidence_page_num": 5,
             "evidence_text_full_page": "second"},
        ],
    }

    question = QuestionRecord.from_mapping(row)

    assert question.gold_pages == (
        PageRef("ACME_2024_10K", 2),
        PageRef("ACME_2024_10K", 5),
    )


def test_retrieved_chunk_rejects_empty_pages() -> None:
    try:
        RetrievedChunk("c1", "text", "ACME_2024_10K", (), 0.8, 1)
    except ValueError as error:
        assert str(error) == "Retrieved chunks must identify at least one page"
    else:
        raise AssertionError("Expected ValueError")


def test_question_record_does_not_retain_mutable_source_references() -> None:
    row = {
        "financebench_id": "q1", "doc_name": "doc", "question": "Question?",
        "answer": "Answer", "question_type": "type", "question_reasoning": None,
        "evidence": [{"doc_name": "doc", "evidence_page_num": 0,
            "evidence_text_full_page": "page"}],
    }
    record = QuestionRecord.from_mapping(row)
    row["evidence"][0]["evidence_page_num"] = 9
    assert record.gold_pages == (PageRef("doc", 0),)
    with pytest.raises(TypeError):
        record.evidence[0]["evidence_page_num"] = 4


@pytest.mark.parametrize("page_index", [True, 1.5, "2"])
def test_page_ref_rejects_non_integer_page_indices(page_index: object) -> None:
    with pytest.raises(TypeError, match="page_index must be an integer"):
        PageRef("doc", page_index)  # type: ignore[arg-type]


def test_retrieved_chunk_normalizes_pages_away_from_mutable_input() -> None:
    pages = [0, 1]
    chunk = RetrievedChunk("c1", "text", "doc", pages, 0.8, 1)  # type: ignore[arg-type]
    pages.clear()
    assert chunk.pages == (0, 1)
```

- [ ] **Step 2: Run the tests and confirm failure**

Run:

```bash
uv run pytest tests/test_records.py -q
```

Expected: collection fails because `finrag_benchmark.records` does not exist.

- [ ] **Step 3: Implement the records**

Create `src/finrag_benchmark/records.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping


def _freeze(value: Any) -> Any:
    """Detach nested JSON-like data and make containers read-only."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class PageRef:
    doc_name: str
    page_index: int

    def __post_init__(self) -> None:
        if type(self.page_index) is not int:
            raise TypeError("page_index must be an integer")
        if self.page_index < 0:
            raise ValueError("page_index must be zero or greater")

    def to_dict(self) -> dict[str, object]:
        return {"doc_name": self.doc_name, "page_index": self.page_index}


@dataclass(frozen=True, slots=True)
class QuestionRecord:
    financebench_id: str
    doc_name: str
    question: str
    answer: str
    question_type: str
    question_reasoning: str | None
    evidence: tuple[Mapping[str, Any], ...]
    raw: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "QuestionRecord":
        frozen = _freeze(row)
        return cls(
            financebench_id=str(frozen["financebench_id"]),
            doc_name=str(frozen["doc_name"]),
            question=str(frozen["question"]),
            answer=str(frozen["answer"]),
            question_type=str(frozen["question_type"]),
            question_reasoning=(
                None if frozen.get("question_reasoning") is None
                else str(frozen["question_reasoning"])
            ),
            evidence=frozen["evidence"],
            raw=frozen,
        )

    @property
    def gold_pages(self) -> tuple[PageRef, ...]:
        ordered: list[PageRef] = []
        seen: set[PageRef] = set()
        for evidence in self.evidence:
            page = PageRef(
                str(evidence.get("doc_name", evidence.get("evidence_doc_name"))),
                evidence["evidence_page_num"],
            )
            if page not in seen:
                seen.add(page)
                ordered.append(page)
        return tuple(ordered)


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    doc_name: str
    pages: tuple[int, ...]
    score: float
    rank: int

    def __post_init__(self) -> None:
        # Runtime callers may supply a list despite the annotation; detach it so
        # later caller mutation cannot change persisted retrieval provenance.
        object.__setattr__(self, "pages", tuple(self.pages))
        if not self.pages:
            raise ValueError("Retrieved chunks must identify at least one page")
        if any(type(page) is not int for page in self.pages):
            raise TypeError("Chunk page indices must be integers")
        if any(page < 0 for page in self.pages):
            raise ValueError("Chunk page indices must be zero or greater")
        if self.rank < 1:
            raise ValueError("Chunk rank must start at one")

    @property
    def page_refs(self) -> tuple[PageRef, ...]:
        return tuple(PageRef(self.doc_name, page) for page in self.pages)


@dataclass(frozen=True, slots=True)
class ConditionInput:
    financebench_id: str
    condition: str
    question: str
    context: str
    context_pages: tuple[PageRef, ...]
    retrieved_chunks: tuple[RetrievedChunk, ...]
```

- [ ] **Step 4: Run record tests**

Run:

```bash
uv run pytest tests/test_records.py -q
```

Expected: `7 passed`.

- [ ] **Step 5: Commit shared records**

```bash
git add src/finrag_benchmark/records.py tests/test_records.py
git commit -m "feat: define benchmark record contracts"
```

### Task 4: Prepare and validate the FinanceBench 10-K subset

**Files:**
- Create: `src/finrag_benchmark/data/__init__.py`
- Create: `src/finrag_benchmark/data/financebench.py`
- Create: `tests/conftest.py`
- Create: `tests/data/test_financebench.py`

- [ ] **Step 1: Create a synthetic source fixture**

Create `tests/conftest.py` with a `financebench_source` fixture that:

```python
import json
from pathlib import Path

import pymupdf
import pytest


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _write_pdf(path: Path, page_texts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = pymupdf.open()
    for text in page_texts:
        page = document.new_page()
        page.insert_text((72, 72), text)
    document.save(path)
    document.close()


@pytest.fixture
def financebench_source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    questions = [
        {
            "financebench_id": "q1", "company": "ACME",
            "doc_name": "ACME_2024_10K", "question_type": "metrics-generated",
            "question_reasoning": "Information extraction", "domain_question_num": None,
            "question": "What was revenue?", "answer": "$10.00",
            "justification": "Reported revenue.", "dataset_subset_label": "OPEN_SOURCE",
            "evidence": [{"evidence_text": "Revenue was $10.",
                "doc_name": "ACME_2024_10K", "evidence_page_num": 0,
                "evidence_text_full_page": "Revenue was $10."}],
        },
        {
            "financebench_id": "q2", "company": "BETA",
            "doc_name": "BETA_2023_10K", "question_type": "domain-relevant",
            "question_reasoning": "Numerical reasoning OR Logical reasoning",
            "domain_question_num": "dg01", "question": "What was the change?",
            "answer": "20.0%", "justification": "Calculated from two pages.",
            "dataset_subset_label": "OPEN_SOURCE",
            "evidence": [
                {"evidence_text": "Prior value 100.", "doc_name": "BETA_2023_10K",
                 "evidence_page_num": 0, "evidence_text_full_page": "Prior value 100."},
                {"evidence_text": "Current value 120.", "doc_name": "BETA_2023_10K",
                 "evidence_page_num": 1, "evidence_text_full_page": "Current value 120."},
            ],
        },
        {
            "financebench_id": "q3", "company": "ACME",
            "doc_name": "ACME_2024Q1_10Q", "question_type": "novel-generated",
            "question_reasoning": None, "domain_question_num": None,
            "question": "Quarterly question?", "answer": "No",
            "justification": "Quarterly evidence.", "dataset_subset_label": "OPEN_SOURCE",
            "evidence": [{"evidence_text": "No.", "doc_name": "ACME_2024Q1_10Q",
                "evidence_page_num": 0, "evidence_text_full_page": "No."}],
        },
    ]
    metadata = [
        {"doc_name": "ACME_2024_10K", "company": "ACME", "gics_sector": "Industrials",
         "doc_type": "10K", "doc_period": 2024, "doc_link": "https://example/acme"},
        {"doc_name": "BETA_2023_10K", "company": "BETA", "gics_sector": "Financials",
         "doc_type": "10k", "doc_period": 2023, "doc_link": "https://example/beta"},
        {"doc_name": "ACME_2024Q1_10Q", "company": "ACME", "gics_sector": "Industrials",
         "doc_type": "10q", "doc_period": 2024, "doc_link": "https://example/acme-q1"},
    ]
    _write_jsonl(source / "data/financebench_open_source.jsonl", questions)
    _write_jsonl(source / "data/financebench_document_information.jsonl", metadata)
    _write_pdf(source / "pdfs/ACME_2024_10K.pdf", ["Revenue was $10."])
    _write_pdf(source / "pdfs/BETA_2023_10K.pdf", ["Prior value 100.", "Current value 120."])
    _write_pdf(source / "pdfs/ACME_2024Q1_10Q.pdf", ["No."])
    return source
```

- [ ] **Step 2: Write failing preparation tests**

Create `tests/data/test_financebench.py` with tests that call:

```python
from dataclasses import replace
import json
from pathlib import Path

import pandas as pd
import pytest

from finrag_benchmark.config import load_config
from finrag_benchmark.data.financebench import (
    DataValidationError,
    load_prepared_questions,
    prepare_financebench,
    validate_prepared_financebench,
)


def test_prepare_preserves_source_schemas_and_multipage_evidence(
    tmp_path: Path, financebench_source: Path
) -> None:
    config_path = Path("configs/financebench.toml")
    config = load_config(config_path)
    dataset = replace(
        config.dataset,
        source_dir=financebench_source,
        output_dir=tmp_path / "prepared",
        expected_questions=2,
        expected_documents=2,
    )

    summary = prepare_financebench(dataset, source_revision="test-fixture")

    source_questions = pd.read_json(
        financebench_source / "data/financebench_open_source.jsonl", lines=True
    )
    prepared_questions = pd.read_json(summary.questions_path, lines=True)
    source_metadata = pd.read_json(
        financebench_source / "data/financebench_document_information.jsonl", lines=True
    )
    prepared_metadata = pd.read_json(summary.metadata_path, lines=True)
    assert list(prepared_questions.columns) == list(source_questions.columns)
    assert list(prepared_metadata.columns) == list(source_metadata.columns)
    assert len(prepared_questions) == 2
    assert len(prepared_questions.loc[prepared_questions.financebench_id == "q2", "evidence"].iloc[0]) == 2
    assert sorted(path.name for path in summary.pdf_dir.glob("*.pdf")) == [
        "ACME_2024_10K.pdf", "BETA_2023_10K.pdf"
    ]
    manifest = json.loads(summary.manifest_path.read_text(encoding="utf-8"))
    assert manifest["question_count"] == 2
    assert manifest["document_count"] == 2
    assert manifest["source_revision"] == "test-fixture"


def test_validation_rejects_out_of_range_evidence_page(
    tmp_path: Path, financebench_source: Path
) -> None:
    config = load_config(Path("configs/financebench.toml"))
    dataset = replace(config.dataset, source_dir=financebench_source,
        output_dir=tmp_path / "prepared", expected_questions=2, expected_documents=2)
    summary = prepare_financebench(dataset, source_revision="test-fixture")
    frame = pd.read_json(summary.questions_path, lines=True)
    frame.at[0, "evidence"][0]["evidence_page_num"] = 99
    frame.to_json(summary.questions_path, orient="records", lines=True, force_ascii=False)

    with pytest.raises(DataValidationError, match="outside PDF page range"):
        validate_prepared_financebench(dataset)


def test_load_prepared_questions_returns_question_records(
    tmp_path: Path, financebench_source: Path
) -> None:
    config = load_config(Path("configs/financebench.toml"))
    dataset = replace(config.dataset, source_dir=financebench_source,
        output_dir=tmp_path / "prepared", expected_questions=2, expected_documents=2)
    prepare_financebench(dataset, source_revision="test-fixture")

    records = load_prepared_questions(dataset.output_dir)

    assert [record.financebench_id for record in records] == ["q1", "q2"]
    assert len(records[1].gold_pages) == 2
```

- [ ] **Step 3: Run preparation tests and confirm failure**

Run:

```bash
uv run pytest tests/data/test_financebench.py -q
```

Expected: collection fails because `finrag_benchmark.data.financebench` does not exist.

- [ ] **Step 4: Implement preparation, validation, and loading**

Create `src/finrag_benchmark/data/financebench.py` with these public definitions and behavior:

```python
class DataValidationError(ValueError):
    pass

@dataclass(frozen=True, slots=True)
class PreparationSummary:
    questions_path: Path
    metadata_path: Path
    manifest_path: Path
    pdf_dir: Path
    question_count: int
    document_count: int

def prepare_financebench(
    config: DatasetConfig, *, source_revision: str | None = None
) -> PreparationSummary:
    """Prepare the selected FinanceBench subset and return its validated paths/counts."""
    raise NotImplementedError

def validate_prepared_financebench(config: DatasetConfig) -> PreparationSummary:
    """Validate existing prepared files without modifying them."""
    raise NotImplementedError

def load_prepared_questions(output_dir: Path) -> list[QuestionRecord]:
    """Load prepared rows into immutable question records in source order."""
    raise NotImplementedError
```

Implement the bodies with the following exact algorithm:

1. Read both JSONL files using `pd.read_json(..., lines=True)`.
2. Compare their columns against these required sets without reordering or dropping any source columns:

```python
QUESTION_COLUMNS = {
    "financebench_id", "company", "doc_name", "question_type",
    "question_reasoning", "domain_question_num", "question", "answer",
    "justification", "dataset_subset_label", "evidence",
}
DOCUMENT_COLUMNS = {
    "doc_name", "company", "gics_sector", "doc_type", "doc_period", "doc_link",
}
```

3. Create a temporary normalized series with `metadata["doc_type"].astype(str).str.strip().str.casefold()`; do not overwrite the original `doc_type` column.
4. Group by `doc_name` and raise `DataValidationError` if a name maps to more than one normalized type.
5. Map each question's `doc_name` to that normalized type and reject missing mappings.
6. Select questions matching `config.document_type`, preserving source order and columns.
7. Select only metadata rows referenced by those questions. Require exactly one metadata row per selected name.
8. Validate expected question/document counts and unique `financebench_id` values.
9. Validate every evidence item has upstream `doc_name` (accept legacy `evidence_doc_name` only in memory), integer `evidence_page_num >= 0`, and `evidence_text_full_page`; require its PDF to exist and its page index to be less than `pymupdf.open(pdf_path).page_count`. Do not add or rename nested evidence keys in prepared JSONL.
10. Write the two filtered frames with `orient="records"`, `lines=True`, and `force_ascii=False` to temporary sibling files, then replace the destination files.
11. Copy exactly the selected PDFs to `output_dir/pdfs`, overwriting selected names and deleting stale `*.pdf` files that are not in the selected set.
12. Hash copied PDFs with SHA-256. Write `manifest.json` with `source_path`, `source_revision`, `filter = {"field": "doc_type", "normalized_equals": "10k"}`, `expected_question_count`, `expected_document_count`, observed `question_count`, observed `document_count`, sorted `selected_pdf_filenames`, and a filename-to-hash `pdf_sha256` mapping.
13. If `source_revision` is absent, resolve it using `git -C <source_dir> rev-parse HEAD`; turn failure into `DataValidationError`.
14. `validate_prepared_financebench` repeats schema, count, uniqueness, PDF, evidence-page, and manifest-hash checks without modifying files.
15. `load_prepared_questions` reads the prepared question JSONL and returns `QuestionRecord.from_mapping(row)` in file order.

Do not flatten the two JSONL files and do not rename `answer` to `gold_answer` in prepared data.

- [ ] **Step 5: Run preparation tests**

Run:

```bash
uv run pytest tests/data/test_financebench.py -q
```

Expected: `3 passed`.

- [ ] **Step 6: Exercise the real FinanceBench clone**

Run:

```bash
uv run python -c "from pathlib import Path; from finrag_benchmark.config import load_config; from finrag_benchmark.data.financebench import prepare_financebench; c=load_config(Path('configs/financebench.toml')); s=prepare_financebench(c.dataset); print(s.question_count, s.document_count)"
```

Expected:

```text
112 64
```

- [ ] **Step 7: Commit data preparation**

```bash
git add src/finrag_benchmark/data tests/conftest.py tests/data/test_financebench.py
git commit -m "feat: prepare validated FinanceBench 10-K data"
```

### Task 5: Build the five condition inputs

**Files:**
- Create: `src/finrag_benchmark/conditions/__init__.py`
- Create: `src/finrag_benchmark/conditions/base.py`
- Create: `src/finrag_benchmark/conditions/financebench.py`
- Create: `src/finrag_benchmark/generation/__init__.py`
- Create: `src/finrag_benchmark/generation/prompts.py`
- Create: `tests/conditions/test_financebench.py`

- [ ] **Step 1: Write failing condition tests**

Create tests covering these exact assertions:

```python
def test_closed_book_has_no_context(question, pdf_repository):
    result = build_condition(question, "closed_book", pdf_repository,
        all_doc_names=(question.doc_name,), retriever=None, top_k=5)
    assert result.context == ""
    assert result.context_pages == ()


def test_oracle_includes_each_gold_page_once(question_with_duplicate_evidence, pdf_repository):
    result = build_condition(question_with_duplicate_evidence, "oracle", pdf_repository,
        all_doc_names=(question_with_duplicate_evidence.doc_name,), retriever=None, top_k=5)
    assert result.context.count("Page index: 0") == 1
    assert result.context_pages == (PageRef(question_with_duplicate_evidence.doc_name, 0),)


def test_long_context_preserves_pdf_page_order(question, pdf_repository):
    result = build_condition(question, "long_context", pdf_repository,
        all_doc_names=(question.doc_name,), retriever=None, top_k=5)
    assert result.context.index("Page index: 0") < result.context.index("Page index: 1")


@pytest.mark.parametrize("condition", ["single_store", "shared_store"])
def test_retrieval_condition_refuses_missing_retriever(condition, question, pdf_repository):
    with pytest.raises(RetrieverUnavailableError, match="No retriever is configured"):
        build_condition(question, condition, pdf_repository,
            all_doc_names=(question.doc_name,), retriever=None, top_k=5)
```

Use the synthetic PDFs from `tests/conftest.py` to construct the `question` and `pdf_repository` fixtures. Include an oracle fixture with two evidence entries for the same `(doc_name, page_index)`.

- [ ] **Step 2: Run condition tests and confirm failure**

Run:

```bash
uv run pytest tests/conditions/test_financebench.py -q
```

Expected: collection fails because the condition modules do not exist.

- [ ] **Step 3: Define the retriever boundary**

Create `src/finrag_benchmark/conditions/base.py`:

```python
from typing import Protocol

from finrag_benchmark.records import RetrievedChunk


class RetrieverUnavailableError(RuntimeError):
    pass


class Retriever(Protocol):
    def retrieve(
        self, *, question: str, document_scope: tuple[str, ...], top_k: int
    ) -> list[RetrievedChunk]:
        raise NotImplementedError
```

- [ ] **Step 4: Implement PDF access and condition construction**

In `src/finrag_benchmark/conditions/financebench.py`, implement:

```python
class PdfTextRepository:
    def __init__(self, pdf_dir: Path):
        self.pdf_dir = pdf_dir
        self._cache: dict[str, tuple[str, ...]] = {}

    def pages(self, doc_name: str) -> tuple[str, ...]:
        if doc_name not in self._cache:
            path = self.pdf_dir / f"{doc_name}.pdf"
            if not path.exists():
                raise FileNotFoundError(f"Missing prepared PDF: {path}")
            with pymupdf.open(path) as document:
                self._cache[doc_name] = tuple(page.get_text("text") for page in document)
        return self._cache[doc_name]


def _page_block(page: PageRef, text: str) -> str:
    return f"[Document: {page.doc_name} | Page index: {page.page_index}]\n{text.strip()}"


def build_condition(
    question: QuestionRecord,
    condition: str,
    pdf_repository: PdfTextRepository,
    *,
    all_doc_names: tuple[str, ...],
    retriever: Retriever | None,
    top_k: int,
) -> ConditionInput:
    if condition == "closed_book":
        return ConditionInput(question.financebench_id, condition, question.question, "", (), ())
    if condition == "oracle":
        blocks: list[str] = []
        pages: list[PageRef] = []
        seen: set[PageRef] = set()
        for evidence in question.evidence:
            page = PageRef(str(evidence.get("doc_name", evidence.get("evidence_doc_name"))), evidence["evidence_page_num"])
            if page not in seen:
                seen.add(page)
                pages.append(page)
                blocks.append(_page_block(page, str(evidence["evidence_text_full_page"])))
        return ConditionInput(question.financebench_id, condition, question.question,
            "\n\n".join(blocks), tuple(pages), ())
    if condition == "long_context":
        pages = tuple(
            PageRef(question.doc_name, index)
            for index, _ in enumerate(pdf_repository.pages(question.doc_name))
        )
        blocks = [
            _page_block(page, text)
            for page, text in zip(pages, pdf_repository.pages(question.doc_name), strict=True)
        ]
        return ConditionInput(question.financebench_id, condition, question.question,
            "\n\n".join(blocks), pages, ())
    if condition in {"single_store", "shared_store"}:
        if retriever is None:
            raise RetrieverUnavailableError(
                f"No retriever is configured for condition {condition}"
            )
        scope = (question.doc_name,) if condition == "single_store" else all_doc_names
        chunks = tuple(retriever.retrieve(
            question=question.question, document_scope=scope, top_k=top_k
        ))
        context = "\n\n".join(
            f"[Chunk: {chunk.chunk_id} | Document: {chunk.doc_name} | "
            f"Pages: {','.join(map(str, chunk.pages))} | Rank: {chunk.rank}]\n{chunk.text.strip()}"
            for chunk in chunks
        )
        pages = tuple(page for chunk in chunks for page in chunk.page_refs)
        return ConditionInput(question.financebench_id, condition, question.question,
            context, pages, chunks)
    raise ValueError(f"Unsupported condition: {condition}")
```

- [ ] **Step 5: Implement one shared generation prompt**

Create `src/finrag_benchmark/generation/prompts.py`:

```python
SYSTEM_PROMPT = (
    "You are a financial analyst answering a benchmark question. "
    "Use the supplied filing context when it is present. Perform any necessary "
    "calculation carefully. Give a concise answer and do not claim that context "
    "contains information that it does not contain."
)


def build_messages(*, question: str, context: str) -> list[dict[str, str]]:
    user = f"<context>\n{context}\n</context>\n\n<question>\n{question}\n</question>"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
```

- [ ] **Step 6: Run condition tests**

Run:

```bash
uv run pytest tests/conditions/test_financebench.py -q
```

Expected: all condition tests pass.

- [ ] **Step 7: Commit condition construction**

```bash
git add src/finrag_benchmark/conditions src/finrag_benchmark/generation tests/conditions
git commit -m "feat: build FinanceBench context conditions"
```

### Task 6: Implement deterministic retrieval metrics and segmentation

**Files:**
- Create: `src/finrag_benchmark/evaluation/__init__.py`
- Create: `src/finrag_benchmark/evaluation/retrieval_metrics.py`
- Create: `src/finrag_benchmark/evaluation/segmentation.py`
- Create: `tests/evaluation/test_retrieval_metrics.py`
- Create: `tests/evaluation/test_segmentation.py`

- [ ] **Step 1: Write failing page-metric tests**

Create `tests/evaluation/test_retrieval_metrics.py`:

```python
import pytest

from finrag_benchmark.evaluation.retrieval_metrics import calculate_page_metrics
from finrag_benchmark.records import PageRef, RetrievedChunk


def chunk(chunk_id: str, doc: str, pages: tuple[int, ...], rank: int) -> RetrievedChunk:
    return RetrievedChunk(chunk_id, "text", doc, pages, 1.0 / rank, rank)


def test_metrics_use_ranked_unique_pages() -> None:
    gold = (PageRef("doc", 2), PageRef("doc", 5))
    chunks = (
        chunk("c1", "doc", (1,), 1),
        chunk("c2", "doc", (2,), 2),
        chunk("c3", "doc", (2, 5), 3),
    )

    result = calculate_page_metrics(gold, chunks, retrieval_depth=3)

    assert result.retrieved_pages == (
        PageRef("doc", 1), PageRef("doc", 2), PageRef("doc", 5)
    )
    assert result.page_recall == 1.0
    assert result.page_precision == pytest.approx(2 / 3)
    assert result.page_mrr == 0.5


def test_wrong_document_page_is_not_relevant() -> None:
    result = calculate_page_metrics(
        (PageRef("correct", 2),),
        (chunk("c1", "wrong", (2,), 1),),
        retrieval_depth=1,
    )
    assert result.page_recall == 0.0
    assert result.page_precision == 0.0
    assert result.page_mrr == 0.0


def test_metrics_reject_empty_gold_pages() -> None:
    with pytest.raises(ValueError, match="Gold pages cannot be empty"):
        calculate_page_metrics((), (), retrieval_depth=5)


def test_mrr_uses_retrieved_chunk_rank_for_a_multipage_chunk() -> None:
    result = calculate_page_metrics(
        (PageRef("doc", 5),),
        (chunk("c1", "doc", (1, 5), 1),),
        retrieval_depth=1,
    )
    assert result.page_mrr == 1.0
```

- [ ] **Step 2: Write failing cognitive-skill tests**

Create `tests/evaluation/test_segmentation.py`:

```python
from finrag_benchmark.evaluation.segmentation import normalize_cognitive_skills


def test_normalizes_composite_reasoning_label() -> None:
    assert normalize_cognitive_skills(
        "Logical reasoning (based on numerical reasoning) OR Information extraction"
    ) == ("information_extraction", "numerical_reasoning", "logical_reasoning")


def test_missing_reasoning_is_unspecified() -> None:
    assert normalize_cognitive_skills(None) == ("unspecified",)
    assert normalize_cognitive_skills("None") == ("unspecified",)
```

- [ ] **Step 3: Run metric tests and confirm failure**

Run:

```bash
uv run pytest tests/evaluation/test_retrieval_metrics.py tests/evaluation/test_segmentation.py -q
```

Expected: collection fails because the evaluation modules do not exist.

- [ ] **Step 4: Implement page retrieval metrics**

Create `src/finrag_benchmark/evaluation/retrieval_metrics.py`:

```python
from dataclasses import dataclass

from finrag_benchmark.records import PageRef, RetrievedChunk


@dataclass(frozen=True, slots=True)
class PageMetrics:
    page_recall: float
    page_precision: float
    page_mrr: float
    gold_pages: tuple[PageRef, ...]
    retrieved_pages: tuple[PageRef, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "page_recall": self.page_recall,
            "page_precision": self.page_precision,
            "page_mrr": self.page_mrr,
            "gold_pages": [page.to_dict() for page in self.gold_pages],
            "retrieved_pages": [page.to_dict() for page in self.retrieved_pages],
        }


def calculate_page_metrics(
    gold_pages: tuple[PageRef, ...],
    chunks: tuple[RetrievedChunk, ...],
    *,
    retrieval_depth: int,
) -> PageMetrics:
    if not gold_pages:
        raise ValueError("Gold pages cannot be empty")
    if retrieval_depth <= 0:
        raise ValueError("retrieval_depth must be positive")
    ranked: list[PageRef] = []
    seen: set[PageRef] = set()
    ranked_chunks = sorted(chunks, key=lambda item: item.rank)[:retrieval_depth]
    for chunk in ranked_chunks:
        for page in chunk.page_refs:
            if page not in seen:
                seen.add(page)
                ranked.append(page)
    gold = set(gold_pages)
    hits = sum(page in gold for page in ranked)
    recall = hits / len(gold)
    precision = hits / len(ranked) if ranked else 0.0
    # Retrieval ranks chunks, not individual pages within a multi-page chunk.
    # Using chunk rank avoids inventing an arbitrary second ranking among its pages.
    reciprocal_rank = next(
        (1.0 / chunk.rank for chunk in ranked_chunks if set(chunk.page_refs) & gold),
        0.0,
    )
    return PageMetrics(recall, precision, reciprocal_rank, gold_pages, tuple(ranked))
```

- [ ] **Step 5: Implement cognitive-skill normalization**

Create `src/finrag_benchmark/evaluation/segmentation.py`:

```python
SKILLS = (
    ("information extraction", "information_extraction"),
    ("numerical reasoning", "numerical_reasoning"),
    ("logical reasoning", "logical_reasoning"),
)


def normalize_cognitive_skills(value: str | None) -> tuple[str, ...]:
    if value is None or value.strip().casefold() in {"", "none", "nan"}:
        return ("unspecified",)
    normalized = value.casefold()
    found = tuple(canonical for phrase, canonical in SKILLS if phrase in normalized)
    return found or ("unspecified",)
```

- [ ] **Step 6: Run metric tests**

Run:

```bash
uv run pytest tests/evaluation/test_retrieval_metrics.py tests/evaluation/test_segmentation.py -q
```

Expected: `6 passed`.

- [ ] **Step 7: Commit metrics and segmentation**

```bash
git add src/finrag_benchmark/evaluation tests/evaluation/test_retrieval_metrics.py tests/evaluation/test_segmentation.py
git commit -m "feat: calculate page metrics and reasoning segments"
```

### Task 7: Implement strict numeric scoring

**Files:**
- Create: `src/finrag_benchmark/evaluation/numeric_scorer.py`
- Create: `tests/evaluation/test_numeric_scorer.py`

- [ ] **Step 1: Write failing numeric-scoring tests**

Create `tests/evaluation/test_numeric_scorer.py`:

```python
from finrag_benchmark.evaluation.numeric_scorer import score_numeric_answer


def test_scores_currency_and_commas() -> None:
    result = score_numeric_answer("$1,577.00", "The answer is 1577.")
    assert result.applicable is True
    assert result.score == 1.0


def test_rounds_candidate_to_gold_significant_figures() -> None:
    result = score_numeric_answer("70.2%", "70.2033598585%")
    assert result.applicable is True
    assert result.score == 1.0


def test_does_not_conflate_fraction_and_percentage_points() -> None:
    result = score_numeric_answer("70.2%", "0.702")
    assert result.applicable is True
    assert result.score == 0.0


def test_declines_ambiguous_multi_number_answer() -> None:
    result = score_numeric_answer("Revenue increased from 100 to 120.", "20")
    assert result.applicable is False
    assert result.score is None
```

- [ ] **Step 2: Run numeric tests and confirm failure**

Run:

```bash
uv run pytest tests/evaluation/test_numeric_scorer.py -q
```

Expected: collection fails because `numeric_scorer` does not exist.

- [ ] **Step 3: Implement numeric parsing and significant-figure comparison**

Create `src/finrag_benchmark/evaluation/numeric_scorer.py`:

```python
from dataclasses import dataclass
import math
import re


NUMBER = re.compile(r"(?<![A-Za-z0-9])\(?\$?-?\d[\d,]*(?:\.\d+)?%?\)?")


@dataclass(frozen=True, slots=True)
class NumericScore:
    applicable: bool
    score: float | None
    gold_value: float | None
    candidate_value: float | None
    reason: str


def _single_token(text: str) -> str | None:
    matches = NUMBER.findall(text)
    return matches[0] if len(matches) == 1 else None


def _parse(token: str) -> tuple[float, bool]:
    stripped = token.strip()
    negative_parentheses = stripped.startswith("(") and stripped.endswith(")")
    is_percent = "%" in stripped
    numeric = stripped.strip("()").replace("$", "").replace(",", "").replace("%", "")
    value = float(numeric)
    return (-value if negative_parentheses else value), is_percent


def _significant_figures(token: str) -> int:
    numeric = token.strip().strip("()").replace("$", "").replace(",", "").replace("%", "")
    numeric = numeric.lstrip("+-")
    digits = numeric.replace(".", "").lstrip("0")
    return max(1, len(digits))


def _round_sigfigs(value: float, figures: int) -> float:
    if value == 0:
        return 0.0
    decimals = figures - int(math.floor(math.log10(abs(value)))) - 1
    return round(value, decimals)


def score_numeric_answer(gold: str, candidate: str) -> NumericScore:
    gold_token = _single_token(gold)
    candidate_token = _single_token(candidate)
    if gold_token is None or candidate_token is None:
        return NumericScore(False, None, None, None, "Expected exactly one numeric value in each answer")
    gold_value, gold_percent = _parse(gold_token)
    candidate_value, candidate_percent = _parse(candidate_token)
    if gold_percent != candidate_percent:
        return NumericScore(True, 0.0, gold_value, candidate_value, "Percentage representation differs")
    rounded_candidate = _round_sigfigs(candidate_value, _significant_figures(gold_token))
    correct = math.isclose(gold_value, rounded_candidate, rel_tol=1e-10, abs_tol=1e-10)
    return NumericScore(
        True, 1.0 if correct else 0.0, gold_value, candidate_value,
        "Candidate matches at the reference answer's significant-figure precision" if correct
        else "Candidate does not match at the reference answer's significant-figure precision",
    )
```

- [ ] **Step 4: Run numeric-scoring tests**

Run:

```bash
uv run pytest tests/evaluation/test_numeric_scorer.py -q
```

Expected: `4 passed`.

- [ ] **Step 5: Commit numeric scoring**

```bash
git add src/finrag_benchmark/evaluation/numeric_scorer.py tests/evaluation/test_numeric_scorer.py
git commit -m "feat: add strict numeric answer scorer"
```

### Task 8: Add context preflight and the OpenRouter generator

**Files:**
- Create: `src/finrag_benchmark/generation/openrouter.py`
- Create: `tests/generation/test_openrouter.py`

- [ ] **Step 1: Write failing generator tests with a fake client**

Create `tests/generation/test_openrouter.py` covering:

```python
from types import SimpleNamespace

import pytest

from finrag_benchmark.config import GenerationConfig
from finrag_benchmark.generation.openrouter import (
    ContextWindowExceeded,
    OpenRouterGenerator,
    conservative_token_upper_bound,
    get_openrouter_client,
)


def generation_config(**changes) -> GenerationConfig:
    values = dict(provider="openrouter", model="z-ai/glm-5.3-flash",
        base_url="https://openrouter.ai/api/v1", upstream_provider="z-ai",
        allow_fallbacks=False, context_window_tokens=1000, max_output_tokens=100,
        token_safety_margin=50, temperature=0.0, reasoning_effort="medium",
        timeout_seconds=30.0, max_retries=2)
    values.update(changes)
    return GenerationConfig(**values)


def test_client_is_lazy_and_requires_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY environment variable not set"):
        get_openrouter_client(generation_config())


def test_byte_upper_bound_counts_complete_messages() -> None:
    messages = [{"role": "user", "content": "£10"}]
    assert conservative_token_upper_bound(messages) >= len("£10".encode("utf-8"))


def test_generator_rejects_oversized_prompt_before_api_call() -> None:
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: None)))
    generator = OpenRouterGenerator(generation_config(context_window_tokens=160), client=client)
    with pytest.raises(ContextWindowExceeded):
        generator.generate([{"role": "user", "content": "x" * 20}])


def test_generator_passes_pinned_provider_and_returns_usage() -> None:
    captured = {}
    completion = SimpleNamespace(
        id="gen-1", model="z-ai/glm-5.3-flash", provider="Z.AI",
        choices=[SimpleNamespace(message=SimpleNamespace(content="The answer is 10."))],
        usage=SimpleNamespace(model_dump=lambda: {
            "prompt_tokens": 20, "completion_tokens": 6, "total_tokens": 26, "cost": 0.001
        }),
    )
    def create(**kwargs):
        captured.update(kwargs)
        return completion
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    result = OpenRouterGenerator(generation_config(), client=client).generate(
        [{"role": "user", "content": "short"}]
    )

    assert captured["model"] == "z-ai/glm-5.3-flash"
    assert captured["extra_body"]["provider"] == {
        "order": ["z-ai"], "allow_fallbacks": False
    }
    assert result.text == "The answer is 10."
    assert result.provider == "Z.AI"
    assert result.usage["total_tokens"] == 26
    assert result.latency_seconds >= 0.0
```

- [ ] **Step 2: Run generator tests and confirm failure**

Run:

```bash
uv run pytest tests/generation/test_openrouter.py -q
```

Expected: collection fails because `generation.openrouter` does not exist.

- [ ] **Step 3: Implement lazy client creation and generation**

Create `src/finrag_benchmark/generation/openrouter.py` with:

```python
from dataclasses import dataclass
import os
import time
from typing import Any

from openai import OpenAI

from finrag_benchmark.config import GenerationConfig


class ContextWindowExceeded(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class GenerationResult:
    text: str
    request_id: str
    model: str
    provider: str | None
    usage: dict[str, Any]
    latency_seconds: float


def conservative_token_upper_bound(messages: list[dict[str, str]]) -> int:
    payload_bytes = sum(
        len(message["role"].encode("utf-8")) + len(message["content"].encode("utf-8"))
        for message in messages
    )
    return payload_bytes + 32 * len(messages)


def get_openrouter_client(config: GenerationConfig) -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")
    return OpenAI(
        base_url=config.base_url,
        api_key=api_key,
        timeout=config.timeout_seconds,
        max_retries=config.max_retries,
    )


class OpenRouterGenerator:
    def __init__(self, config: GenerationConfig, *, client: Any | None = None):
        self.config = config
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = get_openrouter_client(self.config)
        return self._client

    def generate(self, messages: list[dict[str, str]]) -> GenerationResult:
        estimate = conservative_token_upper_bound(messages)
        reserved = self.config.max_output_tokens + self.config.token_safety_margin
        if estimate + reserved > self.config.context_window_tokens:
            raise ContextWindowExceeded(
                f"Conservative prompt bound {estimate} plus reserved {reserved} "
                f"exceeds context window {self.config.context_window_tokens}"
            )
        started = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_output_tokens,
            extra_body={
                "provider": {
                    "order": [self.config.upstream_provider],
                    "allow_fallbacks": self.config.allow_fallbacks,
                },
                "reasoning": {"effort": self.config.reasoning_effort},
            },
        )
        latency_seconds = time.perf_counter() - started
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("OpenRouter returned an empty completion")
        usage = response.usage.model_dump() if response.usage is not None else {}
        return GenerationResult(
            text=content,
            request_id=response.id,
            model=response.model,
            provider=getattr(response, "provider", None),
            usage=usage,
            latency_seconds=latency_seconds,
        )
```

The UTF-8 byte count is deliberately a conservative upper bound, not a tokenizer approximation. It counts the complete prompt and is used only to reject inputs safely; it never slices bytes or characters. Persist OpenRouter's returned token counts as the observed usage after successful requests.

- [ ] **Step 4: Run generator tests**

Run:

```bash
uv run pytest tests/generation/test_openrouter.py -q
```

Expected: `4 passed`.

- [ ] **Step 5: Commit the generator**

```bash
git add src/finrag_benchmark/generation/openrouter.py tests/generation/test_openrouter.py
git commit -m "feat: generate through pinned OpenRouter provider"
```

### Task 9: Add resumable run storage and orchestration

**Files:**
- Create: `src/finrag_benchmark/runner.py`
- Create: `tests/test_runner.py`

- [ ] **Step 1: Write failing run-store tests**

Create `tests/test_runner.py` with complete local fakes and a prepared two-question fixture:

```python
from dataclasses import replace
import json
from pathlib import Path

import pytest

from finrag_benchmark.conditions.base import RetrieverUnavailableError
from finrag_benchmark.config import BenchmarkConfig, load_config
from finrag_benchmark.data.financebench import prepare_financebench
from finrag_benchmark.generation.openrouter import GenerationResult
from finrag_benchmark.runner import run_benchmark


class FakeGenerator:
    def __init__(self) -> None:
        self.call_count = 0

    def generate(self, messages: list[dict[str, str]]) -> GenerationResult:
        self.call_count += 1
        return GenerationResult(
            text="10",
            request_id=f"request-{self.call_count}",
            model="z-ai/glm-5.3-flash",
            provider="Z.AI",
            usage={"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
            latency_seconds=0.01,
        )


class FailingOnceGenerator(FakeGenerator):
    def __init__(self) -> None:
        super().__init__()
        self.has_failed = False

    def generate(self, messages: list[dict[str, str]]) -> GenerationResult:
        if not self.has_failed:
            self.has_failed = True
            raise RuntimeError("temporary provider failure")
        return super().generate(messages)


@pytest.fixture
def runner_setup(
    tmp_path: Path, financebench_source: Path
) -> tuple[BenchmarkConfig, Path]:
    base_path = Path("configs/financebench.toml")
    base = load_config(base_path)
    dataset = replace(
        base.dataset,
        source_dir=financebench_source,
        output_dir=tmp_path / "prepared",
        expected_questions=2,
        expected_documents=2,
    )
    prepare_financebench(dataset, source_revision="test-fixture")
    config = replace(base, dataset=dataset)
    config_path = tmp_path / "runner-config.toml"
    config_path.write_text(base_path.read_text(encoding="utf-8"), encoding="utf-8")
    return config, config_path


def test_run_store_skips_successful_jobs_on_resume(
    tmp_path: Path, runner_setup: tuple[BenchmarkConfig, Path]
) -> None:
    config, config_path = runner_setup
    generator = FakeGenerator()
    run_dir = tmp_path / "results/run-1"
    first = run_benchmark(config, config_path=config_path,
        conditions=("closed_book",), run_dir=run_dir,
        generator=generator, retriever=None)
    second = run_benchmark(config, config_path=config_path,
        conditions=("closed_book",), run_dir=run_dir,
        generator=generator, retriever=None)
    assert first.generated == 2
    assert second.generated == 0
    assert second.skipped == 2
    assert generator.call_count == 2


def test_failed_job_is_recorded_and_retried_on_resume(
    tmp_path: Path, runner_setup: tuple[BenchmarkConfig, Path]
) -> None:
    config, config_path = runner_setup
    generator = FailingOnceGenerator()
    run_dir = tmp_path / "results/run-1"
    first = run_benchmark(config, config_path=config_path,
        conditions=("closed_book",), run_dir=run_dir,
        generator=generator, retriever=None)
    second = run_benchmark(config, config_path=config_path,
        conditions=("closed_book",), run_dir=run_dir,
        generator=generator, retriever=None)
    assert first.failed == 1
    assert second.generated == 1
    assert any(
        json.loads(line)["status"] == "error"
        for line in (run_dir / "errors.jsonl").read_text(encoding="utf-8").splitlines()
    )


def test_run_refuses_retrieval_condition_without_retriever(
    tmp_path: Path, runner_setup: tuple[BenchmarkConfig, Path]
) -> None:
    config, config_path = runner_setup
    with pytest.raises(RetrieverUnavailableError):
        run_benchmark(config, config_path=config_path,
            conditions=("single_store",), run_dir=tmp_path / "results/run-1",
            generator=FakeGenerator(), retriever=None)
```

- [ ] **Step 2: Run runner tests and confirm failure**

Run:

```bash
uv run pytest tests/test_runner.py -q
```

Expected: collection fails because `finrag_benchmark.runner` does not exist.

- [ ] **Step 3: Implement append-only run storage**

In `src/finrag_benchmark/runner.py`, create:

```python
@dataclass(frozen=True, slots=True)
class RunSummary:
    generated: int
    skipped: int
    failed: int
    run_dir: Path


class RunStore:
    def __init__(self, run_dir: Path, config_path: Path):
        self.run_dir = run_dir
        self.predictions_path = run_dir / "predictions.jsonl"
        self.errors_path = run_dir / "errors.jsonl"
        run_dir.mkdir(parents=True, exist_ok=True)
        snapshot = run_dir / "config.toml"
        incoming = config_path.read_bytes()
        if snapshot.exists() and snapshot.read_bytes() != incoming:
            raise ValueError("Run configuration does not match existing config.toml")
        if not snapshot.exists():
            snapshot.write_bytes(incoming)

    def successful_jobs(self) -> set[str]:
        if not self.predictions_path.exists():
            return set()
        jobs: set[str] = set()
        for line in self.predictions_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["status"] == "success":
                jobs.add(row["job_id"])
        return jobs

    @staticmethod
    def _append(path: Path, row: dict[str, object]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def write_prediction(self, row: dict[str, object]) -> None:
        self._append(self.predictions_path, row)

    def write_error(self, row: dict[str, object]) -> None:
        self._append(self.errors_path, row)
```

- [ ] **Step 4: Implement job orchestration**

Implement `run_benchmark` with this signature:

```python
def run_benchmark(
    config: BenchmarkConfig,
    *,
    config_path: Path,
    conditions: tuple[str, ...],
    run_dir: Path,
    generator: OpenRouterGenerator,
    retriever: Retriever | None,
) -> RunSummary:
```

The function must:

1. Reject `single_store`/`shared_store` before opening the run if `retriever is None`.
2. Load prepared questions and the sorted unique document names.
3. Use `PdfTextRepository(config.dataset.output_dir / "pdfs")`.
4. Create `job_id = f"{question.financebench_id}:{condition}"`.
5. Skip only job IDs whose latest saved status is `success`.
6. Build the `ConditionInput`, call `build_messages`, then `generator.generate`.
7. Call `normalize_cognitive_skills` and `score_numeric_answer`.
8. For retrieval conditions, call `calculate_page_metrics`; otherwise store `None` for all three page metrics.
9. Append a success record containing: `job_id`, `status`, `financebench_id`, `question`, `gold_answer`, `model_answer`, `eval_mode`, raw segmentation labels, normalized skills, gold pages, retrieved chunks, retrieval metrics, requested/returned model, provider, request ID, usage (including provider-reported cost when supplied), measured `latency_seconds`, and UTC completion timestamp.
10. Catch per-job exceptions, append an error-status record to `predictions.jsonl`, and append `job_id`, exception class, message, and UTC timestamp to `errors.jsonl`. Do not catch `KeyboardInterrupt` or `SystemExit`.

Use helper serializers for `PageRef`, `RetrievedChunk`, `NumericScore`, and `PageMetrics`; do not use `default=str`, because that can conceal schema mistakes.

- [ ] **Step 5: Run runner tests**

Run:

```bash
uv run pytest tests/test_runner.py -q
```

Expected: all runner tests pass.

- [ ] **Step 6: Commit resumable orchestration**

```bash
git add src/finrag_benchmark/runner.py tests/test_runner.py
git commit -m "feat: persist and resume benchmark generation"
```

### Task 10: Aggregate reports without rerunning models

**Files:**
- Create: `src/finrag_benchmark/evaluation/reporting.py`
- Create: `tests/evaluation/test_reporting.py`

- [ ] **Step 1: Write failing report tests**

Create `tests/evaluation/test_reporting.py` using four synthetic successful prediction rows and one error row. Assert:

```python
summary = write_reports(run_dir)
assert summary["jobs"]["total"] == 5
assert summary["jobs"]["successful"] == 4
assert summary["jobs"]["failed"] == 1
assert summary["jobs"]["completion_rate"] == 0.8

frame = pd.read_csv(run_dir / "summary.csv")
assert {"overall", "generation_method", "cognitive_skill", "cross_tab"}.issubset(
    set(frame["aggregation"])
)
assert set(frame["eval_mode"]) == {"closed_book", "oracle"}
assert "n" in frame.columns
assert "page_recall" in frame.columns
```

Include one row with two `cognitive_skills`; verify it contributes once to each corresponding skill group and that the cross-tab retains its `question_type`.

- [ ] **Step 2: Run reporting tests and confirm failure**

Run:

```bash
uv run pytest tests/evaluation/test_reporting.py -q
```

Expected: collection fails because `evaluation.reporting` does not exist.

- [ ] **Step 3: Implement deterministic aggregation**

Create `src/finrag_benchmark/evaluation/reporting.py` with:

```python
METRIC_COLUMNS = ["numeric_accuracy", "page_recall", "page_precision", "page_mrr"]


def _latest_by_job(rows: list[dict]) -> list[dict]:
    latest: dict[str, dict] = {}
    for row in rows:
        latest[row["job_id"]] = row
    return list(latest.values())


def _aggregate(frame: pd.DataFrame, by: list[str], label: str) -> pd.DataFrame:
    grouped = frame.groupby(by, dropna=False)
    counts = grouped.size().rename("n")
    metrics = grouped[METRIC_COLUMNS].mean(numeric_only=True)
    result = pd.concat([counts, metrics], axis=1).reset_index()
    result.insert(0, "aggregation", label)
    return result


def write_reports(run_dir: Path) -> dict[str, object]:
    prediction_path = run_dir / "predictions.jsonl"
    rows = [json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines()]
    latest = _latest_by_job(rows)
    successes = [row for row in latest if row["status"] == "success"]
    frame = pd.DataFrame(successes)
    for column in METRIC_COLUMNS:
        if column not in frame:
            frame[column] = None

    overall = _aggregate(frame.assign(scope="all"), ["eval_mode", "scope"], "overall")
    generation = _aggregate(frame, ["eval_mode", "question_type"], "generation_method")
    skill_frame = frame.explode("cognitive_skills").rename(
        columns={"cognitive_skills": "cognitive_skill"}
    )
    skills = _aggregate(skill_frame, ["eval_mode", "cognitive_skill"], "cognitive_skill")
    cross = _aggregate(
        skill_frame,
        ["eval_mode", "question_type", "cognitive_skill"],
        "cross_tab",
    )
    report = pd.concat([overall, generation, skills, cross], ignore_index=True, sort=False)
    report.to_csv(run_dir / "summary.csv", index=False)

    total = len(latest)
    successful = len(successes)
    summary = {
        "jobs": {
            "total": total,
            "successful": successful,
            "failed": total - successful,
            "completion_rate": successful / total if total else 0.0,
        },
        "aggregates": report.where(pd.notna(report), None).to_dict(orient="records"),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary
```

Before building `frame`, raise `ValueError("No prediction records found")` for an empty or missing predictions file. Sort the combined report by `aggregation`, `eval_mode`, `question_type`, and `cognitive_skill` using stable sorting before writing it.

- [ ] **Step 4: Run reporting tests**

Run:

```bash
uv run pytest tests/evaluation/test_reporting.py -q
```

Expected: all reporting tests pass.

- [ ] **Step 5: Commit reporting**

```bash
git add src/finrag_benchmark/evaluation/reporting.py tests/evaluation/test_reporting.py
git commit -m "feat: aggregate segmented benchmark reports"
```

### Task 11: Expose the staged command-line workflow

**Files:**
- Create: `src/finrag_benchmark/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write failing CLI tests**

Create `tests/test_cli.py` with direct calls to `main(argv)` and captured output:

```python
from pathlib import Path

import pytest

from finrag_benchmark.cli import main


@pytest.fixture
def cli_config_path(tmp_path: Path, financebench_source: Path) -> Path:
    config_path = tmp_path / "financebench.toml"
    config_path.write_text(
        f'''[dataset]
source_dir = "{financebench_source.as_posix()}"
output_dir = "{(tmp_path / "prepared").as_posix()}"
document_type = "10k"
expected_questions = 2
expected_documents = 2

[generation]
provider = "openrouter"
model = "z-ai/glm-5.3-flash"
base_url = "https://openrouter.ai/api/v1"
upstream_provider = "z-ai"
allow_fallbacks = false
context_window_tokens = 10000
max_output_tokens = 500
token_safety_margin = 100
temperature = 0.0
reasoning_effort = "medium"
timeout_seconds = 30.0
max_retries = 2

[run]
conditions = ["closed_book", "oracle", "long_context"]
retrieval_depth = 5
results_dir = "{(tmp_path / "results").as_posix()}"
''',
        encoding="utf-8",
    )
    return config_path


def test_judge_command_fails_clearly(capsys) -> None:
    with pytest.raises(SystemExit) as error:
        main(["judge", "--run-dir", "results/example"])
    assert error.value.code == 2
    assert "Azure judge is not configured in the FinanceBench MVP" in capsys.readouterr().err


def test_dry_run_does_not_require_openrouter_key(
    monkeypatch, cli_config_path: Path, capsys
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert main(["prepare", "--config", str(cli_config_path),
        "--source-revision", "test-fixture"]) == 0
    capsys.readouterr()
    exit_code = main([
        "run", "--config", str(cli_config_path),
        "--conditions", "closed_book", "oracle", "--limit", "1", "--dry-run",
    ])
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Planned jobs: 2" in output
    assert "API requests sent: 0" in output


def test_prepare_then_validate(cli_config_path: Path, capsys) -> None:
    assert main(["prepare", "--config", str(cli_config_path),
        "--source-revision", "test-fixture"]) == 0
    assert "Prepared 2 questions and 2 documents" in capsys.readouterr().out
    assert main(["validate", "--config", str(cli_config_path)]) == 0
    assert "Validated 2 questions and 2 documents" in capsys.readouterr().out
```

- [ ] **Step 2: Run CLI tests and confirm failure**

Run:

```bash
uv run pytest tests/test_cli.py -q
```

Expected: tests fail because the bootstrap parser does not yet define `prepare`, `validate`, `run`, `report`, or `judge`.

- [ ] **Step 3: Implement the CLI parser and commands**

Create `src/finrag_benchmark/cli.py` with `main(argv: list[str] | None = None) -> int`. Define these subcommands:

```text
finrag-benchmark prepare --config PATH
finrag-benchmark validate --config PATH
finrag-benchmark run --config PATH [--conditions NAME ...] [--run-dir PATH] [--limit N] [--dry-run]
finrag-benchmark report --run-dir PATH
finrag-benchmark judge --run-dir PATH
```

Exact behavior:

- `prepare`: load configuration, accept optional `--source-revision`, run `prepare_financebench`, and print question/document counts and paths.
- `validate`: run `validate_prepared_financebench` and print `Validated 112 questions and 64 documents` for the real config.
- `run --limit N`: require a positive integer and slice the source-ordered questions to the first `N` questions before expanding them into condition jobs. Reject zero and negative values with `parser.error("--limit must be a positive integer")`.
- `run --dry-run`: validate selected conditions; load questions; reject missing prepared data; construct each non-retrieval condition so long-context PDFs are actually read; print total job count, conditions, conservative input bounds by condition, and `API requests sent: 0`; never construct `OpenRouterGenerator.client`.
- `run` without `--dry-run`: use the supplied run directory, or create `results/<UTC timestamp>-<first eight hex characters of SHA-256(config bytes)>`; construct `OpenRouterGenerator`; call `run_benchmark`; print generated/skipped/failed counts and the run path.
- `report`: call `write_reports` and print total/success/failed counts and output paths.
- `judge`: call `parser.error("Azure judge is not configured in the FinanceBench MVP")`, which exits with status 2.
- Catch `ConfigError`, `DataValidationError`, `RetrieverUnavailableError`, and missing-key `RuntimeError` at the outer boundary; print a one-line error to stderr and return 2. Do not expose a traceback for expected user/configuration errors.

Use `argparse.ArgumentParser`; do not add a CLI framework dependency.

- [ ] **Step 4: Run CLI tests**

Run:

```bash
uv run pytest tests/test_cli.py -q
```

Expected: all CLI tests pass.

- [ ] **Step 5: Exercise the no-spend workflow**

Run:

```bash
uv run finrag-benchmark prepare --config configs/financebench.toml
uv run finrag-benchmark validate --config configs/financebench.toml
uv run finrag-benchmark run --config configs/financebench.toml --conditions closed_book oracle long_context --dry-run
```

Expected: preparation and validation report 112 questions and 64 documents; dry-run reports 336 planned jobs and zero API requests.

- [ ] **Step 6: Commit the CLI**

```bash
git add src/finrag_benchmark/cli.py tests/test_cli.py
git commit -m "feat: expose FinanceBench staged CLI"
```

### Task 12: Run the complete local verification and update status documentation

**Files:**
- Modify: `README.md`
- Modify: `docs sys design/Benchmark Progress.md`

- [ ] **Step 1: Run the complete test suite**

Run:

```bash
uv run pytest -q
```

Expected: all tests pass with no warnings from project code.

- [ ] **Step 2: Verify exact dependency pins**

Run:

```bash
uv lock --check
uv tree
```

Expected: the lockfile is current; the tree contains pandas, PyMuPDF, OpenAI, and the pytest development dependency. Confirm direct dependencies in `pyproject.toml` use `==`.

- [ ] **Step 3: Verify ignored artifacts and secrets**

Run:

```bash
git check-ignore .env data/financebench vectorstores results docs/superpowers/plans/2026-09-07-financebench-harness-mvp.md
git grep -nE 'sk-or-v1-|AZURE_OPENAI_API_KEY=.+|OPENROUTER_API_KEY=.+'
```

Expected: the first command identifies every path as ignored. The second prints nothing.

- [ ] **Step 4: Update README from planned to implemented behavior**

In `README.md`:

- Change `## Planned project structure` to `## Project structure`.
- Add the five exact CLI commands from Task 11 under a `## Quick start` section.
- Expand the `runner` entry: it creates stable question-condition jobs, invokes condition construction/generation, saves every result immediately, records failures, and skips successful jobs during resume. Clarify that the OpenAI SDK performs bounded HTTP retries while the runner performs job-level checkpointing/resumption.
- Expand the `configs` entry: these are committed, non-secret experiment choices (model/provider identifiers, selected conditions, retrieval depth, token limits, chunk settings, and prompt versions) that change run behavior without editing Python. Credentials remain environment variables.
- Add a `## Run artifacts` glossary defining `config.toml`, `predictions.jsonl`, `judgments.jsonl`, `summary.json`, `summary.csv`, and `errors.jsonl`, including why generation and judgments are stored separately.
- Add a `## Retrieval metrics` section with the worked example: gold pages `{(A, 2), (A, 5)}` and ranked unique retrieved pages `[(A, 1), (A, 2), (A, 5)]` produce recall `1.0`, precision `2/3`, and MRR `1/2`. State that page identity includes `doc_name`, chunk pages are deduplicated in first-ranked order, and the three non-retrieval conditions report these metrics as not applicable.
- State that `single_store`, `shared_store`, and Azure judging deliberately return unavailable errors in this MVP.
- Explain that the long-context preflight uses a conservative UTF-8 byte upper bound and that successful API responses supply observed token usage.
- Do not claim that an OpenRouter paid run or Azure judge has been executed unless it actually has.

- [ ] **Step 5: Update Benchmark Progress with verified implementation state**

Move these completed items from `Planned next` to `Implemented or verified` in `docs sys design/Benchmark Progress.md`:

- uv package and lockfile
- deterministic 112-question/64-PDF preparation
- shared records and condition interfaces
- closed-book, oracle, and long-context construction
- OpenRouter generation integration
- resumable run artifacts
- deterministic retrieval metrics, numeric scoring, segmentation, and reports
- minimum automated tests

Keep the actual retriever, Azure judge deployment/integration, judge validation, RAGAS, LOFin/HiREC, and broader development suite under planned or deferred scope.

- [ ] **Step 6: Review the final diff**

Run:

```bash
git status --short
git diff --check
git diff -- README.md "docs sys design/Benchmark Progress.md"
```

Expected: only the intended README and progress updates remain unstaged; `git diff --check` prints nothing.

- [ ] **Step 7: Commit verified documentation**

```bash
git add README.md "docs sys design/Benchmark Progress.md"
git commit -m "docs: record FinanceBench MVP workflow"
```

### Task 13: Perform one explicitly authorized OpenRouter smoke call

**Files:**
- Generated and ignored: `results/<run-id>/config.toml`
- Generated and ignored: `results/<run-id>/predictions.jsonl`
- Generated and ignored: `results/<run-id>/errors.jsonl`

- [ ] **Step 1: Confirm the key without printing it**

Run inside a newly created VS Code terminal:

```bash
uv run python -c "import os; print('OPENROUTER_API_KEY set:', bool(os.getenv('OPENROUTER_API_KEY')))"
```

Expected:

```text
OPENROUTER_API_KEY set: True
```

- [ ] **Step 2: Run one closed-book question only after the user approves paid API use**

The `--limit` option and its dry-run test were implemented in Task 11. Run exactly one question into a named smoke directory:

```bash
uv run finrag-benchmark run --config configs/financebench.toml --conditions closed_book --limit 1 --run-dir results/smoke-openrouter
```

Expected: one successful prediction, no error record, and non-empty returned model/provider/usage fields. If OpenRouter rejects the configured provider slug, copy the current Z.AI provider slug from the GLM-5.3-Flash model page, update `upstream_provider`, rerun the generator tests, and repeat this single request.

- [ ] **Step 3: Inspect the generated record without exposing credentials**

Inspect the fixed smoke-run path:

```bash
uv run python -c "import json, pathlib; p=pathlib.Path('results/smoke-openrouter/predictions.jsonl'); r=json.loads(p.read_text().splitlines()[-1]); print({k:r[k] for k in ('status','financebench_id','eval_mode','model_answer','returned_model','provider','usage')})"
```

Expected: `status` is `success`, `eval_mode` is `closed_book`, and usage includes observed prompt/completion token counts. No commit follows because this task creates only ignored run artifacts.

## Final acceptance checks

Run:

```bash
uv run pytest -q
uv lock --check
uv run finrag-benchmark validate --config configs/financebench.toml
uv run finrag-benchmark run --config configs/financebench.toml --conditions closed_book oracle long_context --dry-run
git status --short
```

Expected outcomes:

- All automated tests pass.
- The lockfile is current and all direct dependencies are exact pins.
- Validation reports 112 questions and 64 PDFs.
- Dry-run reports 336 jobs and zero API requests.
- No generated data, results, vector stores, secrets, design specs, or plan files appear in Git status.
- The branch contains small, independently reviewable commits for each completed task.
