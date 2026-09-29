"""Tests for the RAG-system configuration separate from benchmark settings."""

from __future__ import annotations

from pathlib import Path

import pytest

from sec_rag.config import load_config

VALID_CONFIG = """[corpus]
prepared_dir = "data/financebench"
parsed_dir = "data/financebench/parsed"
chunks_dir = "data/financebench/chunks"
indexes_dir = "data/financebench/indexes"
expected_documents = 64

[parsing]
provider = "azure-document-intelligence"
model_id = "prebuilt-layout"
output_content_format = "markdown"

[chunking]
floor_tokens = 250
ceiling_tokens = 1024
overlap_cap_tokens = 100

[bm25]
token_pattern = '(?u)[^\\W\\d_]+|\\d+'
stopwords = "en"
stemmer = "english"

[embedding]
model = "voyage-4-lite"
batch_size = 128
output_dimension = 1024
output_dtype = "float"
usd_per_million_tokens = 0.02

[retrieval]
candidates_per_search = 50
rrf_k = 60
rerank_candidates = 50

[query_enhancement]
model = "z-ai/glm-5.3-flash"
base_url = "https://openrouter.ai/api/v1"
upstream_provider = "z-ai"
reasoning_effort = "low"
temperature = 0.0
max_output_tokens = 4096
timeout_seconds = 120.0
max_retries = 5
prompt_version = "exp1-query-enhancement-v1"
reuse_plans_from = ""

[rerank]
model = "rerank-3-lite"
usd_per_million_tokens = 0.02

[structure]
max_depth = 6
softmax_divisor = 0.05
weight = 0.0
"""


def _write(tmp_path: Path, text: str) -> tuple[Path, Path]:
    """Write a config at <project>/configs/sec_rag.toml; return (project, config path)."""
    project = tmp_path / "project"
    config_dir = project / "configs"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "sec_rag.toml"
    config_path.write_text(text, encoding="utf-8")
    return project, config_path


def test_load_config_resolves_corpus_paths_from_project_root(tmp_path: Path) -> None:
    """The parser config is portable because paths resolve beside configs/."""
    project, config_path = _write(tmp_path, VALID_CONFIG)

    config = load_config(config_path)

    assert config["corpus"]["prepared_dir"] == str(project / "data/financebench")
    assert config["corpus"]["parsed_dir"] == str(project / "data/financebench/parsed")
    assert config["corpus"]["chunks_dir"] == str(project / "data/financebench/chunks")
    assert config["corpus"]["indexes_dir"] == str(project / "data/financebench/indexes")
    assert config["chunking"] == {
        "floor_tokens": 250,
        "ceiling_tokens": 1024,
        "overlap_cap_tokens": 100,
    }
    assert config["bm25"] == {
        "token_pattern": r"(?u)[^\W\d_]+|\d+",
        "stopwords": "en",
        "stemmer": "english",
    }
    assert config["embedding"] == {
        "model": "voyage-4-lite",
        "batch_size": 128,
        "output_dimension": 1024,
        "output_dtype": "float",
        "usd_per_million_tokens": 0.02,
    }


def test_missing_bm25_section_is_refused(tmp_path: Path) -> None:
    text = VALID_CONFIG.replace("[bm25]", "[not_bm25]")
    _, config_path = _write(tmp_path, text)

    with pytest.raises(ValueError, match=r"\[bm25\]"):
        load_config(config_path)


def test_empty_bm25_setting_is_refused(tmp_path: Path) -> None:
    text = VALID_CONFIG.replace('stemmer = "english"', 'stemmer = ""')
    _, config_path = _write(tmp_path, text)

    with pytest.raises(ValueError, match="bm25.stemmer"):
        load_config(config_path)


def test_missing_embedding_section_is_refused(tmp_path: Path) -> None:
    text = VALID_CONFIG.replace("[embedding]", "[not_embedding]")
    _, config_path = _write(tmp_path, text)

    with pytest.raises(ValueError, match=r"\[embedding\]"):
        load_config(config_path)


@pytest.mark.parametrize(
    ("setting", "bad_line", "message"),
    [
        ('model = "voyage-4-lite"', 'model = ""', "embedding.model"),
        ("batch_size = 128", "batch_size = 0", "embedding.batch_size"),
        ("batch_size = 128", "batch_size = 1001", "embedding.batch_size"),
        ("batch_size = 128", "batch_size = true", "embedding.batch_size"),
        (
            "output_dimension = 1024",
            "output_dimension = 1000",
            "embedding.output_dimension",
        ),
        ('output_dtype = "float"', 'output_dtype = "int8"', "embedding.output_dtype"),
    ],
)
def test_invalid_embedding_setting_is_refused(
    tmp_path: Path, setting: str, bad_line: str, message: str
) -> None:
    """Voyage's limits: 1-1,000 texts per call, four lengths, floats for Chroma."""
    _, config_path = _write(tmp_path, VALID_CONFIG.replace(setting, bad_line))

    with pytest.raises(ValueError, match=message):
        load_config(config_path)


def test_missing_chunking_section_is_refused(tmp_path: Path) -> None:
    text = VALID_CONFIG.replace("[chunking]", "[not_chunking]")
    _, config_path = _write(tmp_path, text)

    with pytest.raises(ValueError, match=r"\[chunking\]"):
        load_config(config_path)


@pytest.mark.parametrize("bad_value", ["0", "-5", "true", '"250"'])
def test_chunking_settings_must_be_positive_whole_numbers(
    tmp_path: Path, bad_value: str
) -> None:
    text = VALID_CONFIG.replace("floor_tokens = 250", f"floor_tokens = {bad_value}")
    _, config_path = _write(tmp_path, text)

    with pytest.raises(ValueError, match="chunking.floor_tokens"):
        load_config(config_path)


def test_ceiling_under_twice_the_floor_is_refused(tmp_path: Path) -> None:
    """The halving rule can only promise both halves clear the floor if ceiling >= 2 x floor."""
    text = VALID_CONFIG.replace("ceiling_tokens = 1024", "ceiling_tokens = 400")
    _, config_path = _write(tmp_path, text)

    with pytest.raises(ValueError, match="at least twice"):
        load_config(config_path)


def test_missing_retrieval_section_is_refused(tmp_path: Path) -> None:
    text = VALID_CONFIG.replace("[retrieval]", "[not_retrieval]")
    _, config_path = _write(tmp_path, text)

    with pytest.raises(ValueError, match=r"\[retrieval\]"):
        load_config(config_path)


@pytest.mark.parametrize("bad_value", ["0", "-1", "true", '"60"'])
def test_retrieval_settings_must_be_positive_whole_numbers(
    tmp_path: Path, bad_value: str
) -> None:
    text = VALID_CONFIG.replace("rrf_k = 60", f"rrf_k = {bad_value}")
    _, config_path = _write(tmp_path, text)

    with pytest.raises(ValueError, match="retrieval.rrf_k"):
        load_config(config_path)


@pytest.mark.parametrize("section", ["[query_enhancement]", "[rerank]"])
def test_missing_query_enhancement_or_rerank_section_is_refused(
    tmp_path: Path, section: str
) -> None:
    text = VALID_CONFIG.replace(section, "[something_else]")
    _, config_path = _write(tmp_path, text)

    with pytest.raises(ValueError, match="sections"):
        load_config(config_path)


@pytest.mark.parametrize(
    ("setting", "bad_line", "message"),
    [
        ('reasoning_effort = "low"', 'reasoning_effort = "medium"', "reasoning_effort"),
        ('model = "z-ai/glm-5.3-flash"', 'model = ""', "query_enhancement.model"),
        ("max_output_tokens = 4096", "max_output_tokens = -1", "max_output_tokens"),
        ("temperature = 0.0", "temperature = true", "temperature"),
        ('model = "rerank-3-lite"', 'model = ""', "rerank.model"),
    ],
)
def test_invalid_query_enhancement_or_rerank_setting_is_refused(
    tmp_path: Path, setting: str, bad_line: str, message: str
) -> None:
    _, config_path = _write(tmp_path, VALID_CONFIG.replace(setting, bad_line))

    with pytest.raises(ValueError, match=message):
        load_config(config_path)


@pytest.mark.parametrize("section", ["embedding", "rerank"])
@pytest.mark.parametrize(
    "bad_line",
    ["usd_per_million_tokens = -0.01", "usd_per_million_tokens = true", ""],
)
def test_voyage_list_price_must_be_a_number_of_at_least_zero(
    tmp_path: Path, section: str, bad_line: str
) -> None:
    """Build Order 2.5: Voyage cost = tokens x list price, so the price must exist."""
    head, _, tail = VALID_CONFIG.partition(f"[{section}]")
    tail = tail.replace("usd_per_million_tokens = 0.02", bad_line, 1)
    _, config_path = _write(tmp_path, head + f"[{section}]" + tail)

    with pytest.raises(ValueError, match=f"{section}.usd_per_million_tokens"):
        load_config(config_path)


def test_missing_structure_section_is_refused(tmp_path: Path) -> None:
    _, config_path = _write(tmp_path, VALID_CONFIG.replace("[structure]", "[other]"))

    with pytest.raises(ValueError, match=r"\[structure\]"):
        load_config(config_path)


@pytest.mark.parametrize(
    ("setting", "bad_line", "message"),
    [
        ("max_depth = 6", "max_depth = 0", "structure.max_depth"),
        ("max_depth = 6", "max_depth = true", "structure.max_depth"),
        ("softmax_divisor = 0.05", "softmax_divisor = 0", "structure.softmax_divisor"),
        ("weight = 0.0", "weight = -1.0", "structure.weight"),
        ('reuse_plans_from = ""', "reuse_plans_from = 0", "reuse_plans_from"),
    ],
)
def test_invalid_structure_setting_is_refused(
    tmp_path: Path, setting: str, bad_line: str, message: str
) -> None:
    _, config_path = _write(tmp_path, VALID_CONFIG.replace(setting, bad_line))

    with pytest.raises(ValueError, match=message):
        load_config(config_path)


def test_reuse_plans_from_resolves_from_project_root(tmp_path: Path) -> None:
    """Empty stays empty (call GLM); a run folder becomes an absolute path."""
    text = VALID_CONFIG.replace(
        'reuse_plans_from = ""', 'reuse_plans_from = "results/run-a"'
    )
    project, config_path = _write(tmp_path, text)

    config = load_config(config_path)

    assert config["query_enhancement"]["reuse_plans_from"] == str(
        project / "results/run-a"
    )


def test_the_repository_config_loads() -> None:
    """configs/sec_rag.toml itself passes validation."""
    config = load_config(Path(__file__).parent.parent / "configs" / "sec_rag.toml")

    assert config["query_enhancement"]["reasoning_effort"] == "low"
    assert config["rerank"]["model"] == "rerank-3-lite"
    assert config["retrieval"]["rerank_candidates"] == 50
    # Exp1's own config keeps Exp1 exactly Exp1 (Guide 3.1-3.4 -> Configuration).
    assert config["structure"]["weight"] == 0.0
    assert config["query_enhancement"]["reuse_plans_from"] == ""
