"""Load result-affecting SEC RAG settings from TOML."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate parser, chunker and index settings with repository-relative paths resolved.

    The benchmark and RAG system use separate TOML files because they own
    different result-affecting behaviour. This loader has no credential or data
    dependency, so CLI planning can run without Azure access.
    """
    config_path = Path(path)
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    project_root = config_path.resolve().parent.parent
    corpus = config.get("corpus")
    parsing = config.get("parsing")
    chunking = config.get("chunking")
    bm25 = config.get("bm25")
    embedding = config.get("embedding")
    retrieval = config.get("retrieval")
    query_enhancement = config.get("query_enhancement")
    rerank = config.get("rerank")
    structure = config.get("structure")
    if (
        not isinstance(corpus, dict)
        or not isinstance(parsing, dict)
        or not isinstance(chunking, dict)
        or not isinstance(bm25, dict)
        or not isinstance(embedding, dict)
        or not isinstance(retrieval, dict)
        or not isinstance(query_enhancement, dict)
        or not isinstance(rerank, dict)
        or not isinstance(structure, dict)
    ):
        raise ValueError(
            "SEC RAG configuration needs [corpus], [parsing], [chunking], "
            "[bm25], [embedding], [retrieval], [query_enhancement], "
            "[rerank] and [structure] sections"
        )

    for key in ("prepared_dir", "parsed_dir", "chunks_dir", "indexes_dir"):
        value = corpus.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"corpus.{key} must be a non-empty path")
        configured_path = Path(value)
        if not configured_path.is_absolute():
            configured_path = project_root / configured_path
        corpus[key] = str(configured_path)

    if (
        not isinstance(corpus.get("expected_documents"), int)
        or corpus["expected_documents"] <= 0
    ):
        raise ValueError("corpus.expected_documents must be positive")
    if parsing.get("provider") != "azure-document-intelligence":
        raise ValueError("Stage 1.1 requires the azure-document-intelligence provider")
    if parsing.get("model_id") != "prebuilt-layout":
        raise ValueError("Stage 1.1 requires the prebuilt-layout model")
    if parsing.get("output_content_format") != "markdown":
        raise ValueError("Stage 1.1 requires markdown output")

    for key in ("floor_tokens", "ceiling_tokens", "overlap_cap_tokens"):
        value = chunking.get(key)
        # bool is a subclass of int in Python, so `true` would pass isinstance.
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"chunking.{key} must be a positive whole number")
    # Build Order 1.3: "halving something over 1,024 always leaves both sides
    # over 512, so the floor cannot be violated" only holds while the ceiling
    # is at least twice the floor.
    if chunking["ceiling_tokens"] < 2 * chunking["floor_tokens"]:
        raise ValueError("chunking.ceiling_tokens must be at least twice floor_tokens")

    # Implementation Guide 1.5-1.6 -> Configuration: the word-splitting
    # pattern, stopword list and stemmer are applied both when the index is
    # built and whenever it is reloaded, so they live here, not in code.
    for key in ("token_pattern", "stopwords", "stemmer"):
        value = bm25.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"bm25.{key} must be a non-empty string")

    # Implementation Guide 1.5-1.6 -> Configuration. The limits are Voyage's
    # (docs/libraries/voyage/embeddings.md lines 66-81): at most 1,000 texts
    # per call, and the four lengths voyage-4-lite can return. Only floats,
    # because Chroma stores floats and quantised types would need converting.
    model = embedding.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("embedding.model must be a non-empty string")
    batch_size = embedding.get("batch_size")
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or not 1 <= batch_size <= 1000
    ):
        raise ValueError("embedding.batch_size must be a whole number from 1 to 1000")
    if embedding.get("output_dimension") not in (256, 512, 1024, 2048):
        raise ValueError("embedding.output_dimension must be 256, 512, 1024 or 2048")
    if embedding.get("output_dtype") != "float":
        raise ValueError('embedding.output_dtype must be "float"')

    # Implementation Guide 2.1-2.3 -> Configuration: how many chunks each
    # search returns, the RRF constant, and how many fused chunks reach the
    # reranker. The final 10 is the benchmark's retrieval_depth, not here.
    for key in ("candidates_per_search", "rrf_k", "rerank_candidates"):
        value = retrieval.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"retrieval.{key} must be a positive whole number")

    # Implementation Guide 2.1-2.3 -> Configuration. GLM-5.3-Flash offers
    # only low, high and max reasoning (CLAUDE.md -> Invariants).
    for key in ("model", "base_url", "upstream_provider", "prompt_version"):
        value = query_enhancement.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"query_enhancement.{key} must be a non-empty string")
    if query_enhancement.get("reasoning_effort") not in ("low", "high", "max"):
        raise ValueError(
            'query_enhancement.reasoning_effort must be "low", "high" or "max"'
        )
    for key in ("temperature", "timeout_seconds"):
        value = query_enhancement.get(key)
        if not isinstance(value, int | float) or isinstance(value, bool) or value < 0:
            raise ValueError(f"query_enhancement.{key} must be a number of at least 0")
    for key in ("max_output_tokens", "max_retries"):
        value = query_enhancement.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(
                f"query_enhancement.{key} must be a whole number of at least 0"
            )
    # Implementation Guide 3.1-3.4 -> Configuration: "" means call GLM, as
    # Exp1 does; a run folder means reuse that run's saved plans (rung B).
    reuse_plans_from = query_enhancement.get("reuse_plans_from")
    if not isinstance(reuse_plans_from, str):
        raise ValueError("query_enhancement.reuse_plans_from must be a string")
    if reuse_plans_from:
        plans_path = Path(reuse_plans_from)
        if not plans_path.is_absolute():
            plans_path = project_root / plans_path
        query_enhancement["reuse_plans_from"] = str(plans_path)
    rerank_model = rerank.get("model")
    if not isinstance(rerank_model, str) or not rerank_model:
        raise ValueError("rerank.model must be a non-empty string")

    # Build Order 2.5: Voyage cost is recorded as tokens x list price
    # (docs.voyageai.com/docs/pricing, checked 28 Sep 2026), so each Voyage
    # section carries its price; retrieve_exp1 multiplies by it.
    for section_name, section in (("embedding", embedding), ("rerank", rerank)):
        price = section.get("usd_per_million_tokens")
        if not isinstance(price, int | float) or isinstance(price, bool) or price < 0:
            raise ValueError(
                f"{section_name}.usd_per_million_tokens must be a number of at least 0"
            )

    # Implementation Guide 3.1-3.4 -> Configuration. max_depth and
    # softmax_divisor shape the saved structure vectors (build settings);
    # weight only changes how a run scores (0 = Exp1, Chroma's search).
    max_depth = structure.get("max_depth")
    if not isinstance(max_depth, int) or isinstance(max_depth, bool) or max_depth <= 0:
        raise ValueError("structure.max_depth must be a positive whole number")
    divisor = structure.get("softmax_divisor")
    if (
        not isinstance(divisor, int | float)
        or isinstance(divisor, bool)
        or divisor <= 0
    ):
        raise ValueError("structure.softmax_divisor must be a number above 0")
    weight = structure.get("weight")
    if not isinstance(weight, int | float) or isinstance(weight, bool) or weight < 0:
        raise ValueError("structure.weight must be a number of at least 0")
    return config
