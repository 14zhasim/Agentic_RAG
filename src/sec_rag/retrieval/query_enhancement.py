"""Query enhancement (Build Order 2.1): GLM turns a question into a search plan.

The search plan is GLM's three decisions - which filing to search, a
keyword query for BM25, a semantic query for Chroma - plus whether its reply
was usable and what the call cost. This module holds the ONE prompt, shared
with Exp3's first two tools, so the two experiments cannot drift apart.

The call is our own, not the benchmark's `generate()`: the system under
test must not import the benchmark (Implementation Guide 2.1-2.3 -> Rejected).
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from openai import OpenAI

SYSTEM_PROMPT = (
    "You plan searches over SEC 10-K filings. "
    "Reply with one JSON object and nothing else."
)

# The prompt is a contract (Implementation Guide 2.1-2.3 -> "The
# query-enhancement prompt"); changing it changes results, so bump
# [query_enhancement] prompt_version with it.
USER_PROMPT = """Choose the filing that answers the question, and write two search queries.

Filings you can search:
{filenames}

Reply with this JSON object:
{{"filename": one filename from the list, exactly as written, or null if none clearly fits,
 "keyword_query": keywords for a keyword (BM25) search,
 "semantic_query": one full sentence for a meaning-based search}}

keyword_query: keep the question's company, years and financial terms, and add the wording a 10-K would use for any shorthand, keeping the shorthand too. For example:
- PP&E or PPNE -> property, plant and equipment
- COGS -> cost of goods sold
- DPO -> days payable outstanding
- FCF -> free cash flow
- capex -> capital expenditure, purchases of property, plant and equipment
- FY2018 -> fiscal year 2018

semantic_query: restate the question as one plain sentence, with shorthand written out.

<question>
{question}
</question>"""


def enhance_query(
    question: str, filenames: list[str], config: dict[str, Any], openrouter: Any
) -> dict[str, Any]:
    """Ask GLM for the search plan: filing, keyword query, semantic query.

    `filenames` is the scope: one filing in single-store, all 64 in
    shared-store (Draft -> Retrieve: the SAME call in both conditions).
    Returns the search plan (Guide 2.1-2.3 -> Records) with a "call" record
    of what this GLM call cost and how long it took on its own.
    """
    settings = config["query_enhancement"]

    # Time the GLM call alone. retrieve_exp1 separately times the whole
    # retrieval, so a report can say how much of it was query enhancement -
    # the same per-call timing generation.py does for answers.
    started = time.perf_counter()
    # OpenRouter's Responses endpoint, as the answer model uses it
    # (sec_rag_benchmark/pipeline/generation.py). JSON mode rather than a
    # strict schema: the pinned provider, Z.AI, lists response_format but not
    # structured_outputs (OpenRouter endpoint list for z-ai/glm-5.3-flash,
    # checked 27 Sep 2026), so _read_plan checks the fields itself.
    response = openrouter.responses.create(
        model=settings["model"],
        input=build_enhancement_messages(question, filenames),
        temperature=settings["temperature"],
        max_output_tokens=settings["max_output_tokens"],
        reasoning={"effort": settings["reasoning_effort"]},
        text={"format": {"type": "json_object"}},
        # OpenRouter's Responses endpoint is stateless and rejects store=True.
        store=False,
        # Same host as the answer model, and no silent switch to another.
        extra_body={
            "provider": {
                "order": [settings["upstream_provider"]],
                "allow_fallbacks": False,
            }
        },
    )
    latency_seconds = time.perf_counter() - started

    plan = _read_plan(response.output_text, question, filenames)
    usage = response.usage.model_dump() if response.usage else {}
    plan["call"] = {
        "requested_model": settings["model"],
        "returned_model": response.model,
        "usage": usage,  # input, output and reasoning tokens
        "cost": usage.get("cost"),  # OpenRouter reports the charge in usage
        "latency_seconds": latency_seconds,
        # Exactly what GLM wrote. _read_plan turns a made-up filename into
        # None, and a broken reply into the raw question; keeping the text
        # lets failure diagnosis tell "said null" from "invented a filing",
        # and shows what a broken reply looked like.
        "reply_text": response.output_text,
    }
    return plan


def build_enhancement_messages(
    question: str, filenames: list[str]
) -> list[dict[str, str]]:
    """The ONE query-enhancement prompt, filled in for this question and scope.

    Separate so tests can check the prompt without a call, and so Exp3's
    tools reuse exactly this text (Build Order 2.1).
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT.format(
                filenames="\n".join(filenames), question=question
            ),
        },
    ]


def _read_plan(
    reply_text: str | None, question: str, filenames: list[str]
) -> dict[str, Any]:
    """Turn GLM's reply into a search plan, never failing the question.

    - valid reply: its three fields, "enhancement_status": "ok";
    - a filename that is null, not a string, or not on the list becomes None
      (Draft -> Retrieve: "a name not on the list counts as null"), and the
      reply is still "ok" - GLM declined to choose, it didn't break;
    - not JSON, not an object, or a query missing or blank: both queries
      become the raw question, "invalid_reply". JSON mode promises parseable
      JSON, not our fields; Benchmark.md -> failure diagnosis counts these.
    """
    fallback = {
        "filename": None,
        "keyword_query": question,
        "semantic_query": question,
        "enhancement_status": "invalid_reply",
    }
    try:
        reply = json.loads(reply_text or "")
    except json.JSONDecodeError:
        return fallback
    if not isinstance(reply, dict):
        return fallback

    keyword_query = reply.get("keyword_query")
    semantic_query = reply.get("semantic_query")
    for query in (keyword_query, semantic_query):
        if not isinstance(query, str) or not query.strip():
            return fallback

    filename = reply.get("filename")
    if filename not in filenames:
        filename = None
    return {
        "filename": filename,
        "keyword_query": keyword_query,
        "semantic_query": semantic_query,
        "enhancement_status": "ok",
    }


def openrouter_client(config: dict[str, Any]) -> OpenAI:
    """Create the OpenRouter client; reads OPENROUTER_API_KEY only now.

    Called once by open_exp1, so no-spend commands and tests never need the
    key (CLAUDE.md -> Credentials).
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    settings = config["query_enhancement"]
    return OpenAI(
        base_url=settings["base_url"],
        api_key=api_key,
        timeout=settings["timeout_seconds"],
        max_retries=settings["max_retries"],
    )
