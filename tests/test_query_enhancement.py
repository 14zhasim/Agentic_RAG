"""Tests for query enhancement: the one prompt, the OpenRouter call, and reply checking.

A fake OpenRouter client records what it was sent and returns a canned
reply, so no test can spend or needs a key.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from sec_rag.retrieval.query_enhancement import (
    _read_plan,
    build_enhancement_messages,
    enhance_query,
    openrouter_client,
)

FILENAMES = ["3M_2018_10K", "AMD_2022_10K"]
QUESTION = "What is the FY2018 capex for 3M?"
CONFIG = {
    "query_enhancement": {
        "model": "z-ai/glm-5.3-flash",
        "base_url": "https://openrouter.ai/api/v1",
        "upstream_provider": "z-ai",
        "reasoning_effort": "low",
        "temperature": 0.0,
        "max_output_tokens": 4096,
        "timeout_seconds": 120.0,
        "max_retries": 5,
        "prompt_version": "exp1-query-enhancement-v1",
    }
}
GOOD_REPLY = {
    "filename": "3M_2018_10K",
    "keyword_query": "3M 2018 capex capital expenditure",
    "semantic_query": "How much did 3M spend on capital expenditure in 2018?",
}


class FakeUsage:
    def model_dump(self) -> dict:
        return {"input_tokens": 900, "output_tokens": 120, "cost": 0.0004}


class FakeOpenRouter:
    """Stands in for openai.OpenAI: records the request, returns reply_text."""

    def __init__(self, reply_text: str) -> None:
        self.requests: list[dict] = []
        self.reply_text = reply_text
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kwargs) -> SimpleNamespace:
        self.requests.append(kwargs)
        return SimpleNamespace(
            output_text=self.reply_text, model="z-ai/glm-5.3-flash", usage=FakeUsage()
        )


# --- enhance_query ------------------------------------------------------------


def test_enhance_query_returns_the_plan_and_the_call_record() -> None:
    client = FakeOpenRouter(json.dumps(GOOD_REPLY))

    plan = enhance_query(QUESTION, FILENAMES, CONFIG, client)

    assert plan["filename"] == "3M_2018_10K"
    assert plan["keyword_query"] == GOOD_REPLY["keyword_query"]
    assert plan["semantic_query"] == GOOD_REPLY["semantic_query"]
    assert plan["enhancement_status"] == "ok"
    assert plan["call"]["requested_model"] == "z-ai/glm-5.3-flash"
    assert plan["call"]["returned_model"] == "z-ai/glm-5.3-flash"
    assert plan["call"]["cost"] == 0.0004
    assert plan["call"]["usage"]["output_tokens"] == 120
    assert plan["call"]["latency_seconds"] >= 0


def test_call_record_keeps_glms_exact_reply() -> None:
    """A made-up filing becomes None in the plan, but the reply shows what GLM wrote."""
    reply_text = json.dumps({**GOOD_REPLY, "filename": "3M_2019_10K"})
    client = FakeOpenRouter(reply_text)

    plan = enhance_query(QUESTION, FILENAMES, CONFIG, client)

    assert plan["filename"] is None
    assert plan["call"]["reply_text"] == reply_text


def test_enhance_query_sends_the_agreed_settings() -> None:
    """Draft -> Retrieve: GLM at reasoning low, JSON mode, pinned to z-ai."""
    client = FakeOpenRouter(json.dumps(GOOD_REPLY))

    enhance_query(QUESTION, FILENAMES, CONFIG, client)

    request = client.requests[0]
    assert request["model"] == "z-ai/glm-5.3-flash"
    assert request["reasoning"] == {"effort": "low"}
    assert request["text"] == {"format": {"type": "json_object"}}
    assert request["temperature"] == 0.0
    assert request["max_output_tokens"] == 4096
    assert request["store"] is False
    assert request["extra_body"] == {
        "provider": {"order": ["z-ai"], "allow_fallbacks": False}
    }
    assert request["input"] == build_enhancement_messages(QUESTION, FILENAMES)


def test_openrouter_client_needs_the_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        openrouter_client(CONFIG)


# --- the prompt ---------------------------------------------------------------


def test_prompt_lists_every_filename_and_the_question() -> None:
    messages = build_enhancement_messages(QUESTION, FILENAMES)

    assert messages[0]["role"] == "system"
    assert "JSON" in messages[0]["content"]
    user = messages[1]["content"]
    assert "3M_2018_10K\nAMD_2022_10K" in user
    assert f"<question>\n{QUESTION}\n</question>" in user


def test_prompt_asks_for_shorthand_to_be_expanded() -> None:
    """Build Order 2.1: the keyword query spells shorthand out for BM25."""
    user = build_enhancement_messages(QUESTION, FILENAMES)[1]["content"]

    for shorthand, wording in [
        ("PP&E", "property, plant and equipment"),
        ("COGS", "cost of goods sold"),
        ("DPO", "days payable outstanding"),
        ("FCF", "free cash flow"),
        ("capex", "capital expenditure"),
    ]:
        assert shorthand in user
        assert wording in user


# --- _read_plan: every edge case ----------------------------------------------


def test_valid_reply_is_ok() -> None:
    plan = _read_plan(json.dumps(GOOD_REPLY), QUESTION, FILENAMES)

    assert plan == {**GOOD_REPLY, "enhancement_status": "ok"}


@pytest.mark.parametrize("filename", [None, "3M_2019_10K", "3M 2018", 42])
def test_filename_not_on_the_list_becomes_none_but_the_reply_is_ok(filename) -> None:
    """null, a made-up filing or a wrong type all mean "no filing chosen"."""
    reply = {**GOOD_REPLY, "filename": filename}

    plan = _read_plan(json.dumps(reply), QUESTION, FILENAMES)

    assert plan["filename"] is None
    assert plan["keyword_query"] == GOOD_REPLY["keyword_query"]
    assert plan["enhancement_status"] == "ok"


@pytest.mark.parametrize(
    "reply_text",
    [
        "not json at all",
        "",
        json.dumps(["a", "list"]),
        json.dumps({"filename": "3M_2018_10K", "semantic_query": "q"}),
        json.dumps({**GOOD_REPLY, "keyword_query": "   "}),
        json.dumps({**GOOD_REPLY, "semantic_query": 7}),
    ],
)
def test_unusable_reply_falls_back_to_the_raw_question(reply_text) -> None:
    """Benchmark.md -> failure diagnosis counts these as invalid_reply."""
    plan = _read_plan(reply_text, QUESTION, FILENAMES)

    assert plan == {
        "filename": None,
        "keyword_query": QUESTION,
        "semantic_query": QUESTION,
        "enhancement_status": "invalid_reply",
    }
