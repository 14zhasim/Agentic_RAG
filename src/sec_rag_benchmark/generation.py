"""Build the shared prompt and generate an answer through OpenRouter."""

from __future__ import annotations

from functools import lru_cache
import os
import time
from typing import Any

from openai import OpenAI
from transformers import AutoTokenizer


SYSTEM_PROMPT = (
    "You are a careful financial analyst. Answer accurately and concisely. "
    "When filing context is supplied, use it as the primary evidence."
)

# Pin the tokenizer snapshot as well as the Python dependency. Otherwise a
# future change to the Hugging Face repository could change benchmark preflight
# counts even when this code and uv.lock have not changed.
TOKENIZER_MODEL = "zai-org/GLM-5.3-Flash"
TOKENIZER_REVISION = "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a"


def build_messages(question: str, context: str) -> list[dict[str, str]]:
    """Build the one shared answer prompt used by every context condition."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"<context>\n{context}\n</context>\n<question>\n{question}\n</question>"},
    ]


@lru_cache(maxsize=1)
def get_tokenizer() -> Any:
    """Download once, then reuse the official GLM tokenizer from the local cache."""
    return AutoTokenizer.from_pretrained(TOKENIZER_MODEL, revision=TOKENIZER_REVISION)


def count_prompt_tokens(messages: list[dict[str, str]], reasoning_effort: str) -> int:
    """Count the complete GLM chat template, including roles and generation marker."""
    encoded = get_tokenizer().apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        reasoning_effort=reasoning_effort,
    )
    return len(encoded["input_ids"])


def _selected_provider(response: Any) -> str | None:
    """Read the serving provider from opted-in OpenRouter routing metadata."""
    metadata = getattr(response, "openrouter_metadata", None)
    if hasattr(metadata, "model_dump"):
        metadata = metadata.model_dump()
    if not isinstance(metadata, dict):
        return None
    endpoints = metadata.get("endpoints")
    if not isinstance(endpoints, dict):
        return None
    available = endpoints.get("available")
    if not isinstance(available, list):
        return None
    for endpoint in available:
        if isinstance(endpoint, dict) and endpoint.get("selected") is True:
            provider = endpoint.get("provider")
            return provider if isinstance(provider, str) else None
    return None


def generate(
    messages: list[dict[str, str]],
    config: dict[str, Any],
    *,
    client: Any | None = None,
) -> dict[str, Any]:
    """Check context capacity, call OpenRouter, and return answer provenance."""
    # Reserve both the model's possible answer and an extra accounting margin.
    # Reject the complete prompt rather than silently truncating filing pages.
    reserved = config["max_output_tokens"] + config["token_safety_margin"]
    if count_prompt_tokens(messages, config["reasoning_effort"]) + reserved > config["context_window_tokens"]:
        raise ValueError("Complete prompt and output reserve exceed the context window")
    if client is None:
        # Construct the client only for a real generation call. Preparation,
        # validation, dry runs, reporting, and tests therefore need no API key.
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        client = OpenAI(
            base_url=config["base_url"],
            api_key=key,
            timeout=config["timeout_seconds"],
            max_retries=config["max_retries"],
        )
    started = time.perf_counter()
    # The OpenAI SDK sends this Responses request to OpenRouter because the
    # client uses OpenRouter's base URL. Provider routing remains an OpenRouter-
    # only body extension, while reasoning is a standard Responses parameter.
    response = client.responses.create(
        model=config["model"],
        input=messages,
        temperature=config["temperature"],
        max_output_tokens=config["max_output_tokens"],
        reasoning={"effort": config["reasoning_effort"]},
        # OpenRouter's Responses endpoint is stateless and rejects store=True.
        store=False,
        extra_headers={"X-OpenRouter-Metadata": "enabled"},
        extra_body={
            "provider": {"order": [config["upstream_provider"]], "allow_fallbacks": False},
        },
    )
    text = response.output_text
    if not text:
        raise RuntimeError("OpenRouter returned an empty answer")
    usage = response.usage.model_dump() if response.usage else {}
    return {
        "answer": text,
        "requested_model": config["model"],
        "request_id": response.id,
        "returned_model": response.model,
        # OpenRouter adds this metadata only because the request opted in. Do
        # not guess the serving provider when the metadata is absent.
        "provider": _selected_provider(response),
        "usage": usage,
        # OpenRouter can include request cost inside usage. Keep the key even
        # when a provider/test response does not return a value.
        "cost": usage.get("cost"),
        "latency_seconds": time.perf_counter() - started,
    }
