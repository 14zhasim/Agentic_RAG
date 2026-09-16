"""Judge saved answers twice with Azure DeepSeek and checkpoint one verdict."""

from __future__ import annotations

from datetime import UTC, datetime
import json
import os
from pathlib import Path
import time
from typing import Any

from openai import OpenAI


SYSTEM_PROMPT = (
    "You grade whether a 'candidate' answer is correct for a financial question regarding a financial statement. "
    "Allow equivalent units, harmless rounding, and harmless truncation. "
    "Use the supplied reference 'evidence' and 'human justification'. "
    "Return JSON with exactly two fields: verdict (integer 0 or 1) and reason "
    "(a concise string). Return 1 for the verdice only when the candidate is correct."
)


def _append(path: Path, row: dict[str, Any]) -> None:
    """Checkpoint one judgment or error as an immediately flushed JSON line."""
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")
        file.flush()


def _latest_rows(path: Path) -> dict[str, dict[str, Any]]:
    """Return the latest append-only row for each job ID."""
    if not path.exists():
        return {}
    latest: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            row = json.loads(line)
            latest[row["job_id"]] = row
    return latest


def _azure_base_url(endpoint: str) -> str:
    """Turn a resource or project endpoint into the OpenAI-compatible base URL."""
    endpoint = endpoint.rstrip("/")
    if endpoint.endswith("/openai/v1"):
        return endpoint
    return f"{endpoint}/openai/v1"


def _create_azure_client(config: dict[str, Any]) -> OpenAI:
    """Create the Azure client only when an unjudged answer needs it."""
    api_key = os.getenv("AZURE_DEEPSEEK_API_KEY")
    endpoint = os.getenv("AZURE_DEEPSEEK_ENDPOINT")
    if not api_key or not endpoint:
        raise RuntimeError(
            "AZURE_DEEPSEEK_API_KEY and AZURE_DEEPSEEK_ENDPOINT must be set"
        )
    return OpenAI(
        api_key=api_key,
        base_url=_azure_base_url(endpoint),
        timeout=config["timeout_seconds"],
        max_retries=config["max_retries"],
    )


def _answer_block(label: str, answer: str) -> str:
    return f"## {label}\n{answer}"


def _build_judge_messages(
    prediction: dict[str, Any], prompt_order: str
) -> list[dict[str, str]]:
    """Include all reference fields while changing only answer presentation order."""
    reference = _answer_block("REFERENCE ANSWER", prediction["gold_answer"])
    candidate = _answer_block("CANDIDATE ANSWER", prediction["model_answer"])
    if prompt_order == "reference_first":
        answer_blocks = f"{reference}\n\n{candidate}"
    elif prompt_order == "candidate_first":
        answer_blocks = f"{candidate}\n\n{reference}"
    else:
        raise ValueError(f"Unknown judge prompt order: {prompt_order}")

    evidence = json.dumps(prediction["gold_evidence"], ensure_ascii=False, indent=2)
    user_prompt = (
        f"# QUESTION\n{prediction['question']}\n\n"
        f"{answer_blocks}\n\n"
        f"# REFERENCE EVIDENCE\n{evidence}\n\n"
        f"# HUMAN JUSTIFICATION\n{prediction['human_justification']}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _request_verdict(
    messages: list[dict[str, str]], config: dict[str, Any], client: Any
) -> dict[str, Any]:
    """Make one JSON-mode request and strictly validate its binary verdict."""
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=config["deployment"],
        messages=messages,
        max_completion_tokens=config["max_output_tokens"],
        response_format={"type": "json_object"},
    )
    latency_seconds = time.perf_counter() - started
    if not response.choices or not response.choices[0].message.content:
        raise ValueError("Azure judge returned no JSON content")

    try:
        parsed = json.loads(response.choices[0].message.content)
    except json.JSONDecodeError as error:
        raise ValueError(f"Azure judge returned invalid JSON: {error}") from error
    if set(parsed) != {"verdict", "reason"}:
        raise ValueError("Azure judge output must contain only verdict and reason")
    if type(parsed["verdict"]) is not int or parsed["verdict"] not in {0, 1}:
        raise ValueError("Azure judge verdict must be the integer 0 or 1")
    if not isinstance(parsed["reason"], str) or not parsed["reason"].strip():
        raise ValueError("Azure judge reason must be a non-empty string")

    usage = response.usage.model_dump() if response.usage else {}
    return {
        "verdict": parsed["verdict"],
        "reason": parsed["reason"].strip(),
        "returned_model": response.model,
        "request_id": response.id,
        "usage": usage,
        "latency_seconds": latency_seconds,
    }


def _combine_verdicts(
    job_id: str,
    config: dict[str, Any],
    first_pass: dict[str, Any],
    second_pass: dict[str, Any],
) -> dict[str, Any]:
    """Trust an accuracy value only when both presentation orders agree."""
    first_verdict = first_pass["verdict"]
    second_verdict = second_pass["verdict"]
    agreed = first_verdict == second_verdict
    return {
        "job_id": job_id,
        "status": "complete",
        "accuracy": first_verdict if agreed else None,
        "manual_review": not agreed,
        "prompt_version": config["prompt_version"],
        "requested_model": config["deployment"],
        "passes": [
            {"prompt_order": "reference_first", **first_pass},
            {"prompt_order": "candidate_first", **second_pass},
        ],
        "completed_at": datetime.now(UTC).isoformat(),
    }


def _judge_answer(
    prediction: dict[str, Any], config: dict[str, Any], client: Any
) -> dict[str, Any]:
    """Run both answer orders and combine them into one resumable row."""
    reference_first = _request_verdict(
        _build_judge_messages(prediction, "reference_first"), config, client
    )
    candidate_first = _request_verdict(
        _build_judge_messages(prediction, "candidate_first"), config, client
    )
    return _combine_verdicts(
        prediction["job_id"], config, reference_first, candidate_first
    )


def judge_run(
    run_dir: str | Path,
    config: dict[str, Any],
    *,
    client: Any | None = None,
) -> dict[str, int]:
    """Judge every successful, unjudged prediction and checkpoint each result."""
    run_path = Path(run_dir)
    predictions = _latest_rows(run_path / "predictions.jsonl")
    completed_judgments = _latest_rows(run_path / "judgments.jsonl")
    successful_predictions = [
        row for row in predictions.values() if row.get("status") == "success"
    ]
    pending_predictions = [
        row
        for row in successful_predictions
        if row["job_id"] not in completed_judgments
    ]
    counts = {
        "judged": 0,
        "skipped": len(successful_predictions) - len(pending_predictions),
        "failed": 0,
    }
    if not pending_predictions:
        return counts

    judge_client = client or _create_azure_client(config)
    for prediction in pending_predictions:
        try:
            judgment = _judge_answer(prediction, config, judge_client)
            _append(run_path / "judgments.jsonl", judgment)
            counts["judged"] += 1
        except Exception as error:
            _append(
                run_path / "errors.jsonl",
                {
                    "job_id": prediction["job_id"],
                    "stage": "judge",
                    "status": "error",
                    "error": str(error),
                    "completed_at": datetime.now(UTC).isoformat(),
                },
            )
            counts["failed"] += 1
    return counts
