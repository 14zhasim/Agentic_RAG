"""Parse each FinanceBench PDF with Azure Document Intelligence exactly once.

This is the only code in the project that calls Azure's parser, and each call
costs money (~$1.70 a filing), so the parse run is built around the rules in
`Systems Design Draft.md` -> "How the parse run works":

- skip filings already saved (`parsed/<doc_name>.json`), so a rerun pays for
  nothing it has already bought;
- spend nothing unless `execute_paid` is set;
- one filing at a time, stopping at the first error;
- write to `<doc_name>.json.partial`, then rename, so a crash can never leave a
  half-written file that looks finished;
- check every parse (saved or fresh) before trusting it, and stop rather than
  overwrite or pay again when one fails.

The saved JSON is Azure's result exactly as returned. Nothing here edits it:
every later stage reads positions (spans) that point into its `content`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pymupdf

# Retry settings for the Azure client, so a Wi-Fi drop mid-run is waited out
# rather than stopping the run and costing a repeat parse. They change only how
# long a request is retried, never what Azure returns, so they don't affect
# results and live here rather than in config.
#
# azure-core's defaults give up on a failed connection after 3 retries spaced
# 0, 1.6 and 3.2 seconds (retry_backoff_factor 0.8, doubling), too short for a
# network reconnect. With factor 2, doubling, capped at 60 s, 10 connection
# retries wait about 0, 4, 8, 16, 32, then 60 s each: roughly six minutes.
# Read retries don't apply to the upload itself (a POST): azure-core never
# retries a POST after it was sent, so a PDF can't be billed twice this way.
# Confirmed in .venv/.../azure/core/pipeline/policies/_retry.py
# (RetryPolicy.__init__, get_backoff_time, _is_method_retryable).
AZURE_RETRY_SETTINGS: dict[str, int] = {
    "retry_total": 15,
    "retry_connect": 10,
    "retry_read": 5,
    "retry_backoff_factor": 2,
    "retry_backoff_max": 60,
}


class ParseStateError(ValueError):
    """A saved parse, a fresh Azure result or a filing name can't be trusted."""


def parse_corpus(
    config: dict[str, Any],
    document_names: tuple[str, ...] | None = None,
    *,
    execute_paid: bool = False,
    client: Any | None = None,
) -> dict[str, Any]:
    """Report which filings are parsed, and with execute_paid, parse the missing ones.

    Every selected filing is first sorted into "done" (saved and passes the
    check) or "missing" (no saved JSON). Only then, and only with
    `execute_paid`, are the missing ones sent to Azure one by one, each
    checked and saved before the next is sent.

    `client` exists so tests can pass in a fake Azure client. In real use it's
    left empty, and the real client (and the credentials it needs) is created
    only when a missing filing is about to be sent.
    """
    pdf_paths = select_documents(config, document_names)

    # Sort every filing into done or missing BEFORE anything can spend. A saved
    # file that fails the check raises here, so a bad file stops the run before
    # any Azure call, and is never overwritten or paid for again.
    done: list[str] = []
    missing: list[Path] = []
    for pdf_path in pdf_paths:
        path = json_path(config, pdf_path.stem)
        if not path.is_file():
            missing.append(pdf_path)
            continue
        raw = read_json(path)
        check_parse(raw, pdf_path)
        done.append(pdf_path.stem)

    result: dict[str, Any] = {
        "action": "plan",
        "selected": len(pdf_paths),
        "done": done,
        "missing": [pdf_path.stem for pdf_path in missing],
        "parsed": [],
    }
    if not execute_paid or not missing:
        return result

    result["action"] = "execute_paid"
    api_client = client if client is not None else _create_azure_client()
    for pdf_path in missing:
        # Any exception from Azure itself (wrong key, quota, network) is left
        # uncaught: it stops the loop at once, and earlier filings stay saved.
        raw = _send_to_azure(api_client, pdf_path, config["parsing"]["model_id"])
        # A bad result stops the run before it is saved.
        check_parse(raw, pdf_path)
        _write_json(json_path(config, pdf_path.stem), raw)
        result["missing"].remove(pdf_path.stem)
        result["done"].append(pdf_path.stem)
        result["parsed"].append(pdf_path.stem)
    result["done"].sort()
    return result


def select_documents(
    config: dict[str, Any], document_names: tuple[str, ...] | None
) -> list[Path]:
    """Return the PDFs to work on: all prepared PDFs, or the named ones, in sorted order.

    Names may be given with or without ".pdf". An unknown or repeated name
    stops the run, so a typo can't silently narrow a paid run or send one
    filing twice.
    """
    pdf_dir = Path(config["corpus"]["prepared_dir"]) / "pdfs"
    all_pdfs = sorted(pdf_dir.glob("*.pdf"))
    if document_names is None:
        return all_pdfs

    by_name = {pdf_path.stem: pdf_path for pdf_path in all_pdfs}
    selected: list[Path] = []
    for name in document_names:
        doc_name = name.removesuffix(".pdf")
        if doc_name not in by_name:
            raise ParseStateError(f"Prepared PDF is unknown: {doc_name}")
        if by_name[doc_name] in selected:
            raise ParseStateError(f"Document was selected more than once: {doc_name}")
        selected.append(by_name[doc_name])
    return sorted(selected)


def json_path(config: dict[str, Any], doc_name: str) -> Path:
    """Return where a filing's saved Azure JSON lives: parsed/<doc_name>.json."""
    return Path(config["corpus"]["parsed_dir"]) / f"{doc_name}.json"


def read_json(path: Path) -> dict[str, Any]:
    """Load one saved parse, naming the filing if the file can't be read."""
    try:
        raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ParseStateError(
            f"{path.stem}: cannot read saved parse: {error}"
        ) from error
    return raw


def check_parse(raw: dict[str, Any], pdf_path: Path) -> int:
    """Check a parse is trustworthy, and return its page count.

    Used on a JSON already on disk and on Azure's fresh result before it's
    saved. The page count is compared with PyMuPDF's count of the PDF itself:
    an independent check, since PyMuPDF knows nothing about Azure, so a
    truncated or mismatched result can't pass by agreeing with itself.
    """
    doc_name = pdf_path.stem
    content = raw.get("content")
    pages = raw.get("pages")
    if not isinstance(content, str) or not content.strip():
        raise ParseStateError(f"{doc_name}: parse has no content")
    if not isinstance(pages, list) or not pages:
        raise ParseStateError(f"{doc_name}: parse has no pages")
    with pymupdf.open(pdf_path) as pdf:
        pdf_page_count = pdf.page_count
    if len(pages) != pdf_page_count:
        raise ParseStateError(
            f"{doc_name}: Azure has {len(pages)} pages but PDF has {pdf_page_count}"
        )
    return pdf_page_count


def _create_azure_client() -> Any:
    """Create the real Azure client from the two environment variables."""
    # The Azure SDK is imported here, not at the top of the file, so free
    # commands (the plain parse report, inspection, load_pages) never load it
    # or need credentials.
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.core.credentials import AzureKeyCredential

    key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
    endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    if not key or not endpoint:
        raise RuntimeError(
            "AZURE_DOCUMENT_INTELLIGENCE_KEY and "
            "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT must be set for --execute-paid"
        )
    return DocumentIntelligenceClient(
        endpoint, AzureKeyCredential(key), **AZURE_RETRY_SETTINGS
    )


def _send_to_azure(client: Any, pdf_path: Path, model_id: str) -> dict[str, Any]:
    """Send one PDF to Azure's layout model and return the whole result as a dict.

    `begin_analyze_document` returns a poller because Azure takes minutes;
    `poller.result()` waits for the finished parse. The result object isn't
    JSON-ready, so `.as_dict()` converts it (SDK README, "Parse analyzed
    result to JSON format"). The installed SDK (1.0.2) names the format enum
    `DocumentContentFormat`; Microsoft's layout page calls it `ContentFormat`.
    """
    from azure.ai.documentintelligence.models import DocumentContentFormat

    with pdf_path.open("rb") as pdf_file:
        poller = client.begin_analyze_document(
            model_id,
            body=pdf_file,
            output_content_format=DocumentContentFormat.MARKDOWN,
        )
        raw: dict[str, Any] = poller.result().as_dict()
    return raw


def _write_json(path: Path, raw: dict[str, Any]) -> None:
    """Save one Azure result, unchanged, so it is either whole or absent.

    The JSON is written to `<doc_name>.json.partial` and only then renamed.
    A half-written `<doc_name>.json` would be mistaken for a finished parse by
    the skip rule; the rename is instant, so that can't happen.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = path.with_suffix(".json.partial")
    partial_path.write_text(json.dumps(raw), encoding="utf-8")
    partial_path.replace(path)
