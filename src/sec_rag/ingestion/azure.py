"""Plan Azure Document Intelligence parsing without spending by default."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pymupdf

MANIFEST_FILE = "manifest.json"
CACHE_FILE = "azure-layout.json"


class ParseStateError(ValueError):
    """Raise when local parser inputs or saved state cannot safely be used."""


def parse_corpus(
    config: dict[str, Any],
    document_names: tuple[str, ...] | None = None,
    *,
    execute_paid: bool = False,
    client: Any | None = None,
) -> dict[str, Any]:
    """Plan or run one serial corpus parse without repeating paid work.

    The unflagged path deliberately only reports which prepared PDFs lack a
    final Azure cache. It does not create the parsed directory, read credentials
    or instantiate an Azure client, so it is safe to run while learning the
    corpus state.
    """
    corpus = config["corpus"]
    prepared_dir = Path(corpus["prepared_dir"])
    parsed_dir = Path(corpus["parsed_dir"])
    pdf_paths = sorted((prepared_dir / "pdfs").glob("*.pdf"))
    selected_paths = _select_documents(pdf_paths, document_names)
    manifest = _load_manifest(parsed_dir)
    documents = _classify_documents(selected_paths, parsed_dir)
    counts = {
        "complete": sum(document["status"] == "complete" for document in documents),
        "missing": sum(document["status"] == "missing" for document in documents),
        "invalid": sum(document["status"] == "invalid" for document in documents),
        "parsed": 0,
        "manifest_entries_recreated": 0,
    }
    if counts["invalid"]:
        invalid_names = [
            document["doc_name"]
            for document in documents
            if document["status"] == "invalid"
        ]
        raise ParseStateError(f"Invalid Azure cache: {', '.join(invalid_names)}")

    result = {
        "action": "plan",
        "selected": len(selected_paths),
        "counts": counts,
        "documents": _public_documents(documents),
    }
    if not execute_paid:
        return result

    result["action"] = "execute_paid"
    for document in documents:
        if (
            document["status"] == "complete"
            and document["doc_name"] not in manifest["documents"]
        ):
            manifest["documents"][document["doc_name"]] = _manifest_entry(
                document["pdf_path"],
                document["cache_path"],
                document["pages"],
                config["parsing"],
                recreated=True,
            )
            _write_manifest(parsed_dir, manifest)
            counts["manifest_entries_recreated"] += 1

    missing_documents = [
        document for document in documents if document["status"] == "missing"
    ]
    if not missing_documents:
        return result

    api_client = client if client is not None else _create_azure_client()
    for document in missing_documents:
        raw = _parse_pdf(api_client, document["pdf_path"], config["parsing"])
        page_count = validate_cache(raw, document["pdf_path"])
        _write_json(document["cache_path"], raw)
        manifest["documents"][document["doc_name"]] = _manifest_entry(
            document["pdf_path"],
            document["cache_path"],
            page_count,
            config["parsing"],
        )
        _write_manifest(parsed_dir, manifest)
        document["status"] = "complete"
        document["pages"] = page_count
        document["message"] = "parsed and cached"
        counts["missing"] -= 1
        counts["complete"] += 1
        counts["parsed"] += 1
    result["documents"] = _public_documents(documents)
    return result


def _public_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return JSON-ready status records without the paths used internally."""
    return [
        {
            key: value
            for key, value in document.items()
            if key not in {"pdf_path", "cache_path"}
        }
        for document in documents
    ]


def _select_documents(
    pdf_paths: list[Path], document_names: tuple[str, ...] | None
) -> list[Path]:
    """Return all PDFs or a validated named subset in deterministic order."""
    by_name = {path.stem: path for path in pdf_paths}
    if document_names is None:
        return pdf_paths

    selected: list[Path] = []
    seen: set[str] = set()
    for document_name in document_names:
        normalised = Path(document_name).stem
        if normalised in seen:
            raise ParseStateError(f"Document was selected more than once: {normalised}")
        if normalised not in by_name:
            raise ParseStateError(f"Prepared PDF is unknown: {normalised}")
        seen.add(normalised)
        selected.append(by_name[normalised])
    return sorted(selected)


def _classify_documents(
    pdf_paths: list[Path], parsed_dir: Path
) -> list[dict[str, Any]]:
    """Classify local cache state before any operation can spend money."""
    documents: list[dict[str, Any]] = []
    for pdf_path in pdf_paths:
        cache_path = parsed_dir / pdf_path.stem / CACHE_FILE
        document: dict[str, Any] = {
            "doc_name": pdf_path.stem,
            "pdf_path": pdf_path,
            "cache_path": cache_path,
            "pages": None,
        }
        if not cache_path.is_file():
            document.update(status="missing", message="no final Azure cache")
            documents.append(document)
            continue
        try:
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            document["pages"] = validate_cache(raw, pdf_path)
        except (OSError, json.JSONDecodeError, ParseStateError) as error:
            document.update(status="invalid", message=str(error))
        else:
            document.update(status="complete", message="validated final Azure cache")
        documents.append(document)
    return documents


def validate_cache(raw: dict[str, Any], pdf_path: Path) -> int:
    """Check only the essentials needed before skipping another paid Azure call."""
    content = raw.get("content")
    pages = raw.get("pages")
    if not isinstance(content, str) or not content.strip():
        raise ParseStateError(f"{pdf_path.stem}: Azure cache has no content")
    if not isinstance(pages, list) or not pages:
        raise ParseStateError(f"{pdf_path.stem}: Azure cache has no pages")
    with pymupdf.open(pdf_path) as pdf:
        pdf_page_count = pdf.page_count
    if len(pages) != pdf_page_count:
        raise ParseStateError(
            f"{pdf_path.stem}: Azure has {len(pages)} pages but PDF has {pdf_page_count}"
        )
    return pdf_page_count


def _load_manifest(parsed_dir: Path) -> dict[str, Any]:
    """Load completed parse provenance, treating a missing manifest as empty."""
    path = parsed_dir / MANIFEST_FILE
    if not path.exists():
        return {"schema_version": 1, "documents": {}}
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ParseStateError(f"Cannot read parse manifest: {error}") from error
    if not isinstance(manifest.get("documents"), dict):
        raise ParseStateError("Parse manifest has no documents mapping")
    return manifest


def _parse_pdf(client: Any, pdf_path: Path, parsing: dict[str, Any]) -> dict[str, Any]:
    """Submit one PDF to Azure and return its complete serialisable result."""
    from azure.ai.documentintelligence.models import DocumentContentFormat

    with pdf_path.open("rb") as pdf_file:
        poller = client.begin_analyze_document(
            parsing["model_id"],
            body=pdf_file,
            output_content_format=DocumentContentFormat.MARKDOWN,
        )
    raw = poller.result().as_dict()
    if not isinstance(raw, dict):
        raise ParseStateError(f"{pdf_path.stem}: Azure result is not a dictionary")
    return raw


def _create_azure_client() -> Any:
    """Create the paid Azure client only after a document actually needs parsing."""
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.core.credentials import AzureKeyCredential

    key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
    endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    if not key or not endpoint:
        raise RuntimeError(
            "AZURE_DOCUMENT_INTELLIGENCE_KEY and "
            "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT must be set for --execute-paid"
        )
    return DocumentIntelligenceClient(endpoint, AzureKeyCredential(key))


def _manifest_entry(
    pdf_path: Path,
    cache_path: Path,
    page_count: int,
    parsing: dict[str, Any],
    *,
    recreated: bool = False,
) -> dict[str, Any]:
    """Record enough provenance to identify the source and immutable raw cache."""
    now = datetime.now(UTC).isoformat()
    return {
        "source_pdf": str(Path("pdfs") / pdf_path.name),
        "source_pdf_sha256": _sha256(pdf_path),
        "cache_file": str(Path(pdf_path.stem) / CACHE_FILE),
        "cache_sha256": _sha256(cache_path),
        "page_count": page_count,
        "model_id": parsing["model_id"],
        "output_content_format": parsing["output_content_format"],
        "sdk_package": "azure-ai-documentintelligence",
        "sdk_version": version("azure-ai-documentintelligence"),
        "parse_completed_at": None if recreated else now,
        "recorded_at": now,
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    """Atomically save one final raw Azure response without transforming it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.partial")
    temporary_path.write_text(json.dumps(value), encoding="utf-8")
    temporary_path.replace(path)


def _write_manifest(parsed_dir: Path, manifest: dict[str, Any]) -> None:
    """Atomically checkpoint parse provenance after each completed document."""
    parsed_dir.mkdir(parents=True, exist_ok=True)
    path = parsed_dir / MANIFEST_FILE
    temporary_path = path.with_suffix(f"{path.suffix}.partial")
    temporary_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_path.replace(path)


def _sha256(path: Path) -> str:
    """Return a file hash used only for provenance in the parse manifest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()
