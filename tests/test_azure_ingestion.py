"""Tests for Stage 1.1's safe Azure parser, using no real API calls."""

from __future__ import annotations

import json
from pathlib import Path

import pymupdf
import pytest

from sec_rag.ingestion.azure import parse_corpus


class _FakeResult:
    """Return an Azure-like raw dictionary without making a network request."""

    def as_dict(self) -> dict[str, object]:
        return {
            "content": "A parsed filing page",
            "pages": [{"pageNumber": 1}],
            "paragraphs": [],
            "tables": [],
            "sections": [],
            "styles": [],
        }


class _FakePoller:
    """Expose the long-running-operation result method used by the SDK."""

    def result(self) -> _FakeResult:
        return _FakeResult()


class _FakeAzureClient:
    """Record paid calls while returning a deterministic fake Azure result."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def begin_analyze_document(self, model_id, *, body, output_content_format):
        self.calls.append(model_id)
        return _FakePoller()


def _write_pdf(path: Path, page_count: int) -> None:
    """Create a small machine-readable PDF with a known physical page count."""
    pdf = pymupdf.open()
    for number in range(page_count):
        page = pdf.new_page()
        page.insert_text((72, 72), f"Page {number + 1}")
    pdf.save(path)
    pdf.close()


def _config(tmp_path: Path) -> dict[str, object]:
    """Build the resolved parser settings used by an isolated test corpus."""
    prepared_dir = tmp_path / "financebench"
    pdf_dir = prepared_dir / "pdfs"
    pdf_dir.mkdir(parents=True)
    _write_pdf(pdf_dir / "alpha_2024_10K.pdf", page_count=1)
    _write_pdf(pdf_dir / "beta_2024_10K.pdf", page_count=2)
    return {
        "corpus": {
            "prepared_dir": str(prepared_dir),
            "parsed_dir": str(prepared_dir / "parsed"),
            "expected_documents": 2,
        },
        "parsing": {
            "provider": "azure-document-intelligence",
            "model_id": "prebuilt-layout",
            "output_content_format": "markdown",
        },
    }


def test_plan_selects_all_pdfs_without_creating_client_or_files(tmp_path: Path) -> None:
    """A no-flag plan reports missing PDFs and leaves generated storage absent."""
    config = _config(tmp_path)

    result = parse_corpus(config)

    assert result["action"] == "plan"
    assert result["selected"] == 2
    assert result["counts"]["missing"] == 2
    assert [document["doc_name"] for document in result["documents"]] == [
        "alpha_2024_10K",
        "beta_2024_10K",
    ]
    assert all("pdf_path" not in document for document in result["documents"])
    assert all("cache_path" not in document for document in result["documents"])
    assert not (Path(config["corpus"]["parsed_dir"]) / "manifest.json").exists()


def test_execute_paid_saves_raw_result_and_skips_it_on_resume(tmp_path: Path) -> None:
    """Paid mode saves one completed result then does not submit it again."""
    config = _config(tmp_path)
    config["parsing"]["model_id"] = "test-layout"
    config["parsing"]["output_content_format"] = "test-markdown"
    client = _FakeAzureClient()

    first = parse_corpus(
        config,
        document_names=("alpha_2024_10K",),
        execute_paid=True,
        client=client,
    )

    cache_path = (
        Path(config["corpus"]["parsed_dir"]) / "alpha_2024_10K" / "azure-layout.json"
    )
    assert first["counts"]["parsed"] == 1
    assert client.calls == ["test-layout"]
    assert json.loads(cache_path.read_text())["content"] == "A parsed filing page"
    manifest_entry = json.loads(
        (Path(config["corpus"]["parsed_dir"]) / "manifest.json").read_text()
    )["documents"]["alpha_2024_10K"]
    assert manifest_entry["model_id"] == "test-layout"
    assert manifest_entry["output_content_format"] == "test-markdown"
    assert "api_version" not in manifest_entry

    second = parse_corpus(
        config,
        document_names=("alpha_2024_10K",),
        execute_paid=True,
        client=client,
    )

    assert second["counts"]["complete"] == 1
    assert client.calls == ["test-layout"]


def test_execute_paid_recreates_missing_manifest_without_azure_call(
    tmp_path: Path,
) -> None:
    """A final cache is enough to recover its provenance without repeat spend."""
    config = _config(tmp_path)
    first_client = _FakeAzureClient()
    parse_corpus(
        config,
        document_names=("alpha_2024_10K",),
        execute_paid=True,
        client=first_client,
    )
    manifest_path = Path(config["corpus"]["parsed_dir"]) / "manifest.json"
    manifest_path.unlink()
    second_client = _FakeAzureClient()

    result = parse_corpus(
        config,
        document_names=("alpha_2024_10K",),
        execute_paid=True,
        client=second_client,
    )

    assert result["counts"]["manifest_entries_recreated"] == 1
    assert second_client.calls == []
    repaired_entry = json.loads(manifest_path.read_text())["documents"][
        "alpha_2024_10K"
    ]
    assert repaired_entry["parse_completed_at"] is None


def test_invalid_final_cache_stops_before_submitting_missing_document(
    tmp_path: Path,
) -> None:
    """A corrupt cache stops the batch before a separate missing PDF can cost money."""
    config = _config(tmp_path)
    invalid_cache = (
        Path(config["corpus"]["parsed_dir"]) / "alpha_2024_10K" / "azure-layout.json"
    )
    invalid_cache.parent.mkdir(parents=True)
    invalid_cache.write_text('{"content": "", "pages": []}', encoding="utf-8")
    client = _FakeAzureClient()

    try:
        parse_corpus(config, execute_paid=True, client=client)
    except ValueError as error:
        assert "alpha_2024_10K" in str(error)
    else:
        raise AssertionError("An invalid final cache must stop the corpus operation")

    assert client.calls == []


def test_named_selection_rejects_unknown_and_duplicate_documents(
    tmp_path: Path,
) -> None:
    """Typos cannot narrow a paid run silently or submit a filing twice."""
    config = _config(tmp_path)

    with pytest.raises(ValueError, match="unknown"):
        parse_corpus(config, document_names=("not_a_filing",))
    with pytest.raises(ValueError, match="more than once"):
        parse_corpus(
            config,
            document_names=("alpha_2024_10K", "alpha_2024_10K.pdf"),
        )


def test_credentials_are_required_only_when_a_paid_parse_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fully cached paid resume does not need Azure environment variables."""
    config = _config(tmp_path)
    client = _FakeAzureClient()
    parse_corpus(
        config,
        document_names=("alpha_2024_10K",),
        execute_paid=True,
        client=client,
    )
    monkeypatch.delenv("AZURE_DOCUMENT_INTELLIGENCE_KEY", raising=False)
    monkeypatch.delenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", raising=False)

    cached_result = parse_corpus(
        config,
        document_names=("alpha_2024_10K",),
        execute_paid=True,
    )

    assert cached_result["counts"]["complete"] == 1
    with pytest.raises(RuntimeError, match="AZURE_DOCUMENT_INTELLIGENCE_KEY"):
        parse_corpus(
            config,
            document_names=("beta_2024_10K",),
            execute_paid=True,
        )
