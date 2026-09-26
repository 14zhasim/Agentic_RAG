"""Tests for Stage 1.1's parse run, using a fake Azure client and no network."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pymupdf
import pytest

from sec_rag.ingestion.parse import ParseStateError, json_path, parse_corpus


class _FakeResult:
    """Stand in for Azure's result object, which the real code turns into a dict."""

    def __init__(self, raw: dict[str, Any]) -> None:
        self._raw = raw

    def as_dict(self) -> dict[str, Any]:
        return self._raw


class _FakePoller:
    """Stand in for the poller that begin_analyze_document returns."""

    def __init__(self, raw: dict[str, Any]) -> None:
        self._raw = raw

    def result(self) -> _FakeResult:
        return _FakeResult(self._raw)


class _FakeAzureClient:
    """Record every 'paid' call and return an Azure-shaped result.

    `page_counts` says how many pages to report for each filing, so a test can
    make Azure's answer disagree with the PDF. By default it reports the right
    number, read from the PDF being sent.
    """

    def __init__(self, page_counts: dict[str, int] | None = None) -> None:
        self.calls: list[str] = []
        self._page_counts = page_counts or {}

    def begin_analyze_document(self, model_id, *, body, output_content_format):
        doc_name = Path(body.name).stem
        self.calls.append(doc_name)
        with pymupdf.open(body.name) as pdf:
            page_count = self._page_counts.get(doc_name, pdf.page_count)
        raw = {
            "content": f"Parsed text of {doc_name}",
            "pages": [{"pageNumber": n + 1} for n in range(page_count)],
            "paragraphs": [],
            "sections": [],
        }
        return _FakePoller(raw)


def _write_pdf(path: Path, page_count: int) -> None:
    """Create a small real PDF with a known number of pages."""
    pdf = pymupdf.open()
    for number in range(page_count):
        page = pdf.new_page()
        page.insert_text((72, 72), f"Page {number + 1}")
    pdf.save(path)
    pdf.close()


def _config(tmp_path: Path, pdfs: dict[str, int]) -> dict[str, Any]:
    """Build a config pointing at a temporary corpus of the given PDFs and page counts."""
    prepared_dir = tmp_path / "financebench"
    pdf_dir = prepared_dir / "pdfs"
    pdf_dir.mkdir(parents=True)
    for doc_name, page_count in pdfs.items():
        _write_pdf(pdf_dir / f"{doc_name}.pdf", page_count)
    return {
        "corpus": {
            "prepared_dir": str(prepared_dir),
            "parsed_dir": str(prepared_dir / "parsed"),
            "expected_documents": len(pdfs),
        },
        "parsing": {
            "provider": "azure-document-intelligence",
            "model_id": "prebuilt-layout",
            "output_content_format": "markdown",
        },
    }


def _save_parse(config: dict[str, Any], doc_name: str, page_count: int) -> Path:
    """Put a saved parse on disk, as a previous paid run would have left it."""
    path = json_path(config, doc_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = {
        "content": "Saved text",
        "pages": [{"pageNumber": n + 1} for n in range(page_count)],
    }
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def _clear_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AZURE_DOCUMENT_INTELLIGENCE_KEY", raising=False)
    monkeypatch.delenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", raising=False)


def test_plan_reports_without_spending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A plain run sorts filings into done and missing, and creates nothing."""
    _clear_credentials(monkeypatch)
    config = _config(tmp_path, {"alpha_2024_10K": 1, "beta_2024_10K": 2})
    _save_parse(config, "alpha_2024_10K", page_count=1)
    files_before = sorted(tmp_path.rglob("*"))

    result = parse_corpus(config)

    assert result == {
        "action": "plan",
        "selected": 2,
        "done": ["alpha_2024_10K"],
        "missing": ["beta_2024_10K"],
        "parsed": [],
    }
    assert sorted(tmp_path.rglob("*")) == files_before


def test_json_path_is_flat() -> None:
    """Saved parses sit directly in parsed_dir, matching the 3M file on disk."""
    config = {"corpus": {"parsed_dir": "/data/parsed"}}

    assert json_path(config, "3M_2018_10K") == Path("/data/parsed/3M_2018_10K.json")


def test_execute_paid_saves_and_then_skips(tmp_path: Path) -> None:
    """Paid mode saves Azure's result unchanged, and a rerun pays for nothing."""
    config = _config(tmp_path, {"alpha_2024_10K": 1, "beta_2024_10K": 2})
    client = _FakeAzureClient()

    first = parse_corpus(config, execute_paid=True, client=client)

    assert first["action"] == "execute_paid"
    assert first["parsed"] == ["alpha_2024_10K", "beta_2024_10K"]
    assert first["done"] == ["alpha_2024_10K", "beta_2024_10K"]
    assert first["missing"] == []
    saved = json.loads(json_path(config, "beta_2024_10K").read_text())
    assert saved["content"] == "Parsed text of beta_2024_10K"
    assert len(saved["pages"]) == 2
    assert list(tmp_path.rglob("*.partial")) == []

    second = parse_corpus(config, execute_paid=True, client=client)

    assert client.calls == ["alpha_2024_10K", "beta_2024_10K"]
    assert second["parsed"] == []


def test_bad_saved_file_stops_before_spending(tmp_path: Path) -> None:
    """A saved parse that fails the check stops the run before any Azure call."""
    config = _config(tmp_path, {"alpha_2024_10K": 2, "beta_2024_10K": 1})
    bad_path = _save_parse(config, "alpha_2024_10K", page_count=1)
    bad_text = bad_path.read_text()
    client = _FakeAzureClient()

    with pytest.raises(ParseStateError, match="alpha_2024_10K"):
        parse_corpus(config, execute_paid=True, client=client)

    assert client.calls == []
    assert bad_path.read_text() == bad_text


def test_bad_azure_result_stops_the_run(tmp_path: Path) -> None:
    """A wrong page count from Azure stops the run before that result is saved."""
    config = _config(
        tmp_path, {"alpha_2024_10K": 1, "beta_2024_10K": 1, "gamma_2024_10K": 1}
    )
    client = _FakeAzureClient(page_counts={"beta_2024_10K": 3})

    with pytest.raises(ParseStateError, match="beta_2024_10K"):
        parse_corpus(config, execute_paid=True, client=client)

    assert json_path(config, "alpha_2024_10K").is_file()
    assert not json_path(config, "beta_2024_10K").exists()
    assert client.calls == ["alpha_2024_10K", "beta_2024_10K"]


def test_leftover_partial_counts_as_missing(tmp_path: Path) -> None:
    """A half-written file from a crashed run is not mistaken for a finished parse."""
    config = _config(tmp_path, {"alpha_2024_10K": 1})
    partial = json_path(config, "alpha_2024_10K").with_suffix(".json.partial")
    partial.parent.mkdir(parents=True)
    partial.write_text('{"content": "half', encoding="utf-8")

    result = parse_corpus(config)

    assert result["missing"] == ["alpha_2024_10K"]


def test_documents_selection(tmp_path: Path) -> None:
    """Named filings may end in .pdf; unknown or repeated names stop the run."""
    config = _config(tmp_path, {"alpha_2024_10K": 1, "beta_2024_10K": 1})

    result = parse_corpus(config, ("beta_2024_10K.pdf",))

    assert result["selected"] == 1
    assert result["missing"] == ["beta_2024_10K"]
    with pytest.raises(ParseStateError, match="unknown"):
        parse_corpus(config, ("not_a_filing",))
    with pytest.raises(ParseStateError, match="more than once"):
        parse_corpus(config, ("alpha_2024_10K", "alpha_2024_10K.pdf"))


def test_credentials_needed_only_when_paying(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Credentials are read only when a missing filing is about to be sent."""
    _clear_credentials(monkeypatch)
    config = _config(tmp_path, {"alpha_2024_10K": 1, "beta_2024_10K": 1})
    _save_parse(config, "alpha_2024_10K", page_count=1)

    parse_corpus(config)
    parse_corpus(config, ("alpha_2024_10K",), execute_paid=True)

    with pytest.raises(RuntimeError) as error:
        parse_corpus(config, ("beta_2024_10K",), execute_paid=True)
    assert "AZURE_DOCUMENT_INTELLIGENCE_KEY" in str(error.value)
    assert "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT" in str(error.value)
