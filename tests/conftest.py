import json
from pathlib import Path

import pymupdf
import pytest


@pytest.fixture
def sample(tmp_path: Path):
    source, output = tmp_path / "source", tmp_path / "prepared"
    (source / "data").mkdir(parents=True)
    (source / "pdfs").mkdir()
    questions = [
        {
            "financebench_id": "q1",
            "company": "A",
            "doc_name": "a.pdf",
            "question_type": "reported",
            "question_reasoning": "information extraction",
            "domain_question_num": 1,
            "question": "Revenue?",
            "answer": "$42.0",
            "justification": "page",
            "dataset_subset_label": "open_source",
            "evidence": [
                {
                    "doc_name": "a.pdf",
                    "evidence_page_num": 0,
                    "evidence_text_full_page": "Revenue was $42.",
                }
            ],
        },
        {
            "financebench_id": "q2",
            "company": "B",
            "doc_name": "b.pdf",
            "question_type": "calculated",
            "question_reasoning": "numerical reasoning",
            "domain_question_num": 2,
            "question": "Total?",
            "answer": "84",
            "justification": "pages",
            "dataset_subset_label": "open_source",
            "evidence": [
                {
                    "doc_name": "b.pdf",
                    "evidence_page_num": 0,
                    "evidence_text_full_page": "Part one",
                },
                {
                    "doc_name": "b.pdf",
                    "evidence_page_num": 1,
                    "evidence_text_full_page": "Part two",
                },
            ],
        },
        {
            "financebench_id": "q3",
            "company": "C",
            "doc_name": "c.pdf",
            "question_type": "reported",
            "question_reasoning": "information extraction",
            "domain_question_num": 3,
            "question": "Quarter?",
            "answer": "1",
            "justification": "page",
            "dataset_subset_label": "open_source",
            "evidence": [
                {
                    "doc_name": "c.pdf",
                    "evidence_page_num": 0,
                    "evidence_text_full_page": "Quarter",
                }
            ],
        },
    ]
    metadata = [
        {
            "doc_name": "a.pdf",
            "company": "A",
            "gics_sector": "Tech",
            "doc_type": "10K",
            "doc_period": "2023",
            "doc_link": "a",
        },
        {
            "doc_name": "b.pdf",
            "company": "B",
            "gics_sector": "Tech",
            "doc_type": " 10k ",
            "doc_period": "2023",
            "doc_link": "b",
        },
        {
            "doc_name": "c.pdf",
            "company": "C",
            "gics_sector": "Tech",
            "doc_type": "10-Q",
            "doc_period": "2023",
            "doc_link": "c",
        },
    ]
    for name, texts in {
        "a.pdf": ["A page"],
        "b.pdf": ["B zero", "B one"],
        "c.pdf": ["C page"],
    }.items():
        pdf = pymupdf.open()
        for text in texts:
            page = pdf.new_page()
            page.insert_text((72, 72), text)
        pdf.save(source / "pdfs" / name)
        pdf.close()
    for filename, rows in (
        ("financebench_open_source.jsonl", questions),
        ("financebench_document_information.jsonl", metadata),
    ):
        (source / "data" / filename).write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
    return {
        "dataset": {
            "source_dir": str(source),
            "output_dir": str(output),
            "document_type": "10k",
            "expected_questions": 2,
            "expected_documents": 2,
        },
        "generation": {
            "model": "z-ai/glm-5.3-flash",
            "base_url": "https://openrouter.ai/api/v1",
            "upstream_provider": "z-ai",
            "context_window_tokens": 1048576,
            "max_output_tokens": 8192,
            "token_safety_margin": 1024,
            "temperature": 0.0,
            "reasoning_effort": "high",
            "timeout_seconds": 30,
            "max_retries": 2,
        },
        "judge": {
            "provider": "azure",
            "model": "DeepSeek-V4-Flash",
            "deployment": "DeepSeek-V4-Flash",
            "prompt_version": "financebench-binary-judge-v2",
            "temperature": 0.0,
            "max_output_tokens": 512,
            "timeout_seconds": 30.0,
            "max_retries": 2,
        },
        "judge_validation": {
            "source_commit": "cc39aeb4afdf33909ee1412188bf89035950c2eb",
            "seed": 42,
            "correct_examples": 15,
            "incorrect_examples": 10,
            "refusal_examples": 5,
            "minimum_agreement": 0.90,
        },
        "development_subsets": {"seed": 42, "smoke_size": 2, "pattern_size": 2},
        "run": {
            "experiment": "financebench",
            "variant": "baseline-context-conditions-v1",
            "conditions": ["closed_book", "oracle", "long_context"],
            "retrieval_depth": 10,
            "results_dir": str(tmp_path / "results"),
        },
    }


def _azure_paragraph(text: str, offset: int, page: int, role: str | None) -> dict:
    """Build one Azure paragraph with its span and start page."""
    paragraph: dict = {
        "content": text,
        "spans": [{"offset": offset, "length": len(text)}],
        "boundingRegions": [{"pageNumber": page}],
    }
    if role is not None:
        paragraph["role"] = role
    return paragraph


@pytest.fixture
def worked_example_parse() -> dict:
    """The implementation guide's miniature of 3M's first pages, as Azure JSON.

    Six paragraphs (one of them body text) and six sections, over 13 pages:
    PART I > Item 1 on p4, Item 1A on p10, PART II > Item 5 on p13.
    """
    return {
        "content": "",
        "pages": [{"pageNumber": n} for n in range(1, 14)],
        "paragraphs": [
            _azure_paragraph("PART I", 0, 4, "title"),
            _azure_paragraph("Item 1. Business", 10, 4, "sectionHeading"),
            _azure_paragraph("3M is a diversified ...", 30, 4, None),
            _azure_paragraph("Item 1A. Risk Factors", 60, 10, "sectionHeading"),
            _azure_paragraph("PART II", 90, 13, "title"),
            _azure_paragraph("Item 5. Market for ...", 100, 13, "sectionHeading"),
        ],
        "sections": [
            {"elements": ["/sections/1", "/sections/4"]},
            {"elements": ["/paragraphs/0", "/sections/2", "/sections/3"]},
            {"elements": ["/paragraphs/1", "/paragraphs/2"]},
            {"elements": ["/paragraphs/3"]},
            {"elements": ["/paragraphs/4", "/sections/5"]},
            {"elements": ["/paragraphs/5"]},
        ],
    }


def chunk_record(doc_name: str, number: int, text: str, **fields) -> dict:
    """One chunk record as `chunk_files` writes it, with only the fields indexing reads."""
    company, year, doc_type = doc_name.split("_")
    record = {
        "chunk_id": f"{doc_name}:p0:c{number}",
        "doc_name": doc_name,
        "company": company,
        "year": int(year),
        "doc_type": doc_type,
        "page_index": 0,
        "kind": "prose",
        "text": text,
    }
    record.update(fields)
    return record


@pytest.fixture
def index_config(tmp_path: Path):
    """A config for the indexing modules, plus a helper that adds one chunked filing.

    `select_documents` only lists PDF names, so an empty stand-in PDF is
    enough; the chunk file is written directly, as `build_chunks` would.
    """
    (tmp_path / "pdfs").mkdir()
    (tmp_path / "chunks").mkdir()
    config = {
        "corpus": {
            "prepared_dir": str(tmp_path),
            "chunks_dir": str(tmp_path / "chunks"),
            "indexes_dir": str(tmp_path / "indexes"),
        },
        "bm25": {
            "token_pattern": r"(?u)[^\W\d_]+|\d+",
            "stopwords": "en",
            "stemmer": "english",
        },
        "embedding": {
            "model": "voyage-4-lite",
            "batch_size": 2,
            "output_dimension": 3,
            "output_dtype": "float",
        },
    }

    def add_filing(doc_name: str, texts: list[str] | None, **fields) -> None:
        """Add a filing's PDF and, unless texts is None, its chunk file."""
        (tmp_path / "pdfs" / f"{doc_name}.pdf").write_bytes(b"")
        if texts is None:
            return
        lines = [
            json.dumps(chunk_record(doc_name, number, text, **fields))
            for number, text in enumerate(texts)
        ]
        (tmp_path / "chunks" / f"{doc_name}.jsonl").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    return config, add_filing
