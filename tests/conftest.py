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
        {"financebench_id": "q1", "company": "A", "doc_name": "a.pdf", "question_type": "reported",
         "question_reasoning": "information extraction", "domain_question_num": 1, "question": "Revenue?",
         "answer": "$42.0", "justification": "page", "dataset_subset_label": "open_source",
         "evidence": [{"doc_name": "a.pdf", "evidence_page_num": 0, "evidence_text_full_page": "Revenue was $42."}]},
        {"financebench_id": "q2", "company": "B", "doc_name": "b.pdf", "question_type": "calculated",
         "question_reasoning": "numerical reasoning", "domain_question_num": 2, "question": "Total?",
         "answer": "84", "justification": "pages", "dataset_subset_label": "open_source",
         "evidence": [{"doc_name": "b.pdf", "evidence_page_num": 0, "evidence_text_full_page": "Part one"},
                      {"doc_name": "b.pdf", "evidence_page_num": 1, "evidence_text_full_page": "Part two"}]},
        {"financebench_id": "q3", "company": "C", "doc_name": "c.pdf", "question_type": "reported",
         "question_reasoning": "information extraction", "domain_question_num": 3, "question": "Quarter?",
         "answer": "1", "justification": "page", "dataset_subset_label": "open_source",
         "evidence": [{"doc_name": "c.pdf", "evidence_page_num": 0, "evidence_text_full_page": "Quarter"}]},
    ]
    metadata = [
        {"doc_name": "a.pdf", "company": "A", "gics_sector": "Tech", "doc_type": "10K", "doc_period": "2023", "doc_link": "a"},
        {"doc_name": "b.pdf", "company": "B", "gics_sector": "Tech", "doc_type": " 10k ", "doc_period": "2023", "doc_link": "b"},
        {"doc_name": "c.pdf", "company": "C", "gics_sector": "Tech", "doc_type": "10-Q", "doc_period": "2023", "doc_link": "c"},
    ]
    for name, texts in {"a.pdf": ["A page"], "b.pdf": ["B zero", "B one"], "c.pdf": ["C page"]}.items():
        pdf = pymupdf.open()
        for text in texts:
            page = pdf.new_page(); page.insert_text((72, 72), text)
        pdf.save(source / "pdfs" / name); pdf.close()
    for filename, rows in (("financebench_open_source.jsonl", questions), ("financebench_document_information.jsonl", metadata)):
        (source / "data" / filename).write_text("".join(json.dumps(row) + "\n" for row in rows))
    return {
        "dataset": {"source_dir": str(source), "output_dir": str(output), "document_type": "10k",
                    "expected_questions": 2, "expected_documents": 2},
        "generation": {"model": "z-ai/glm-5.3-flash", "base_url": "https://openrouter.ai/api/v1",
                       "upstream_provider": "z-ai", "context_window_tokens": 1048576,
                       "max_output_tokens": 2048, "token_safety_margin": 1024, "temperature": 0.0,
                       "reasoning_effort": "low", "timeout_seconds": 30, "max_retries": 2},
        "run": {"conditions": ["closed_book", "oracle", "long_context"], "retrieval_depth": 5,
                "results_dir": str(tmp_path / "results")},
    }
