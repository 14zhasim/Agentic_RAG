import csv
import json

import pytest

from sec_rag_benchmark.cli import main
from sec_rag_benchmark.evaluation.manual_review import (
    export_manual_review,
    import_manual_review,
)


def _prediction(job_id: str, *, status: str = "success") -> dict:
    return {
        "job_id": job_id,
        "status": status,
        "financebench_id": job_id.split(":")[1],
        "eval_mode": job_id.rsplit(":", 1)[-1],
        "question": "What was revenue?",
        "gold_answer": "$42 million",
        "gold_evidence": [
            {
                "doc_name": "a.pdf",
                "evidence_page_num": 7,
                "evidence_text_full_page": "Revenue was $42 million.",
            },
            {
                "doc_name": "a.pdf",
                "evidence_page_num": 8,
                "evidence_text": "The note confirms the value.",
            },
        ],
        "human_justification": "The filing directly reports the value.",
        "model_answer": "$42m" if status == "success" else None,
    }


def _judgment(job_id: str, first: int, second: int) -> dict:
    return {
        "job_id": job_id,
        "status": "complete",
        "accuracy": first if first == second else None,
        "manual_review": first != second,
        "passes": [
            {
                "prompt_order": "reference_first",
                "verdict": first,
                "reason": f"Reference-first verdict {first}",
            },
            {
                "prompt_order": "candidate_first",
                "verdict": second,
                "reason": f"Candidate-first verdict {second}",
            },
        ],
    }


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _make_run(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    disputed = "run:q1:oracle"
    agreed = "run:q2:oracle"
    did_not_fit = "run:q3:long_context"
    _write_jsonl(
        run_dir / "predictions.jsonl",
        [
            _prediction(disputed),
            _prediction(agreed),
            _prediction(did_not_fit, status="did_not_fit"),
        ],
    )
    _write_jsonl(
        run_dir / "judgments.jsonl",
        [_judgment(disputed, 1, 0), _judgment(agreed, 1, 1)],
    )
    return run_dir, disputed


def _read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def _write_review_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_export_selects_disagreements_and_formats_review_context(tmp_path):
    run_dir, disputed = _make_run(tmp_path)

    result = export_manual_review(run_dir)
    rows = _read_csv(run_dir / "manual_review.csv")

    assert result == {
        "path": run_dir / "manual_review.csv",
        "disagreements": 1,
        "prefilled": 0,
    }
    assert len(rows) == 1
    assert rows[0]["job_id"] == disputed
    assert rows[0]["reference_first_verdict"] == "1"
    assert rows[0]["candidate_first_verdict"] == "0"
    assert (
        "Document: a.pdf\nPage: 7\nRevenue was $42 million." in rows[0]["gold_evidence"]
    )
    assert "---" in rows[0]["gold_evidence"]
    assert "Page: 8\nThe note confirms the value." in rows[0]["gold_evidence"]
    assert rows[0]["human_accuracy"] == ""
    assert rows[0]["review_reason"] == ""
    assert (run_dir / "manual_review.csv").read_bytes().startswith(b"\xef\xbb\xbf")

    with pytest.raises(ValueError, match="already exists"):
        export_manual_review(run_dir)


def test_export_prefills_latest_manual_decision_in_stable_order(tmp_path):
    run_dir, disputed = _make_run(tmp_path)
    second_job = "run:q0:closed_book"
    with (run_dir / "predictions.jsonl").open("a") as file:
        file.write(json.dumps(_prediction(second_job)) + "\n")
    with (run_dir / "judgments.jsonl").open("a") as file:
        file.write(json.dumps(_judgment(second_job, 0, 1)) + "\n")
    _write_jsonl(
        run_dir / "manual_reviews.jsonl",
        [
            {"job_id": disputed, "human_accuracy": 0, "review_reason": "old"},
            {"job_id": disputed, "human_accuracy": 1, "review_reason": "corrected"},
        ],
    )

    result = export_manual_review(run_dir)
    rows = _read_csv(run_dir / "manual_review.csv")

    assert [row["job_id"] for row in rows] == sorted([disputed, second_job])
    reviewed = next(row for row in rows if row["job_id"] == disputed)
    assert reviewed["human_accuracy"] == "1"
    assert reviewed["review_reason"] == "corrected"
    assert result["prefilled"] == 1


def test_import_supports_partial_unchanged_and_corrected_reviews(tmp_path):
    run_dir, disputed = _make_run(tmp_path)
    export_manual_review(run_dir)
    rows = _read_csv(run_dir / "manual_review.csv")

    assert import_manual_review(run_dir) == {
        "imported": 0,
        "skipped": 0,
        "unresolved": 1,
    }

    rows[0]["human_accuracy"] = "1"
    _write_review_csv(run_dir / "manual_review.csv", rows)
    assert import_manual_review(run_dir) == {
        "imported": 1,
        "skipped": 0,
        "unresolved": 0,
    }
    assert import_manual_review(run_dir) == {
        "imported": 0,
        "skipped": 1,
        "unresolved": 0,
    }

    rows[0]["human_accuracy"] = "0"
    rows[0]["review_reason"] = "Corrected after checking the table."
    _write_review_csv(run_dir / "manual_review.csv", rows)
    assert import_manual_review(run_dir)["imported"] == 1

    saved = [
        json.loads(line)
        for line in (run_dir / "manual_reviews.jsonl").read_text().splitlines()
    ]
    assert [row["human_accuracy"] for row in saved] == [1, 0]
    assert all(row["job_id"] == disputed for row in saved)
    assert all(row["reviewer"] == "author" for row in saved)
    assert all(row["reviewed_at"] for row in saved)


def test_import_accepts_an_excel_utf8_bom(tmp_path):
    run_dir, _ = _make_run(tmp_path)
    export_manual_review(run_dir)
    rows = _read_csv(run_dir / "manual_review.csv")
    rows[0]["human_accuracy"] = "1"
    with (run_dir / "manual_review.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    assert import_manual_review(run_dir)["imported"] == 1


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda row: row.update(human_accuracy="1.0"), "human_accuracy"),
        (lambda row: row.update(review_reason="reason only"), "human_accuracy"),
        (
            lambda row: row.update(job_id="run:missing:oracle", human_accuracy="1"),
            "unknown",
        ),
    ],
)
def test_import_rejects_invalid_rows_before_writing(tmp_path, mutate, message):
    run_dir, _ = _make_run(tmp_path)
    export_manual_review(run_dir)
    rows = _read_csv(run_dir / "manual_review.csv")
    mutate(rows[0])
    _write_review_csv(run_dir / "manual_review.csv", rows)

    with pytest.raises(ValueError, match=message):
        import_manual_review(run_dir)
    assert not (run_dir / "manual_reviews.jsonl").exists()


def test_import_rejects_agreed_and_duplicate_job_ids(tmp_path):
    run_dir, _ = _make_run(tmp_path)
    export_manual_review(run_dir)
    rows = _read_csv(run_dir / "manual_review.csv")
    rows[0]["human_accuracy"] = "1"
    agreed_row = {**rows[0], "job_id": "run:q2:oracle"}
    _write_review_csv(run_dir / "manual_review.csv", [rows[0], agreed_row])
    with pytest.raises(ValueError, match="not a disputed judgment"):
        import_manual_review(run_dir)
    assert not (run_dir / "manual_reviews.jsonl").exists()

    _write_review_csv(run_dir / "manual_review.csv", [rows[0], rows[0]])
    with pytest.raises(ValueError, match="duplicate"):
        import_manual_review(run_dir)
    assert not (run_dir / "manual_reviews.jsonl").exists()


def test_import_rejects_did_not_fit_even_with_a_fabricated_judgment(tmp_path):
    run_dir, _ = _make_run(tmp_path)
    did_not_fit = "run:q3:long_context"
    export_manual_review(run_dir)
    rows = _read_csv(run_dir / "manual_review.csv")

    with (run_dir / "judgments.jsonl").open("a") as file:
        file.write(json.dumps(_judgment(did_not_fit, 1, 0)) + "\n")
    invalid_row = {**rows[0], "job_id": did_not_fit}
    invalid_row["human_accuracy"] = "1"
    _write_review_csv(run_dir / "manual_review.csv", [invalid_row])

    with pytest.raises(ValueError, match="no answer to review"):
        import_manual_review(run_dir)
    assert not (run_dir / "manual_reviews.jsonl").exists()


def test_cli_delegates_manual_review_commands(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "run"
    captured = []

    def fake_export(selected_run_dir, *, overwrite=False):
        captured.append(("export", selected_run_dir, overwrite))
        return {
            "path": selected_run_dir / "manual_review.csv",
            "disagreements": 2,
            "prefilled": 0,
        }

    def fake_import(selected_run_dir):
        captured.append(("import", selected_run_dir))
        return {"imported": 1, "skipped": 0, "unresolved": 1}

    monkeypatch.setattr("sec_rag_benchmark.cli.export_manual_review", fake_export)
    monkeypatch.setattr("sec_rag_benchmark.cli.import_manual_review", fake_import)

    assert main(["export-manual-review", "--run-dir", str(run_dir), "--overwrite"]) == 0
    assert main(["import-manual-review", "--run-dir", str(run_dir)]) == 0
    assert captured == [
        ("export", run_dir, True),
        ("import", run_dir),
    ]
    output = capsys.readouterr().out
    assert "disagreements" in output
    assert "imported" in output
