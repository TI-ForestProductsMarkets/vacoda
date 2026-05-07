from pathlib import Path

from vacoda.comtrade.manifest import summarize_manifest
from vacoda.comtrade.manifest import summarize_manifest
from vacoda.comtrade.manifest import find_invalid_manifest_rows
from vacoda.comtrade.manifest import validate_output_file


def test_summarize_manifest_counts_single_rows(tmp_path: Path) -> None:
    file_path = tmp_path / "download.csv"
    file_path.write_text("a,b\n1,2\n", encoding="utf-8")

    rows = [
        {
            "status": "downloaded",
            "rows_downloaded": 10,
            "output_path": str(file_path),
        },
        {
            "status": "skipped",
            "rows_downloaded": None,
            "output_path": str(file_path),
        },
        {
            "status": "error",
            "error": "timeout",
        },
    ]

    summary = summarize_manifest(rows, tmp_path / "manifest.json")

    assert summary.total == 3
    assert summary.downloaded == 1
    assert summary.skipped == 1
    assert summary.error == 1
    assert summary.rows_downloaded_total == 10
    assert summary.output_file_count == 2
    assert summary.existing_file_count == 2
    assert summary.missing_file_count == 0


def test_summarize_manifest_detects_missing_bulk_file(tmp_path: Path) -> None:
    existing = tmp_path / "file_001.gz"
    missing = tmp_path / "file_002.gz"

    existing.write_bytes(b"abc")

    rows = [
        {
            "status": "downloaded",
            "output_paths": [str(existing), str(missing)],
        }
    ]

    summary = summarize_manifest(rows, tmp_path / "manifest.json")

    assert summary.output_file_count == 2
    assert summary.existing_file_count == 1
    assert summary.missing_file_count == 1
    assert summary.ok is False


def test_manifest_marks_zero_byte_skipped_file_as_suspicious(tmp_path):
    output_path = tmp_path / "comtrade_441012_2023.csv"
    output_path.write_bytes(b"")

    rows = [
        {
            "job": "wood_panel_direct_441012_2023",
            "mode": "single",
            "status": "skipped_existing",
            "output_path": str(output_path),
        }
    ]

    summary = summarize_manifest(rows, tmp_path / "manifest.json")

    assert summary.total == 1
    assert summary.zero_byte_file_count == 1
    assert summary.suspicious_zero_byte_file_count == 1
    assert summary.ok is False


def test_manifest_valid_skipped_file_is_ok(tmp_path):
    output_path = tmp_path / "comtrade_441012_2023.csv"
    output_path.write_text("a,b\n1,2\n", encoding="utf-8")

    rows = [
        {
            "job": "wood_panel_direct_441012_2023",
            "mode": "single",
            "status": "skipped_existing",
            "output_path": str(output_path),
        }
    ]

    summary = summarize_manifest(rows, tmp_path / "manifest.json")

    assert summary.status_counts == {"skipped_existing": 1}
    assert summary.zero_byte_file_count == 0
    assert summary.suspicious_zero_byte_file_count == 0
    assert summary.ok is True



def test_find_invalid_manifest_rows_returns_zero_byte_skipped_row(tmp_path):
    output_path = tmp_path / "bad.csv"
    output_path.write_bytes(b"")

    rows = [
        {
            "job": "bad_job",
            "status": "skipped_existing",
            "output_path": str(output_path),
        }
    ]

    invalid = find_invalid_manifest_rows(rows)

    assert len(invalid) == 1
    assert invalid[0]["job"] == "bad_job"
    assert invalid[0]["validation_status"] == "zero_byte"


def test_find_invalid_manifest_rows_ignores_no_data_without_file(tmp_path):
    rows = [
        {
            "job": "no_data_job",
            "status": "no_data",
            "output_path": None,
        }
    ]

    invalid = find_invalid_manifest_rows(rows)

    assert invalid == []



def test_validate_output_file_missing(tmp_path):
    path = tmp_path / "missing.csv"

    valid, reason, size = validate_output_file(path)

    assert valid is False
    assert reason == "missing"
    assert size == 0


def test_validate_output_file_zero_byte(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_bytes(b"")

    valid, reason, size = validate_output_file(path)

    assert valid is False
    assert reason == "zero_byte"
    assert size == 0


def test_validate_output_file_too_small(tmp_path):
    path = tmp_path / "tiny.csv"
    path.write_text("a,b\n", encoding="utf-8")

    valid, reason, size = validate_output_file(path, min_size_bytes=20)

    assert valid is False
    assert reason == "too_small"
    assert size > 0


def test_validate_output_file_valid(tmp_path):
    path = tmp_path / "valid.csv"
    path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    valid, reason, size = validate_output_file(path, min_size_bytes=5)

    assert valid is True
    assert reason == "valid"
    assert size > 0

