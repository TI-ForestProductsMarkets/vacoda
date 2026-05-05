from pathlib import Path

from vacoda.comtrade.manifest import summarize_manifest


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