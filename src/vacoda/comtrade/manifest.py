"""
vacoda.comtrade.manifest

Utilities for reading, writing, and auditing Comtrade download manifests.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vacoda.utils.logger import get_logger, rel_path

logger = get_logger("COMTRADE.MANIFEST")


@dataclass(frozen=True)
class ManifestSummary:
    manifest_path: Path
    total: int
    downloaded: int
    skipped: int
    error: int
    rows_downloaded_total: int = 0

    # Bulk/file-oriented fields
    output_file_count: int = 0
    existing_file_count: int = 0
    missing_file_count: int = 0
    zero_byte_file_count: int = 0
    total_size_bytes: int = 0
    suffix_counts: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error == 0 and self.missing_file_count == 0 and self.zero_byte_file_count == 0


def write_manifest(rows: list[dict[str, Any]], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    logger.info("Manifest written to: %s", rel_path(manifest_path))


def load_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file does not exist: {manifest_path}")

    rows = json.loads(manifest_path.read_text(encoding="utf-8"))

    if not isinstance(rows, list):
        raise ValueError(f"Expected manifest to contain a list: {manifest_path}")

    return rows


def _collect_output_paths(rows: list[dict[str, Any]]) -> list[Path]:
    paths: list[Path] = []

    for row in rows:
        if not isinstance(row, dict):
            continue

        output_paths = row.get("output_paths")
        if isinstance(output_paths, list):
            paths.extend(Path(path) for path in output_paths if path)
            continue

        output_path = row.get("output_path")
        if output_path:
            paths.append(Path(output_path))

    return paths


def _audit_output_files(paths: list[Path]) -> dict[str, Any]:
    existing = 0
    missing = 0
    zero_byte = 0
    total_size = 0
    suffix_counter: Counter[str] = Counter()

    for path in paths:
        suffix = path.suffix.lower() or "<no_suffix>"
        suffix_counter[suffix] += 1

        if not path.exists():
            missing += 1
            continue

        existing += 1
        size = path.stat().st_size
        total_size += size

        if size == 0:
            zero_byte += 1

    return {
        "output_file_count": len(paths),
        "existing_file_count": existing,
        "missing_file_count": missing,
        "zero_byte_file_count": zero_byte,
        "total_size_bytes": total_size,
        "suffix_counts": dict(sorted(suffix_counter.items())),
    }


def summarize_manifest(
    rows: list[dict[str, Any]],
    manifest_path: Path,
) -> ManifestSummary:
    output_paths = _collect_output_paths(rows)
    file_audit = _audit_output_files(output_paths)

    return ManifestSummary(
        manifest_path=manifest_path,
        total=len(rows),
        downloaded=sum(row.get("status") == "downloaded" for row in rows),
        skipped=sum(row.get("status") == "skipped" for row in rows),
        error=sum(row.get("status") == "error" for row in rows),
        rows_downloaded_total=sum((row.get("rows_downloaded") or 0) for row in rows),
        output_file_count=file_audit["output_file_count"],
        existing_file_count=file_audit["existing_file_count"],
        missing_file_count=file_audit["missing_file_count"],
        zero_byte_file_count=file_audit["zero_byte_file_count"],
        total_size_bytes=file_audit["total_size_bytes"],
        suffix_counts=file_audit["suffix_counts"],
    )


def summarize_manifest_file(manifest_path: Path) -> ManifestSummary:
    rows = load_manifest(manifest_path)
    return summarize_manifest(rows, manifest_path)


def _format_bytes(n: int) -> str:
    value = float(n)
    units = ["B", "KB", "MB", "GB", "TB"]

    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024

    return f"{n} B"


def log_manifest_summary(summary: ManifestSummary) -> None:
    logger.info("-" * 72)
    logger.info("Manifest audit: %s", rel_path(summary.manifest_path))
    logger.info("  total manifest rows: %d", summary.total)
    logger.info("  downloaded jobs: %d", summary.downloaded)
    logger.info("  skipped jobs: %d", summary.skipped)
    logger.info("  error jobs: %d", summary.error)

    if summary.rows_downloaded_total:
        logger.info("  rows_downloaded_total: %d", summary.rows_downloaded_total)

    if summary.output_file_count:
        logger.info("  output_file_count: %d", summary.output_file_count)
        logger.info("  existing_file_count: %d", summary.existing_file_count)
        logger.info("  missing_file_count: %d", summary.missing_file_count)
        logger.info("  zero_byte_file_count: %d", summary.zero_byte_file_count)
        logger.info("  total_size: %s", _format_bytes(summary.total_size_bytes))
        logger.info("  suffix_counts: %s", summary.suffix_counts)

    logger.info("  ok: %s", summary.ok)
    logger.info("-" * 72)


def audit_manifest_file(
    manifest_path: Path,
    fail_on_error: bool = False,
) -> ManifestSummary:
    summary = summarize_manifest_file(manifest_path)
    log_manifest_summary(summary)

    if fail_on_error and not summary.ok:
        raise RuntimeError(
            "Manifest audit failed. "
            f"errors={summary.error}, "
            f"missing_files={summary.missing_file_count}, "
            f"zero_byte_files={summary.zero_byte_file_count}, "
            f"manifest={manifest_path}"
        )

    return summary

def audit_bulk_index_count(row: dict[str, Any]) -> tuple[int | None, int]:
    index_path_raw = row.get("bulk_index_path")
    output_paths = row.get("output_paths") or []

    if not index_path_raw:
        return None, len(output_paths)

    index_path = Path(index_path_raw)
    if not index_path.exists():
        return None, len(output_paths)

    data = json.loads(index_path.read_text(encoding="utf-8"))
    index_count = data.get("count")

    return int(index_count), len(output_paths)