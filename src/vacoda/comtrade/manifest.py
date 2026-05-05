"""
vacoda.comtrade.manifest

Utilities for reading, writing, and auditing Comtrade download manifests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
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
    rows_downloaded_total: int

    @property
    def ok(self) -> bool:
        return self.error == 0


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


def summarize_manifest(
    rows: list[dict[str, Any]],
    manifest_path: Path,
) -> ManifestSummary:
    return ManifestSummary(
        manifest_path=manifest_path,
        total=len(rows),
        downloaded=sum(row.get("status") == "downloaded" for row in rows),
        skipped=sum(row.get("status") == "skipped" for row in rows),
        error=sum(row.get("status") == "error" for row in rows),
        rows_downloaded_total=sum((row.get("rows_downloaded") or 0) for row in rows),
    )


def summarize_manifest_file(manifest_path: Path) -> ManifestSummary:
    rows = load_manifest(manifest_path)
    return summarize_manifest(rows, manifest_path)


def log_manifest_summary(summary: ManifestSummary) -> None:
    logger.info("-" * 72)
    logger.info("Manifest audit: %s", rel_path(summary.manifest_path))
    logger.info("  total: %d", summary.total)
    logger.info("  downloaded: %d", summary.downloaded)
    logger.info("  skipped: %d", summary.skipped)
    logger.info("  error: %d", summary.error)
    logger.info("  rows_downloaded_total: %d", summary.rows_downloaded_total)
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
            f"Manifest audit failed: {summary.error} job(s) failed. "
            f"Manifest: {manifest_path}"
        )

    return summary