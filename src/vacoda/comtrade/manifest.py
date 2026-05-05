"""
vacoda.comtrade.audit_manifest

Utilities for reading, writing, and auditing Comtrade download manifests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ManifestSummary:
    total: int
    downloaded: int
    skipped: int
    error: int
    rows_downloaded_total: int


def load_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file does not exist: {manifest_path}")

    rows = json.loads(manifest_path.read_text(encoding="utf-8"))

    if not isinstance(rows, list):
        raise ValueError(f"Expected manifest to contain a list: {manifest_path}")

    return rows


def summarize_manifest(rows: list[dict[str, Any]]) -> ManifestSummary:
    return ManifestSummary(
        total=len(rows),
        downloaded=sum(row.get("status") == "downloaded" for row in rows),
        skipped=sum(row.get("status") == "skipped" for row in rows),
        error=sum(row.get("status") == "error" for row in rows),
        rows_downloaded_total=sum((row.get("rows_downloaded") or 0) for row in rows),
    )


def summarize_manifest_file(manifest_path: Path) -> ManifestSummary:
    rows = load_manifest(manifest_path)
    return summarize_manifest(rows)


def print_manifest_summary(manifest_path: Path) -> ManifestSummary:
    summary = summarize_manifest_file(manifest_path)

    print(f"Manifest: {manifest_path}")
    print(f"total: {summary.total}")
    print(f"downloaded: {summary.downloaded}")
    print(f"skipped: {summary.skipped}")
    print(f"error: {summary.error}")
    print(f"rows_downloaded total: {summary.rows_downloaded_total}")

    return summary