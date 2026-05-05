"""
Shared utility functions for Comtrade download modules.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from vacoda.io.paths import resolve_path
from vacoda.utils.logger import get_logger

logger = get_logger("COMTRADE.UTILS")


def load_yaml(path: Path) -> dict[str, Any]:
    logger.debug("Loading YAML config from %s", path)

    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at root of YAML: {path}")

    return data


def get_output_dir(section_cfg: dict[str, Any], default: str) -> Path:
    output_cfg = section_cfg.get("output", {})
    if not isinstance(output_cfg, dict):
        output_cfg = {}

    directory = output_cfg.get("directory", default)
    return resolve_path(str(directory))


def get_manifest_name(section_cfg: dict[str, Any], default: str) -> str:
    output_cfg = section_cfg.get("output", {})
    if not isinstance(output_cfg, dict):
        output_cfg = {}

    return str(output_cfg.get("manifest_file", default))


def get_manifest_path(section_cfg: dict[str, Any], default_dir: str, default_name: str) -> Path:
    output_dir = get_output_dir(section_cfg, default_dir)
    manifest_name = get_manifest_name(section_cfg, default_name)
    return output_dir / manifest_name


def get_single_manifest_path(single_cfg: dict[str, Any]) -> Path:
    return get_manifest_path(
        section_cfg=single_cfg,
        default_dir="data/raw/api/single",
        default_name="download_manifest_single.json",
    )


def get_bulk_manifest_path(bulk_cfg: dict[str, Any]) -> Path:
    return get_manifest_path(
        section_cfg=bulk_cfg,
        default_dir="data/raw/api/bulk",
        default_name="download_manifest_bulk.json",
    )


def write_manifest(rows: list[dict[str, Any]], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    logger.info("Manifest written to: %s", manifest_path)


def load_api_files_from_manifest(manifest_path: Path) -> list[Path]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file does not exist: {manifest_path}")

    rows = json.loads(manifest_path.read_text(encoding="utf-8"))

    if not isinstance(rows, list):
        raise ValueError(f"Expected manifest to contain a list: {manifest_path}")

    files: list[Path] = []

    for row in rows:
        if not isinstance(row, dict):
            continue

        if row.get("status") not in {"downloaded", "skipped"}:
            continue

        output_paths = row.get("output_paths")
        if isinstance(output_paths, list):
            files.extend(Path(path) for path in output_paths)
            continue

        output_path = row.get("output_path")
        if output_path:
            files.append(Path(output_path))

    return files

## Usage: output_file = format_output_file(job.output_file, job.params)
def format_output_file(template: str, params: dict[str, Any]) -> str:
    try:
        return template.format(**params)
    except KeyError as exc:
        missing = exc.args[0]
        raise ValueError(
            f"output_file template references missing parameter: {missing}"
        ) from exc