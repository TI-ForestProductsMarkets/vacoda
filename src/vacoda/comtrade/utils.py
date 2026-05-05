"""
Shared config and utility functions for Comtrade API pipeline and download modules.
"""
from pathlib import Path
from typing import Any
import yaml
import json
from vacoda.io.paths import resolve_path
from vacoda.utils.logger import get_logger

logger = get_logger("COMTRADE.UTILS")


def _load_yaml(path: Path) -> dict[str, Any]:
    logger.debug("Loading YAML config from %s", path)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at root of YAML: {path}")
    return data

def _resolve_output_dir(paths_cfg: dict[str, Any], output_cfg) -> Path:
    rel_dir = output_cfg.directory if str(output_cfg.directory).strip() else paths_cfg.get("api_data_dir")
    return resolve_path(str(rel_dir))

def load_api_files_from_manifest(manifest_path: Path) -> list[Path]:

    rows = json.loads(manifest_path.read_text(encoding="utf-8"))

    files = []
    for row in rows:
        if row.get("status") in {"downloaded", "skipped"}:
            files.append(Path(row["output_path"]))

    return files


def get_single_manifest_path(single_cfg: dict[str, Any]) -> Path:
    single_output = single_cfg.get("output", {})
    output_dir = resolve_path(single_output.get("directory", "data/raw/api/single"))
    manifest_name = single_output.get("manifest_file", "download_manifest.json")
    return output_dir / manifest_name


def get_bulk_manifest_path(bulk_cfg: dict[str, Any]) -> Path:
    bulk_output = bulk_cfg.get("output", {})
    output_dir = resolve_path(bulk_output.get("directory", "data/raw/api/bulk"))
    manifest_name = bulk_output.get("manifest_file", "download_manifest_bulk.json")
    return output_dir / manifest_name