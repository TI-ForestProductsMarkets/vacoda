"""
Shared config and utility functions for Comtrade API pipeline and download modules.
"""
from pathlib import Path
from typing import Any
import yaml
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
