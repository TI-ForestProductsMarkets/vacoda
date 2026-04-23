"""
This file reads the config and resolves paths safely
"""

from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "paths.yaml"

def load_paths_config(config_path: Path | None = None) -> dict:
    """
    Loads the YAML config containing project paths. If no path is provided, it defaults to the CONFIG_PATH.
    """
    path = config_path or CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_project_root() -> Path:
    """
    Returns the root directory of the project.
    """
    return PROJECT_ROOT

def resolve_path(relative_path: str) -> Path:
    """
    Resolves a relative path to an absolute path based on the project root.
    """
    return get_project_root() / relative_path

def get_api_data_dir() -> Path:
    """
    Returns the path to the API data directory.
    """
    config = load_paths_config()
    return resolve_path(config["api_data_dir"])


def get_raw_data_dir() -> Path:
    """
    Returns the path to the raw data directory.
    """
    config = load_paths_config()
    return resolve_path(config["raw_data_dir"])

def get_rawdata_file_path(source: str = "local") -> Path:
    """
    Returns the path to the raw data file (first file in 'files' list for single mode).
    If source is 'api', uses the API data directory; otherwise, uses the raw data directory.
    """
    config = load_paths_config()
    files = config["raw_data"].get("files")
    if isinstance(files, list) and files:
        filename = files[0]
    elif isinstance(files, str):
        filename = files
    else:
        raise KeyError("No files specified in raw_data.files in config")
    if source == "api":
        data_dir = get_api_data_dir()
    else:
        data_dir = get_raw_data_dir()
    return data_dir / filename


def get_interim_data_dir() -> Path:
    """
    Returns the path to the interim data directory.
    """
    config = load_paths_config()
    return resolve_path(config["interim_data_dir"])

def get_processed_data_dir() -> Path:
    """
    Returns the path to the processed data directory.
    """
    config = load_paths_config()
    return resolve_path(config["processed_data_dir"])

def get_external_data_dir() -> Path:
    """
    Returns the path to the external data directory.
    """
    config = load_paths_config()
    return resolve_path(config["external_data_dir"])

def get_external_continent_mapping_file_path() -> Path:
    """
    Returns the path to the external continent mapping file.
    """
    config = load_paths_config()
    return resolve_path(config["external_data_dir"]) / config["external_data"]["country_codes_file"]

def get_processed_data_file_path() -> Path:
    """
    Returns the path to the processed data file.
    """
    config = load_paths_config()
    processed_cfg = config.get("processed_data", {})
    files = processed_cfg.get("files")
    if isinstance(files, list) and files:
        return get_processed_data_dir() / files[0]
    if isinstance(files, str):
        return get_processed_data_dir() / files
    if "file" in processed_cfg:
        return get_processed_data_dir() / processed_cfg["file"]
    raise KeyError("No file configured under processed_data.files (or legacy processed_data.file)")

def get_rawdata_sheet_name() -> str:
    """
    Returns the sheet name for the raw Un-Comtrade data.
    """
    config = load_paths_config()
    return config["raw_data"]["sheet_name"]

def get_figure_dir() -> Path:
    """
    Returns the path to the figure directory.
    """
    config = load_paths_config()
    return resolve_path(config["figure_dir"])

def get_processed_file_path_from_raw(raw_file_path: Path, processed_dir: Path) -> Path:
    """
    Given a raw CSV file path, returns the corresponding processed parquet file path in the processed directory.
    For example, if raw_file_path is 'data/raw/trade_data.csv' and processed_dir is 'data/processed',
    it will return 'data/processed/trade_data.parquet'.
    """
    return processed_dir / raw_file_path.with_suffix(".parquet").name