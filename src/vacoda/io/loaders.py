
"""
Data loading utilities for the vacoda pipeline.
Loads raw data from CSV files and metadata from parquet files.
All paths are controlled by configs/paths.yaml.
"""

from pathlib import Path
import pandas as pd
import yaml
from vacoda.io.paths import (
    get_api_data_dir,
    get_raw_data_dir,
    get_rawdata_file_path,
    CONFIG_PATH,
    resolve_path
)
from vacoda.utils.decorators import require_file
from vacoda.utils.logger import get_logger, rel_path

logger = get_logger("IO.LOADERS")
_config_cache = None

def load_config():
    """
    Loads the configuration for the project. Caches the result to avoid redundant file reads.
    """
    global _config_cache
    if _config_cache is None:
        if not CONFIG_PATH.exists():
            raise FileNotFoundError(
                f"File not found at: {CONFIG_PATH}\n"
                f"Make sure configs/paths.yaml exists in the project root."
            )
        logger.info(f"[Config] Loading config from: {rel_path(CONFIG_PATH)}")
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            _config_cache = yaml.safe_load(f)
    return _config_cache


def _read_csv_with_diagnostics(path: Path) -> pd.DataFrame:
    """
    Reads a CSV file into a DataFrame, logs shape and columns, and checks for single-column errors.
    """
    df = pd.read_csv(path)
    logger.info(f"[Data Load] {path.name}: {df.shape[0]:,} rows × {df.shape[1]} cols")
    logger.debug(f"[Data Load] First columns: {list(df.columns)[:8]}")
    if len(df.columns) <= 1:
        logger.error(f"[CSV Loader] Failed to properly parse {path.name}. Only one column found. Please check the file format and delimiter.")
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            preview = f.read(200)
        logger.error(f"[CSV Loader] File preview (first 200 chars):\n{preview}")
        raise SystemExit(f"[CSV Loader] Stopped: {path.name} could not be parsed correctly. Only one column found.")
    return df


def get_raw_csv_file_list(source: str = "local") -> list[Path]:
    """
    Returns a list of Path objects for all CSV files matching the patterns in raw_data.files in paths.yaml.
    If source is 'api', uses the API data directory; otherwise, uses the raw data directory.
    """
    cfg = load_config()
    if source == "api":
        raw_dir = get_api_data_dir()
    else:
        raw_dir = get_raw_data_dir()
    patterns = cfg["raw_data"].get("files", ["*.csv"])
    if isinstance(patterns, str):
        patterns = [patterns]
    files = []
    for pattern in patterns:
        files.extend(sorted(raw_dir.glob(pattern)))
    return files


def load_metadata_from_config() -> dict[str, pd.DataFrame]:
    """
    Loads all metadata parquet files as specified in paths.yaml under metadata.
    Returns a dict of DataFrames keyed by metadata name.
    """
    from vacoda.preprocessing.metadata import standardize_metadata_dtypes

    cfg = load_config()
    meta_cfg = cfg.get("metadata", {})
    meta = {}
    for key, rel_path_str in meta_cfg.items():
        path = resolve_path(rel_path_str)
        logger.info(f"Loading metadata '{key}' from: {rel_path(path)}")
        df = pd.read_parquet(path)
        meta[key] = standardize_metadata_dtypes(df)
    return meta

def get_processed_parquet_file_list() -> list[Path]:
    """
    Returns processed parquet files configured under processed_data.files.
    Supports exact filenames and glob patterns.
    """
    cfg = load_config()
    processed_dir = resolve_path(cfg.get("processed_data_dir", "data/processed"))
    patterns = cfg.get("processed_data", {}).get("files", ["*.parquet"])
    if isinstance(patterns, str):
        patterns = [patterns]

    files: dict[str, Path] = {}
    for pattern in patterns:
        for path in processed_dir.glob(pattern):
            files[str(path)] = path
    return sorted(files.values())


def load_processed_trade_data(file_path: Path | str | None = None) -> pd.DataFrame:
    """
    Load processed trade data parquet.

    Priority:
    1. explicit file_path argument
    2. first existing file from processed_data.files in config
    3. first parquet file in processed_data_dir
    """
    if file_path is not None:
        path = Path(file_path)
        logger.info(f"Loading processed trade data from: {rel_path(path)}")
        return pd.read_parquet(path)

    configured_files = get_processed_parquet_file_list()
    if configured_files:
        path = configured_files[0]
        logger.info(f"Loading processed trade data from configured file: {rel_path(path)}")
        return pd.read_parquet(path)

    cfg = load_config()
    processed_dir = resolve_path(cfg.get("processed_data_dir", "data/processed"))
    fallback_files = sorted(processed_dir.glob("*.parquet"))
    if not fallback_files:
        raise FileNotFoundError(
            f"No processed parquet file found in {rel_path(processed_dir)}. "
            f"Set processed_data.files in configs/paths.yaml or create a parquet output first."
        )

    path = fallback_files[0]
    logger.warning(
        f"No configured processed_data.files match found. "
        f"Falling back to first parquet file: {rel_path(path)}"
    )
    return pd.read_parquet(path)

def load_data_from_config(file_path: Path = None) -> pd.DataFrame:
    """
    Loads the main input CSV(s) based on config (pipeline.mode and pipeline.source) or explicit file_path.
    - If file_path is provided, loads that file (per_file mode).
    - Otherwise, reads config to determine mode and source.
    - For per_file: loads a single CSV.
    - For merged: loads and concatenates all matching CSVs.
    Returns a DataFrame.
    """
    cfg = load_config()
    pipeline_cfg = cfg.get("pipeline", {})
    mode = pipeline_cfg.get("mode", "per_file").strip().lower()
    source = pipeline_cfg.get("source", "api").strip().lower()
    if file_path is not None:
        logger.info(f"[Config] Loading explicit file: {file_path}")
        return _read_csv_with_diagnostics(file_path)
    if mode == "per_file":
        from vacoda.io.paths import get_rawdata_file_path
        csv_path = get_rawdata_file_path(source=source)
        logger.info(f"[Config] Loading single file for mode=per_file: {csv_path}")
        return _read_csv_with_diagnostics(csv_path)
    elif mode == "merged":
        files = get_raw_csv_file_list(source=source)
        logger.info(f"[Config] Loading {len(files)} files for mode=merged from {source}")
        frames = []
        for path in files:
            logger.debug(f"  Loading {path.name}")
            frames.append(_read_csv_with_diagnostics(path))
        combined = pd.concat(frames, ignore_index=True)
        logger.info(f"[Config] Combined shape: {combined.shape[0]:,} rows × {combined.shape[1]} cols")
        return combined
    else:
        raise ValueError(f"Unknown pipeline mode: {mode}")