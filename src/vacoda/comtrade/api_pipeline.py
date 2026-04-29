from pathlib import Path
from typing import Any

import pandas as pd
from vacoda.pipeline.build_processed_api_dataset import build_processed_api_dataset
#from vacoda.io.paths import get_rawdata_file_path, get_processed_data_file_path
import vacoda.io.loaders as loaders_mod
from vacoda.comtrade.utils import load_api_files_from_manifest
from vacoda.io.paths import get_project_root, resolve_path, get_processed_data_file_path
from vacoda.comtrade.comtrade_pipeline import run_comtrade_api_pipeline
from vacoda.utils.logger import get_logger, rel_path
from vacoda.io.paths import get_project_root, load_paths_config, resolve_path


logger = get_logger("PIPELINE.API")


def run_full_api_pipeline(
    paths_config_path: Path | None = None,
    api_config_path: Path | None = None,
) -> Path:
    project_root = get_project_root()

    resolved_paths_cfg = paths_config_path or project_root / "configs" / "paths.yaml"
    paths_cfg = load_paths_config(resolved_paths_cfg)

    # 1. Download API data
    manifest_rows = run_comtrade_api_pipeline(
        paths_config_path=resolved_paths_cfg,
        api_config_path=api_config_path,
    )

    # 2. Resolve downloaded files from manifest
    api_data_dir = project_root / paths_cfg["api_data_dir"]
    manifest_path = api_data_dir / "download_manifest.json"

    api_csv_paths = load_api_files_from_manifest(manifest_path)

    if not api_csv_paths:
        raise FileNotFoundError(f"No API files found in manifest: {manifest_path}")

    logger.info("Found %d API file(s) from manifest", len(api_csv_paths))

    # 3. Process files
    dfs = []
    for path in api_csv_paths:
        logger.info("Processing API file: %s", rel_path(path))
        dfs.append(build_processed_api_dataset(file_path=path))

    df = pd.concat(dfs, ignore_index=True)

    # 4. Save processed output
    output_path = get_processed_data_file_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    logger.info("Processed API data saved to: %s", rel_path(output_path))

    return output_path