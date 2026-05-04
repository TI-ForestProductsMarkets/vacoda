from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from vacoda.pipeline.build_processed_api_dataset import build_processed_api_dataset
from vacoda.comtrade.utils import load_api_files_from_manifest
from vacoda.io.paths import get_project_root, resolve_path, get_processed_data_file_path
from vacoda.comtrade.comtrade_pipeline import run_comtrade_api_pipeline
from vacoda.utils.logger import get_logger, rel_path
from vacoda.io.paths import get_project_root, load_paths_config, resolve_path


logger = get_logger("PIPELINE.API")


def _load_bulk_config(api_config_path: Path | None) -> dict[str, Any]:
    if api_config_path is None:
        return {}

    with open(api_config_path, "r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file) or {}

    bulk_config = raw_config.get("bulk", {})
    return bulk_config if isinstance(bulk_config, dict) else {}


def _get_bulk_manifest_path(bulk_cfg: dict[str, Any], api_data_dir: str) -> Path:
    bulk_output_dir = resolve_path(str(bulk_cfg.get("output_directory", api_data_dir)))
    bulk_manifest_name = str(bulk_cfg.get("manifest_file", "bulk_download_manifest.json"))
    return bulk_output_dir / bulk_manifest_name


def _process_api_files(api_paths: list[Path]) -> Path:
    if not api_paths:
        raise FileNotFoundError("No API files were provided for processing.")

    logger.info("Found %d API file(s) for processing", len(api_paths))
    dfs = []
    for path in api_paths:
        logger.info("Processing API file: %s", rel_path(path))
        dfs.append(build_processed_api_dataset(file_path=path, confirm_settings=False))

    df = pd.concat(dfs, ignore_index=True)
    output_path = get_processed_data_file_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info("Processed API data saved to: %s", rel_path(output_path))
    return output_path


def run_full_api_pipeline(
    paths_config_path: Path | None = None,
    api_config_path: Path | None = None,
) -> Path:
    project_root = get_project_root()

    resolved_paths_cfg = paths_config_path or project_root / "configs" / "paths.yaml"
    resolved_api_cfg = api_config_path or project_root / "configs" / "comtrade.yaml"
    paths_cfg = load_paths_config(resolved_paths_cfg)
    bulk_cfg = _load_bulk_config(resolved_api_cfg)

    # 1. Download API data
    manifest_rows = run_comtrade_api_pipeline(
        paths_config_path=resolved_paths_cfg,
        api_config_path=resolved_api_cfg,
    )

    if bulk_cfg.get("enabled", False):
        bulk_manifest_path = _get_bulk_manifest_path(bulk_cfg, paths_cfg["api_data_dir"])
        failed_jobs = [row for row in manifest_rows if row.get("status") == "error"]
        downloaded_jobs = [row for row in manifest_rows if row.get("status") == "downloaded"]

        if failed_jobs:
            failed_job_summary = "; ".join(
                f"{row.get('job', 'bulk_job')}: {row.get('error', 'unknown error')}"
                for row in failed_jobs
            )
            raise RuntimeError(
                "Bulk download failed for one or more jobs. "
                f"See manifest {bulk_manifest_path}. Errors: {failed_job_summary}"
            )

        bulk_file_paths = [
            Path(path)
            for row in downloaded_jobs
            for path in row.get("output_paths", [])
        ]


        if not bulk_file_paths:
            raise FileNotFoundError(
                f"Bulk download completed but no output files were recorded in manifest {bulk_manifest_path}."
            )

        logger.info(
            "Bulk download completed with %d job(s). Processing %d referenced file(s). Bulk manifest: %s",
            len(downloaded_jobs),
            len(bulk_file_paths),
            rel_path(bulk_manifest_path),
        )
        return _process_api_files(bulk_file_paths)

    # 2. Resolve downloaded files from manifest
    api_data_dir = project_root / paths_cfg["api_data_dir"]
    manifest_path = api_data_dir / "download_manifest.json"

    api_csv_paths = load_api_files_from_manifest(manifest_path)

    if not api_csv_paths:
        raise FileNotFoundError(f"No API files found in manifest: {manifest_path}")

    return api_csv_paths
    #return _process_api_files(api_csv_paths)