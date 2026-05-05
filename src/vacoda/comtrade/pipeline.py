"""
vacoda.comtrade.pipeline

Clean Comtrade download pipeline.

Responsibilities:
- Load Comtrade config
- Validate authentication
- Run bulk downloads if enabled
- Run single API downloads if enabled
- Write separate manifests for bulk and single downloads

This file replaces the messy overlap between:
- comtrade_pipeline.py
- api_pipeline.py
- parts of download.py
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from vacoda.comtrade.api import (
    build_auth_headers,
    build_bulk_params,
    build_bulk_url,
    build_query_params,
    detect_auth_error_in_payload,
    download_bulk,
    download_paginated_csv,
    download_with_retry,
    is_csv_format,
    redact_url,
    select_resolvable_endpoint,
    validate_auth_preflight,
)
from vacoda.comtrade.config import ComtradeConfig, JobConfig, ModeConfig, load_comtrade_config
from vacoda.io.paths import get_project_root, resolve_path
from vacoda.utils.logger import get_logger, rel_path

logger = get_logger("COMTRADE.PIPELINE")


# ---------------------------------------------------------------------
# Small generic helpers
# ---------------------------------------------------------------------


def _write_manifest(rows: list[dict[str, Any]], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    logger.info("Manifest written to: %s", rel_path(manifest_path))


def _mode_output_dir(mode_cfg: ModeConfig) -> Path:
    return resolve_path(mode_cfg.output.directory)


def _mode_manifest_path(mode_cfg: ModeConfig) -> Path:
    return _mode_output_dir(mode_cfg) / mode_cfg.output.manifest_file


def _format_output_file(template: str, params: dict[str, Any]) -> str:
    """
    Format output file names such as:

        Comtrade_bulk_HS_{period}_all_reporters.zip

    using values from params.
    """
    try:
        return template.format(**params)
    except KeyError as exc:
        missing = exc.args[0]
        raise ValueError(
            f"output_file template references missing parameter: {missing!r}. "
            f"Template was: {template!r}"
        ) from exc


def _expand_year_jobs(job: JobConfig) -> list[JobConfig]:
    """
    Expand a job with all_years_start/all_years_end into one job per year.

    Example YAML:

        output_file: "wood_panel_direct_{period}.csv"
        all_years_start: 2007
        all_years_end: 2025

    becomes jobs with period=2007, period=2008, etc.

    If no year range is defined, the job is returned unchanged.
    """
    if job.all_years_start is None or job.all_years_end is None:
        return [job]

    start = int(job.all_years_start)
    end = int(job.all_years_end)

    if end < start:
        raise ValueError(
            f"Invalid year range for job {job.name!r}: "
            f"all_years_end={end} is smaller than all_years_start={start}"
        )

    expanded_jobs: list[JobConfig] = []

    for year in range(start, end + 1):
        params = dict(job.params)
        params["period"] = str(year)

        expanded_jobs.append(
            JobConfig(
                name=f"{job.name}_{year}",
                params=params,
                output_file=job.output_file,
                local_filter=job.local_filter,
                all_years_start=None,
                all_years_end=None,
            )
        )

    return expanded_jobs


def _expand_jobs(jobs: list[JobConfig]) -> list[JobConfig]:
    expanded: list[JobConfig] = []

    for job in jobs:
        expanded.extend(_expand_year_jobs(job))

    return expanded


def _log_step(step: int, total: int, title: str) -> None:
    logger.info("=" * 72)
    logger.info("STEP %d/%d — %s", step, total, title)
    logger.info("=" * 72)


# ---------------------------------------------------------------------
# Bulk mode
# ---------------------------------------------------------------------


def run_bulk_downloads(cfg: ComtradeConfig) -> list[dict[str, Any]]:
    """
    Run Comtrade Bulk API downloads.

    Bulk downloads usually return broad files. Commodity/flow filtering should
    happen locally after download, not at the bulk API request level.
    """
    if not cfg.bulk.enabled:
        logger.info("[bulk] Bulk mode is disabled.")
        return []

    output_dir = _mode_output_dir(cfg.bulk)
    manifest_path = _mode_manifest_path(cfg.bulk)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[bulk] Output directory: %s", rel_path(output_dir))
    logger.info("[bulk] Manifest path: %s", rel_path(manifest_path))

    manifest_rows: list[dict[str, Any]] = []

    jobs = _expand_jobs(cfg.bulk.jobs)

    for idx, job in enumerate(jobs, start=1):
        logger.info("[bulk] Job %d/%d START: %s", idx, len(jobs), job.name)

        try:
            output_file = _format_output_file(job.output_file, job.params)
            output_path = output_dir / output_file

            if output_path.exists() and not cfg.runtime.overwrite_existing:
                logger.info(
                    "[bulk] Skipping existing file because overwrite_existing=false: %s",
                    rel_path(output_path),
                )
                manifest_rows.append(
                    {
                        "job": job.name,
                        "mode": "bulk",
                        "status": "skipped",
                        "output_path": str(output_path),
                        "output_paths": [str(output_path)],
                        "local_filter": job.local_filter,
                    }
                )
                continue

            saved_paths = download_bulk(
                endpoint=cfg.endpoint.bulk,
                job_params=job.params,
                retry=cfg.retry,
                runtime=cfg.runtime,
                auth=cfg.auth,
                output_path=output_path,
            )

            manifest_rows.append(
                {
                    "job": job.name,
                    "mode": "bulk",
                    "status": "downloaded",
                    "output_path": saved_paths[0] if saved_paths else None,
                    "output_paths": saved_paths,
                    "local_filter": job.local_filter,
                }
            )

            logger.info("[bulk] Job %s completed with %d file(s).", job.name, len(saved_paths))

        except Exception as exc:
            logger.exception("[bulk] Job failed: %s", job.name)
            manifest_rows.append(
                {
                    "job": job.name,
                    "mode": "bulk",
                    "status": "error",
                    "error": str(exc),
                    "local_filter": job.local_filter,
                }
            )

    _write_manifest(manifest_rows, manifest_path)
    return manifest_rows


# ---------------------------------------------------------------------
# Single normal API mode
# ---------------------------------------------------------------------


def _download_single_job(
    *,
    cfg: ComtradeConfig,
    endpoint: str,
    job: JobConfig,
    output_path: Path,
) -> dict[str, Any]:
    """
    Download one normal Comtrade Data API job.
    """
    params = build_query_params(
        default_params=cfg.default_params,
        job_params=job.params,
        auth=cfg.auth,
    )
    headers = build_auth_headers(cfg.auth)

    urls_used: list[str]
    rows_downloaded: int | None = None

    if cfg.pagination.enabled:
        if not is_csv_format(params):
            raise ValueError("Pagination supports CSV only. Set default_params.format='csv'.")

        payload, urls_used, rows_downloaded = download_paginated_csv(
            endpoint=endpoint,
            params=params,
            retry=cfg.retry,
            runtime=cfg.runtime,
            pagination=cfg.pagination,
            headers=headers,
        )

    else:
        url = f"{endpoint}?{urlencode(params, doseq=True)}"
        payload = download_with_retry(
            url=url,
            retry=cfg.retry,
            runtime=cfg.runtime,
            headers=headers,
        )

        auth_error = detect_auth_error_in_payload(payload)
        if auth_error:
            raise PermissionError(f"API returned authentication/authorization payload: {auth_error}")

        urls_used = [redact_url(url)]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(payload)

    return {
        "job": job.name,
        "mode": "single",
        "status": "downloaded",
        "output_path": str(output_path),
        "url": urls_used[0] if len(urls_used) == 1 else None,
        "url_count": len(urls_used),
        "rows_downloaded": rows_downloaded,
        "params": params,
    }


def run_single_downloads(cfg: ComtradeConfig) -> list[dict[str, Any]]:
    """
    Run normal Comtrade Data API downloads.

    Uses controlled parallelism because these jobs are network-bound.
    """
    if not cfg.single.enabled:
        logger.info("[single] Single API mode is disabled.")
        return []

    output_dir = _mode_output_dir(cfg.single)
    manifest_path = _mode_manifest_path(cfg.single)
    output_dir.mkdir(parents=True, exist_ok=True)

    endpoint = select_resolvable_endpoint(
        primary=cfg.endpoint.single,
        fallbacks=cfg.endpoint.single_fallbacks,
    )

    logger.info("[single] Endpoint: %s", endpoint)
    logger.info("[single] Output directory: %s", rel_path(output_dir))
    logger.info("[single] Manifest path: %s", rel_path(manifest_path))

    jobs = _expand_jobs(cfg.single.jobs)

    # Start conservatively. Increase later if stable.
    max_workers = max(1, int(cfg.runtime.max_workers))

    logger.info(
        "[single] Starting %d job(s) with max_workers=%d",
        len(jobs),
        max_workers,
    )

    manifest_rows: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_job = {
            executor.submit(
                _run_single_job_safe,
                cfg=cfg,
                endpoint=endpoint,
                output_dir=output_dir,
                job=job,
            ): job
            for job in jobs
        }

        for idx, future in enumerate(as_completed(future_to_job), start=1):
            job = future_to_job[future]
            logger.info(
                "[single] Completed future %d/%d: %s",
                idx,
                len(jobs),
                job.name,
            )

            row = future.result()
            manifest_rows.append(row)

            # Write incremental manifest after every completed job.
            # Useful if the run is interrupted.
            _write_manifest(manifest_rows, manifest_path)

    downloaded = [row for row in manifest_rows if row.get("status") == "downloaded"]
    skipped = [row for row in manifest_rows if row.get("status") == "skipped"]
    failed = [row for row in manifest_rows if row.get("status") == "error"]

    logger.info(
        "[single] Finished: downloaded=%d skipped=%d failed=%d total=%d",
        len(downloaded),
        len(skipped),
        len(failed),
        len(manifest_rows),
    )

    _write_manifest(manifest_rows, manifest_path)
    return manifest_rows

def run_single_downloads_deprecated(cfg: ComtradeConfig) -> list[dict[str, Any]]:
    """
    Run normal Comtrade Data API downloads.
    """
    if not cfg.single.enabled:
        logger.info("[single] Single API mode is disabled.")
        return []

    output_dir = _mode_output_dir(cfg.single)
    manifest_path = _mode_manifest_path(cfg.single)
    output_dir.mkdir(parents=True, exist_ok=True)

    endpoint = select_resolvable_endpoint(
        primary=cfg.endpoint.single,
        fallbacks=cfg.endpoint.single_fallbacks,
    )

    logger.info("[single] Endpoint: %s", endpoint)
    logger.info("[single] Output directory: %s", rel_path(output_dir))
    logger.info("[single] Manifest path: %s", rel_path(manifest_path))

    manifest_rows: list[dict[str, Any]] = []
    jobs = _expand_jobs(cfg.single.jobs)

    for idx, job in enumerate(jobs, start=1):
        logger.info("[single] Job %d/%d START: %s", idx, len(jobs), job.name)

        try:
            output_file = _format_output_file(job.output_file, job.params)
            output_path = output_dir / output_file

            if output_path.exists() and not cfg.runtime.overwrite_existing:
                logger.info(
                    "[single] Skipping existing file because overwrite_existing=false: %s",
                    rel_path(output_path),
                )

                manifest_rows.append(
                    {
                        "job": job.name,
                        "mode": "single",
                        "status": "skipped",
                        "output_path": str(output_path),
                        "url": None,
                        "params": job.params,
                    }
                )
                continue

            row = _download_single_job(
                cfg=cfg,
                endpoint=endpoint,
                job=job,
                output_path=output_path,
            )

            manifest_rows.append(row)

            logger.info(
                "[single] Job %s completed. Rows downloaded: %s",
                job.name,
                row.get("rows_downloaded", "unknown"),
            )

        except Exception as exc:
            logger.exception("[single] Job failed: %s", job.name)

            manifest_rows.append(
                {
                    "job": job.name,
                    "mode": "single",
                    "status": "error",
                    "error": str(exc),
                    "params": job.params,
                }
            )

    _write_manifest(manifest_rows, manifest_path)
    return manifest_rows


# ---------------------------------------------------------------------
# Public orchestration function
# ---------------------------------------------------------------------


def run_comtrade_download_pipeline(
    config_path: Path | None = None,
) -> list[dict[str, Any]]:
    """
    Main Comtrade download pipeline.

    This is the main function your script should call.

    It can run:
    - bulk only
    - single only
    - both bulk and single

    depending on comtrade.yaml.
    """
    start_time = time.perf_counter()

    project_root = get_project_root()
    resolved_config_path = config_path or project_root / "configs" / "comtrade.yaml"

    _log_step(1, 5, "Load Comtrade config")
    logger.info("Using Comtrade config: %s", rel_path(resolved_config_path))
    cfg = load_comtrade_config(resolved_config_path)

    _log_step(2, 5, "Validate authentication")
    validate_auth_preflight(cfg.auth)

    all_manifest_rows: list[dict[str, Any]] = []

    _log_step(3, 5, "Run bulk downloads")
    bulk_rows = run_bulk_downloads(cfg)
    all_manifest_rows.extend(bulk_rows)

    _log_step(4, 5, "Run single API downloads")
    single_rows = run_single_downloads(cfg)
    all_manifest_rows.extend(single_rows)

    _log_step(5, 5, "Finish")
    elapsed = time.perf_counter() - start_time
    mins, secs = divmod(elapsed, 60)

    failed = [row for row in all_manifest_rows if row.get("status") == "error"]
    downloaded = [row for row in all_manifest_rows if row.get("status") == "downloaded"]
    skipped = [row for row in all_manifest_rows if row.get("status") == "skipped"]

    logger.info(
        "Comtrade download pipeline finished in %.2f seconds (%.0f min %.1f sec).",
        elapsed,
        mins,
        secs,
    )
    logger.info(
        "Summary: downloaded=%d skipped=%d failed=%d total=%d",
        len(downloaded),
        len(skipped),
        len(failed),
        len(all_manifest_rows),
    )

    if failed:
        failed_summary = "; ".join(
            f"{row.get('job', 'unknown')}: {row.get('error', 'unknown error')}"
            for row in failed
        )
        logger.warning("Some Comtrade jobs failed: %s", failed_summary)

    return all_manifest_rows


# Optional backward-compatible alias while you migrate old imports.
def run_comtrade_api_pipeline(
    paths_config_path: Path | None = None,
    api_config_path: Path | None = None,
) -> list[dict[str, Any]]:
    """
    Temporary compatibility wrapper.

    Old code may call:

        run_comtrade_api_pipeline(paths_config_path=..., api_config_path=...)

    The new pipeline only needs the Comtrade config path.
    """
    return run_comtrade_download_pipeline(config_path=api_config_path)




def _run_single_job_safe(
    *,
    cfg: ComtradeConfig,
    endpoint: str,
    output_dir: Path,
    job: JobConfig,
) -> dict[str, Any]:
    """
    Wrapper around one single API job.

    Returns a manifest row whether the job succeeds, is skipped, or fails.
    This makes it safe to use inside ThreadPoolExecutor.
    """
    try:
        output_file = _format_output_file(job.output_file, job.params)
        output_path = output_dir / output_file

        if output_path.exists() and not cfg.runtime.overwrite_existing:
            logger.info(
                "[single] Skipping existing file because overwrite_existing=false: %s",
                rel_path(output_path),
            )

            return {
                "job": job.name,
                "mode": "single",
                "status": "skipped",
                "output_path": str(output_path),
                "url": None,
                "params": job.params,
            }

        row = _download_single_job(
            cfg=cfg,
            endpoint=endpoint,
            job=job,
            output_path=output_path,
        )

        logger.info(
            "[single] Job %s completed. Rows downloaded: %s",
            job.name,
            row.get("rows_downloaded", "unknown"),
        )

        return row

    except Exception as exc:
        logger.exception("[single] Job failed: %s", job.name)

        return {
            "job": job.name,
            "mode": "single",
            "status": "error",
            "error": str(exc),
            "params": job.params,
        }