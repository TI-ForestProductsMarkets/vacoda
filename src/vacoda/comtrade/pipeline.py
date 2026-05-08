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
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from click import confirm

from vacoda.comtrade.api import (
    build_auth_headers,
    build_query_params,
    detect_auth_error_in_payload,
    download_bulk,
    download_paginated_csv,
    download_with_retry,
    is_csv_format,
    redact_url,
    select_resolvable_endpoint,
    validate_auth_preflight,
    RateLimitExceededError,
)
from vacoda.comtrade.config import ComtradeConfig, JobConfig, ModeConfig, load_comtrade_config
from vacoda.io.paths import get_project_root, resolve_path
from vacoda.utils.logger import get_logger, rel_path
from vacoda.reference.hs_sets import get_hs_valid_year_range
from vacoda.comtrade.manifest import (
    audit_manifest_file,
    find_invalid_manifest_rows,
    load_manifest,
    quarantine_invalid_outputs,
    write_manifest,
    validate_output_file
)
from vacoda.utils.progress import progress_bar
## Set up a module-level logger
logger = get_logger("COMTRADE.PIPELINE")


# ---------------------------------------------------------------------
# Small generic helpers
# ---------------------------------------------------------------------

def _mode_output_dir(mode_cfg: ModeConfig) -> Path:
    return resolve_path(mode_cfg.output.directory)


def _mode_manifest_path(mode_cfg: ModeConfig) -> Path:
    return _mode_output_dir(mode_cfg) / mode_cfg.output.manifest_file

def _mode_quarantine_dir(mode_cfg: ModeConfig) -> Path:
    if mode_cfg.output.quarantine_directory:
        return resolve_path(mode_cfg.output.quarantine_directory)

    return _mode_output_dir(mode_cfg) / "_quarantine_invalid"

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


def _short_error(error: str | None, max_chars: int = 120) -> str:
    if not error:
        return "<no error message>"
    return error if len(error) <= max_chars else error[:max_chars] + "..."

def _handle_existing_output(
    *,
    output_path: Path,
    cfg: ComtradeConfig,
    mode: str,
    job: JobConfig,) -> dict[str, Any] | None:
    if not output_path.exists():
        return None

    if cfg.runtime.overwrite_existing:
        logger.info("[%s] Existing file will be overwritten: %s", mode, rel_path(output_path))
        return None

    valid, reason, size = validate_output_file(output_path)

    if valid:
        logger.info(
            "[%s] Skipping valid existing file: %s",
            mode,
            rel_path(output_path),
        )
        return {
            "job": job.name,
            "mode": mode,
            "status": "skipped_existing",
            "output_path": str(output_path),
            "file_size_bytes": size,
            "validation_reason": reason,
            "params": job.params,
        }

    policy = cfg.runtime.existing_file_policy

    logger.warning(
        "[%s] Existing output is invalid: %s (%s, %d bytes). Policy=%s",
        mode,
        rel_path(output_path),
        reason,
        size,
        policy,
    )

    if policy == "redownload":
        if reason == "zero_byte":
            output_path.unlink(missing_ok=True)
        return None

    if policy == "fail":
        raise RuntimeError(f"Invalid existing file: {output_path} ({reason})")

    if policy == "warn":
        return {
            "job": job.name,
            "mode": mode,
            "status": "invalid_existing",
            "output_path": str(output_path),
            "file_size_bytes": size,
            "validation_reason": reason,
            "params": job.params,
        }

    if policy == "skip":
        return {
            "job": job.name,
            "mode": mode,
            "status": "skipped_invalid_existing",
            "output_path": str(output_path),
            "file_size_bytes": size,
            "validation_reason": reason,
            "params": job.params,
        }

    raise ValueError(
    f"Unknown existing_file_policy={policy!r}. "
    "Expected one of: skip, redownload, fail, warn."
)


def _as_list(value: Any) -> list[Any]:
    """
    Normalize scalar/list config values to a list.

    Example:
        "441012" -> ["441012"]
        ["441012", "441011"] -> ["441012", "441011"]
    """
    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    if value is None:
        return [None]

    return [value]

def _expand_cmd_code_jobs(job: JobConfig) -> list[JobConfig]:
    """
    Expand jobs where cmdCode is a list.

    Even a one-item list like ["441012"] is converted to scalar "441012",
    because downstream year expansion expects one cmdCode per job.
    """
    if "cmdCode" not in job.params:
        return [job]

    cmd_codes = _as_list(job.params.get("cmdCode"))

    expanded_jobs: list[JobConfig] = []

    for cmd_code in cmd_codes:
        if cmd_code is None or str(cmd_code).strip() == "":
            params = dict(job.params)
            params.pop("cmdCode", None)
            name = job.name
        else:
            params = dict(job.params)
            params["cmdCode"] = str(cmd_code).strip()
            name = f"{job.name}_{params['cmdCode']}"

        expanded_jobs.append(
            JobConfig(
                name=name,
                params=params,
                output_file=job.output_file,
                local_filter=job.local_filter,
                all_years_start=job.all_years_start,
                all_years_end=job.all_years_end,
                use_reference_years=job.use_reference_years,
                fallback_years_start=job.fallback_years_start,
                fallback_years_end=job.fallback_years_end,
            )
        )

    return expanded_jobs


def _get_year_range_for_job(job: JobConfig) -> tuple[int, int] | None:
    """
    Determine year range for a job.

    Priority:
    1. Explicit all_years_start/all_years_end
    2. Reference-driven HS validity years
    3. No expansion
    """
    if job.all_years_start is not None and job.all_years_end is not None:
        return int(job.all_years_start), int(job.all_years_end)

    if job.use_reference_years:
        cmd_code = job.params.get("cmdCode")

        if isinstance(cmd_code, list):
            if len(cmd_code) == 1:
                cmd_code = cmd_code[0]
            else:
                raise ValueError(
                "use_reference_years=True requires one cmdCode per expanded job. "
                f"Got multiple cmdCode values: {cmd_code}"
                )

        if cmd_code is None:
            raise ValueError("use_reference_years=True requires params.cmdCode.")

        return get_hs_valid_year_range(
            cmd_code=str(cmd_code),
            fallback_start=job.fallback_years_start,
            fallback_end=job.fallback_years_end,
        )

    return None


def _expand_year_jobs(job: JobConfig) -> list[JobConfig]:
    """
    Expand a job into one job per year.

    Supports either:
    - explicit all_years_start/all_years_end
    - use_reference_years=True based on cmdCode

    If no year range is available, return the original job unchanged.
    If year range is invalid, log one clean error and skip the job.
    """
    year_range = _get_year_range_for_job(job)

    if year_range is None:
        return [job]

    if not isinstance(year_range, (list, tuple)) or len(year_range) != 2:
        logger.error(
            "Invalid year_range for job %r: expected [start_year, end_year], "
            "got %r. Skipping this job.",
            job.name,
            year_range,
        )
        return []

    start, end = year_range

    if not isinstance(start, int) or not isinstance(end, int):
        logger.error(
            "Invalid year_range for job %r: start and end must be integers, "
            "got %r. Skipping this job.",
            job.name,
            year_range,
        )
        return []

    if start > end:
        logger.error(
            "Invalid year_range for job %r: start_year %s is greater than end_year %s. "
            "Skipping this job.",
            job.name,
            start,
            end,
        )
        return []

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
                use_reference_years=False,
                fallback_years_start=job.fallback_years_start,
                fallback_years_end=job.fallback_years_end,
            )
        )

    return expanded_jobs

def _expand_jobs(jobs: list[JobConfig]) -> list[JobConfig]:
    """
    Expand jobs in this order:

    1. Expand cmdCode lists into one job per commodity code.
    2. Expand each commodity job into one job per valid year.

    This makes filenames like {cmdCode}_{period} safe and meaningful.
    """
    expanded: list[JobConfig] = []

    for job in jobs:
        cmd_jobs = _expand_cmd_code_jobs(job)

        for cmd_job in cmd_jobs:
            expanded.extend(_expand_year_jobs(cmd_job))

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
                if output_path.stat().st_size > 0:
                    status = "skipped_existing"
                    logger.info("[bulk] Skipping existing file because overwrite_existing=false: %s", rel_path(output_path),)
                else:
                    logger.warning("Existing output is zero-byte; re-downloading: %s", output_path)


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
            bulk_index_path = output_path.with_suffix(".json")

            manifest_rows.append(
                {
                    "job": job.name,
                    "mode": "bulk",
                    "status": "downloaded",
                    "bulk_index_path": str(bulk_index_path),
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

   
    write_manifest(manifest_rows, manifest_path)
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
            raise PermissionError(
                f"API returned authentication/authorization payload: {auth_error}"
            )

        urls_used = [redact_url(url)]

    if rows_downloaded == 0 or not payload:
        return {
            "job": job.name,
            "mode": "single",
            "status": "no_data",
            "output_path": None,
            "url": urls_used[0] if len(urls_used) == 1 else None,
            "url_count": len(urls_used),
            "rows_downloaded": 0,
            "params": params,
            "validation_reason": "api_returned_no_rows",
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    try:
        tmp_path.write_bytes(payload)

        valid, reason, size = validate_output_file(tmp_path)
        if not valid:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(f"Downloaded file failed validation: {reason}")

        tmp_path.replace(output_path)

    finally:
        tmp_path.unlink(missing_ok=True)

    return {
        "job": job.name,
        "mode": "single",
        "status": "downloaded",
        "output_path": str(output_path),
        "file_size_bytes": output_path.stat().st_size,
        "validation_reason": "valid",
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

        
        for idx, future in enumerate(progress_bar(as_completed(future_to_job), total=len(future_to_job), desc="[single] Comtrade jobs", unit="job", enabled=cfg.runtime.show_progress), start=1):
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
            write_manifest(manifest_rows, manifest_path)

    downloaded = [row for row in manifest_rows if row.get("status") == "downloaded"]
    skipped = [row for row in manifest_rows if row.get("status") in {"skipped", "skipped_existing", "skipped_invalid_existing"}]
    failed = [row for row in manifest_rows if row.get("status") == "error"]

    logger.info(
        "[single] Finished: downloaded=%d skipped=%d failed=%d total=%d",
        len(downloaded),
        len(skipped),
        len(failed),
        len(manifest_rows),
    )

    # Write incremental manifest every few jobs.
    write_manifest(manifest_rows, manifest_path)
    return manifest_rows

# ---------------------------------------------------------------------
# Public orchestration function
# ---------------------------------------------------------------------
def _format_bool(value: bool) -> str:
    return "yes" if value else "no"


def _summarize_job_years(job: JobConfig) -> str:
    if job.all_years_start is not None and job.all_years_end is not None:
        return f"{job.all_years_start}–{job.all_years_end}"

    if job.use_reference_years:
        return "reference-driven"

    period = job.params.get("period")
    if period:
        return str(period)

    return "not specified"


def _summarize_local_filter(local_filter: dict[str, Any] | None) -> str:
    if not local_filter:
        return "none"

    parts = []
    for key, value in local_filter.items():
        if isinstance(value, list):
            preview = value[:5]
            suffix = "..." if len(value) > 5 else ""
            parts.append(f"{key}={preview}{suffix}")
        else:
            parts.append(f"{key}={value}")

    return "; ".join(parts)


def log_run_configuration(
    *,
    cfg: ComtradeConfig,
    config_path: Path,
    log_path: Path | None,
) -> None:
    logger.info("=" * 72)
    logger.info("VACODA COMTRADE DOWNLOAD PIPELINE")
    logger.info("=" * 72)

    logger.info("Config file:        %s", rel_path(config_path))
    logger.info("Log file:           %s", rel_path(log_path) if log_path else "console only")
    logger.info("Log level:          %s", cfg.log_level)

    logger.info("")
    logger.info("Modes:")
    logger.info("  Bulk enabled:     %s", _format_bool(cfg.bulk.enabled))
    logger.info("  Single enabled:   %s", _format_bool(cfg.single.enabled))

    logger.info("")
    logger.info("Runtime:")
    logger.info("  max_workers:      %s", cfg.runtime.max_workers)
    logger.info("  timeout_seconds:  %s", cfg.runtime.timeout_seconds)
    logger.info("  overwrite:        %s", _format_bool(cfg.runtime.overwrite_existing))

    if cfg.bulk.enabled:
        logger.info("")
        logger.info("Bulk:")
        logger.info("  endpoint:         %s", cfg.endpoint.bulk)
        logger.info("  output directory: %s", rel_path(_mode_output_dir(cfg.bulk)))
        logger.info("  manifest file:    %s", cfg.bulk.output.manifest_file)
        logger.info("  jobs:             %d", len(cfg.bulk.jobs))

        for idx, job in enumerate(cfg.bulk.jobs, start=1):
            logger.info("    %d. %s", idx, job.name)
            logger.info("       years:        %s", _summarize_job_years(job))
            logger.info("       output_file:  %s", job.output_file)
            logger.info("       local_filter: %s", _summarize_local_filter(job.local_filter))

    if cfg.single.enabled:
        logger.info("")
        logger.info("Single API:")
        logger.info("  endpoint:         %s", cfg.endpoint.single)
        logger.info("  output directory: %s", rel_path(_mode_output_dir(cfg.single)))
        logger.info("  manifest file:    %s", cfg.single.output.manifest_file)
        logger.info("  jobs:             %d", len(cfg.single.jobs))

        expanded_jobs = _expand_jobs(cfg.single.jobs)
        logger.info("  expanded jobs:    %d", len(expanded_jobs))

        for idx, job in enumerate(cfg.single.jobs, start=1):
            logger.info("    %d. %s", idx, job.name)
            logger.info("       years:        %s", _summarize_job_years(job))
            logger.info("       output_file:  %s", job.output_file)
            logger.info("       params:       %s", job.params)

    logger.info("=" * 72)

def _confirm_run(confirm: bool) -> None:
    if not confirm:
        return

    response = input("Proceed with this Comtrade download configuration? [y/N]: ").strip().lower()

    if response not in {"y", "yes"}:
        logger.warning("Pipeline aborted by user before downloads started.")
        raise SystemExit("Pipeline stopped by user.")

def run_comtrade_download_pipeline(
    config_path: Path | None = None,
    log_path: Path | None = None,
    confirm: bool = True,
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

    total_step = 6

    pipeline_steps = [
        "Load and summarize configuration",
        "Validate authentication",
        "Run bulk downloads",
        "Run single API downloads",
        "Audit manifest",
        "Finish"
    ]

    ################## Load config 
    _log_step(1, total_step, pipeline_steps[0])
    logger.info("Using Comtrade config: %s", rel_path(resolved_config_path))
    cfg = load_comtrade_config(resolved_config_path)

    log_run_configuration(
        cfg=cfg,
        config_path=resolved_config_path,
        log_path=log_path,
        )
    _confirm_run(confirm)
    
    ################## Validate authentication
    _log_step(2, total_step, pipeline_steps[1])
    validate_auth_preflight(cfg.auth)

    all_manifest_rows: list[dict[str, Any]] = []


    ################# Run Bulk Downloads and Single API Downloads
    _log_step(3, total_step, pipeline_steps[2])
    bulk_rows = run_bulk_downloads(cfg)
    all_manifest_rows.extend(bulk_rows)

    _log_step(4, total_step, pipeline_steps[3])
    single_rows = run_single_downloads(cfg)
    all_manifest_rows.extend(single_rows)


    ################# Audit Manifest
    _log_step(5, total_step, pipeline_steps[4])

    invalid_rows_all: list[dict[str, Any]] = []

    if cfg.bulk.enabled:
        bulk_manifest_path = _mode_manifest_path(cfg.bulk)
        audit_manifest_file(bulk_manifest_path, fail_on_error=False)

        bulk_rows_for_audit = load_manifest(bulk_manifest_path)
        invalid_bulk_rows = find_invalid_manifest_rows(bulk_rows_for_audit)
        invalid_rows_all.extend(invalid_bulk_rows)

        if invalid_bulk_rows:
            logger.warning("Found %d invalid bulk manifest rows.", len(invalid_bulk_rows))

            if cfg.runtime.existing_file_policy == "redownload":
                moved = quarantine_invalid_outputs(
                invalid_bulk_rows,
                _mode_quarantine_dir(cfg.bulk),
            )
                logger.warning(
                    "Moved %d invalid bulk output files to quarantine: %s",
                    moved,
                    rel_path(_mode_quarantine_dir(cfg.bulk)),
                )

    if cfg.single.enabled:
        single_manifest_path = _mode_manifest_path(cfg.single)
        audit_manifest_file(single_manifest_path, fail_on_error=False)

        single_rows_for_audit = load_manifest(single_manifest_path)
        invalid_single_rows = find_invalid_manifest_rows(single_rows_for_audit)
        invalid_rows_all.extend(invalid_single_rows)

        if invalid_single_rows:
            logger.warning("Found %d invalid single manifest rows.", len(invalid_single_rows))

            if cfg.runtime.existing_file_policy == "redownload":
                moved = quarantine_invalid_outputs(
                    invalid_single_rows,
                    _mode_quarantine_dir(cfg.single),
                )
                logger.warning(
                    "Moved %d invalid single output files to quarantine: %s",
                    moved,
                    rel_path(_mode_quarantine_dir(cfg.single)),
                )

    if invalid_rows_all:
        if cfg.runtime.existing_file_policy == "fail":
            raise RuntimeError(f"Manifest audit failed: {len(invalid_rows_all)} invalid rows found.")

        if cfg.runtime.existing_file_policy == "warn":
            logger.warning("Continuing despite invalid manifest rows.")

        if cfg.runtime.existing_file_policy == "redownload":
            logger.warning("Invalid outputs were quarantined. Rerun the pipeline to re-download them.")

    ################ Finish and summarize
    _log_step(6, total_step, pipeline_steps[5])
    elapsed = time.perf_counter() - start_time
    mins, secs = divmod(elapsed, 60)
    
    ### Summarize manifest rows
    failed = [row for row in all_manifest_rows if row.get("status") in {"error", "failed"}]
    downloaded = [row for row in all_manifest_rows if row.get("status") == "downloaded"]
    skipped = [row for row in all_manifest_rows if row.get("status") in {"skipped", "skipped_existing"}]
    no_data = [row for row in all_manifest_rows if row.get("status") == "no_data"]
    invalid = [row for row in all_manifest_rows if row.get("status") in {"invalid_existing", "invalid_file"}]

    logger.info(
        "Comtrade download pipeline finished in %.2f seconds (%.0f min %.1f sec).",
        elapsed,
        mins,
        secs,
    )
    logger.info(
        "Summary: downloaded=%d skipped=%d no_data=%d invalid=%d failed=%d total=%d",
        len(downloaded),
        len(skipped),
        len(no_data),
        len(invalid),
        len(failed),
        len(all_manifest_rows),
        )

    if failed:
        logger.warning("Some Comtrade jobs failed: %d job(s)", len(failed))

        for row in failed[:20]:
            logger.warning(
                "  - %s | type=%s | %s",
                row.get("job"),
                row.get("error_type", "unknown"),
                _short_error(row.get("error")),
                )

        if len(failed) > 20:
            logger.warning("  ... and %d more failed job(s)", len(failed) - 20)

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
    output_path: Path | None = None

    try:
        output_file = _format_output_file(job.output_file, job.params)
        output_path = output_dir / output_file

        existing_row = _handle_existing_output(
            output_path=output_path,
            cfg=cfg,
            mode="single",
            job=job,
        )

        if existing_row is not None:
            return existing_row

        row = _download_single_job(
            cfg=cfg,
            endpoint=endpoint,
            job=job,
            output_path=output_path,
        )

        logger.info(
            "[single] Job %s finished with status=%s. Rows downloaded: %s",
            job.name,
            row.get("status", "unknown"),
            row.get("rows_downloaded", "unknown"),
        )

        return row

    except RateLimitExceededError as exc:
        logger.warning("[single] Job rate-limited: %s | %s", job.name, exc)

        return {
            "job": job.name,
            "mode": "single",
            "status": "error",
            "output_path": str(output_path) if output_path else None,
            "error_type": "rate_limit",
            "error": str(exc),
            "params": job.params,
        }

    except Exception as exc:
        logger.error("[single] Job failed: %s | %s", job.name, exc)

        return {
            "job": job.name,
            "mode": "single",
            "status": "error",
            "output_path": str(output_path) if output_path else None,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "params": job.params,
        }