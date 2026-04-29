"""
vacoda.comtrade.api_pipeline
Comtrade API pipeline runner
"""

import json
from pathlib import Path
from typing import Any
from vacoda.io.paths import get_project_root, load_paths_config, resolve_path
from vacoda.utils.logger import get_logger, rel_path
from vacoda.comtrade.utils import _resolve_output_dir
from vacoda.comtrade.download import load_comtrade_config
from vacoda.comtrade.api import select_resolvable_endpoint, validate_auth_preflight, build_query_params, build_auth_headers, is_csv_format, download_paginated_csv, download_with_retry, detect_auth_error_in_payload, redact_url

logger = get_logger("COMTRADE.API_PIPELINE")

def run_comtrade_api_pipeline(
    paths_config_path: Path | None = None,
    api_config_path: Path | None = None,
) -> list[dict[str, Any]]:
    import time
    start_time = time.perf_counter()

    def log_step(n: int, total: int, label: str, detail: str = None):
        logger.info("=" * 60)
        logger.info(f"STEP {n}/{total} — {label}")
        logger.info("=" * 60)
        if detail:
            logger.info(f"  [detail] {detail}")

    def log_query_start(idx: int, total: int, name: str, output_path: Path):
        logger.info(f"  [query {idx}/{total}] START: {name}")
        logger.info(f"    Output file: {output_path}")

    def log_query_result(status: str, output_path: Path, extra: str = None):
        logger.info(f"    [result] {status.upper()}: {output_path}")
        if extra:
            logger.info(f"      {extra}")

    steps = [
        "Load config and paths",
        "Prepare output directory",
        "Validate authentication",
        "Download queries",
        "Write manifest",
        "Finish",
    ]
    total_steps = len(steps)
    step = 1

    #################################################################### Load config and paths
    project_root = get_project_root()
    resolved_paths_cfg = paths_config_path or (project_root / "configs" / "paths.yaml")
    resolved_api_cfg = api_config_path or (project_root / "configs" / "comtrade.yaml")

    logger.info("Starting comtrade API pipeline with config: %s", rel_path(resolved_api_cfg))
    paths_cfg = load_paths_config(resolved_paths_cfg)
    cfg = load_comtrade_config(resolved_api_cfg)
    resolved_endpoint = select_resolvable_endpoint(cfg.endpoint, cfg.endpoint_fallbacks)

    logger.debug("Resolved endpoint for multi-hs mode: %s", resolved_endpoint)
    logger.info(f"Using API config: {resolved_api_cfg}")

    
    #################################################################### Prepare output directory
    step += 1
    log_step(step, total_steps, steps[step-1], f"Output directory: {output_dir if 'output_dir' in locals() else '(not resolved yet)'}")
    output_dir = _resolve_output_dir(paths_cfg, cfg.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")


    #################################################################### Validate authentication
    step += 1
    log_step(step, total_steps, steps[step-1], f"Auth method: {cfg.auth.method}, Env var: {cfg.auth.env_var_name}")
    validate_auth_preflight(cfg.auth)

    # Show important config parameters at the start
    print("\n[Comtrade API Pipeline] Configuration summary:")
    print(f"  Endpoint: {cfg.endpoint}")
    print(f"  Output directory: {output_dir if 'output_dir' in locals() else cfg.output.directory}")
    print(f"  Number of queries: {len(cfg.queries)}")
    print(f"  Auth method: {cfg.auth.method}")
    print(f"  Auth env file: {getattr(cfg.auth, 'env_file', None)}")
    print(f"  Log level: {getattr(cfg, 'log_level', 'INFO')}")
    print(f"  Manifest file: {cfg.output.manifest_file}")
    print("")
    for idx, query_item in enumerate(cfg.queries, 1):
        params = query_item.get("params", {})
        print(f"Query {idx}:")
        print(f"  reporterCode: {params.get('reporterCode', '')}")
        print(f"  partnerCode: {params.get('partnerCode', '')}")
        print(f"  cmdCode: {params.get('cmdCode', '')}")
        print(f"  flowCode: {params.get('flowCode', '')}")
        print(f"  period: {params.get('period', '')}")
        print(f"  output_file: {query_item.get('output_file', '')}")
        print("")
    response = input("Proceed with these settings? [y/N]: ").strip().lower()
    if response not in ("y", "yes"):
        logger.error("Comtrade API pipeline aborted by user after configuration prompt.")
        raise SystemExit("Pipeline stopped by user.")
    


    #################################################################### Download queries
    step += 1
    log_step(step, total_steps, steps[step-1], f"Total queries: {len(cfg.queries)}")
    manifest_rows: list[dict[str, Any]] = []

    for idx, query_item in enumerate(cfg.queries, 1):
        query_name = str(query_item["name"])
        output_path = output_dir / str(query_item["output_file"])
        logger.debug("Processing query=%s output=%s", query_name, output_path)

        log_query_start(idx, len(cfg.queries), query_name, output_path)

        if output_path.exists() and not cfg.runtime.overwrite_existing:
            log_query_result("skipped", rel_path(output_path), "File exists and overwrite is disabled.")
            manifest_rows.append({"query": query_name, "status": "skipped", "output_path": str(output_path), "url": None})
            continue
        elif not output_path.exists():
            logger.info(f"    [info] File does not exist, will download.")

        params = build_query_params(cfg.default_params, query_item["params"], cfg.auth)
        headers = build_auth_headers(cfg.auth)
        logger.debug("Query=%s params keys=%s", query_name, sorted(params.keys()))
        logger.info(f"    [info] Downloading query '{query_name}'")

        urls_used: list[str]
        rows_downloaded: int | None = None
        try:
            if cfg.pagination.enabled:
                if not is_csv_format(params):
                    raise ValueError("Pagination supports CSV only. Set default_params.format='csv'.")
                payload, urls_used, rows_downloaded = download_paginated_csv(
                    endpoint=resolved_endpoint,
                    params=params,
                    retry=cfg.retry,
                    runtime=cfg.runtime,
                    pagination=cfg.pagination,
                    headers=headers,
                )
            else:
                from urllib.parse import urlencode
                url = f"{resolved_endpoint}?{urlencode(params, doseq=True)}"
                payload = download_with_retry(url, cfg.retry, cfg.runtime, headers=headers)
                auth_error = detect_auth_error_in_payload(payload)
                if auth_error:
                    raise PermissionError(f"API returned authentication/authorization payload: {auth_error}")
                urls_used = [redact_url(url)]

            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(payload)
            log_query_result("downloaded", rel_path(output_path), f"Rows downloaded: {rows_downloaded if rows_downloaded is not None else 'unknown'}")
            manifest_rows.append(
                {
                    "query": query_name,
                    "status": "downloaded",
                    "output_path": str(output_path),
                    "url": urls_used[0] if len(urls_used) == 1 else None,
                    "url_count": len(urls_used),
                    "rows_downloaded": rows_downloaded,
                }
            )
        except Exception as e:
            log_query_result("error", rel_path(output_path), f"Error: {e}")
            manifest_rows.append({"query": query_name, "status": "error", "output_path": str(output_path), "error": str(e)})

    #################################################################### Write manifest
    step += 1
    log_step(step, total_steps, steps[step-1], f"Manifest file: {rel_path(cfg.output.manifest_file)}")
    manifest_path = output_dir / cfg.output.manifest_file
    manifest_path.write_text(json.dumps(manifest_rows, indent=2), encoding="utf-8")
    logger.info("Download manifest written to: %s", rel_path(manifest_path))


    #################################################################### Finish pipeline
    step += 1
    log_step(step, total_steps, steps[step-1], "Pipeline finished successfully.")
    elapsed = time.perf_counter() - start_time
    mins, secs = divmod(elapsed, 60)
    logger.info(
        "Comtrade API pipeline completed in %.2f seconds (%.0f min %.1f sec)",
        elapsed, mins, secs
    )
    # Summary of saved files
    saved_files = [row["output_path"] for row in manifest_rows if row["status"] == "downloaded"]
    if saved_files:
        logger.info(f"Downloaded data files saved to: {output_dir}")
        for f in saved_files:
            logger.info(f"  - {f}")
        logger.info("=" * 60)
    return manifest_rows

# def run_download_pipeline(
#     paths_config_path: Path | None = None,
#     download_config_path: Path | None = None,
# ) -> list[dict[str, Any]]:
#     """Backward-compatible alias for the comtrade API pipeline."""
#     return run_comtrade_api_pipeline(
#         paths_config_path=paths_config_path,
#         download_config_path=download_config_path,
#     )

