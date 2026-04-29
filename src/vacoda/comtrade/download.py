"""Config-driven data download pipeline for external APIs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import yaml
from vacoda.reference import flow_sets
from vacoda.reference import hs_sets

from vacoda.comtrade.api import (
    AuthConfig,
    DownloadRuntimeConfig,
    PaginationConfig,
    RetryConfig,
    build_auth_headers,
    build_query_params,
    detect_auth_error_in_payload,
    download_paginated_csv,
    download_with_retry,
    is_csv_format,
    redact_url,
    select_resolvable_endpoint,
    validate_auth_preflight,
)
from vacoda.io.paths import get_project_root, load_paths_config, resolve_path
from vacoda.utils.logger import get_logger, rel_path

logger = get_logger("COMTRADE.DOWNLOAD")


@dataclass
class OutputConfig:
    directory: str
    manifest_file: str


@dataclass
class DownloadConfig:
    endpoint: str
    endpoint_fallbacks: list[str]
    default_params: dict[str, Any]
    queries: list[dict[str, Any]]
    output: OutputConfig
    retry: RetryConfig
    runtime: DownloadRuntimeConfig
    auth: AuthConfig
    pagination: PaginationConfig


REQUIRED_QUERY_KEYS = {"name", "params", "output_file"}


def _resolve_reference_token(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    token = value.strip()
    if not token:
        return value

    resolved = getattr(hs_sets, token, None)
    if resolved is None:
        resolved = getattr(flow_sets, token, None)
    if resolved is None:
        return value

    if isinstance(resolved, tuple):
        return list(resolved)
    return resolved


def _get_query_items(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    queries = cfg.get("queries")
    if queries is None:
        queries = cfg.get("query")
    if queries is None:
        queries = cfg.get("jobs", [])

    if isinstance(queries, dict):
        return [queries]
    if isinstance(queries, list):
        return queries
    return []


def resolve_download_references(cfg: dict[str, Any]) -> dict[str, Any]:
    queries = _get_query_items(cfg)
    if not queries:
        return cfg

    resolved_queries: list[dict[str, Any]] = []
    for query_item in queries:
        if not isinstance(query_item, dict):
            resolved_queries.append(query_item)
            continue

        params = query_item.get("params")
        if not isinstance(params, dict):
            resolved_queries.append(query_item)
            continue

        resolved_params: dict[str, Any] = {}
        for key, value in params.items():
            resolved_value = _resolve_reference_token(value)
            if resolved_value is not value:
                logger.debug("Resolved config token %s=%s", key, value)
            resolved_params[key] = resolved_value

        resolved_query = dict(query_item)
        resolved_query["params"] = resolved_params
        resolved_queries.append(resolved_query)

    resolved_cfg = dict(cfg)
    resolved_cfg["queries"] = resolved_queries
    if "jobs" in resolved_cfg:
        resolved_cfg.pop("jobs", None)
    return resolved_cfg


def _load_yaml(path: Path) -> dict[str, Any]:
    logger.debug("Loading YAML config from %s", path)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at root of YAML: {path}")
    return data


def _validate_query(query_item: dict[str, Any], idx: int) -> None:
    missing = REQUIRED_QUERY_KEYS - set(query_item)
    if missing:
        raise ValueError(f"queries[{idx}] missing required keys: {', '.join(sorted(missing))}")
    if not isinstance(query_item["params"], dict):
        raise ValueError(f"queries[{idx}].params must be a mapping")


def load_comtrade_config(download_config_path: Path) -> DownloadConfig:
    """
    Load and validate the comtrade download configuration from a YAML file, resolving any references to predefined sets.
    """
    cfg = _load_yaml(download_config_path)
    scoped = cfg.get("download") if isinstance(cfg.get("download"), dict) else cfg
    if not isinstance(scoped, dict):
        raise ValueError("Expected mapping for download config")
    scoped = resolve_download_references(scoped)

    endpoint = str(scoped.get("endpoint", "")).strip()
    if not endpoint:
        raise ValueError("endpoint must be provided")

    endpoint_fallbacks = scoped.get("endpoint_fallbacks", [])
    if not isinstance(endpoint_fallbacks, list):
        raise ValueError("endpoint_fallbacks must be a list")

    default_params = scoped.get("default_params", {})
    if not isinstance(default_params, dict):
        raise ValueError("default_params must be a mapping")

    queries = _get_query_items(scoped)
    if not isinstance(queries, list) or not queries:
        raise ValueError("queries must be a non-empty list")
    for idx, query_item in enumerate(queries):
        if not isinstance(query_item, dict):
            raise ValueError(f"queries[{idx}] must be a mapping")
        _validate_query(query_item, idx)

    output_cfg     = scoped.get("output", {})
    retry_cfg      = scoped.get("retry", {})
    runtime_cfg    = scoped.get("runtime", {})
    auth_cfg       = scoped.get("auth", {})
    pagination_cfg = scoped.get("pagination", {})

    cfg_obj = DownloadConfig(
        endpoint=endpoint,
        endpoint_fallbacks=[str(e).strip() for e in endpoint_fallbacks if str(e).strip()],
        default_params=default_params,
        queries=queries,
        output=OutputConfig(
            directory=str(output_cfg.get("directory", "data/api")),
            manifest_file=str(output_cfg.get("manifest_file", "download_manifest.json")),
        ),
        retry=RetryConfig(
            max_attempts=int(retry_cfg.get("max_attempts", retry_cfg.get("retries", 3))),
            backoff_seconds=float(retry_cfg.get("backoff_seconds", 2.0)),
        ),
        runtime=DownloadRuntimeConfig(
            timeout_seconds=int(runtime_cfg.get("timeout_seconds", 120)),
            overwrite_existing=bool(runtime_cfg.get("overwrite_existing", False)),
        ),
        auth=AuthConfig(
            enabled=bool(auth_cfg.get("enabled", False)),
            method=str(auth_cfg.get("method", "query")).strip().lower(),
            header_name=str(auth_cfg.get("header_name", "Ocp-Apim-Subscription-Key")),
            env_var_name=str(auth_cfg.get("env_var_name", "COMTRADE_API_KEY")),
            query_param_name=str(auth_cfg.get("query_param_name", "token")),
            env_file=(str(auth_cfg.get("env_file", "")).strip() or None),
        ),
        pagination=PaginationConfig(
            enabled=bool(pagination_cfg.get("enabled", False)),
            limit_param=str(pagination_cfg.get("limit_param", "maxRecords")),
            offset_param=str(pagination_cfg.get("offset_param", "offset")),
            start_offset=int(pagination_cfg.get("start_offset", 0)),
        ),
    )
    logger.debug(
        "Loaded download config: endpoint=%s jobs=%d pagination=%s auth=%s",
        cfg_obj.endpoint,
        len(cfg_obj.queries),
        cfg_obj.pagination.enabled,
        cfg_obj.auth.enabled,
    )
    return cfg_obj


def run_comtrade_bulk_pipeline(
    paths_config_path: Path | None = None,
    download_config_path: Path | None = None,
) -> list[dict[str, Any]]:
    """
    Run the Comtrade Bulk API download pipeline using the bulk section of the config.
    """
    import time

    from vacoda.comtrade.api import download_bulk

    project_root = get_project_root()
    resolved_paths_cfg = paths_config_path or (project_root / "configs" / "paths.yaml")
    resolved_download_cfg = download_config_path or (project_root / "configs" / "comtrade.yaml")
    cfg = _load_yaml(resolved_download_cfg)
    bulk_cfg = cfg.get("bulk", {})
    if not bulk_cfg.get("enabled", False):
        logger.info("Bulk download is not enabled in config.")
        return []
    endpoint = bulk_cfg.get("endpoint", "https://comtradeapi.un.org/bulk/v1/get/")
    jobs = bulk_cfg.get("jobs", [])
    output_dir = bulk_cfg.get("output_directory") or cfg.get("output", {}).get("directory", "data/raw/api/bulk")
    manifest_file = bulk_cfg.get("manifest_file", "bulk_download_manifest.json")
    retry_cfg = cfg.get("retry", {})
    runtime_cfg = cfg.get("runtime", {})
    auth_cfg = cfg.get("auth", {})
    retry = RetryConfig(
        max_attempts=int(retry_cfg.get("max_attempts", retry_cfg.get("retries", 3))),
        backoff_seconds=float(retry_cfg.get("backoff_seconds", 2.0)),
    )
    runtime = DownloadRuntimeConfig(
        timeout_seconds=int(runtime_cfg.get("timeout_seconds", 120)),
        overwrite_existing=bool(runtime_cfg.get("overwrite_existing", False)),
    )
    auth = AuthConfig(
        enabled=bool(auth_cfg.get("enabled", False)),
        method=str(auth_cfg.get("method", "query")).strip().lower(),
        header_name=str(auth_cfg.get("header_name", "Ocp-Apim-Subscription-Key")),
        env_var_name=str(auth_cfg.get("env_var_name", "COMTRADE_API_KEY")),
        query_param_name=str(auth_cfg.get("query_param_name", "token")),
        env_file=(str(auth_cfg.get("env_file", "")).strip() or None),
    )
    output_dir = resolve_path(str(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("[bulk] Output directory: %s", output_dir)
    manifest_rows = []
    for job in jobs:
        name = job.get("name", "bulk_job")
        params = job.get("params", {})
        output_file = job.get("output_file", f"{name}.zip")
        output_path = output_dir / output_file
        logger.info(f"[bulk] Starting job: {name}")
        try:
            saved_path = download_bulk(endpoint, params, retry, runtime, auth, output_path)
            manifest_rows.append({"job": name, "status": "downloaded", "output_path": str(saved_path)})
        except Exception as e:
            logger.error(f"[bulk] Error in job {name}: {e}")
            manifest_rows.append({"job": name, "status": "error", "error": str(e)})
    manifest_path = output_dir / manifest_file
    manifest_path.write_text(json.dumps(manifest_rows, indent=2), encoding="utf-8")
    logger.info("[bulk] Bulk download manifest written to: %s", manifest_path)
    return manifest_rows





