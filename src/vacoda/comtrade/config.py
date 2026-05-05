from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vacoda.comtrade.api import (
    AuthConfig,
    DownloadRuntimeConfig,
    PaginationConfig,
    RetryConfig,
)
from vacoda.comtrade.utils import load_yaml
from vacoda.reference import flow_sets, hs_sets
from vacoda.utils.logger import get_logger

logger = get_logger("COMTRADE.CONFIG")


@dataclass
class EndpointConfig:
    single: str
    single_fallbacks: list[str]
    bulk: str


@dataclass
class OutputConfig:
    directory: str
    manifest_file: str


@dataclass
class JobConfig:
    name: str
    params: dict[str, Any]
    output_file: str
    local_filter: dict[str, Any] | None = None
    all_years_start: int | None = None
    all_years_end: int | None = None
    use_reference_years: bool = False
    fallback_years_start: int = 1990
    fallback_years_end: int = 2025


@dataclass
class ModeConfig:
    enabled: bool
    jobs: list[JobConfig]
    output: OutputConfig


@dataclass
class ComtradeConfig:
    endpoint: EndpointConfig
    default_params: dict[str, Any]
    single: ModeConfig
    bulk: ModeConfig
    retry: RetryConfig
    runtime: DownloadRuntimeConfig
    auth: AuthConfig
    pagination: PaginationConfig
    log_level: str
    log_file: str | None


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


def _resolve_params(params: dict[str, Any]) -> dict[str, Any]:
    resolved: dict[str, Any] = {}

    for key, value in params.items():
        resolved[key] = _resolve_reference_token(value)

    return resolved


def _load_jobs(section_name: str, section_cfg: dict[str, Any]) -> list[JobConfig]:
    raw_jobs = section_cfg.get("jobs", [])

    if not isinstance(raw_jobs, list):
        raise ValueError(f"{section_name}.jobs must be a list")

    jobs: list[JobConfig] = []

    for idx, raw_job in enumerate(raw_jobs):
        if not isinstance(raw_job, dict):
            raise ValueError(f"{section_name}.jobs[{idx}] must be a mapping")

        name = str(raw_job.get("name", "")).strip()
        if not name:
            raise ValueError(f"{section_name}.jobs[{idx}].name is required")

        params = raw_job.get("params", {})
        if not isinstance(params, dict):
            raise ValueError(f"{section_name}.jobs[{idx}].params must be a mapping")

        output_file = str(raw_job.get("output_file", "")).strip()
        if not output_file:
            raise ValueError(f"{section_name}.jobs[{idx}].output_file is required")

        local_filter = raw_job.get("local_filter")
        if local_filter is not None and not isinstance(local_filter, dict):
            raise ValueError(f"{section_name}.jobs[{idx}].local_filter must be a mapping")

        jobs.append(
        JobConfig(
            name=name,
            params=_resolve_params(params),
            output_file=output_file,
            local_filter=_resolve_params(local_filter) if isinstance(local_filter, dict) else None,
            all_years_start=raw_job.get("all_years_start"),
            all_years_end=raw_job.get("all_years_end"),
            use_reference_years=bool(raw_job.get("use_reference_years", False)),
            fallback_years_start=int(raw_job.get("fallback_years_start", 1990)),
            fallback_years_end=int(raw_job.get("fallback_years_end", 2025)),
    )
)

    return jobs


def _load_output(section_cfg: dict[str, Any], default_dir: str, default_manifest: str) -> OutputConfig:
    output_cfg = section_cfg.get("output", {})
    if not isinstance(output_cfg, dict):
        output_cfg = {}

    return OutputConfig(
        directory=str(output_cfg.get("directory", default_dir)),
        manifest_file=str(output_cfg.get("manifest_file", default_manifest)),
    )


def _load_mode(
    section_name: str,
    cfg: dict[str, Any],
    default_dir: str,
    default_manifest: str,
) -> ModeConfig:
    section_cfg = cfg.get(section_name, {})
    if not isinstance(section_cfg, dict):
        section_cfg = {}

    return ModeConfig(
        enabled=bool(section_cfg.get("enabled", False)),
        jobs=_load_jobs(section_name, section_cfg),
        output=_load_output(section_cfg, default_dir, default_manifest),
    )


def load_comtrade_config(path: Path) -> ComtradeConfig:
    raw = load_yaml(path)

    endpoint_cfg = raw.get("endpoint", {})
    if not isinstance(endpoint_cfg, dict):
        raise ValueError("endpoint must be a mapping with single, single_fallbacks, bulk")

    default_params = raw.get("default_params", {})
    if not isinstance(default_params, dict):
        raise ValueError("default_params must be a mapping")

    retry_cfg = raw.get("retry", {})
    runtime_cfg = raw.get("runtime", {})
    auth_cfg = raw.get("auth", {})
    pagination_cfg = raw.get("pagination", {})

    if not isinstance(retry_cfg, dict):
        retry_cfg = {}
    if not isinstance(runtime_cfg, dict):
        runtime_cfg = {}
    if not isinstance(auth_cfg, dict):
        auth_cfg = {}
    if not isinstance(pagination_cfg, dict):
        pagination_cfg = {}

    cfg = ComtradeConfig(
        endpoint=EndpointConfig(
            single=str(endpoint_cfg.get("single", "")).strip(),
            single_fallbacks=[
                str(item).strip()
                for item in endpoint_cfg.get("single_fallbacks", [])
                if str(item).strip()
            ],
            bulk=str(endpoint_cfg.get("bulk", "")).strip(),
        ),
        default_params=default_params,
        single=_load_mode(
            "single",
            raw,
            default_dir="data/raw/api/single",
            default_manifest="download_manifest_single.json",
        ),
        bulk=_load_mode(
            "bulk",
            raw,
            default_dir="data/raw/api/bulk",
            default_manifest="download_manifest_bulk.json",
        ),
        retry=RetryConfig(
            max_attempts=int(retry_cfg.get("max_attempts", retry_cfg.get("retries", 3))),
            backoff_seconds=float(retry_cfg.get("backoff_seconds", 2.0)),
        ),
        runtime=DownloadRuntimeConfig(
            timeout_seconds=int(runtime_cfg.get("timeout_seconds", 120)),
            overwrite_existing=bool(runtime_cfg.get("overwrite_existing", False)),
            max_workers=int(runtime_cfg.get("max_workers", 4)),

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
        log_level=str(raw.get("log_level", "INFO")),
        log_file=raw.get("log_file"),
    )

    if cfg.single.enabled and not cfg.endpoint.single:
        raise ValueError("single.enabled=true but endpoint.single is missing")

    if cfg.bulk.enabled and not cfg.endpoint.bulk:
        raise ValueError("bulk.enabled=true but endpoint.bulk is missing")

    return cfg