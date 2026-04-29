"""
/vacoda/src/vacoda/comtrade/api.py
Core API interaction logic for downloading data from the comtrade API, including 
authentication handling, pagination, retries, and CSV merging.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import socket
from time import sleep
from typing import Any

from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from vacoda.io.paths import resolve_path
from vacoda.utils.logger import get_logger, rel_path

logger = get_logger("COMTRADE.API")

SENSITIVE_QUERY_KEYS = {"token", "apikey", "api_key", "subscription-key", "key"}


@dataclass
class RetryConfig:
    max_attempts: int
    backoff_seconds: float


@dataclass
class DownloadRuntimeConfig:
    timeout_seconds: int
    overwrite_existing: bool


@dataclass
class AuthConfig:
    enabled: bool
    method: str
    header_name: str
    env_var_name: str
    query_param_name: str
    env_file: str | None


@dataclass
class PaginationConfig:
    enabled: bool
    limit_param: str
    offset_param: str
    start_offset: int


def redact_url(url: str) -> str:
    split = urlsplit(url)
    if not split.query:
        return url

    redacted_pairs = []
    for key, value in parse_qsl(split.query, keep_blank_values=True):
        redacted_pairs.append((key, "***" if key.lower() in SENSITIVE_QUERY_KEYS else value))

    return urlunsplit((split.scheme, split.netloc, split.path, urlencode(redacted_pairs, doseq=True), split.fragment))


def select_resolvable_endpoint(primary: str, fallbacks: list[str]) -> str:
    candidates = [primary, *fallbacks]
    errors: list[str] = []

    for candidate in candidates:
        endpoint = str(candidate).strip()
        if not endpoint:
            continue

        split = urlsplit(endpoint)
        host = split.hostname
        if not host:
            errors.append(f"invalid endpoint: {endpoint}")
            continue

        try:
            socket.getaddrinfo(host, split.port or (443 if split.scheme == "https" else 80))
            if endpoint != primary:
                logger.warning("Primary endpoint not resolvable. Falling back to: %s", endpoint)
            logger.debug("Resolved endpoint host=%s endpoint=%s", host, endpoint)
            return endpoint
        except socket.gaierror as exc:
            errors.append(f"{host}: {exc}")

    raise ConnectionError(
        "No resolvable API endpoint found. Check DNS/VPN/proxy/firewall settings. "
        f"Tried: {'; '.join(errors)}"
    )


def load_env_from_file(env_file: str | None) -> None:
    if not env_file:
        return

    path = resolve_path(env_file)
    logger.debug("Loading env file from %s", path)
    if not path.exists():
        logger.debug("Env file does not exist: %s", path)
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            continue
        os.environ.setdefault(key, value.strip().strip('"').strip("'"))


def validate_auth_preflight(auth: AuthConfig) -> None:
    if not auth.enabled:
        logger.info("Auth is disabled. Running unauthenticated download mode.")
        return

    if auth.env_file:
        load_env_from_file(auth.env_file)
        env_path = resolve_path(auth.env_file)
        if env_path.exists():
            logger.info("Loaded local env file: %s", rel_path(env_path))
        else:
            logger.warning("Configured env_file not found: %s", rel_path(env_path))

    token = os.getenv(auth.env_var_name, "").strip()
    if not token:
        logger.error("Auth is enabled but token is missing. Set %s in OS env or env_file.", auth.env_var_name)
        raise ValueError(f"Missing token variable {auth.env_var_name}. Set it in env or .env file.")

    logger.info("Auth preflight passed using env var '%s' (method=%s).", auth.env_var_name, auth.method)


def build_query_params(default_params: dict[str, Any], job_params: dict[str, Any], auth: AuthConfig) -> dict[str, Any]:
    params = {**default_params, **job_params}

    def normalize_param_value(key: str, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (list, tuple, set)):
            parts = [str(v).strip() for v in value if str(v).strip()]
            if key == "flowCode" and "-1" in parts:
                return "-1"
            return ",".join(parts)
        if isinstance(value, str) and not value.strip():
            return None
        return value

    params = {key: normalize_param_value(key, value) for key, value in params.items()}
    params = {key: value for key, value in params.items() if value is not None}
    logger.debug("Building query params from defaults=%d job=%d", len(default_params), len(job_params))

    flow_raw = str(params.get("flowCode", "")).strip()
    flow = flow_raw.lower()
    if flow_raw == "-1" or flow in {"a", "all"}:
        logger.info("flowCode=%s treated as all-flows filter; omitting flowCode parameter", params.get("flowCode"))
        params.pop("flowCode", None)

    for key in ("reporterCode", "partnerCode", "period"):
        value = params.get(key)
        if isinstance(value, str) and value.strip().lower() == "all":
            logger.warning("%s=all can be rejected by this endpoint; omitting %s", key, key)
            params.pop(key, None)

    if auth.enabled and auth.method == "query":
        token = os.getenv(auth.env_var_name, "").strip()
        if not token:
            raise ValueError(f"Authentication enabled but missing {auth.env_var_name}.")
        params[auth.query_param_name] = token

    return params


def build_auth_headers(auth: AuthConfig) -> dict[str, str]:
    headers = {"User-Agent": "vacoda-downloader/1.0"}
    if not auth.enabled:
        return headers

    token = os.getenv(auth.env_var_name, "").strip()
    if not token:
        raise ValueError(f"Authentication enabled but missing {auth.env_var_name}.")

    if auth.method == "header":
        headers[auth.header_name] = token
    return headers


def download_once(url: str, timeout_seconds: int, headers: dict[str, str] | None = None) -> bytes:
    request = Request(url, headers=headers or {"User-Agent": "vacoda-downloader/1.0"})
    with urlopen(request, timeout=timeout_seconds) as response:
        return response.read()


def detect_auth_error_in_payload(payload: bytes) -> str | None:
    text = payload.decode("utf-8-sig", errors="replace")
    head = text[:4000].lower()
    markers = ("unauthorized", "forbidden", "invalid token", "invalid api key", "access denied", "authentication", "subscription", "premium")
    if not any(m in head for m in markers):
        return None

    if head.lstrip().startswith("{") or head.lstrip().startswith("["):
        try:
            return json.dumps(json.loads(text))[:800]
        except json.JSONDecodeError:
            return text[:800]

    first_line = text.splitlines()[0] if text.splitlines() else text[:200]
    return first_line[:800]


def download_with_retry(url: str, retry: RetryConfig, runtime: DownloadRuntimeConfig, headers: dict[str, str] | None = None) -> bytes:
    safe_url = redact_url(url)
    attempt = 1
    while True:
        logger.debug("Downloading URL attempt %d/%d: %s", attempt, retry.max_attempts, safe_url)
        try:
            return download_once(url, runtime.timeout_seconds, headers=headers)
        except HTTPError as exc:
            if exc.code in (401, 403):
                try:
                    body_preview = exc.read().decode("utf-8-sig", errors="replace")[:600]
                except Exception:
                    body_preview = "<response body unavailable>"
                logger.exception("Authentication HTTP error %s for %s", exc.code, safe_url)
                raise PermissionError(f"API auth failed (HTTP {exc.code}). Response: {body_preview}") from exc
            if attempt >= retry.max_attempts:
                body_preview = "<response body unavailable>"
                try:
                    body_preview = exc.read().decode("utf-8-sig", errors="replace")[:600]
                except Exception:
                    pass
                logger.exception("HTTP retry exhausted for %s", safe_url)
                raise RuntimeError(
                    f"Download failed after {attempt} attempt(s): {safe_url}. "
                    f"HTTP {exc.code}. Response: {body_preview}"
                ) from exc
            logger.warning("Attempt %d/%d failed HTTP %s for %s. Retrying in %.1fs", attempt, retry.max_attempts, exc.code, safe_url, retry.backoff_seconds)
        except (URLError, TimeoutError) as exc:
            if attempt >= retry.max_attempts:
                reason = getattr(exc, "reason", str(exc))
                logger.exception("Network retry exhausted for %s", safe_url)
                raise RuntimeError(f"Download failed after {attempt} attempt(s): {safe_url}. Reason: {reason}") from exc
            logger.warning("Attempt %d/%d failed for %s. Retrying in %.1fs", attempt, retry.max_attempts, safe_url, retry.backoff_seconds)
        sleep(retry.backoff_seconds)
        attempt += 1


def is_csv_format(params: dict[str, Any]) -> bool:
    return str(params.get("format", "csv")).strip().lower() == "csv"


def _csv_row_count(payload: bytes) -> int:
    lines = [line for line in payload.decode("utf-8-sig", errors="replace").splitlines() if line.strip()]
    return max(0, len(lines) - 1) if lines else 0


def _merge_csv_pages(pages: list[bytes]) -> bytes:
    merged_lines: list[str] = []
    header_line: str | None = None

    for page_payload in pages:
        lines = [line for line in page_payload.decode("utf-8-sig", errors="replace").splitlines() if line.strip()]
        if not lines:
            continue
        if header_line is None:
            header_line = lines[0]
            merged_lines.append(header_line)
            merged_lines.extend(lines[1:])
        else:
            merged_lines.extend(lines[1:])

    return ("\n".join(merged_lines) + "\n").encode("utf-8") if merged_lines else b""


def download_paginated_csv(
    endpoint: str,
    params: dict[str, Any],
    retry: RetryConfig,
    runtime: DownloadRuntimeConfig,
    pagination: PaginationConfig,
    headers: dict[str, str],
) -> tuple[bytes, list[str], int]:
    if pagination.limit_param not in params:
        raise ValueError(f"Pagination enabled but '{pagination.limit_param}' not present in params.")

    try:
        page_size = int(params[pagination.limit_param])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {pagination.limit_param}: {params[pagination.limit_param]!r}") from exc

    if page_size <= 0:
        raise ValueError(f"{pagination.limit_param} must be > 0")

    offset = pagination.start_offset
    page_payloads: list[bytes] = []
    urls: list[str] = []
    total_rows = 0

    while True:
        page_params = dict(params)
        page_params[pagination.offset_param] = offset
        page_url = f"{endpoint}?{urlencode(page_params, doseq=True)}"
        urls.append(redact_url(page_url))
        logger.debug("Fetching paginated page with offset=%d", offset)

        payload = download_with_retry(page_url, retry, runtime, headers=headers)
        auth_error = detect_auth_error_in_payload(payload)
        if auth_error:
            raise PermissionError(f"API returned auth/authorization payload: {auth_error}")

        rows = _csv_row_count(payload)
        if rows == 0:
            break

        page_payloads.append(payload)
        total_rows += rows
        logger.info("Fetched page offset=%d with %d row(s)", offset, rows)

        if rows < page_size:
            break
        offset += page_size

    return _merge_csv_pages(page_payloads), urls, total_rows


def download_bulk(
    endpoint: str,
    job_params: dict[str, Any],
    retry: RetryConfig,
    runtime: DownloadRuntimeConfig,
    auth: AuthConfig,
    output_path: Path,
) -> str:
    """
    Download a bulk file from the Comtrade Bulk API and save to output_path.
    Returns the path to the saved file.
    """
    import os
    from urllib.parse import urlencode
    headers = build_auth_headers(auth)
    params = build_query_params({}, job_params, auth)
    url = f"{endpoint}?{urlencode(params, doseq=True)}"
    logger.info(f"[bulk] Downloading bulk file: {redact_url(url)}")
    payload = download_with_retry(url, retry, runtime, headers=headers)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(payload)
    logger.info(f"[bulk] Saved bulk file: {output_path}")
    return str(output_path)


