from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from dataclasses import asdict, is_dataclass

import pandas as pd


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()

def _to_json_safe(
    value: Any,
) -> Any:
    """
    Recursively convert checkpoint values into JSON-safe objects.
    """
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(key): _to_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            _to_json_safe(item)
            for item in value
        ]

    if isinstance(value, datetime):
        return value.isoformat()

    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass

    return value

def _remove_nested_key(
    value: dict[str, Any],
    key_path: tuple[str, ...],
) -> None:
    """
    Remove one nested key from a dictionary if it exists.

    Example:
        ("engine", "threads")
    """
    current: Any = value

    for key in key_path[:-1]:
        if not isinstance(current, dict):
            return

        current = current.get(key)

        if current is None:
            return

    if isinstance(current, dict):
        current.pop(key_path[-1], None)

def _jsonable_config(
    config: Any,
) -> dict[str, Any]:
    """
    Convert a resolved configuration object into JSON-compatible data.

    Supports:
      - Pydantic models;
      - dataclasses;
      - dictionaries;
      - regular objects exposing public attributes.
    """
    if hasattr(config, "model_dump"):
        raw = config.model_dump(
            mode="json",
        )

    elif is_dataclass(config):
        raw = asdict(config)

    elif isinstance(config, dict):
        raw = config

    elif hasattr(config, "__dict__"):
        raw = {
            key: value
            for key, value in vars(config).items()
            if not key.startswith("_")
        }

    else:
        raise TypeError(
            "Semantic fingerprinting requires a Pydantic model, "
            "dataclass, dictionary, or object with public attributes. "
            f"Received {type(config).__name__!r}."
        )

    result = json.loads(
        json.dumps(
            raw,
            default=str,
        )
    )

    if not isinstance(result, dict):
        raise TypeError(
            "Resolved configuration must serialize to a dictionary."
        )

    return result

def _jsonable_config_deprecated(
    config: Any,
) -> dict[str, Any]:
    """
    Convert a parsed configuration object into JSON-compatible data.
    """
    if hasattr(config, "model_dump"):
        result = config.model_dump(
            mode="json",
        )
    elif isinstance(config, dict):
        result = json.loads(
            json.dumps(
                config,
                default=str,
            )
        )
    else:
        raise TypeError(
            "Semantic fingerprinting requires a Pydantic model "
            "or dictionary."
        )

    if not isinstance(result, dict):
        raise TypeError(
            "Resolved configuration must serialize to a dictionary."
        )

    return result


def semantic_config_fingerprint(
    *,
    config: Any,
    excluded_paths: tuple[tuple[str, ...], ...] = (),
) -> dict[str, Any]:
    """
    Fingerprint the parsed meaning of a configuration.

    YAML comments, spacing, key order and formatting do not affect this
    fingerprint. Explicitly excluded runtime-only keys are removed before
    hashing.
    """
    normalized = _jsonable_config(
        config
    )

    for key_path in excluded_paths:
        _remove_nested_key(
            normalized,
            key_path,
        )

    canonical_text = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )

    return {
        "sha256": sha256_text(canonical_text),
        "excluded_paths": [
            ".".join(path)
            for path in excluded_paths
        ],
    }

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def config_fingerprint(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": file_sha256(path),
    }

def file_list_fingerprint(
    *,
    paths: list[Path],
    label: str,
) -> dict[str, Any]:
    files = sorted(paths)

    manifest_lines: list[str] = []
    total_size = 0
    latest_mtime_ns = 0
    missing_files: list[str] = []

    for path in files:
        if not path.exists():
            missing_files.append(str(path))
            manifest_lines.append(f"{path}|MISSING")
            continue

        stat = path.stat()

        total_size += stat.st_size
        latest_mtime_ns = max(latest_mtime_ns, stat.st_mtime_ns)

        manifest_lines.append(
            f"{path}|{stat.st_size}|{stat.st_mtime_ns}"
        )

    manifest_text = "\n".join(manifest_lines)

    return {
        "kind": "file_list",
        "label": label,
        "n_files": len(files),
        "n_missing_files": len(missing_files),
        "missing_files": missing_files,
        "total_size_bytes": total_size,
        "latest_mtime_ns": latest_mtime_ns,
        "manifest_sha256": sha256_text(manifest_text),
    }
    
def file_fingerprint(path: Path, *, hash_file: bool = False) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "size_bytes": None,
            "mtime_ns": None,
            "sha256": None,
        }

    stat = path.stat()

    return {
        "path": str(path),
        "exists": True,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": file_sha256(path) if hash_file else None,
    }


def file_manifest_fingerprint(
    *,
    root: Path,
    pattern: str,
) -> dict[str, Any]:
    files = sorted(p for p in root.glob(pattern) if p.is_file())

    manifest_lines: list[str] = []
    total_size = 0
    latest_mtime_ns = 0

    for path in files:
        stat = path.stat()
        rel = path.relative_to(root).as_posix()

        total_size += stat.st_size
        latest_mtime_ns = max(latest_mtime_ns, stat.st_mtime_ns)

        manifest_lines.append(
            f"{rel}|{stat.st_size}|{stat.st_mtime_ns}"
        )

    manifest_text = "\n".join(manifest_lines)

    return {
        "kind": "file_tree",
        "path": str(root),
        "pattern": pattern,
        "n_files": len(files),
        "total_size_bytes": total_size,
        "latest_mtime_ns": latest_mtime_ns,
        "manifest_sha256": sha256_text(manifest_text),
    }


def _read_audit(path: Path) -> pd.DataFrame:
    """Read a supported pipeline audit artifact."""

    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)

    raise ValueError(
        "Unsupported audit format "
        f"{suffix!r} for audit artifact: {path}"
    )


def audit_summary(
    *,
    audit_path: Path,
    ready_check: str | None = None,
) -> dict[str, Any]:
    """
    Summarize either:

    1. a long audit with `check`/`check_name` and `status`; or
    2. a wide one-row audit with an explicit `audit_passed` flag or
       zero-valued failure counters.
    """

    if not audit_path.exists():
        return {
            "exists": False,
            "n_rows": 0,
            "n_failed_audit_rows": None,
            "ready_check": ready_check,
            "ready_check_status": None,
        }

    audit = _read_audit(audit_path)

    ready_status: str | None = None

    # --------------------------------------------------------------
    # Long-format audit
    # --------------------------------------------------------------

    if "status" in audit.columns:
        normalized_status = (
            audit["status"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        n_failed = int(
            normalized_status.isin(
                {"fail", "failed", "error"}
            ).sum()
        )

        check_column: str | None = None

        if "check" in audit.columns:
            check_column = "check"
        elif "check_name" in audit.columns:
            check_column = "check_name"

        if (
            ready_check is not None
            and check_column is not None
        ):
            ready_rows = audit.loc[
                audit[check_column].astype(str)
                == str(ready_check)
            ]

            if not ready_rows.empty:
                ready_status = str(
                    ready_rows["status"].iloc[-1]
                ).strip().lower()

    # --------------------------------------------------------------
    # Wide-format audit with explicit pass flag
    # --------------------------------------------------------------

    elif "audit_passed" in audit.columns:
        passed = (
            audit["audit_passed"]
            .fillna(False)
            .astype(bool)
        )

        n_failed = int((~passed).sum())

        if ready_check is not None:
            ready_status = (
                "pass"
                if bool(passed.all())
                else "fail"
            )

    # --------------------------------------------------------------
    # Trade Sync wide audit
    # --------------------------------------------------------------

    else:
        failure_columns = [
            "n_invalid_reconciled_values",
            "n_missing_expected_reconciled_values",
            "n_duplicate_keys",
        ]

        available_failure_columns = [
            column
            for column in failure_columns
            if column in audit.columns
        ]

        if len(available_failure_columns) != len(
            failure_columns
        ):
            missing = sorted(
                set(failure_columns)
                - set(available_failure_columns)
            )

            raise RuntimeError(
                "Wide audit has neither a `status` column nor an "
                "`audit_passed` column and is missing required "
                "failure counters: "
                + ", ".join(missing)
                + f". Audit: {audit_path}"
            )

        failure_mask = pd.Series(
            False,
            index=audit.index,
        )

        for column in failure_columns:
            values = pd.to_numeric(
                audit[column],
                errors="coerce",
            )

            # Missing/non-numeric failure counters must not silently pass.
            failure_mask |= values.isna() | values.ne(0)

        n_failed = int(failure_mask.sum())

        if ready_check is not None:
            ready_status = (
                "pass"
                if n_failed == 0
                else "fail"
            )

    return {
        "exists": True,
        "n_rows": int(len(audit)),
        "n_failed_audit_rows": n_failed,
        "ready_check": ready_check,
        "ready_check_status": ready_status,
    }


def checkpoint_path(
    *,
    checkpoint_dir: Path,
    step_name: str,
) -> Path:
    return checkpoint_dir / f"{step_name}.json"

def source_files_fingerprint(
    *,
    paths: list[Path],
    label: str,
) -> dict[str, Any]:
    """
    Fingerprint source files by content.

    Unlike ordinary data-file fingerprints, source reuse depends on
    file contents rather than mtime.
    """
    files = sorted(Path(path) for path in paths)

    manifest_lines: list[str] = []
    missing_files: list[str] = []

    for path in files:
        if not path.is_file():
            missing_files.append(str(path))
            manifest_lines.append(
                f"{path}|MISSING"
            )
            continue

        manifest_lines.append(
            f"{path.resolve()}|{file_sha256(path)}"
        )

    manifest_text = "\n".join(manifest_lines)

    return {
        "kind": "source_files",
        "label": label,
        "n_files": len(files),
        "n_missing_files": len(missing_files),
        "missing_files": missing_files,
        "manifest_sha256": sha256_text(
            manifest_text
        ),
    }

def write_checkpoint(
    *,
    path: Path,
    run_id: str,
    step_name: str,
    status: str,
    ready_flag: str | None,
    ready_value: Any,
    config: dict[str, Any],
    input_fingerprint: dict[str, Any],
    output_path: Path,
    audit_path: Path,
    result: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    output_fp = file_fingerprint(output_path, hash_file=False)
    audit_fp = file_fingerprint(audit_path, hash_file=True)
    audit_info = audit_summary(
        audit_path=audit_path,
        ready_check=ready_flag,
    )

    payload = {
        "schema_version": 2,
        "step_name": step_name,
        "run_id": run_id,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "ready_flag": ready_flag,
        "ready_value": ready_value,
        "config": config,
        "input": input_fingerprint,
        "output": output_fp,
        "audit": {
            **audit_fp,
            **audit_info,
        },
        "result":result,
    }

    path.write_text(
        json.dumps(_to_json_safe(payload), 
                    indent=2),
        encoding="utf-8",
    )


def load_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def validate_checkpoint(
    *,
    checkpoint: dict[str, Any],
    expected_status: list[str],
    required_ready_flag: str | None,
    current_config: dict[str, Any],
    current_input: dict[str, Any],
) -> tuple[bool, str]:
    

    if checkpoint.get("status") not in expected_status:
        return False, "checkpoint status is not acceptable"

    if required_ready_flag is not None:
        if checkpoint.get("ready_flag") != required_ready_flag:
            return False, "checkpoint ready flag name changed"

        if checkpoint.get("ready_value") is not True:
            return False, "checkpoint ready value is not true"

    if checkpoint.get("schema_version") != 2:
        return (
            False,
            "checkpoint schema version changed",
        )

    config_matches, config_reason = (
        _config_fingerprints_match(
            checkpoint_config=checkpoint.get(
                "config",
                {},
            ),
            current_config=current_config,
        )
    )

    if not config_matches:
        return False, config_reason

    if checkpoint.get("input") != current_input:
        return False, "input fingerprint changed"

    output = checkpoint.get("output", {})
    output_path = Path(str(output.get("path", "")))

    if not output_path.exists():
        return False, "output file is missing"

    current_output = file_fingerprint(output_path, hash_file=False)

    if output.get("size_bytes") != current_output.get("size_bytes"):
        return False, "output file size changed"

    if output.get("mtime_ns") != current_output.get("mtime_ns"):
        return False, "output file mtime changed"

    audit = checkpoint.get("audit", {})
    audit_path = Path(str(audit.get("path", "")))

    if not audit_path.exists():
        return False, "audit file is missing"

    current_audit = file_fingerprint(audit_path, hash_file=True)

    if audit.get("sha256") != current_audit.get("sha256"):
        return False, "audit file hash changed"

    current_audit_summary = audit_summary(
        audit_path=audit_path,
        ready_check=required_ready_flag,
    )

    if (
        required_ready_flag is not None
        and current_audit_summary.get(
            "ready_check_status"
        ) not in {None, "pass"}
    ):
        return False, "current audit readiness check did not pass"

    if current_audit_summary.get("n_failed_audit_rows") != 0:
        return False, "current audit has failed rows"

    checkpoint_failed = audit.get("n_failed_audit_rows")
    if checkpoint_failed != 0:
        return False, "checkpoint audit had failed rows"

    return True, "valid checkpoint"

def _config_fingerprints_match(
    *,
    checkpoint_config: dict[str, Any],
    current_config: dict[str, Any],
) -> tuple[bool, str]:
    """
    Compare semantic configuration fingerprints.

    Raw file hashes remain provenance metadata and do not control reuse.
    """
    checkpoint_step = checkpoint_config.get(
        "step_config",
        {}
    )
    current_step = current_config.get(
        "step_config",
        {}
    )

    checkpoint_semantic = checkpoint_step.get(
        "semantic_sha256"
    )
    current_semantic = current_step.get(
        "semantic_sha256"
    )

    if checkpoint_semantic is None:
        return (
            False,
            "checkpoint predates semantic config fingerprinting",
        )

    if current_semantic is None:
        return (
            False,
            "current semantic config fingerprint is missing",
        )

    if checkpoint_semantic != current_semantic:
        return (
            False,
            "semantic config fingerprint changed",
        )

    checkpoint_scope = checkpoint_config.get(
        "pipeline_scope"
    )
    current_scope = current_config.get(
        "pipeline_scope"
    )

    if checkpoint_scope != current_scope:
        return (
            False,
            "pipeline scope changed",
        )

    return True, ""