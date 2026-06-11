from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


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


def audit_summary(
    *,
    audit_path: Path,
    ready_check: str | None = None,
) -> dict[str, Any]:
    if not audit_path.exists():
        return {
            "exists": False,
            "n_rows": 0,
            "n_failed_audit_rows": None,
            "ready_check": ready_check,
            "ready_check_status": None,
        }

    audit = pd.read_csv(audit_path)

    if "status" in audit.columns:
        n_failed = int((audit["status"] == "fail").sum())
    else:
        n_failed = -1

    ready_status = None
    if ready_check is not None and "check" in audit.columns and "status" in audit.columns:
        ready_rows = audit.loc[audit["check"] == ready_check]
        if not ready_rows.empty:
            ready_status = str(ready_rows["status"].iloc[-1])

    return {
        "exists": True,
        "n_rows": len(audit),
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
        "schema_version": 1,
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
        "result": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in result.items()
        },
    }

    path.write_text(
        json.dumps(payload, indent=2),
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

    if checkpoint.get("config") != current_config:
        return False, "config fingerprint changed"

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

    if current_audit_summary.get("n_failed_audit_rows") != 0:
        return False, "current audit has failed rows"

    checkpoint_failed = audit.get("n_failed_audit_rows")
    if checkpoint_failed != 0:
        return False, "checkpoint audit had failed rows"

    return True, "valid checkpoint"