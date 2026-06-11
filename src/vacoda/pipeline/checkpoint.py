from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

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

    n_failed = int((audit["status"] == "fail").sum()) if "status" in audit.columns else -1

    ready_status = None
    if ready_check is not None and "check" in audit.columns:
        ready_rows = audit.loc[audit["check"] == ready_check]
        if not ready_rows.empty and "status" in audit.columns:
            ready_status = str(ready_rows["status"].iloc[-1])

    return {
        "exists": True,
        "n_rows": len(audit),
        "n_failed_audit_rows": n_failed,
        "ready_check": ready_check,
        "ready_check_status": ready_status,
    }

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

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

def file_sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def config_fingerprint(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": file_sha256(path),
    }

def input_tree_fingerprint(
    *,
    input_dir: Path,
    pattern: str,
) -> dict[str, Any]:
    files = sorted(p for p in input_dir.glob(pattern) if p.is_file())

    total_size = sum(p.stat().st_size for p in files)
    latest_mtime_ns = max((p.stat().st_mtime_ns for p in files), default=0)

    return {
        "input_dir": str(input_dir),
        "pattern": pattern,
        "n_files": len(files),
        "total_size_bytes": total_size,
        "latest_mtime_ns": latest_mtime_ns,
    }


def single_file_fingerprint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "size_bytes": None,
            "mtime_ns": None,
        }

    stat = path.stat()

    return {
        "path": str(path),
        "exists": True,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }

def checkpoint_is_valid(
    *,
    checkpoint: dict[str, Any],
    current_config: dict[str, Any],
    current_input: dict[str, Any],
    expected_status: list[str],
    required_ready_flag: str | None,
) -> tuple[bool, str]:
    if checkpoint.get("status") not in expected_status:
        return False, "status does not match expected status"

    if required_ready_flag is not None:
        if checkpoint.get("ready_flag") != required_ready_flag:
            return False, "ready flag name changed"

        if checkpoint.get("ready_value") is not True:
            return False, "ready flag is not true"

    if checkpoint.get("config") != current_config:
        return False, "config fingerprint changed"

    if checkpoint.get("input") != current_input:
        return False, "input fingerprint changed"

    output = checkpoint.get("output", {})
    output_path = Path(output.get("path", ""))

    if not output_path.exists():
        return False, "output file no longer exists"

    current_output = file_fingerprint(output_path, hash_file=False)

    if (
        output.get("size_bytes") != current_output.get("size_bytes")
        or output.get("mtime_ns") != current_output.get("mtime_ns")
    ):
        return False, "output file fingerprint changed"

    audit = checkpoint.get("audit", {})
    audit_path = Path(audit.get("path", ""))

    if not audit_path.exists():
        return False, "audit file no longer exists"

    current_audit_file = file_fingerprint(audit_path, hash_file=True)

    if audit.get("sha256") != current_audit_file.get("sha256"):
        return False, "audit file hash changed"

    if audit.get("n_failed_audit_rows") not in (0, None):
        return False, "checkpoint audit had failed rows"

    return True, "valid checkpoint"