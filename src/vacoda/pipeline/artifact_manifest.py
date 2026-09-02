"""
vacoda.pipeline.artifact_manifest.py

Reusable provenance manifests for CIF/FOB artifacts.

An artifact is reusable only when:

1. the artifact exists and is non-empty;
2. its manifest exists;
3. the stored semantic fingerprint matches the current expected
   fingerprint.

Fingerprints are derived from JSON-safe semantic payloads. Runtime-only
settings such as DuckDB thread count should not be included in those
payloads.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any
import json

import numpy as np


MANIFEST_SCHEMA_VERSION = 1


def json_safe(value: Any) -> Any:
    """Recursively convert a value into deterministic JSON-safe data."""
    if isinstance(value, Path):
        return str(value)

    if is_dataclass(value):
        return json_safe(asdict(value))

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in sorted(
                value.items(),
                key=lambda pair: str(pair[0]),
            )
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(item)
            for item in value
        ]

    if isinstance(value, set):
        return sorted(
            json_safe(item)
            for item in value
        )

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, float):
        if np.isnan(value):
            return None

        if np.isposinf(value):
            return "Infinity"

        if np.isneginf(value):
            return "-Infinity"

        return value

    if value is None or isinstance(
        value,
        (str, int, bool),
    ):
        return value

    if hasattr(value, "__dict__"):
        return json_safe(
            {
                key: item
                for key, item in vars(value).items()
                if not key.startswith("_")
            }
        )

    return str(value)


def semantic_fingerprint(
    payload: Any,
) -> str:
    """Return a deterministic SHA-256 fingerprint for a semantic payload."""
    encoded = json.dumps(
        json_safe(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")

    return sha256(encoded).hexdigest()


def file_stat_fingerprint(
    path: Path,
) -> dict[str, Any]:
    """
    Return a lightweight file fingerprint.

    This avoids hashing multi-gigabyte Parquet files while still detecting
    normal replacement and modification through size and nanosecond mtime.
    """
    path = Path(path)

    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "size_bytes": None,
            "mtime_ns": None,
        }

    stat = path.stat()

    return {
        "path": str(path.resolve()),
        "exists": True,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }

def file_content_list_fingerprint(
    *,
    paths: list[Path],
    label: str,
) -> dict[str, Any]:
    """
    Fingerprint a collection of semantic input files by content.
    """
    files = sorted(
        Path(path)
        for path in paths
    )

    entries = [
        file_content_fingerprint(path)
        for path in files
    ]

    return {
        "kind": "content_file_list",
        "label": label,
        "n_files": len(files),
        "n_missing_files": sum(
            not item["exists"]
            for item in entries
        ),
        "files": entries,
        "semantic_sha256": semantic_fingerprint(
            entries
        ),
    }

def file_content_fingerprint(
    path: Path,
) -> dict[str, Any]:
    """
    Return a content-based fingerprint for a semantic input.

    Intended for repository-managed reference/config/source files where
    modification time must not affect reuse.
    """
    path = Path(path)

    if not path.is_file():
        return {
            "path": str(path.resolve()),
            "exists": False,
            "size_bytes": None,
            "sha256": None,
        }

    stat = path.stat()

    return {
        "path": str(path.resolve()),
        "exists": True,
        "size_bytes": int(stat.st_size),
        "sha256": file_content_sha256(path),
    }

def file_content_sha256(
    path: Path,
    *,
    chunk_size: int = 1024 * 1024,
) -> str:
    """
    Hash a relatively small source, JSON, YAML, or CSV file.

    Do not use this for the large CIF/FOB Parquet products unless a full
    content hash is explicitly required.
    """
    path = Path(path)
    digest = sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def manifest_path_for(
    artifact_path: Path,
) -> Path:
    """
    Return the sidecar manifest path for an artifact.

    Example:
      estimated_rates.parquet
      estimated_rates.manifest.json
    """
    artifact_path = Path(artifact_path)

    return artifact_path.with_suffix(
        ".manifest.json"
    )


def read_manifest(
    manifest_path: Path,
) -> dict[str, Any] | None:
    """Read a manifest safely, returning None when absent or invalid."""
    manifest_path = Path(manifest_path)

    if not manifest_path.is_file():
        return None

    try:
        value = json.loads(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ):
        return None

    return value if isinstance(value, dict) else None


def write_manifest(
    *,
    manifest_path: Path,
    artifact_name: str,
    semantic_payload: dict[str, Any],
    artifact_path: Path,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write an artifact manifest atomically."""
    manifest_path = Path(manifest_path)
    artifact_path = Path(artifact_path)

    fingerprint = semantic_fingerprint(
        semantic_payload
    )

    payload: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifact_name": artifact_name,
        "created_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "semantic_fingerprint": fingerprint,
        "semantic_payload": json_safe(
            semantic_payload
        ),
        "artifact": file_stat_fingerprint(
            artifact_path
        ),
    }

    if extra:
        payload["extra"] = json_safe(extra)

    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = manifest_path.with_suffix(
        manifest_path.suffix + ".tmp"
    )

    temporary_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    temporary_path.replace(
        manifest_path
    )

    return payload


def validate_manifest(
    *,
    artifact_path: Path,
    manifest_path: Path,
    expected_payload: dict[str, Any],
) -> tuple[bool, str]:
    """
    Validate an artifact and its manifest against current semantics.
    """
    artifact_path = Path(artifact_path)
    manifest_path = Path(manifest_path)

    if not artifact_path.is_file():
        return False, "artifact does not exist"

    if artifact_path.stat().st_size <= 0:
        return False, "artifact is empty"

    manifest = read_manifest(
        manifest_path
    )

    if manifest is None:
        return False, "manifest is missing or invalid"

    if (
        manifest.get("schema_version")
        != MANIFEST_SCHEMA_VERSION
    ):
        return False, "manifest schema version changed"

    expected_fingerprint = semantic_fingerprint(
        expected_payload
    )

    if (
        manifest.get("semantic_fingerprint")
        != expected_fingerprint
    ):
        return False, "semantic fingerprint changed"

    stored_artifact = manifest.get(
        "artifact",
        {}
    )
    current_artifact = file_stat_fingerprint(
        artifact_path
    )

    if (
        stored_artifact.get("size_bytes")
        != current_artifact.get("size_bytes")
    ):
        return False, "artifact size changed"

    if (
        stored_artifact.get("mtime_ns")
        != current_artifact.get("mtime_ns")
    ):
        return False, "artifact modification time changed"

    return True, "manifest fingerprint matches"