"""Publish audited VACODA Release candidates without changing their contents.

This is the publication step, not the scientific audit. Candidates must have
passed audit_release_candidates first. Published dataset paths are immutable:
reruns never silently replace a completed or partial publication.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

import duckdb

from vacoda.config.steps.release import ReleaseConfig


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _audit_details(audit_path: Path, run_id: str, release_id: str) -> dict[str, Any]:
    with duckdb.connect() as con:
        description = con.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)", [str(audit_path)]
        ).fetchall()
        columns = {str(row[0]) for row in description}
        required = {"run_id", "release_id", "audit_passed", "standard_schema_exact",
                    "enriched_schema_exact", "standard_prediction_policy_consistent",
                    "enriched_prediction_preserved", "manual_review_rule_consistent"}
        if required - columns:
            raise RuntimeError(f"Release audit missing columns: {sorted(required - columns)}")
        rows = con.execute("SELECT * FROM read_parquet(?)", [str(audit_path)]).fetchall()
        if len(rows) != 1:
            raise RuntimeError("Release audit must contain exactly one summary row.")
        record = dict(zip([row[0] for row in description], rows[0]))
    if record["run_id"] != run_id or record["release_id"] != release_id:
        raise RuntimeError("Release audit identity does not match this publication.")
    if record["audit_passed"] is not True:
        raise RuntimeError("Release audit did not pass; publication blocked.")
    failed = [name for name, value in record.items()
              if name.endswith("_consistent") or name.endswith("_preserved")
              or name.endswith("_exact")]
    if any(record[name] is not True for name in failed):
        raise RuntimeError("Release audit contains failed publication invariants.")
    return record


def publish_release_datasets(
    *,
    cfg: ReleaseConfig,
    candidate_standard_path: Path,
    candidate_enriched_path: Path,
    audit_path: Path,
    standard_path: Path,
    enriched_path: Path,
    run_id: str,
    release_id: str,
    force: bool = False,
    logger,
) -> dict[str, Any]:
    """Promote staged candidates after an affirmative Release audit.

    Published outputs are immutable, including when force=True. To publish a
    different dataset, select a new release_id and rerun candidate building.
    """
    start = perf_counter()
    standard_candidate = Path(candidate_standard_path)
    enriched_candidate = Path(candidate_enriched_path)
    audit_path = Path(audit_path)
    standard_path = Path(standard_path)
    enriched_path = Path(enriched_path)

    expected_standard = cfg.product_path("release", release_id=release_id)
    expected_enriched = cfg.enriched_path(release_id=release_id)
    if standard_path != expected_standard or enriched_path != expected_enriched:
        raise ValueError("Published paths do not match Release configuration.")
    if audit_path != cfg.product_path("release_audit", release_id=release_id):
        raise ValueError("Audit path does not match the current Release identity.")
    staging_dir = cfg.output_dir() / ".staging" / release_id
    if (standard_candidate != staging_dir / cfg.products.release.filename
            or enriched_candidate != (staging_dir / cfg.products.release.filename).with_name(
                f"{Path(cfg.products.release.filename).stem}_enriched"
                f"{Path(cfg.products.release.filename).suffix}"
            )):
        raise ValueError("Release candidates are not from this release identity staging directory.")
    metadata_standard = cfg.metadata_path(release_id=release_id)
    metadata_enriched = cfg.metadata_path(release_id=release_id, enriched=True)
    targets = (standard_path, enriched_path, metadata_standard, metadata_enriched)
    sources = (standard_candidate, enriched_candidate)

    for source in (*sources, audit_path):
        if not source.is_file() or source.stat().st_size <= 0:
            raise FileNotFoundError(f"Missing/empty Release publication input: {source}")
    if standard_candidate.resolve() == standard_path.resolve() or enriched_candidate.resolve() == enriched_path.resolve():
        raise ValueError("Candidate and published paths must differ.")
    if any(path.exists() for path in targets):
        raise FileExistsError(
            "Release already has published artifacts. Publication is immutable; "
            "use a new release_id instead of force=True."
        )
    if standard_path.parent != enriched_path.parent or standard_path.parent != metadata_standard.parent:
        raise ValueError("Release artifacts must share a publication directory.")
    if audit_path == standard_path or audit_path == enriched_path:
        raise ValueError("Audit artifact cannot be the published dataset.")

    audit_record = _audit_details(audit_path, run_id, release_id)
    # An audit is not valid for candidate files modified after it was written.
    audited_at = audit_path.stat().st_mtime_ns
    if any(path.stat().st_mtime_ns > audited_at for path in sources):
        raise RuntimeError("A Release candidate changed after the audit; rerun the audit.")

    output_dir = standard_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_paths: list[Path] = []
    committed: list[Path] = []
    try:
        candidate_info = []
        for source, destination, schema in (
            (standard_candidate, standard_path, cfg.schema.standard),
            (enriched_candidate, enriched_path, cfg.schema.enriched),
        ):
            temporary = output_dir / f".{destination.name}.{uuid4().hex}.tmp"
            temp_paths.append(temporary)
            with source.open("rb") as reader, temporary.open("xb") as writer:
                shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
                writer.flush()
                os.fsync(writer.fileno())
            if _sha256(source) != _sha256(temporary):
                raise RuntimeError(f"Candidate changed during copy: {source}")
            candidate_info.append({
                "file": destination.name,
                "sha256": _sha256(temporary),
                "size_bytes": temporary.stat().st_size,
                "columns": list(schema),
            })

        audit_hash = _sha256(audit_path)
        for info, metadata_path, kind in (
            (candidate_info[0], metadata_standard, "standard"),
            (candidate_info[1], metadata_enriched, "enriched"),
        ):
            payload = {
                "dataset": "VACODA", "variant": kind,
                "release_id": release_id, "run_id": run_id,
                "scope": {"target_hs": cfg.scope.target_hs,
                          "min_year": cfg.scope.min_year,
                          "max_year": cfg.scope.max_year},
                "artifact": info,
                "audit": {"path": str(audit_path), "sha256": audit_hash,
                          "passed": True, "n_rows": audit_record.get("standard_rows"),
                          "manual_review_rows": audit_record.get("manual_review_rows"),
                          "manual_review_hs6_codes": audit_record.get("manual_review_hs6_codes")},
                "publication_policy": {
                    "manual_review_rule": "hs6_seen IS FALSE AND hs4_seen IS FALSE",
                    "standard_prediction": "NULL when manual_review_required is true",
                    "enriched_prediction": "original model prediction preserved",
                    "model_prediction_available": "original inference availability preserved",
                },
                "unit_value_reference_basis": "Must be verified against the Validation reference methodology",
            }
            _atomic_json(metadata_path, payload)
            committed.append(metadata_path)

        # Commit datasets after metadata sidecars are ready. If either commit
        # fails, remove only artifacts created in this attempt.
        for temporary, destination in zip(temp_paths, (standard_path, enriched_path)):
            if destination.exists():
                raise FileExistsError(f"Publication target unexpectedly exists: {destination}")
            os.replace(temporary, destination)
            committed.append(destination)
        logger.info("Published standard Release dataset: %s", standard_path)
        logger.info("Published enriched Release dataset: %s", enriched_path)
        logger.info("Publication audit: %s", audit_path)
        return {
            "status": "completed", "run_id": run_id, "release_id": release_id,
            "standard_path": standard_path, "enriched_path": enriched_path,
            "output_path": standard_path,
            "standard_metadata_path": metadata_standard,
            "enriched_metadata_path": metadata_enriched,
            "audit_path": audit_path,
            "publication_completed": True,
            "duration_seconds": perf_counter() - start,
        }
    except Exception:
        for committed_path in reversed(committed):
            committed_path.unlink(missing_ok=True)
        raise
    finally:
        for temporary in temp_paths:
            temporary.unlink(missing_ok=True)
