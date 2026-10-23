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

from vacoda.steps.release.release_data import PUBLIC_COLUMN_NAMES
from vacoda.config.steps.release import ReleaseConfig
from vacoda.pipeline.checkpoint import (
    file_content_fingerprint,
    semantic_config_fingerprint,
    load_checkpoint,
    validate_local_checkpoint,
    write_local_checkpoint,
)

def _published_schema(columns) -> list[str]:
    return [
        PUBLIC_COLUMN_NAMES.get(column, column)
        for column in columns
    ]

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

def _verify_existing_publication(
    *,
    cfg: ReleaseConfig,
    standard_path: Path,
    enriched_path: Path,
    metadata_standard: Path,
    metadata_enriched: Path,
    audit_path: Path,
    release_id: str,
    dictionary_standard: Path,  
    dictionary_enriched: Path,
) -> None:
    """Verify an immutable publication against its metadata and audit."""

    products = (
        (standard_path, metadata_standard, "standard", cfg.schema.standard),
        (enriched_path, metadata_enriched, "enriched", cfg.schema.enriched),
    )

    original_run_ids = set()

    for metadata_path, dictionary_path in (
        (metadata_standard, dictionary_standard),
        (metadata_enriched, dictionary_enriched),
    ):
        if not dictionary_path.is_file():
            raise RuntimeError(f"Published data dictionary missing: {dictionary_path}")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        info = metadata["data_dictionary"]

        if info["file"] != dictionary_path.name:
            raise RuntimeError("Published dictionary filename mismatch.")

        if info["size_bytes"] != dictionary_path.stat().st_size:
            raise RuntimeError("Published dictionary size mismatch.")

        if info["sha256"] != _sha256(dictionary_path):
            raise RuntimeError("Published dictionary checksum mismatch.")

    for dataset_path, metadata_path, variant, expected_schema in products:
        if not dataset_path.is_file() or not metadata_path.is_file():
            raise RuntimeError(
                f"Incomplete Release publication: {dataset_path}"
            )

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        artifact = metadata["artifact"]
        audit = metadata["audit"]

        if metadata.get("release_id") != release_id:
            raise RuntimeError("Published Release identity mismatch.")

        if metadata.get("variant") != variant:
            raise RuntimeError("Published Release variant mismatch.")

        if artifact.get("file") != dataset_path.name:
            raise RuntimeError("Published Release filename mismatch.")

        if artifact.get("columns") != _published_schema(expected_schema):
            raise RuntimeError("Published Release schema mismatch.")

        if artifact.get("size_bytes") != dataset_path.stat().st_size:
            raise RuntimeError("Published Release size mismatch.")

        if artifact.get("sha256") != _sha256(dataset_path):
            raise RuntimeError("Published Release checksum mismatch.")
        
        if audit.get("passed") is not True:
            raise RuntimeError("Original publication was not audit-approved.")
        
        if audit.get("path") != str(audit_path):
            raise RuntimeError("Published Release audit path mismatch.")

        original_run_ids.add(metadata.get("run_id"))

    if len(original_run_ids) != 1:
        raise RuntimeError("Published datasets have different originating runs.")

    original_run_id = original_run_ids.pop()

    if not original_run_id:
        raise RuntimeError("Published Release has no originating run ID.")


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
    checkpoint_dir: Path,
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

    staging_dir = cfg.output_dir() / ".staging" / release_id

    metadata_standard = cfg.metadata_path(release_id=release_id)
    metadata_enriched = cfg.metadata_path(release_id=release_id, enriched=True)

    dictionary_standard_candidate = staging_dir / "data_dictionary_standard.md"
    dictionary_enriched_candidate = staging_dir / "data_dictionary_enriched.md"

    dictionary_standard = standard_path.parent / "data_dictionary_standard.md"
    dictionary_enriched = standard_path.parent / "data_dictionary_enriched.md"

    expected_standard = cfg.product_path("release", release_id=release_id)
    expected_enriched = cfg.enriched_path(release_id=release_id)
    if standard_path != expected_standard or enriched_path != expected_enriched:
        raise ValueError("Published paths do not match Release configuration.")
    if audit_path != cfg.product_path("release_audit", release_id=release_id):
        raise ValueError("Audit path does not match the current Release identity.")
    if (standard_candidate != staging_dir / cfg.products.release.filename
            or enriched_candidate != (staging_dir / cfg.products.release.filename).with_name(
                f"{Path(cfg.products.release.filename).stem}_enriched"
                f"{Path(cfg.products.release.filename).suffix}"
            )):
        raise ValueError("Release candidates are not from this release identity staging directory.")
    metadata_standard = cfg.metadata_path(release_id=release_id)
    metadata_enriched = cfg.metadata_path(release_id=release_id, enriched=True)
    targets = (
        standard_path,
        enriched_path,
        metadata_standard,
        metadata_enriched,
        dictionary_standard,
        dictionary_enriched,
    )

    dictionaries = (
        (dictionary_standard_candidate, dictionary_standard),
        (dictionary_enriched_candidate, dictionary_enriched),
    )
    sources = (standard_candidate, enriched_candidate)

    # ==============================================================
    # LOCAL CHECKPOINT — IMMUTABLE PUBLICATION
    # ==============================================================

    checkpoint_path = checkpoint_dir / "publication.json"

    config_fp = semantic_config_fingerprint(
        config={
            "release_id": release_id,
            "scope": cfg.scope,
            "standard_schema": tuple(cfg.schema.standard),
            "enriched_schema": tuple(cfg.schema.enriched),
        }
    )

    input_fp = {
        "candidate_standard": file_content_fingerprint(standard_candidate),
        "candidate_enriched": file_content_fingerprint(enriched_candidate),
        "dictionary_standard": file_content_fingerprint(dictionary_standard_candidate),
        "dictionary_enriched": file_content_fingerprint(dictionary_enriched_candidate),
    }

    output_paths = {
        "standard": standard_path,
        "enriched": enriched_path,
        "standard_metadata": metadata_standard,
        "enriched_metadata": metadata_enriched,
        "dictionary_standard": dictionary_standard,
        "dictionary_enriched": dictionary_enriched,
    }

    # Every execution requires a passing current audit.
    _audit_details(audit_path, run_id, release_id)

    existing = any(path.exists() for path in targets)

    if existing and not all(path.is_file() for path in targets):
        raise RuntimeError("Existing publication is incomplete.")

    same_publication = existing and all(
        source.is_file()
        and _sha256(source) == _sha256(destination)
        for source, destination in (
            (standard_candidate, standard_path),
            (enriched_candidate, enriched_path),
            *dictionaries,
        )
    )

    if same_publication:
        if not all(path.is_file() for path in targets):
            raise RuntimeError(
                "Release publication is incomplete; refusing reuse."
            )

        _verify_existing_publication(
            cfg=cfg,
            standard_path=standard_path,
            enriched_path=enriched_path,
            metadata_standard=metadata_standard,
            metadata_enriched=metadata_enriched,
            audit_path=audit_path,
            release_id=release_id,
            dictionary_standard=dictionary_standard,
            dictionary_enriched=dictionary_enriched,
        )

        # Existing candidates must correspond exactly to the published datasets.
        for candidate, published in (
            (standard_candidate, standard_path),
            (enriched_candidate, enriched_path),
        ):
            if not candidate.is_file() or _sha256(candidate) != _sha256(published):
                raise RuntimeError(
                    f"Release candidate differs from published dataset: {candidate}"
                )

        reusable, reason = validate_local_checkpoint(
            checkpoint_path=checkpoint_path,
            current_config=config_fp,
            current_input=input_fp,
            output_paths=output_paths,
        )

        for candidate, published in dictionaries:
            if not candidate.is_file() or _sha256(candidate) != _sha256(published):
                raise RuntimeError(
                    f"Release dictionary differs from published dictionary: {candidate}"
                )

        if reusable:
            logger.info("REUSE — Publication local checkpoint matches.")
        else:
            logger.info(
                "REFRESH — Publication checkpoint: %s",
                reason,
            )

            # Publication integrity and current audit were verified above.
            # Refresh the checkpoint without modifying published artifacts.
            write_local_checkpoint(
                checkpoint_path=checkpoint_path,
                step_name="publication",
                run_id=run_id,
                config=config_fp,
                input_fingerprint=input_fp,
                output_paths=output_paths,
            )

            logger.info("Created checkpoint for existing verified publication.")


        logger.info("REUSE — Existing immutable Release publication verified.")

        return {
            "status": "reused",
            "run_id": run_id,
            "release_id": release_id,
            "standard_path": standard_path,
            "enriched_path": enriched_path,
            "output_path": standard_path,
            "standard_metadata_path": metadata_standard,
            "enriched_metadata_path": metadata_enriched,
            "standard_dictionary_path": dictionary_standard,
            "enriched_dictionary_path": dictionary_enriched,
            "audit_path": audit_path,
            "publication_completed": True,
            "duration_seconds": perf_counter() - start,
        }

    for source in (*sources, audit_path):
        if not source.is_file() or source.stat().st_size <= 0:
            raise FileNotFoundError(f"Missing/empty Release publication input: {source}")
    if standard_candidate.resolve() == standard_path.resolve() or enriched_candidate.resolve() == enriched_path.resolve():
        raise ValueError("Candidate and published paths must differ.")
    if existing:
        logger.info(
            "REBUILD — Published Release differs from audited candidates. "
            "Replacing outdated publication."
        )
    if standard_path.parent != enriched_path.parent or standard_path.parent != metadata_standard.parent:
        raise ValueError("Release artifacts must share a publication directory.")
    if audit_path == standard_path or audit_path == enriched_path:
        raise ValueError("Audit artifact cannot be the published dataset.")

    audit_record = _audit_details(audit_path, run_id, release_id)
    # An audit is not valid for candidate files modified after it was written.
    audited_at = audit_path.stat().st_mtime_ns
    if any(
        path.stat().st_mtime_ns > audited_at
        for path in (
            *sources,
            dictionary_standard_candidate,
            dictionary_enriched_candidate,
        )
    ):
        raise RuntimeError("A Release candidate changed after the audit; rerun the audit.")

    output_dir = standard_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_paths: list[Path] = []
    committed: list[Path] = []
    backups: dict[Path, Path] = {}

    with duckdb.connect() as con:
        for path, expected in (
            (standard_candidate, _published_schema(cfg.schema.standard)),
            (enriched_candidate, _published_schema(cfg.schema.enriched)),
        ):
            actual = [
                row[0]
                for row in con.execute(
                    "DESCRIBE SELECT * FROM read_parquet(?)",
                    [str(path)],
                ).fetchall()
            ]

            if actual != expected:
                raise RuntimeError(f"Release schema mismatch: {path}")

    try:
        candidate_info = []
        for source, destination, schema in (
            (standard_candidate, standard_path, _published_schema(cfg.schema.standard)),
            (enriched_candidate, enriched_path, _published_schema(cfg.schema.enriched)),
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

        dictionary_info = []

        for source, destination in dictionaries:

            if not source.is_file() or source.stat().st_size <= 0:
                raise FileNotFoundError(f"Missing Release data dictionary: {source}")

            temporary = output_dir / f".{destination.name}.{uuid4().hex}.tmp"
            temp_paths.append(temporary)

            with source.open("rb") as reader, temporary.open("xb") as writer:
                shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
                writer.flush()
                os.fsync(writer.fileno())

            if _sha256(source) != _sha256(temporary):
                raise RuntimeError(f"Dictionary changed during copy: {source}")

            dictionary_info.append({
                "file": destination.name,
                "sha256": _sha256(temporary),
                "size_bytes": temporary.stat().st_size,
            })

        if existing:
            for destination in targets:
                backup = output_dir / f".{destination.name}.{uuid4().hex}.bak"
                shutil.copy2(destination, backup)
                backups[destination] = backup

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
                "data_dictionary": dictionary_info[0 if kind == "standard" else 1],
            }
            _atomic_json(metadata_path, payload)
            committed.append(metadata_path)

        # Commit datasets after metadata sidecars are ready. If either commit
        # fails, remove only artifacts created in this attempt.
        for temporary, destination in zip(
                temp_paths,
                (
                    standard_path,
                    enriched_path,
                    dictionary_standard,
                    dictionary_enriched,
                ),
            ):
            os.replace(temporary, destination)
            committed.append(destination)

        write_local_checkpoint(
            checkpoint_path=checkpoint_path,
            step_name="publication",
            run_id=run_id,
            config=config_fp,
            input_fingerprint=input_fp,
            output_paths=output_paths,
        )

        committed.append(checkpoint_path)

        logger.info("Published standard Release dataset: %s", standard_path)
        logger.info("Published enriched Release dataset: %s", enriched_path)
        logger.info("Published standard Release metadata: %s", metadata_standard)
        logger.info("Published enriched Release metadata: %s", metadata_enriched)
        logger.info("Published standard Release data dictionary: %s", dictionary_standard)
        logger.info("Published enriched Release data dictionary: %s", dictionary_enriched)
        logger.info("Publication audit: %s", audit_path)
        return {
            "status": "completed", "run_id": run_id, "release_id": release_id,
            "standard_path": standard_path, "enriched_path": enriched_path,
            "output_path": standard_path,
            "standard_metadata_path": metadata_standard,
            "enriched_metadata_path": metadata_enriched,
            "standard_dictionary_path": dictionary_standard,
            "enriched_dictionary_path": dictionary_enriched,
            "audit_path": audit_path,
            "publication_completed": True,
            "duration_seconds": perf_counter() - start,
        }
    except Exception:
        for committed_path in reversed(committed):
            committed_path.unlink(missing_ok=True)

        for destination, backup in backups.items():
            os.replace(backup, destination)

        raise

    finally:
        for temporary in temp_paths:
            temporary.unlink(missing_ok=True)

        for backup in backups.values():
            backup.unlink(missing_ok=True)
