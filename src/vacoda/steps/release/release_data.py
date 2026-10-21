"""VACODA Release candidate construction (no publication or approval).

Combine canonical Validation Prepared and Inference on bilateral keys, write
standard and enriched Parquet candidates to the workflow staging directory.
Release audits and publication are separate workflow steps.
"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import duckdb

from vacoda.config.steps.release import ReleaseConfig
from vacoda.utils.sql import sql_literal
from vacoda.pipeline.checkpoint import (
    file_fingerprint,
    semantic_config_fingerprint,
    validate_local_checkpoint,
    write_local_checkpoint,
)

KEY = ("year", "exporter_iso", "importer_iso", "target_cmd")
PREPARED_EXTRA = {
    "export_value_fob", "import_value_cif", "import_value_fob_corrected",
    "export_synchronized_quantity_kg", "import_synchronized_quantity_kg",
    "export_synchronized_quantity_source", "import_synchronized_quantity_source",
    "pair_uses_estimated_weight", "pair_uses_weight_proxy",
    "trade_value_reconciled_constant_usd", "cif_fob_factor_applied",
    "cif_fob_factor_applied_source", "reconciliation_status",
    "reconciliation_method_applied", "trade_synced_quantity_method",
    "trade_synced_quantity_status", "quantity_availability_case",
    "product_year_median_unit_value", "product_year_log_unit_value_mad",
    "unit_value_reference_observations",
}
INFERENCE_EXTRA = {
    "training_selected", "model_input_valid", "quantity_log_residual",
    "quantity_abs_log_residual", "exporter_seen", "importer_seen",
    "hs6_seen", "hs4_seen", "validation_model_name", "validation_model_type",
}
DERIVED = {
    "trade_sync_uv_usd_per_kg", "model_uv_usd_per_kg",
    "trade_sync_uv_mad_distance", "model_uv_mad_distance",
}
PREPARED_FIELDS = {
    "trade_value_reconciled", "trade_synced_quantity_kg", *PREPARED_EXTRA,
}
INFERENCE_FIELDS = {
    "model_prediction_quantity_kg", "model_prediction_available",
    "manual_review_required", *INFERENCE_EXTRA,
}


def _columns(con: duckdb.DuckDBPyConnection, path: Path) -> set[str]:
    return {
        str(row[0]) for row in con.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]
        ).fetchall()
    }


def _check_keys(con: duckdb.DuckDBPyConnection, table: str) -> int:
    keys = ", ".join(KEY)
    total, missing, duplicates = con.execute(f"""
        SELECT COUNT(*),
               COUNT(*) FILTER (WHERE {" OR ".join(k + ' IS NULL' for k in KEY)}),
               COUNT(*) - COUNT(DISTINCT ({keys}))
        FROM {table}
    """).fetchone()
    if not total or missing or duplicates:
        raise RuntimeError(
            f"Release {table} key check failed: rows={total}, "
            f"missing_keys={missing}, duplicate_keys={duplicates}"
        )
    return int(total)


def build_release_candidates(
    *,
    cfg: ReleaseConfig,
    prepared_input_path: Path,
    inference_input_path: Path,
    staging_dir: Path,
    run_id: str,
    checkpoint_dir: Path,
    force: bool = False,
    logger,
) -> dict[str, Any]:
    """Build both unpublished candidate datasets; return paths for the audit.

    Both source datasets must represent the same canonical population. This
    function never writes to cfg.output_dir()/release_id and never approves a
    release. Existing candidate paths require force=True to replace.
    """
    start = perf_counter()
    prepared_path = Path(prepared_input_path)
    inference_path = Path(inference_input_path)
    staging_dir = Path(staging_dir)
    for label, path in (("prepared", prepared_path), ("inference", inference_path)):
        if not path.is_file():
            raise FileNotFoundError(f"Validation {label} input not found: {path}")

    standard_cols = tuple(cfg.schema.standard)
    enriched_cols = tuple(cfg.schema.enriched)
    if len(standard_cols) != 9 or len(enriched_cols) != 43:
        raise ValueError("Release schema must contain 9 standard and 43 enriched columns.")
    available = set(KEY) | PREPARED_FIELDS | INFERENCE_FIELDS | DERIVED
    unknown = set(enriched_cols) - available
    if unknown:
        raise ValueError(f"Release schema has unmapped columns: {sorted(unknown)}")

    filename = cfg.products.release.filename
    standard_path = staging_dir / filename
    enriched_path = standard_path.with_name(
        f"{standard_path.stem}_enriched{standard_path.suffix}"
    )

    # ==============================================================
    # LOCAL CHECKPOINT — RELEASE CANDIDATES
    # ==============================================================

    checkpoint_path = checkpoint_dir / "candidates.json"

    config_fp = semantic_config_fingerprint(
        config={
            "schema_standard": standard_cols,
            "schema_enriched": enriched_cols,
            "candidate_filename": filename,
            "candidate_builder_version": 1,
        }
    )

    input_fp = {
        "prepared": file_fingerprint(
            prepared_path,
            hash_file=True,
        ),
        "inference": file_fingerprint(
            inference_path,
            hash_file=True,
        ),
    }

    output_paths = {
        "standard": standard_path,
        "enriched": enriched_path,
    }

    reusable, reason = validate_local_checkpoint(
        checkpoint_path=checkpoint_path,
        current_config=config_fp,
        current_input=input_fp,
        output_paths=output_paths,
    )

    if reusable and not force:
        logger.info("REUSE — Release candidate checkpoint matches.")

        return {
            "status": "reused",
            "run_id": run_id,
            "candidate_standard_path": standard_path,
            "candidate_enriched_path": enriched_path,
            "duration_seconds": perf_counter() - start,
        }

    logger.info(
        "REBUILD — Release candidates: %s",
        "forced" if force else reason,
    )

    for path in (standard_path, enriched_path):
        if path.exists() and not force:
            raise FileExistsError(f"Release candidate exists (force required): {path}")
    staging_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    try:
        con.execute(f"PRAGMA threads={int(cfg.engine.threads)}")
        prepared_cols = _columns(con, prepared_path)
        inference_cols = _columns(con, inference_path)
        required_prepared = set(KEY) | PREPARED_FIELDS
        required_inference = set(KEY) | INFERENCE_FIELDS | {"trade_synced_quantity_kg"}
        for label, missing in (
            ("prepared", required_prepared - prepared_cols),
            ("inference", required_inference - inference_cols),
        ):
            if missing:
                raise ValueError(f"Validation {label} missing columns: {sorted(missing)}")

        con.execute(f"CREATE TEMP VIEW p AS SELECT * FROM read_parquet({sql_literal(prepared_path)})")
        con.execute(f"CREATE TEMP VIEW i AS SELECT * FROM read_parquet({sql_literal(inference_path)})")
        prepared_rows = _check_keys(con, "p")
        inference_rows = _check_keys(con, "i")
        if prepared_rows != inference_rows:
            raise RuntimeError(
                f"Validation populations differ: Prepared={prepared_rows}, "
                f"Inference={inference_rows}"
            )
        join_clause = " AND ".join(f"p.{key} = i.{key}" for key in KEY)
        unmatched = con.execute(f"""
            SELECT COUNT(*) FROM p
            ANTI JOIN i ON {join_clause}
        """).fetchone()[0]
        if unmatched:
            raise RuntimeError(f"Inference is missing {unmatched} Prepared keys")
        quantity_conflicts = con.execute(f"""
            SELECT COUNT(*) FROM p JOIN i ON {join_clause}
            WHERE p.trade_synced_quantity_kg
                  IS DISTINCT FROM i.trade_synced_quantity_kg
        """).fetchone()[0]
        if quantity_conflicts:
            raise RuntimeError(
                f"Prepared/Inference quantity disagreement: {quantity_conflicts} rows"
            )
        review_mismatches = con.execute("""
            SELECT COUNT(*) FROM i
            WHERE manual_review_required IS DISTINCT FROM
                  (hs6_seen IS FALSE AND hs4_seen IS FALSE)
        """).fetchone()[0]
        if review_mismatches:
            raise RuntimeError(
                f"Inference manual-review flag mismatches: {review_mismatches}"
            )

        # All four UV diagnostics use nominal reconciled USD / kg. The
        # reference's value basis must be confirmed to match before publication.
        value = "p.trade_value_reconciled"
        qsync = "p.trade_synced_quantity_kg"
        qmodel = "i.model_prediction_quantity_kg"
        median = "p.product_year_median_unit_value"
        mad = "p.product_year_log_unit_value_mad"

        def uv_expression(quantity: str) -> str:
            return (f"CASE WHEN {value} > 0 AND {quantity} > 0 "
                    f"AND isfinite({value}) AND isfinite({quantity}) "
                    f"THEN {value} / {quantity} ELSE NULL END")

        def mad_expression(quantity: str) -> str:
            return (f"CASE WHEN {value} > 0 AND {quantity} > 0 "
                    f"AND {median} > 0 AND {mad} > 0 "
                    f"AND isfinite({value}) AND isfinite({quantity}) "
                    f"AND isfinite({median}) AND isfinite({mad}) "
                    f"THEN ABS(LN({value}/{quantity}) - LN({median}))/{mad} "
                    f"ELSE NULL END")

        expression = {
            **{key: f'p."{key}"' for key in KEY},
            **{key: f'p."{key}"' for key in PREPARED_FIELDS},
            **{key: f'i."{key}"' for key in INFERENCE_FIELDS},
            "trade_sync_uv_usd_per_kg": uv_expression(qsync),
            "model_uv_usd_per_kg": uv_expression(qmodel),
            "trade_sync_uv_mad_distance": mad_expression(qsync),
            "model_uv_mad_distance": mad_expression(qmodel),
        }
        fields = ",\n".join(f'{expression[key]} AS "{key}"' for key in enriched_cols)
        con.execute(f"""
            CREATE TEMP VIEW release_evidence AS
            SELECT {fields} FROM p JOIN i ON {join_clause}
        """)
        preview_count = con.execute(
            "SELECT COUNT(*) FROM release_evidence"
        ).fetchone()[0]
        if preview_count != prepared_rows:
            raise RuntimeError("Candidate population does not match Prepared input")

        select_standard = ", ".join(
            ('CASE WHEN manual_review_required IS TRUE THEN NULL '
             'ELSE model_prediction_quantity_kg END AS model_prediction_quantity_kg')
            if col == "model_prediction_quantity_kg" else f'"{col}"'
            for col in standard_cols
        )
        select_enriched = ", ".join(f'"{col}"' for col in enriched_cols)
        temporary = []
        try:
            for path, sql in (
                (standard_path, select_standard),
                (enriched_path, select_enriched),
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_name(path.name + ".tmp")
                temporary.append(tmp)
                tmp.unlink(missing_ok=True)
                con.execute(
                    f"COPY (SELECT {sql} FROM release_evidence) "
                    f"TO {sql_literal(tmp)} (FORMAT PARQUET, COMPRESSION ZSTD)"
                )
                rows = con.execute(
                    "SELECT COUNT(*) FROM read_parquet(?)", [str(tmp)]
                ).fetchone()[0]
                if rows != prepared_rows:
                    raise RuntimeError(f"Candidate row count mismatch: {path}")
            for tmp, path in zip(temporary, (standard_path, enriched_path)):
                tmp.replace(path)
        finally:
            for tmp in temporary:
                tmp.unlink(missing_ok=True)

        review_rows, review_codes = con.execute("""
            SELECT COUNT(*) FILTER (WHERE manual_review_required IS TRUE),
                   COUNT(DISTINCT target_cmd) FILTER
                     (WHERE manual_review_required IS TRUE)
            FROM release_evidence
        """).fetchone()

        # ==============================================================
        # WRITE LOCAL CHECKPOINT
        # ==============================================================

        write_local_checkpoint(
            checkpoint_path=checkpoint_path,
            step_name="release_candidates",
            run_id=run_id,
            config=config_fp,
            input_fingerprint=input_fp,
            output_paths=output_paths,
        )

        logger.info("Release candidates built: rows=%s, manual-review=%s (%s HS6 codes)",
            f"{prepared_rows:,}", f"{review_rows:,}", f"{review_codes:,}",
        )
        logger.info("  Candidate standard: %s", standard_path)
        logger.info("  Candidate enriched: %s", enriched_path)
        return {
            "status": "completed",
            "run_id": run_id,
            "candidate_standard_path": standard_path,
            "candidate_enriched_path": enriched_path,
            "n_rows": prepared_rows,
            "n_manual_review_rows": int(review_rows),
            "n_manual_review_hs6_codes": int(review_codes),
            "duration_seconds": perf_counter() - start,
        }
    finally:
        con.close()
