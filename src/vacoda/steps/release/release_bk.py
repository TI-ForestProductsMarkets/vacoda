"""VACODA final trade-data release assembly.

Standalone release builder for integration into vacoda.steps.validation.release.
Inputs: Validation Prepared and Validation Inference Parquet artifacts.

Does not adjudicate between Trade-Sync and model quantities.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

KEY = ("year", "exporter_iso", "importer_iso", "target_cmd")
STANDARD = (
    *KEY,
    "trade_value_reconciled",
    "trade_synced_quantity_kg",
    "model_prediction_quantity_kg",
    "model_prediction_available",
    "manual_review_required",
)

# All source fields in this list occur in the provided Validation Prepared schema.
PREPARED_EXTRA = (
    "export_value_fob",
    "import_value_cif",
    "import_value_fob_corrected",
    "export_synchronized_quantity_kg",
    "import_synchronized_quantity_kg",
    "export_synchronized_quantity_source",
    "import_synchronized_quantity_source",
    "pair_uses_estimated_weight",
    "pair_uses_weight_proxy",
    "trade_value_reconciled_constant_usd",
    "cif_fob_factor_applied",
    "cif_fob_factor_applied_source",
    "reconciliation_status",
    "reconciliation_method_applied",
    "trade_synced_quantity_method",
    "trade_synced_quantity_status",
    "quantity_availability_case",
    "product_year_median_unit_value",
    "product_year_log_unit_value_mad",
    "unit_value_reference_observations",
)
DERIVED_EXTRA = (
    "trade_sync_uv_usd_per_kg",
    "model_uv_usd_per_kg",
    "trade_sync_uv_mad_distance",
    "model_uv_mad_distance",
)
INFERENCE_EXTRA = (
    "training_selected",
    "model_input_valid",
    "quantity_log_residual",
    "quantity_abs_log_residual",
    "exporter_seen",
    "importer_seen",
    "hs6_seen",
    "hs4_seen",
    "validation_model_name",
    "validation_model_type",
)
ENRICHED = (*STANDARD, *PREPARED_EXTRA, *DERIVED_EXTRA, *INFERENCE_EXTRA)
assert len(STANDARD) == 9 and len(ENRICHED) == 43


def _columns(con: duckdb.DuckDBPyConnection, path: Path) -> set[str]:
    return {r[0] for r in con.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]).fetchall()}


def _validated_file(path: Path) -> Path:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _source_info(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def build_release_datasets(
    *,
    prepared_path: Path,
    inference_path: Path,
    output_dir: Path,
    target_hs: str,
    run_id: str,
    force: bool = False,
    logger: Any = None,
) -> dict[str, Any]:
    """Write standard/enriched Parquet files and metadata in an explicit output_dir.

    The caller must choose a configuration-specific directory, e.g.
    data/release/target_h5/<full_pipeline_config_id>/.

    Existing outputs are never silently overwritten; pass force=True to replace.
    """
    prepared_path = _validated_file(Path(prepared_path))
    inference_path = _validated_file(Path(inference_path))
    output_dir = Path(output_dir)
    stem = f"vacoda_target_{target_hs.lower()}_dataset"
    targets = {
        "standard": output_dir / f"{stem}.parquet",
        "enriched": output_dir / f"{stem}_enriched.parquet",
    }
    metadata_paths = {name: path.with_suffix(".metadata.json") for name, path in targets.items()}
    existing = [str(p) for p in (*targets.values(), *metadata_paths.values()) if p.exists()]
    if existing and not force:
        raise FileExistsError("Release outputs already exist; use force=True deliberately: " + ", ".join(existing))
    output_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    try:
        required_prepared = set(KEY) | {"trade_value_reconciled", "trade_synced_quantity_kg", *PREPARED_EXTRA}
        required_inference = set(KEY) | {
            "model_prediction_quantity_kg", "model_prediction_available", "manual_review_required",
            "trade_synced_quantity_kg", *INFERENCE_EXTRA,
        }
        for label, path, required in (
            ("prepared", prepared_path, required_prepared),
            ("inference", inference_path, required_inference),
        ):
            missing = required - _columns(con, path)
            if missing:
                raise ValueError(f"{label}: missing required columns {sorted(missing)}")

        def sql_path(path: Path) -> str:
            return "'" + str(path).replace("'", "''") + "'"

        con.execute(f"CREATE TEMP VIEW prepared AS SELECT * FROM read_parquet({sql_path(prepared_path)})")
        con.execute(f"CREATE TEMP VIEW inference AS SELECT * FROM read_parquet({sql_path(inference_path)})")
        key_sql = ", ".join(KEY)
        for label in ("prepared", "inference"):
            count, distinct_count = con.execute(
                f"SELECT COUNT(*), COUNT(DISTINCT ({key_sql})) FROM {label}"
            ).fetchone()
            if count != distinct_count:
                raise ValueError(f"{label} has duplicate canonical trade keys")

        prepared_n = con.execute("SELECT COUNT(*) FROM prepared").fetchone()[0]
        inference_n = con.execute("SELECT COUNT(*) FROM inference").fetchone()[0]
        if prepared_n != inference_n:
            raise ValueError(f"Prepared/inference counts differ: {prepared_n} vs {inference_n}")
        unmatched = con.execute(f"""
            SELECT COUNT(*) FROM prepared p
            LEFT JOIN inference i USING ({key_sql})
            WHERE i.year IS NULL
        """).fetchone()[0]
        if unmatched:
            raise ValueError(f"Inference missing {unmatched} canonical prepared keys")

        conflicts = con.execute(f"""
            SELECT COUNT(*) FROM prepared p JOIN inference i USING ({key_sql})
            WHERE p.trade_synced_quantity_kg IS DISTINCT FROM i.trade_synced_quantity_kg
        """).fetchone()[0]
        if conflicts:
            raise ValueError(f"Trade-Sync quantity differs between prepared and inference: {conflicts} rows")

        # Nullable support indicators are *not* equivalent to unseen categories.
        errors = con.execute("""
            SELECT COUNT(*) FROM inference
            WHERE (hs6_seen IS NOT NULL AND hs4_seen IS NOT NULL
                   AND manual_review_required IS DISTINCT FROM
                       (hs6_seen = FALSE AND hs4_seen = FALSE))
               OR ((hs6_seen IS NULL OR hs4_seen IS NULL)
                   AND manual_review_required IS NOT NULL)
        """).fetchone()[0]
        if errors:
            raise ValueError(f"Manual-review rule mismatch in {errors} inference rows")

        # Only preserve the input trade value; never define a merged 'final' quantity.
        prepared_select = ",\n".join(f'p."{c}"' for c in PREPARED_EXTRA)
        inferred_select = ",\n".join(f'i."{c}"' for c in INFERENCE_EXTRA)
        con.execute(f"""
            CREATE TEMP VIEW release_evidence AS
            SELECT
                {', '.join('p."' + c + '"' for c in KEY)},
                p.trade_value_reconciled,
                p.trade_synced_quantity_kg,
                i.model_prediction_quantity_kg,
                i.model_prediction_available,
                i.manual_review_required,
                {prepared_select},
                CASE WHEN p.trade_value_reconciled > 0
                          AND p.trade_synced_quantity_kg > 0
                     THEN p.trade_value_reconciled / p.trade_synced_quantity_kg
                END AS trade_sync_uv_usd_per_kg,
                CASE WHEN p.trade_value_reconciled > 0
                          AND i.model_prediction_quantity_kg > 0
                     THEN p.trade_value_reconciled / i.model_prediction_quantity_kg
                END AS model_uv_usd_per_kg,
                CASE WHEN p.trade_value_reconciled > 0
                          AND p.trade_synced_quantity_kg > 0
                          AND p.product_year_median_unit_value > 0
                          AND p.product_year_log_unit_value_mad > 0
                     THEN ABS(LN(p.trade_value_reconciled / p.trade_synced_quantity_kg)
                              - LN(p.product_year_median_unit_value))
                          / p.product_year_log_unit_value_mad
                END AS trade_sync_uv_mad_distance,
                CASE WHEN p.trade_value_reconciled > 0
                          AND i.model_prediction_quantity_kg > 0
                          AND p.product_year_median_unit_value > 0
                          AND p.product_year_log_unit_value_mad > 0
                     THEN ABS(LN(p.trade_value_reconciled / i.model_prediction_quantity_kg)
                              - LN(p.product_year_median_unit_value))
                          / p.product_year_log_unit_value_mad
                END AS model_uv_mad_distance,
                {inferred_select}
            FROM prepared p
            INNER JOIN inference i USING ({key_sql})
        """)

        review_summary = con.execute("""
            SELECT target_cmd, COUNT(*) AS n_rows
            FROM release_evidence
            WHERE manual_review_required IS TRUE
            GROUP BY 1 ORDER BY 1
        """).fetchall()
        counts = con.execute("""
            SELECT COUNT(*) AS n_rows,
                   COUNT(*) FILTER (WHERE manual_review_required IS TRUE) AS manual_review_rows,
                   COUNT(*) FILTER (WHERE manual_review_required IS NULL) AS unknown_review_status_rows,
                   COUNT(*) FILTER (WHERE model_prediction_available IS TRUE) AS prediction_generated_rows,
                   COUNT(*) FILTER (WHERE model_prediction_available IS NOT TRUE) AS prediction_unavailable_rows
            FROM release_evidence
        """).fetchone()
        if counts[0] != prepared_n:
            raise ValueError("Release row count does not match prepared input")

        # The standard dataset withholds unreviewed categorical-extrapolation predictions.
        standard_expressions = [f'"{name}"' for name in STANDARD]
        standard_expressions[STANDARD.index("model_prediction_quantity_kg")] = (
            'CASE WHEN manual_review_required IS TRUE THEN NULL '
            'ELSE model_prediction_quantity_kg END AS model_prediction_quantity_kg'
        )
        select_standard = ", ".join(standard_expressions)
        select_enriched = ", ".join(f'"{name}"' for name in ENRICHED)
        temp_paths = {name: path.with_name(path.name + ".tmp") for name, path in targets.items()}
        try:
            for name, query in (
                ("standard", f"SELECT {select_standard} FROM release_evidence"),
                ("enriched", f"SELECT {select_enriched} FROM release_evidence"),
            ):
                temp_paths[name].unlink(missing_ok=True)
                con.execute(
                    f"COPY ({query}) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)",
                    [str(temp_paths[name])],
                )
                actual = con.execute("SELECT COUNT(*) FROM read_parquet(?)", [str(temp_paths[name])]).fetchone()[0]
                if actual != prepared_n:
                    raise ValueError(f"{name} release row-count mismatch")

            # The release metadata is for users; source manifests remain the pipeline truth.
            metadata_common = {
                "release_schema_version": "1.0",
                "target_hs": target_hs,
                "run_id": run_id,
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "row_key": list(KEY),
                "n_rows": int(prepared_n),
                "sources": {
                    "validation_prepared": _source_info(prepared_path),
                    "inference": _source_info(inference_path),
                },
                "interpretation": {
                    "trade_value_reconciled": "Nominal reconciled trade value, USD.",
                    "trade_value_reconciled_constant_usd": "Inflation-adjusted reconciled value (enriched only).",
                    "trade_synced_quantity_kg": "Reconciled Trade-Sync quantity, kg; not ground truth.",
                    "model_prediction_quantity_kg": "Model-derived quantity, kg; conditional on input data and fitted methodology, not ground truth.",
                    "model_prediction_available": "True when the model generated a prediction, even if standard release suppresses its numeric value.",
                    "manual_review_required": "True exactly when hs6_seen is false and hs4_seen is false. Unknown if categorical support could not be determined.",
                    "unit_value_distance": "Absolute difference of log implied UV and log HS-year median UV, divided by log-UV MAD; diagnostic, not accuracy.",
                },
                "manual_review_rule": "hs6_seen IS FALSE AND hs4_seen IS FALSE",
                "manual_review_rows": int(counts[1]),
                "unknown_review_status_rows": int(counts[2]),
                "prediction_generated_rows": int(counts[3]),
                "prediction_unavailable_rows": int(counts[4]),
                "manual_review_hs6_counts": [
                    {"target_cmd": str(hs), "n_rows": int(n)} for hs, n in review_summary
                ],
                "quantity_adjudication": "none: the two estimators remain distinct",
                "hs_harmonization_lineage": "Source HS mapping detail is not row-level in this release; consult upstream lineage artifacts.",
            }
            temp_meta = {}
            for name, path in metadata_paths.items():
                payload = dict(metadata_common)
                payload["product"] = name
                payload["columns"] = list(STANDARD if name == "standard" else ENRICHED)
                payload["model_prediction_release_policy"] = (
                    "NULL if manual_review_required is TRUE" if name == "standard"
                    else "Raw model prediction retained, including review-required rows"
                )
                temp_meta[name] = path.with_name(path.name + ".tmp")
                temp_meta[name].write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

            for name in targets:
                temp_paths[name].replace(targets[name])
                temp_meta[name].replace(metadata_paths[name])
        finally:
            for path in temp_paths.values():
                path.unlink(missing_ok=True)
        if logger is not None:
            logger.info("VACODA release written: %s rows, %s manual-review", prepared_n, counts[1])
        return {
            "status": "completed", "rows": int(prepared_n),
            "manual_review_rows": int(counts[1]),
            "standard_path": targets["standard"],
            "enriched_path": targets["enriched"],
            "metadata_paths": metadata_paths,
        }
    finally:
        con.close()
