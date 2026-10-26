"""Descriptive diagnostics for published VACODA Release datasets.
Step 4 of Release. These summaries are informational only and never
authorize publication or change the published datasets.
"""
from __future__ import annotations
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping
import duckdb
from vacoda.config.steps.release import ReleaseConfig
from vacoda.utils.sql import sql_literal
from vacoda.pipeline.checkpoint import (
    file_fingerprint,
    semantic_config_fingerprint,
    validate_local_checkpoint,
    write_local_checkpoint,
)
from vacoda.steps.release.release_data import PUBLIC_COLUMN_NAMES

# Sidecars are intentionally separate from the original overall/year table so
# existing report consumers remain compatible. No per-run counts are embedded.

_STAGE_ORDER = (
    "ingestion",
    "hs_sync",
    "quantity",
    "cif_fob",
    "trade_sync",
    "value",
    "validation_prepared",
    "validation_inference",
    "release_enriched",
)

def _sidecar_paths(main_path: Path) -> dict[str, Path]:
    names = (
        "journey",
        "availability",
        "review",
        "missing_reasons",
        "distributions",
        "annual_trade",
        "country_trade",
        "product_trade",
        "feature_distributions",
    )
    return {
        name: main_path.with_name(f"{main_path.stem}_{name}.parquet")
        for name in names
    }


def _build_exploration_sidecars(
    con: duckdb.DuckDBPyConnection,
    sidecar_paths: dict[str, Path],
    quantity_col: str,
) -> dict[str, Path]:
    """Aggregate the published enriched dataset for Release exploration."""
    temps = {}
    quantity_sql = '"' + quantity_col.replace('"', '""') + '"'
    valid_value = """
        trade_value_reconciled_constant_usd > 0
        AND isfinite(trade_value_reconciled_constant_usd)
    """
    valid_quantity = f"""
        {quantity_sql} > 0
        AND isfinite({quantity_sql})
    """
    valid_prediction = """
        model_prediction_quantity_kg > 0
        AND isfinite(model_prediction_quantity_kg)
    """

    # 1. Annual trade population and constant-price values.
    annual_sql = f"""
        SELECT
            year,
            COUNT(*) AS rows,
            COUNT(*) FILTER (WHERE {valid_value}) AS usable_value_rows,
            COUNT(*) FILTER (WHERE {valid_quantity}) AS usable_quantity_rows,
            COUNT(*) FILTER (WHERE {valid_prediction}) AS prediction_rows,
            SUM(trade_value_reconciled_constant_usd)
                FILTER (WHERE {valid_value})
                AS total_trade_value_constant_usd,
            MEDIAN(trade_value_reconciled_constant_usd)
                FILTER (WHERE {valid_value})
                AS median_trade_value_constant_usd,
            MEDIAN({quantity_sql})
                FILTER (WHERE {valid_quantity})
                AS median_trade_synced_quantity_kg,
            MEDIAN(model_prediction_quantity_kg)
                FILTER (WHERE {valid_prediction})
                AS median_model_prediction_quantity_kg
        FROM enriched_release
        GROUP BY year
        ORDER BY year
    """
    temps["annual_trade"] = _write_frame(
        con, annual_sql, sidecar_paths["annual_trade"]
    )

    # 2. Country totals, separately for exporters and importers.
    country_sql = f"""
        WITH country_totals AS (
            SELECT
                'exporter' AS role,
                exporter_iso AS iso3,
                COUNT(*) AS rows,
                SUM(trade_value_reconciled_constant_usd)
                    AS total_trade_value_constant_usd
            FROM enriched_release
            WHERE {valid_value}
            GROUP BY exporter_iso

            UNION ALL

            SELECT
                'importer' AS role,
                importer_iso AS iso3,
                COUNT(*) AS rows,
                SUM(trade_value_reconciled_constant_usd)
                    AS total_trade_value_constant_usd
            FROM enriched_release
            WHERE {valid_value}
            GROUP BY importer_iso
        )
        SELECT
            role,
            iso3,
            rows,
            total_trade_value_constant_usd,
            100.0 * total_trade_value_constant_usd
                / NULLIF(
                    SUM(total_trade_value_constant_usd)
                        OVER (PARTITION BY role),
                    0
                ) AS share_of_total_value_pct
        FROM country_totals
        ORDER BY role, total_trade_value_constant_usd DESC
    """
    temps["country_trade"] = _write_frame(
        con, country_sql, sidecar_paths["country_trade"]
    )

    # 3. HS2 chapter and HS6 product composition.
    product_sql = f"""
        WITH product_totals AS (

            SELECT
                'hs4' AS level,
                SUBSTR(CAST(target_cmd AS VARCHAR), 1, 4) AS product_code,
                COUNT(*) AS rows,
                SUM(trade_value_reconciled_constant_usd)
                    AS total_trade_value_constant_usd
            FROM enriched_release
            WHERE trade_value_reconciled_constant_usd > 0
            AND isfinite(trade_value_reconciled_constant_usd)
            GROUP BY 2

            UNION ALL
            
            SELECT
                'hs2' AS level,
                SUBSTR(CAST(target_cmd AS VARCHAR), 1, 2)
                    AS product_code,
                COUNT(*) AS rows,
                SUM(trade_value_reconciled_constant_usd)
                    AS total_trade_value_constant_usd
            FROM enriched_release
            WHERE {valid_value}
            GROUP BY 2

            UNION ALL

            SELECT
                'hs6' AS level,
                CAST(target_cmd AS VARCHAR) AS product_code,
                COUNT(*) AS rows,
                SUM(trade_value_reconciled_constant_usd)
                    AS total_trade_value_constant_usd
            FROM enriched_release
            WHERE {valid_value}
            GROUP BY 2
        )
        SELECT
            level,
            product_code,
            rows,
            total_trade_value_constant_usd,
            100.0 * total_trade_value_constant_usd
                / NULLIF(
                    SUM(total_trade_value_constant_usd)
                        OVER (PARTITION BY level),
                    0
                ) AS share_of_total_value_pct
        FROM product_totals
        ORDER BY level, total_trade_value_constant_usd DESC
    """
    temps["product_trade"] = _write_frame(
        con, product_sql, sidecar_paths["product_trade"]
    )

    # 4. Distribution quantiles across published observations.
    measures = (
        "trade_value_reconciled_constant_usd",
        quantity_col,
        "model_prediction_quantity_kg",
        "trade_sync_uv_usd_per_kg",
        "model_uv_usd_per_kg",
        "quantity_abs_log_residual",
    )
    available_columns = {
        row[0] for row in con.execute(
            "DESCRIBE SELECT * FROM enriched_release"
        ).fetchall()
    }
    missing_columns = set(measures) - available_columns
    if missing_columns:
        raise RuntimeError(
            f"Release exploration missing columns: {sorted(missing_columns)}"
        )

    distribution_queries = []
    for measure in measures:
        value = f"TRY_CAST({measure} AS DOUBLE)"
        comparison = ">=" if measure == "quantity_abs_log_residual" else ">"
        condition = f"{value} {comparison} 0 AND isfinite({value})"
        distribution_queries.append(f"""
            SELECT
                '{measure}' AS measure,
                COUNT(*) AS usable_rows,
                MIN({value}) AS minimum,
                quantile_cont({value}, 0.01) AS p01,
                quantile_cont({value}, 0.05) AS p05,
                quantile_cont({value}, 0.10) AS p10,
                quantile_cont({value}, 0.25) AS p25,
                quantile_cont({value}, 0.50) AS p50,
                quantile_cont({value}, 0.75) AS p75,
                quantile_cont({value}, 0.90) AS p90,
                quantile_cont({value}, 0.95) AS p95,
                quantile_cont({value}, 0.99) AS p99,
                MAX({value}) AS maximum
            FROM enriched_release
            WHERE {condition}
        """)

    distribution_sql = " UNION ALL ".join(distribution_queries)
    temps["feature_distributions"] = _write_frame(
        con, distribution_sql, sidecar_paths["feature_distributions"]
    )
    return temps


def _schema(con: duckdb.DuckDBPyConnection, path: Path) -> set[str]:
    return {
        row[0]
        for row in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet({sql_literal(path)})"
        ).fetchall()
    }

def _release_view_sql(
    path: Path,
    columns: set[str],
    public_column_names: dict[str, str],
) -> str:
    aliases = [
        f'"{public}" AS "{internal}"'
        for internal, public in public_column_names.items()
        if public in columns and internal != public
    ]

    select = "*, " + ", ".join(aliases) if aliases else "*"

    return f"SELECT {select} FROM read_parquet({sql_literal(path)})"

def _write_frame(con, sql: str, destination: Path) -> Path:
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.unlink(missing_ok=True)
    con.execute(
        f"COPY ({sql}) TO {sql_literal(temporary)} "
        "(FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    return temporary


def _build_sidecars(
    con: duckdb.DuckDBPyConnection,
    *,
    output_path: Path,
    sidecar_paths: dict[str, Path],
    source_paths: Mapping[str, Path],
    enriched_path: Path,
    quantity_col: str,
) -> dict[str, Path]:
    """Return temporary paths; caller promotes only after every query succeeds.
    Optional source_paths keys:
      ingestion, hs_sync, quantity, cif_fob, trade_sync, value,
      validation_prepared, validation_inference.
    These are actual source artifacts, not file paths reconstructed here.
    Missing optional stages are omitted and explicitly reported by stage_status.
    """
    unknown = set(source_paths) - set(_STAGE_ORDER)
    if unknown:
        raise ValueError(f"Unknown release diagnostic source stage(s): {sorted(unknown)}")
    for stage, path in source_paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"{stage} source missing or empty: {path}")
    sources = dict(source_paths)
    sources["release_enriched"] = enriched_path
    temps: dict[str, Path] = {}
    # 1. Real stage row counts, including the observation-grain boundary.
    # A stage may be absent if the workflow has not supplied that artifact.
    journey_selects = []
    for ordinal, stage in enumerate(_STAGE_ORDER):
        path = sources.get(stage)
        if path is None:
            journey_selects.append(
                f"SELECT {ordinal} AS stage_order, '{stage}' AS stage, "
                "CAST(NULL AS BIGINT) AS rows, 'not_provided' AS stage_status"
            )
        else:
            journey_selects.append(
                f"SELECT {ordinal} AS stage_order, '{stage}' AS stage, "
                f"COUNT(*) AS rows, 'read' AS stage_status "
                f"FROM read_parquet({sql_literal(path)})"
            )
    temps["journey"] = _write_frame(
        con, " UNION ALL ".join(journey_selects), sidecar_paths["journey"]
    )
    # 2. Availability: shared definition means positive finite quantity/value,
    # but 'prediction available' flag can differ from published non-null output.
    # Optional fields are omitted rather than confused with zero counts.
    specs = {
        "quantity": ("synchronized_quantity_kg",),
        "cif_fob": ("export_value_fob", "import_value_fob_corrected"),
        "trade_sync": ("trade_value_reconciled", quantity_col),
        "value": ("trade_value_reconciled", "trade_value_reconciled_constant_usd"),
        "validation_prepared": ("trade_value_reconciled_constant_usd", quantity_col),
        "validation_inference": ("model_prediction_quantity_kg",),
        "release_enriched": (
            "trade_value_reconciled", 
            "trade_value_reconciled_constant_usd",
            quantity_col, 
            "model_prediction_quantity_kg",
        ),
    }
    selects = []
    for stage, fields in specs.items():
        path = sources.get(stage)
        if path is None:
            continue
        columns = (
                {row[0] for row in con.execute(
                    "DESCRIBE SELECT * FROM enriched_release"
                ).fetchall()}
                if stage == "release_enriched"
                else _schema(con, path)
            )
        for field in fields:
            if field not in columns:
                continue
            check = f"TRY_CAST({field} AS DOUBLE)"
            usable = f"({check} > 0 AND isfinite({check}))"
            selects.append(f"""
                SELECT '{stage}' AS stage, '{field}' AS field,
                       COUNT(*) AS total_rows,
                       COUNT(*) FILTER (WHERE {field} IS NULL) AS missing,
                       COUNT(*) FILTER (WHERE {field} IS NOT NULL
                          AND NOT {usable}) AS nonpositive_or_invalid,
                       COUNT(*) FILTER (WHERE {usable}) AS usable,
                       ROUND(100.0 * COUNT(*) FILTER (WHERE {usable}) /
                             NULLIF(COUNT(*), 0), 3) AS coverage_pct
                FROM {"enriched_release" if stage == "release_enriched" else f"read_parquet({sql_literal(path)})"}
            """)
    if not selects:
        raise RuntimeError("No source fields available for Release coverage diagnostics")
    temps["availability"] = _write_frame(
        con, " UNION ALL ".join(selects), sidecar_paths["availability"]
    )
    # 3. Unseen-category review records: discover codes and years from the
    # published enriched dataset. NULL seen-flags stay distinct from FALSE.
    ecols = {
            row[0]
            for row in con.execute(
                "DESCRIBE SELECT * FROM enriched_release"
            ).fetchall()
        }
    review_required = {
        "target_cmd", "year", "manual_review_required", "hs6_seen",
        "hs4_seen", "model_prediction_quantity_kg",
    }
    missing = review_required - ecols
    if missing:
        raise RuntimeError(f"Release enriched missing review fields: {sorted(missing)}")
    temps["review"] = _write_frame(con, """
        SELECT CAST(target_cmd AS VARCHAR) AS target_cmd,
               MIN(year) AS first_year, MAX(year) AS last_year,
               COUNT(*) AS review_rows,
               COUNT(*) FILTER (WHERE model_prediction_quantity_kg IS NOT NULL)
                   AS raw_prediction_rows,
               COUNT(*) FILTER (WHERE hs6_seen IS FALSE) AS hs6_unseen_rows,
               COUNT(*) FILTER (WHERE hs4_seen IS FALSE) AS hs4_unseen_rows
        FROM enriched_release
        WHERE manual_review_required IS TRUE
        GROUP BY 1 ORDER BY review_rows DESC, target_cmd
    """, sidecar_paths["review"])
    # 4. Explanations: derive cases from source-side CIF/FOB values if supplied.
    # This classifies every missing reconciled value by available side evidence.
    # Counts are not the same thing as proof of source causation. Expose
    # 'source_not_provided' explicitly rather than inventing a reason.
    cif_path = source_paths.get("cif_fob")
    cols = _schema(con, cif_path) if cif_path is not None else set()
    if {"export_value_fob", "import_value_fob_corrected"}.issubset(cols):
        # Key join assures that status is attributed to the SAME bilateral row.
        keys = {"year", "exporter_iso", "importer_iso", "target_cmd"}
        if not keys.issubset(cols) or not keys.issubset(ecols):
            raise RuntimeError("CIF/FOB and Release need bilateral identity columns")
        path_sql = sql_literal(cif_path)
        factor_col = "estimated_cif_fob_factor" if "estimated_cif_fob_factor" in cols else None
        # Factor explanation is not inferred from an arbitrarily named column.
        factor_case = (
            f"WHEN c.{factor_col} IS NULL THEN 'no_estimated_factor' "
            if factor_col else ""
        )
        reasons_query = f"""
            WITH missing AS (
                SELECT
                    e.year,
                    e.exporter_iso,
                    e.importer_iso,
                    e.target_cmd,
                    c.year AS matched_source_year,
                    c.export_value_fob,
                    c.import_value_fob_corrected
                FROM enriched_release e
                LEFT JOIN read_parquet({path_sql}) c
                USING (year, exporter_iso, importer_iso, target_cmd)
                WHERE e.trade_value_reconciled IS NULL
            ),
            grouped AS (
                SELECT
                    CASE
                        WHEN matched_source_year IS NULL
                            THEN 'source_row_not_matched'
                        WHEN export_value_fob IS NULL
                            AND import_value_fob_corrected IS NULL
                            THEN 'export_missing_and_import_corrected_missing'
                        WHEN export_value_fob IS NULL
                            THEN 'export_missing_import_corrected_present'
                        WHEN export_value_fob <= 0
                            THEN 'export_nonpositive'
                        ELSE 'other_missing_reconciled_value'
                    END AS reason,
                    COUNT(*) AS rows
                FROM missing
                GROUP BY 1
            )
            SELECT reason, rows
            FROM grouped
        """
        # Use LEFT JOIN with missing-side checks in the notebook for exact
        # attribution if source artifacts have changed between runs.
    else:
        reasons_query = """
            SELECT 'source_not_provided' AS reason, COUNT(*) AS rows
            FROM enriched_release
            WHERE trade_value_reconciled IS NULL
        """
    temps["missing_reasons"] = _write_frame(
        con, reasons_query, sidecar_paths["missing_reasons"]
    )
    # 5. Compact long-form quantiles for plots; positive finite values only.
    distribution_fields = (
        "trade_value_reconciled_constant_usd", 
        quantity_col,
        "model_prediction_quantity_kg", 
        "trade_sync_uv_usd_per_kg",
        "model_uv_usd_per_kg", 
        "quantity_abs_log_residual",
    )
    dist = []
    for field in distribution_fields:
        if field not in ecols:
            continue
        value = f"TRY_CAST({field} AS DOUBLE)"
        dist.append(f"""
            SELECT '{field}' AS field, CASE WHEN GROUPING(year)=1
                  THEN 'all_years' ELSE 'year' END AS period_type,
                   CASE WHEN GROUPING(year)=1 THEN NULL ELSE year END AS year,
                   COUNT(*) AS usable_rows,
                   MIN({value}) AS minimum,
                   quantile_cont({value}, 0.05) AS p05,
                   quantile_cont({value}, 0.25) AS p25,
                   MEDIAN({value}) AS median,
                   quantile_cont({value}, 0.75) AS p75,
                   quantile_cont({value}, 0.95) AS p95,
                   MAX({value}) AS maximum
            FROM enriched_release
            WHERE {value} {">=" if field == "quantity_abs_log_residual" else ">"} 0
                AND isfinite({value})
            GROUP BY GROUPING SETS ((year), ())
        """)
    if not dist:
        raise RuntimeError("Release enriched has no distribution fields")

    temps["distributions"] = _write_frame(
        con, " UNION ALL ".join(dist), sidecar_paths["distributions"]
    )

    temps.update(
        _build_exploration_sidecars(
            con=con,
            sidecar_paths=sidecar_paths,
            quantity_col=quantity_col,
        )
    )
    return temps


def build_release_diagnostics(
    *,
    cfg: ReleaseConfig,
    quantity_col: str,
    standard_path: Path,
    enriched_path: Path,
    audit_path: Path,
    run_id: str,
    release_id: str,
    checkpoint_dir: Path,
    logger,
    source_paths: Mapping[str, Path] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Write Release summaries and compact lineage/availability sidecar Parquets.
    The audit is a provenance input, not an approval decision at this stage:
    approval was enforced earlier by the Release workflow and publisher.
    """
    start = perf_counter()
    public_column_names = {
            **PUBLIC_COLUMN_NAMES,
            quantity_col: "trade_quantity_kg",
        }
    
    standard_path = Path(standard_path)
    enriched_path = Path(enriched_path)
    audit_path = Path(audit_path)
    output_path = cfg.product_path("diagnostics", release_id=release_id)
    sidecar_paths = _sidecar_paths(output_path)
    source_paths = {k: Path(v) for k, v in (source_paths or {}).items()}

    quantity_sql = '"' + quantity_col.replace('"', '""') + '"'
    ## Local Checkpoints
    checkpoint_path = checkpoint_dir / "diagnostics.json"

    output_paths = {
        "main": output_path,
        **_sidecar_paths(output_path),
    }

    config_fp = semantic_config_fingerprint(
        config={
            "quantity_column": quantity_col,
            "release_id": release_id,
            "scope": cfg.scope,
            "code_fingerprint": file_fingerprint(
                Path(__file__),
                hash_file=True,
            ),
        }
    )

    input_fp = {
        "standard": file_fingerprint(standard_path),
        "enriched": file_fingerprint(enriched_path),
        "sources": {
            name: file_fingerprint(path)
            for name, path in sorted(source_paths.items())
        },
    }

    expected_standard = cfg.product_path("release", release_id=release_id)
    expected_enriched = cfg.enriched_path(release_id=release_id)
    expected_audit = cfg.product_path("release_audit", release_id=release_id)
    if (standard_path != expected_standard
            or enriched_path != expected_enriched
            or audit_path != expected_audit):
        raise ValueError("Release diagnostics input paths do not match release identity.")
    for label, path in (
        ("standard dataset", standard_path),
        ("enriched dataset", enriched_path),
        ("Release audit", audit_path),
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Release {label} missing or empty: {path}")


    ## Checkpoint Validation
    reusable, reason = validate_local_checkpoint(
        checkpoint_path=checkpoint_path,
        current_config=config_fp,
        current_input=input_fp,
        output_paths=output_paths,
    )

    if reusable and not force:
        logger.info("REUSE — Release diagnostics checkpoint matches.")

        return {
            "status": "reused",
            "run_id": run_id,
            "release_id": release_id,
            "output_path": output_path,
            "diagnostics_path": output_path,
            **{
                f"{name}_path": path
                for name, path in sidecar_paths.items()
            },
            "duration_seconds": perf_counter() - start,
        }

    logger.info(
        "REBUILD — Release diagnostics: %s",
        "forced" if force else reason,
    )

    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    standard_sql = sql_literal(standard_path)
    enriched_sql = sql_literal(enriched_path)
    audit_sql = sql_literal(audit_path)
    con = duckdb.connect()
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    try:
        con.execute(f"PRAGMA threads={int(cfg.engine.threads)}")
        # Only read the published artifacts; do not materialize the full dataset
        # into pandas, and do not assign publication permissions here.
        audit_row = con.execute(f"""
            SELECT run_id, release_id, audit_passed
            FROM read_parquet({audit_sql})
        """).fetchall()
        if (
            len(audit_row) != 1
            or audit_row[0][1] != release_id
            or audit_row[0][2] is not True
        ):
            raise RuntimeError("Release audit identity/status does not match.")
        con.execute(
            "CREATE TEMP VIEW standard_release AS "
            + _release_view_sql(
                standard_path,
                _schema(con, standard_path),
                public_column_names,
            )
        )

        con.execute(
            "CREATE TEMP VIEW enriched_release AS "
            + _release_view_sql(
                enriched_path,
                _schema(con, enriched_path),
                public_column_names,
            )
        )
        # Two aggregate sets permit descriptive cross-product comparisons
        # without joining millions of underlying bilateral observations.
        query = f"""
            WITH s AS (
                SELECT
                    CASE WHEN GROUPING(year) = 1 THEN 'all_years'
                         ELSE 'year' END AS period_type,
                    CASE WHEN GROUPING(year) = 1 THEN NULL
                         ELSE year END AS year,
                    COUNT(*) AS standard_rows,
                    COUNT(DISTINCT target_cmd) AS hs6_codes,
                    COUNT(*) FILTER (
                        WHERE trade_value_reconciled IS NOT NULL
                    ) AS nominal_value_rows,
                    COUNT(*) FILTER (
                        WHERE {quantity_sql} IS NOT NULL
                    ) AS synced_quantity_rows,
                    COUNT(*) FILTER (
                        WHERE model_prediction_available IS TRUE
                    ) AS model_prediction_available_rows,
                    COUNT(*) FILTER (
                        WHERE model_prediction_quantity_kg IS NOT NULL
                    ) AS standard_prediction_rows,
                    COUNT(*) FILTER (
                        WHERE manual_review_required IS TRUE
                    ) AS manual_review_rows,
                    COUNT(DISTINCT target_cmd) FILTER (
                        WHERE manual_review_required IS TRUE
                    ) AS manual_review_hs6_codes,
                    MEDIAN(trade_value_reconciled) AS median_trade_value_usd,
                    MEDIAN({quantity_sql}) AS median_synced_quantity_kg,
                    MEDIAN(model_prediction_quantity_kg) AS median_standard_prediction_kg
                FROM standard_release
                GROUP BY GROUPING SETS ((year), ())
            ),
            e AS (
                SELECT
                    CASE WHEN GROUPING(year) = 1 THEN 'all_years'
                         ELSE 'year' END AS period_type,
                    CASE WHEN GROUPING(year) = 1 THEN NULL
                         ELSE year END AS year,
                    COUNT(*) AS enriched_rows,
                    COUNT(*) FILTER (
                        WHERE model_prediction_quantity_kg IS NOT NULL
                    ) AS raw_model_prediction_rows,
                    COUNT(*) FILTER (
                        WHERE model_input_valid IS TRUE
                    ) AS model_input_valid_rows,
                    COUNT(*) FILTER (
                        WHERE training_selected IS TRUE
                    ) AS training_selected_rows,
                    COUNT(*) FILTER (
                        WHERE hs6_seen IS FALSE
                    ) AS unseen_hs6_rows,
                    COUNT(*) FILTER (
                        WHERE hs4_seen IS FALSE
                    ) AS unseen_hs4_rows,
                    COUNT(*) FILTER (
                        WHERE hs6_seen IS NULL OR hs4_seen IS NULL
                    ) AS support_not_evaluated_rows,
                    COUNT(*) FILTER (
                        WHERE quantity_abs_log_residual IS NOT NULL
                    ) AS comparable_residual_rows,
                    MEDIAN(quantity_abs_log_residual) AS median_abs_log_residual,
                    MEDIAN(trade_sync_uv_usd_per_kg) AS median_trade_sync_uv_usd_per_kg,
                    MEDIAN(model_uv_usd_per_kg) AS median_model_uv_usd_per_kg,
                    MEDIAN(trade_sync_uv_mad_distance) AS median_trade_sync_uv_mad_distance,
                    MEDIAN(model_uv_mad_distance) AS median_model_uv_mad_distance
                FROM enriched_release
                GROUP BY GROUPING SETS ((year), ())
            )
            SELECT
                s.period_type,
                s.year,
                s.standard_rows,
                e.enriched_rows,
                s.hs6_codes,
                s.nominal_value_rows,
                s.synced_quantity_rows,
                s.model_prediction_available_rows,
                s.standard_prediction_rows,
                e.raw_model_prediction_rows,
                s.manual_review_rows,
                s.manual_review_hs6_codes,
                e.model_input_valid_rows,
                e.training_selected_rows,
                e.unseen_hs6_rows,
                e.unseen_hs4_rows,
                e.support_not_evaluated_rows,
                e.comparable_residual_rows,
                s.median_trade_value_usd,
                s.median_synced_quantity_kg,
                s.median_standard_prediction_kg,
                e.median_abs_log_residual,
                e.median_trade_sync_uv_usd_per_kg,
                e.median_model_uv_usd_per_kg,
                e.median_trade_sync_uv_mad_distance,
                e.median_model_uv_mad_distance
            FROM s JOIN e
              ON s.period_type = e.period_type
             AND s.year IS NOT DISTINCT FROM e.year
            ORDER BY CASE WHEN s.period_type = 'all_years' THEN 1 ELSE 0 END,
                     s.year
        """
        # Create a temporary Parquet before promoting the diagnostics artifact.
        temporary_path.unlink(missing_ok=True)
        con.execute(
            f"COPY ({query}) TO {sql_literal(temporary_path)} "
            "(FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        summary = con.execute(f"""
            SELECT standard_rows, enriched_rows, manual_review_rows,
                   manual_review_hs6_codes, standard_prediction_rows,
                   raw_model_prediction_rows
            FROM read_parquet({sql_literal(temporary_path)})
            WHERE period_type = 'all_years'
        """).fetchall()
        if len(summary) != 1:
            raise RuntimeError("Release diagnostics missing all-years summary.")
        (n_standard, n_enriched, n_review, n_review_codes,
         n_standard_prediction, n_raw_prediction) = summary[0]
        if not n_standard or not n_enriched:
            raise RuntimeError("Published Release dataset is empty.")
        sidecar_temp = _build_sidecars(
            con,
            output_path=output_path,
            sidecar_paths=sidecar_paths,
            source_paths=source_paths,
            enriched_path=enriched_path,
            quantity_col=quantity_col,
        )

        for key, path in sidecar_paths.items():
            sidecar_temp[key].replace(path)

        temporary_path.replace(output_path)

        # Write checkpoint only after all diagnostics outputs are finalized.
        write_local_checkpoint(
            checkpoint_path=checkpoint_path,
            step_name="diagnostics",
            run_id=run_id,
            config=config_fp,
            input_fingerprint=input_fp,
            output_paths=output_paths,
        )
        
        logger.info("Release diagnostics: %s", output_path)
        logger.info(
            "  Records: standard=%s enriched=%s", f"{n_standard:,}", f"{n_enriched:,}"
        )
        logger.info(
            "  Manual review: %s rows (%s HS6 codes)",
            f"{n_review:,}", f"{n_review_codes:,}"
        )
        logger.info(
            "  Predictions: standard=%s enriched raw=%s",
            f"{n_standard_prediction:,}", f"{n_raw_prediction:,}"
        )
        return {
            "status": "completed",
            "run_id": run_id,
            "release_id": release_id,
            "output_path": output_path,
            "diagnostics_path": output_path,
            "journey_path": sidecar_paths["journey"],
            "availability_path": sidecar_paths["availability"],
            "annual_trade_path": sidecar_paths["annual_trade"],
            "country_trade_path": sidecar_paths["country_trade"],
            "product_trade_path": sidecar_paths["product_trade"],
            "feature_distributions_path": sidecar_paths["feature_distributions"],
            "review_path": sidecar_paths["review"],
            "missing_reasons_path": sidecar_paths["missing_reasons"],
            "distributions_path": sidecar_paths["distributions"],
            "n_rows": int(n_standard),
            "n_manual_review_rows": int(n_review),
            "n_manual_review_hs6_codes": int(n_review_codes),
            "duration_seconds": perf_counter() - start,
        }
    finally:
        temporary_path.unlink(missing_ok=True)
        con.close()
