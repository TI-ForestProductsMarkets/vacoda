"""
vacoda.steps.cif_fob.impute
-------------------------
Impute missing CIF/FOB factors for bilateral trade flows using observed direct evidence and eligible mirror pairs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

from vacoda.config.steps.cif_fob import CifFobConfig
from vacoda.pipeline.checkpoint import (
    validate_local_checkpoint,
    file_fingerprint,
    write_local_checkpoint,
    semantic_config_fingerprint,
    audit_summary
)
from vacoda.utils.logger import log_checkpoint_reuse, log_checkpoint_rebuild

KEY = ("year", "exporter_iso", "importer_iso", "target_cmd")


def _sql_path(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def _columns(con: duckdb.DuckDBPyConnection, relation: str) -> set[str]:
    return {row[0] for row in con.execute(f"DESCRIBE {relation}").fetchall()}


def _inspect_existing_imputation(*, output_path: Path, audit_path: Path, threads: int) -> dict[str, Any]:
    """Revalidate a checkpointed product rather than trusting the file name."""
    con = duckdb.connect()
    try:
        con.execute(f"SET threads = {int(threads)}")
        row = con.execute("""
            SELECT COUNT(*),
                   COUNT(DISTINCT (year, exporter_iso, importer_iso, target_cmd)),
                   COUNT(*) FILTER (WHERE factor_source = 'imputed'),
                   COUNT(*) FILTER (
                       WHERE factor_source = 'imputed' AND (
                           imputation_method IS NULL OR
                           cif_fob_factor_applied IS NULL OR
                           import_value_fob_corrected IS NULL OR
                           import_value_fob_corrected <= 0
                       )
                   )
            FROM read_parquet(?)
        """, [str(output_path)]).fetchone()
    finally:
        con.close()
    if row is None or row[0] != row[1] or row[3] != 0 or not audit_path.is_file():
        raise RuntimeError("Checkpointed CIF/FOB imputation product failed inspection.")

    ## The audit summary is a separate artifact that is not guaranteed to exist, but if it does exist, it must pass the ready check.
    audit = audit_summary(
        audit_path=audit_path,
        ready_check="ready_for_trade_sync",
    )

    if (
        audit["n_rows"] != 1
        or audit["n_failed_audit_rows"] != 0
        or audit["ready_check_status"] != "pass"
    ):
        raise RuntimeError(
            "Checkpointed CIF/FOB imputation audit is invalid."
        )
    return {
        "status": "reused",
        "audit_passed": True,
        "ready_for_trade_sync": True,
        "output_path": output_path,
        "imputed_trade_path": output_path,
        "corrected_trade_path": output_path,
        "audit_path": audit_path,
        "imputation_audit_path": audit_path,
        "total_rows": int(row[0]),
        "imputed_rows": int(row[2]),
    }


def impute_cif_fob_factors(
    *,
    cfg: CifFobConfig,
    corrected_trade_path: Path,
    estimated_rates_path: Path,
    mirror_pairs_path: Path,
    maximum_factor: float,
    run_id: str,
    force: bool = False,
    logger,
) -> dict[str, Any]:
    """

    Product configuration: ``cfg.products.imputed_trade``. Reference factors
    are observed direct evidence or eligible mirrors, never previous imputations.
    This stage does not modify the Step 8 artifact.
    """
    corrected_trade_path = Path(corrected_trade_path)
    estimated_rates_path = Path(estimated_rates_path)
    mirror_pairs_path = Path(mirror_pairs_path)
    
    # The estimated-rate artifact determines the active CIF/FOB run.
    run_data_dir = estimated_rates_path.resolve().parent

    output_path = (
        run_data_dir
        / cfg.product_path("imputed_trade").name
    )

    audit_path = output_path.with_suffix(".audit.parquet")
    methods_path = output_path.with_suffix(".methods.csv")

    checkpoint_path = (
        run_data_dir
        / "checkpoints"
        / "imputed_trade.json"
    )

    maximum_factor = float(maximum_factor)

    if not 1.0 <= maximum_factor < float("inf"):
        raise ValueError(
            f"Invalid resolved maximum CIF/FOB factor: {maximum_factor}"
        )
    
    # The upper bound is frozen to the tested run until the existing cfg
    # exposes the selected application limit as a stable public accessor.
    imputation_cfg = getattr(cfg.method, "imputation", None)
    if imputation_cfg is not None and hasattr(
        imputation_cfg,
        "minimum_mirror_concordance",
    ):
        mirror_min_concordance = float(
            imputation_cfg.minimum_mirror_concordance
        )
    else:
        mirror_min_concordance = float(
            cfg.method.training.minimum_quantity_concordance
        )
    threads = int(cfg.engine.threads)

    for path in (corrected_trade_path, estimated_rates_path, mirror_pairs_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required CIF/FOB artifact missing: {path}")
    if output_path.resolve() in {
        path.resolve() for path in (corrected_trade_path, estimated_rates_path, mirror_pairs_path)
    }:
        raise ValueError("Impute must not overwrite an upstream artifact")
    if threads < 1:
        raise ValueError("CIF/FOB engine threads must be positive")

    config_fp = semantic_config_fingerprint(
        config=cfg,
        excluded_paths=(("engine", "threads"),),
    )
    input_fp = {
        "corrected_trade": file_fingerprint(corrected_trade_path, hash_file=True),
        "estimated_rates": file_fingerprint(estimated_rates_path, hash_file=True),
        "mirror_pairs": file_fingerprint(mirror_pairs_path, hash_file=True),
        "maximum_factor": maximum_factor,
    }
    reusable, reason = validate_local_checkpoint(
        checkpoint_path=checkpoint_path,
        current_config=config_fp,
        current_input=input_fp,
        output_path=output_path,
    )
    if not force and reusable and audit_path.is_file():
        try:
            inspected = _inspect_existing_imputation(
                output_path=output_path,
                audit_path=audit_path,
                threads=threads,
            )
        except RuntimeError:
            logger.info("REBUILD — CIF/FOB imputation checkpoint failed output audit")
        else:
            log_checkpoint_reuse(
                logger, "CIF/FOB imputed trade", reason,
                cached_output_path=output_path,
            )
            return inspected
    else:
        log_checkpoint_rebuild(
            logger, "CIF/FOB imputed trade",
            reason=reason if not force else "forced",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("CIF/FOB — imputing missing factors")
    logger.info("  Run ID: %s", run_id)
    logger.info("  input: %s", corrected_trade_path)
    logger.info("  Observed factor reference: %s", estimated_rates_path)
    logger.info("  Mirror reference: %s", mirror_pairs_path)
    logger.info("  Output: %s", output_path)

    con = duckdb.connect()
    try:
        con.execute(f"SET threads = {int(threads)}")
        con.execute(f"CREATE VIEW original AS SELECT * FROM read_parquet({_sql_path(corrected_trade_path)})")
        con.execute(f"CREATE VIEW rates AS SELECT * FROM read_parquet({_sql_path(estimated_rates_path)})")
        con.execute(f"CREATE VIEW mirror AS SELECT * FROM read_parquet({_sql_path(mirror_pairs_path)})")
        required = {
            "original": set(KEY) | {"model_eligible", "factor_source", "import_value_cif",
                "export_value_fob", "import_value_fob_corrected", "cif_fob_factor_applied",
                "cif_fob_factor_applied_source", "cif_fob_correction_applied",
                "factor_for_application"},
            "rates": set(KEY) | {"model_eligible", "factor_source", "observed_cif_fob_factor"},
            "mirror": set(KEY) | {"training_eligible", "weight_concordance", "raw_value_ratio"},
        }
        for name, wanted in required.items():
            missing = wanted - _columns(con, name)
            if missing:
                raise ValueError(f"{name} missing required columns: {sorted(missing)}")
        if "imputation_method" in _columns(con, "original"):
            raise ValueError("Input already has imputation_method; refuse accidental reapplication")
        for name in ("original", "rates", "mirror"):
            rows, keys = con.execute(
                f"SELECT COUNT(*), COUNT(DISTINCT ({', '.join(KEY)})) FROM {name}"
            ).fetchone()
            if rows != keys:
                raise ValueError(f"{name}: duplicate bilateral keys ({rows - keys})")

        # The original correction field is the authority: never replace an
        # existing correction or a positive exporter-side FOB value.
        con.execute("""
            CREATE TEMP TABLE targets AS
            SELECT year, exporter_iso, importer_iso, target_cmd, import_value_cif
            FROM original
            WHERE factor_for_application IS NULL
              AND cif_fob_factor_applied IS NULL
              AND import_value_fob_corrected IS NULL
              AND COALESCE(cif_fob_correction_applied, FALSE) IS FALSE
              AND import_value_cif > 0
              AND (export_value_fob IS NULL OR export_value_fob <= 0)
        """)
        n_targets = con.execute("SELECT COUNT(*) FROM targets").fetchone()[0]
        logger.info("  Imputation targets: %s", f"{n_targets:,}")

        con.execute(f"""
            CREATE TEMP TABLE direct AS
            SELECT year, exporter_iso, importer_iso, target_cmd,
                   observed_cif_fob_factor AS factor
            FROM rates
            WHERE factor_source = 'direct'
              AND model_eligible IS TRUE
              AND observed_cif_fob_factor BETWEEN 1.0 AND {float(maximum_factor)}
        """)
        con.execute(f"""
            CREATE TEMP TABLE eligible_mirror AS
            SELECT year, exporter_iso, importer_iso, target_cmd,
                   raw_value_ratio AS factor
            FROM mirror
            WHERE training_eligible IS TRUE
              AND weight_concordance >= {float(mirror_min_concordance)}
              AND raw_value_ratio BETWEEN 1.0 AND {float(maximum_factor)}
        """)
        con.execute("""
            CREATE TEMP TABLE target_groups AS
            SELECT DISTINCT year, importer_iso, target_cmd FROM targets
        """)
        # Excluding a target exporter is essential to match the notebook LOEO.
        con.execute("""
            CREATE TEMP TABLE local_direct AS
            SELECT t.year, t.exporter_iso, t.importer_iso, t.target_cmd,
                   COUNT(d.factor) AS n_local_direct, MEDIAN(d.factor) AS local_direct_factor
            FROM targets t LEFT JOIN direct d
              ON t.year = d.year AND t.importer_iso = d.importer_iso
             AND t.target_cmd = d.target_cmd AND t.exporter_iso <> d.exporter_iso
            GROUP BY 1,2,3,4
        """)
        con.execute("""
            CREATE TEMP TABLE cross_direct AS
            SELECT g.year, g.importer_iso, g.target_cmd,
                   COUNT(d.factor) AS n_cross_direct,
                   COUNT(DISTINCT d.importer_iso) AS n_cross_importers,
                   MEDIAN(d.factor) AS cross_direct_factor
            FROM target_groups g LEFT JOIN direct d
              ON g.year = d.year AND g.target_cmd = d.target_cmd
             AND g.importer_iso <> d.importer_iso
            GROUP BY 1,2,3
        """)
        con.execute("""
            CREATE TEMP TABLE local_mirror AS
            SELECT t.year, t.exporter_iso, t.importer_iso, t.target_cmd,
                   COUNT(m.factor) AS n_local_mirror, MEDIAN(m.factor) AS local_mirror_factor
            FROM targets t LEFT JOIN eligible_mirror m
              ON t.year = m.year AND t.importer_iso = m.importer_iso
             AND t.target_cmd = m.target_cmd AND t.exporter_iso <> m.exporter_iso
            GROUP BY 1,2,3,4
        """)
        con.execute("""
            CREATE TEMP TABLE unresolved_groups AS
            SELECT DISTINCT t.year, t.target_cmd,
                   LEFT(CAST(t.target_cmd AS VARCHAR),4) AS hs4
            FROM targets t
            JOIN local_direct l USING(year, exporter_iso, importer_iso, target_cmd)
            JOIN cross_direct c USING(year, importer_iso, target_cmd)
            JOIN local_mirror m USING(year, exporter_iso, importer_iso, target_cmd)
            WHERE l.n_local_direct < 2
              AND NOT (c.n_cross_direct >= 10 AND c.n_cross_importers >= 3)
              AND m.n_local_mirror < 1
        """)
        con.execute("""
            CREATE TEMP TABLE historical AS
            SELECT g.year, g.target_cmd, COUNT(d.factor) AS n_historical,
                   COUNT(DISTINCT d.importer_iso) AS n_historical_importers,
                   MEDIAN(d.factor) AS historical_factor
            FROM unresolved_groups g LEFT JOIN direct d
              ON g.target_cmd = d.target_cmd AND d.year < g.year
            GROUP BY 1,2
        """)
        con.execute("""
            CREATE TEMP TABLE hs4 AS
            SELECT g.year, g.target_cmd, COUNT(d.factor) AS n_hs4,
                   COUNT(DISTINCT d.importer_iso) AS n_hs4_importers,
                   MEDIAN(d.factor) AS hs4_factor
            FROM unresolved_groups g LEFT JOIN direct d
              ON g.year = d.year
             AND g.hs4 = LEFT(CAST(d.target_cmd AS VARCHAR),4)
             AND g.target_cmd <> d.target_cmd
            GROUP BY 1,2
        """)
        con.execute("""
            CREATE TEMP TABLE annual AS
            SELECT year, COUNT(*) AS n_annual, MEDIAN(factor) AS annual_factor
            FROM direct GROUP BY 1
        """)
        con.execute("""
            CREATE TEMP TABLE candidates AS
            SELECT t.*, l.n_local_direct, l.local_direct_factor,
                   c.n_cross_direct, c.n_cross_importers, c.cross_direct_factor,
                   m.n_local_mirror, m.local_mirror_factor,
                   h.n_historical, h.n_historical_importers, h.historical_factor,
                   s.n_hs4, s.n_hs4_importers, s.hs4_factor,
                   a.n_annual, a.annual_factor
            FROM targets t
            JOIN local_direct l USING(year, exporter_iso, importer_iso, target_cmd)
            JOIN cross_direct c USING(year, importer_iso, target_cmd)
            JOIN local_mirror m USING(year, exporter_iso, importer_iso, target_cmd)
            LEFT JOIN historical h USING(year, target_cmd)
            LEFT JOIN hs4 s USING(year, target_cmd)
            LEFT JOIN annual a USING(year)
        """)
        # Original notebook priority: direct local, cross-importer direct,
        # local mirror, historical HS6, same-year other-HS6 HS4, annual.
        logic = """
            CASE
              WHEN n_local_direct >= 2 THEN 'local_direct'
              WHEN n_cross_direct >= 10 AND n_cross_importers >= 3 THEN 'cross_importer_direct'
              WHEN n_local_mirror >= 1 THEN 'local_mirror'
              WHEN n_historical >= 10 AND n_historical_importers >= 3 THEN 'historical_hs6'
              WHEN n_hs4 >= 30 AND n_hs4_importers >= 5 THEN 'same_year_hs4'
              WHEN n_annual > 0 THEN 'annual_direct'
              ELSE NULL END
        """
        factor_logic = """
            CASE
              WHEN n_local_direct >= 2 THEN local_direct_factor
              WHEN n_cross_direct >= 10 AND n_cross_importers >= 3 THEN cross_direct_factor
              WHEN n_local_mirror >= 1 THEN local_mirror_factor
              WHEN n_historical >= 10 AND n_historical_importers >= 3 THEN historical_factor
              WHEN n_hs4 >= 30 AND n_hs4_importers >= 5 THEN hs4_factor
              WHEN n_annual > 0 THEN annual_factor
              ELSE NULL END
        """
        con.execute(f"""
            CREATE TEMP TABLE imputed AS
            SELECT year, exporter_iso, importer_iso, target_cmd,
                   ({logic}) AS imputation_method,
                   ({factor_logic}) AS imputed_factor,
                   import_value_cif
            FROM candidates
        """)
        invalid = con.execute(f"""
            SELECT COUNT(*) FROM imputed
            WHERE imputation_method IS NULL OR imputed_factor IS NULL
               OR NOT isfinite(imputed_factor)
               OR imputed_factor NOT BETWEEN 1.0 AND {float(maximum_factor)}
               OR import_value_cif / imputed_factor <= 0
        """).fetchone()[0]
        if invalid:
            raise RuntimeError(f"Impute has {invalid} unresolvable/invalid rows; no output written")
        if con.execute("SELECT COUNT(*) FROM imputed").fetchone()[0] != n_targets:
            raise RuntimeError("Impute candidate row count mismatch")

        # Strictly preserve ALL columns except fields changed by inpute
        # 'factor_source' is the agreed final classification in this output.
        con.execute("""
            CREATE TEMP VIEW recovered AS
            SELECT c.* REPLACE (
                CASE WHEN i.imputed_factor IS NOT NULL THEN 'imputed' ELSE c.factor_source END AS factor_source,
                CASE WHEN i.imputed_factor IS NOT NULL THEN i.imputed_factor ELSE c.cif_fob_factor_applied END AS cif_fob_factor_applied,
                CASE WHEN i.imputed_factor IS NOT NULL THEN 'imputed' ELSE c.cif_fob_factor_applied_source END AS cif_fob_factor_applied_source,
                CASE WHEN i.imputed_factor IS NOT NULL THEN c.import_value_cif / i.imputed_factor ELSE c.import_value_fob_corrected END AS import_value_fob_corrected,
                CASE WHEN i.imputed_factor IS NOT NULL THEN TRUE ELSE c.cif_fob_correction_applied END AS cif_fob_correction_applied
            ), i.imputation_method
            FROM original c LEFT JOIN imputed i USING(year, exporter_iso, importer_iso, target_cmd)
        """)
        stats = con.execute("""
            SELECT COUNT(*) AS total_rows,
                   COUNT(DISTINCT (year, exporter_iso, importer_iso, target_cmd)) AS distinct_keys,
                   COUNT(*) FILTER (WHERE factor_source = 'imputed') AS imputed_rows,
                   COUNT(*) FILTER (WHERE factor_source = 'imputed' AND cif_fob_factor_applied IS NULL) AS missing_applied_factor,
                   COUNT(*) FILTER (WHERE factor_source = 'imputed' AND imputation_method IS NULL) AS missing_method,
                   COUNT(*) FILTER (WHERE factor_source = 'imputed' AND (import_value_fob_corrected IS NULL OR import_value_fob_corrected <= 0)) AS invalid_fob
            FROM recovered
        """).fetchone()
        if stats[0] != stats[1] or stats[2] != n_targets or any(stats[3:]):
            raise RuntimeError(f"Recovered-output audit failed: {stats}")
        # Never change non-targeted rows, including all existing model outputs.
        drift = con.execute("""
            SELECT COUNT(*) FROM recovered r
            JOIN original o USING(year, exporter_iso, importer_iso, target_cmd)
            LEFT JOIN imputed i USING(year, exporter_iso, importer_iso, target_cmd)
            WHERE i.year IS NULL AND (
                r.factor_source IS DISTINCT FROM o.factor_source OR
                r.cif_fob_factor_applied IS DISTINCT FROM o.cif_fob_factor_applied OR
                r.cif_fob_factor_applied_source IS DISTINCT FROM o.cif_fob_factor_applied_source OR
                r.import_value_fob_corrected IS DISTINCT FROM o.import_value_fob_corrected OR
                r.cif_fob_correction_applied IS DISTINCT FROM o.cif_fob_correction_applied 
            )
        """).fetchone()[0]
        if drift:
            raise RuntimeError(f"Unexpected changes on {drift} non-target rows")
        original_rows = con.execute(
            "SELECT COUNT(*) FROM original"
        ).fetchone()[0]

        candidate_rows = con.execute(
            "SELECT COUNT(*) FROM imputed"
        ).fetchone()[0]

        formula_failures, invalid_provenance, invalid_factors = con.execute(f"""
            SELECT
                COUNT(*) FILTER (
                    WHERE factor_source = 'imputed'
                    AND (
                        import_value_fob_corrected IS NULL
                        OR NOT isfinite(import_value_fob_corrected)
                        OR ABS(
                            import_value_fob_corrected
                            - import_value_cif / NULLIF(cif_fob_factor_applied, 0)
                        ) > GREATEST(
                            1e-8,
                            ABS(import_value_fob_corrected) * 1e-10
                        )
                    )
                ),

                COUNT(*) FILTER (
                    WHERE factor_source = 'imputed'
                    AND (
                        imputation_method IS NULL
                        OR cif_fob_factor_applied_source IS DISTINCT FROM 'imputed'
                        OR cif_fob_correction_applied IS DISTINCT FROM TRUE
                    )
                ),

                COUNT(*) FILTER (
                    WHERE factor_source = 'imputed'
                    AND (
                        cif_fob_factor_applied IS NULL
                        OR NOT isfinite(cif_fob_factor_applied)
                        OR cif_fob_factor_applied
                            NOT BETWEEN 1.0 AND {float(maximum_factor)}
                    )
                )

            FROM recovered
        """).fetchone()

        checks = {
            "row_count_preserved": stats[0] == original_rows,
            "bilateral_keys_unique": stats[0] == stats[1],
            "imputation_coverage_complete": (
                candidate_rows == n_targets
                and stats[2] == n_targets
            ),
            "candidate_factors_valid": invalid == 0,
            "applied_factors_valid": invalid_factors == 0,
            "corrected_fob_formula_valid": formula_failures == 0,
            "imputation_provenance_valid": invalid_provenance == 0,
            "existing_rows_preserved": drift == 0,
            "imputed_values_complete": all(value == 0 for value in stats[3:]),
        }

        audit_passed = all(checks.values())
        audit_values = {
            "run_id": run_id,
            "n_rows": int(stats[0]),
            "n_targets": int(n_targets),
            "n_imputed_rows": int(stats[2]),
            "n_invalid_candidates": int(invalid),
            "n_invalid_applied_factors": int(invalid_factors),
            "n_formula_failures": int(formula_failures),
            "n_invalid_provenance": int(invalid_provenance),
            "n_non_target_drift": int(drift),
            **checks,
            "audit_passed": audit_passed,
        }

        columns_sql = []

        for name, value in audit_values.items():
            if isinstance(value, bool):
                sql_value = "TRUE" if value else "FALSE"
            elif isinstance(value, str):
                sql_value = _sql_path(value)
            else:
                sql_value = str(value)

            columns_sql.append(f"{sql_value} AS {name}")

        con.execute(f"""
            COPY (
                SELECT {", ".join(columns_sql)}
            )
            TO {_sql_path(audit_path)}
            (FORMAT PARQUET, COMPRESSION ZSTD)
        """)

        if not audit_passed:
            failed_checks = [
                name for name, passed in checks.items()
                if not passed
            ]

            logger.error(
                "AUDIT FAILED — CIF/FOB imputation: %s",
                ", ".join(failed_checks),
            )

            raise RuntimeError(
                "CIF/FOB imputation integrity audit failed: "
                + ", ".join(failed_checks)
            )
        logger.info("  Imputation eligible rows: %s", f"{n_targets:,}")
        # Output is written only after audits pass.
        con.execute(f"COPY (SELECT * FROM recovered) TO {_sql_path(output_path)} (FORMAT PARQUET, COMPRESSION ZSTD)")
        con.execute(f"""
            COPY (
                SELECT imputation_method, COUNT(*) AS observations,
                       SUM(import_value_cif) AS original_cif_usd,
                       SUM(import_value_cif / imputed_factor) AS recovered_fob_usd,
                       MIN(imputed_factor) AS min_factor,
                       MEDIAN(imputed_factor) AS median_factor,
                       MAX(imputed_factor) AS max_factor
                FROM imputed GROUP BY 1 ORDER BY observations DESC
            ) TO {_sql_path(methods_path)} (HEADER, DELIMITER ',')
        """)
        write_local_checkpoint(
            checkpoint_path=checkpoint_path,
            step_name="imputed_trade",
            run_id=run_id,
            config=config_fp,
            input_fingerprint=input_fp,
            output_path=output_path,
        )
        logger.info("  Imputed rows: %s", f"{n_targets:,}")
        logger.info("  Total output rows: %s", f"{stats[0]:,}")
        logger.info("  Audit passed: %s", audit_passed)
        logger.info("  Ready for Trade-sync: %s", audit_passed)
        return {
            "status": "completed",
            "audit_passed": audit_passed,
            "ready_for_trade_sync": audit_passed,
            "output_path": output_path,
            "methods_summary_path": methods_path,
            "imputed_trade_path": output_path,
            "corrected_trade_path": output_path,
            "audit_path": audit_path,
            "imputation_audit_path": audit_path,
            "imputed_rows": n_targets,
            "total_rows": stats[0],
        }
    finally:
        con.close()
