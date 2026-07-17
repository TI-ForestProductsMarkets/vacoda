"""
vacoda.cif_fob.apply_rates

Apply the primary HistGradientBoosting CIF/FOB factor and attach empirically
calibrated prediction intervals.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import duckdb
import numpy as np
import pandas as pd

from vacoda.config.cif_fob import CifFobConfig
from vacoda.cif_fob.prepare_input import valid_output

_REQUIRED_COLUMNS = {
    "year", "exporter_iso", "importer_iso", "target_cmd",
    "prediction_eligible", "primary_model_name", "primary_estimation_run",
    "predicted_log_cif_fob_factor", "estimated_cif_fob_factor",
    "holdout_predicted_log_factor", "observed_cif_fob_factor",
    "import_value_cif",
}


def _sql_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _read_columns(con: duckdb.DuckDBPyConnection, path: Path) -> set[str]:
    rows = con.execute(
        "DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]
    ).fetchall()
    return {str(row[0]) for row in rows}


def _normalize_coverage_levels(levels: Sequence[float]) -> tuple[float, ...]:
    result = tuple(sorted({float(level) for level in levels}))
    if not result or any(not 0.0 < level < 1.0 for level in result):
        raise ValueError("Coverage levels must lie strictly between 0 and 1.")
    return result


def _coverage_suffix(level: float) -> str:
    percentage = 100.0 * float(level)
    if not np.isclose(percentage, round(percentage)):
        raise ValueError("Coverage levels must be whole percentages.")
    return str(int(round(percentage)))


def _finite_sample_quantile(scores: np.ndarray, coverage: float) -> float:
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        raise RuntimeError("No finite holdout conformity scores are available.")
    n = int(scores.size)
    rank = min(n, int(np.ceil((n + 1) * float(coverage))))
    return float(np.quantile(scores, rank / n, method="higher"))


def _calibrate_log_error_quantiles(
    frame: pd.DataFrame,
    levels: tuple[float, ...],
) -> tuple[dict[float, float], int]:
    observed = pd.to_numeric(frame["observed_cif_fob_factor"], errors="coerce")
    predicted_log = pd.to_numeric(frame["holdout_predicted_log_factor"], errors="coerce")
    valid = observed.notna() & observed.gt(0) & predicted_log.notna() & np.isfinite(predicted_log)
    if not valid.any():
        raise RuntimeError("No valid primary-model holdout rows exist for uncertainty calibration.")
    scores = np.abs(
        np.log(observed.loc[valid].to_numpy(dtype=float))
        - predicted_log.loc[valid].to_numpy(dtype=float)
    )
    return ({level: _finite_sample_quantile(scores, level) for level in levels}, int(scores.size))


def _write_parquet(frame: pd.DataFrame, output_path: Path, threads: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    con = duckdb.connect()
    try:
        con.execute(f"SET threads = {int(threads)}")
        con.register("corrected_trade", frame)
        con.execute(
            f"""
            COPY (SELECT * FROM corrected_trade)
            TO {_sql_literal(temporary_path)}
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
    finally:
        con.close()
    temporary_path.replace(output_path)


def _audit_corrected_trade_output(
    *,
    output_path: Path,
    expected_input_rows: int | None,
    coverage_levels: tuple[float, ...],
    threads: int,
) -> dict[str, Any]:
    """
    Audit the corrected CIF/FOB trade artifact.

    This function calculates audit metrics only. It does not decide whether
    the artifact is valid; validation is handled separately.
    """
    con = duckdb.connect()

    try:
        con.execute(
            f"SET threads = {int(threads)}"
        )

        columns = _read_columns(
            con,
            output_path,
        )

        required_output_columns = {
            "year",
            "exporter_iso",
            "importer_iso",
            "target_cmd",
            "prediction_eligible",
            "estimated_cif_fob_factor",
            "cif_fob_factor_applied",
            "import_value_cif",
            "import_value_fob_corrected",
            "cif_fob_correction_applied",
        }

        for level in coverage_levels:
            suffix = _coverage_suffix(level)

            required_output_columns.update(
                {
                    f"cif_fob_factor_pi{suffix}_lower",
                    f"cif_fob_factor_pi{suffix}_upper",
                    f"import_value_fob_corrected_pi{suffix}_lower",
                    f"import_value_fob_corrected_pi{suffix}_upper",
                }
            )

        missing_columns = sorted(
            required_output_columns - columns
        )

        if missing_columns:
            raise RuntimeError(
                "Corrected-trade output is missing required columns: "
                + ", ".join(missing_columns)
            )

        interval_failure_expressions = []

        for level in coverage_levels:
            suffix = _coverage_suffix(level)

            interval_failure_expressions.append(
                f"""
                (
                    cif_fob_correction_applied
                    AND (
                           cif_fob_factor_pi{suffix}_lower IS NULL
                        OR cif_fob_factor_pi{suffix}_upper IS NULL
                        OR cif_fob_factor_pi{suffix}_lower <= 0
                        OR cif_fob_factor_pi{suffix}_upper <= 0
                        OR cif_fob_factor_pi{suffix}_lower
                           > cif_fob_factor_applied
                        OR cif_fob_factor_applied
                           > cif_fob_factor_pi{suffix}_upper
                        OR import_value_fob_corrected_pi{suffix}_lower
                           > import_value_fob_corrected
                        OR import_value_fob_corrected
                           > import_value_fob_corrected_pi{suffix}_upper
                    )
                )
                """
            )

        interval_failure_sql = " OR ".join(
            interval_failure_expressions
        )

        row = con.execute(
            f"""
            WITH duplicate_keys AS (
                SELECT
                    year,
                    exporter_iso,
                    importer_iso,
                    target_cmd,
                    COUNT(*) AS n_rows
                FROM read_parquet(?)
                GROUP BY
                    year,
                    exporter_iso,
                    importer_iso,
                    target_cmd
                HAVING COUNT(*) > 1
            )
            SELECT
                COUNT(*) AS n_rows,

                (
                    SELECT COUNT(*)
                    FROM duplicate_keys
                ) AS n_duplicate_keys,

                COUNT(*) FILTER (
                    WHERE prediction_eligible
                ) AS n_prediction_eligible,

                COUNT(*) FILTER (
                    WHERE cif_fob_correction_applied
                ) AS n_corrected_rows,

                COUNT(*) FILTER (
                    WHERE prediction_eligible
                      AND import_value_cif IS NOT NULL
                      AND import_value_cif >= 0
                      AND estimated_cif_fob_factor IS NOT NULL
                      AND estimated_cif_fob_factor > 0
                      AND NOT cif_fob_correction_applied
                ) AS n_eligible_not_corrected,

                COUNT(*) FILTER (
                    WHERE NOT prediction_eligible
                      AND cif_fob_correction_applied
                ) AS n_ineligible_corrected,

                COUNT(*) FILTER (
                    WHERE cif_fob_correction_applied
                      AND (
                            cif_fob_factor_applied IS NULL
                         OR cif_fob_factor_applied <= 0
                         OR import_value_fob_corrected IS NULL
                         OR import_value_fob_corrected < 0
                      )
                ) AS n_invalid_corrected_rows,

                COUNT(*) FILTER (
                    WHERE NOT cif_fob_correction_applied
                      AND (
                            cif_fob_factor_applied IS NOT NULL
                         OR import_value_fob_corrected IS NOT NULL
                      )
                ) AS n_values_when_not_applied,

                COUNT(*) FILTER (
                    WHERE cif_fob_correction_applied
                      AND ABS(
                            import_value_fob_corrected
                            - import_value_cif
                              / cif_fob_factor_applied
                      ) > GREATEST(
                            1e-8,
                            ABS(import_value_fob_corrected) * 1e-10
                      )
                ) AS n_formula_failures,

                COUNT(*) FILTER (
                    WHERE cif_fob_correction_applied
                      AND cif_fob_factor_applied >= 1
                      AND import_value_fob_corrected
                          > import_value_cif
                ) AS n_direction_failures,

                COUNT(*) FILTER (
                    WHERE {interval_failure_sql}
                ) AS n_interval_failures,

                SUM(import_value_cif) FILTER (
                    WHERE cif_fob_correction_applied
                ) AS original_import_cif_value,

                SUM(import_value_fob_corrected) FILTER (
                    WHERE cif_fob_correction_applied
                ) AS corrected_import_fob_value,

                MEDIAN(cif_fob_factor_applied) FILTER (
                    WHERE cif_fob_correction_applied
                ) AS median_factor_applied,

                QUANTILE_CONT(
                    cif_fob_factor_applied,
                    0.05
                ) FILTER (
                    WHERE cif_fob_correction_applied
                ) AS factor_p05,

                QUANTILE_CONT(
                    cif_fob_factor_applied,
                    0.95
                ) FILTER (
                    WHERE cif_fob_correction_applied
                ) AS factor_p95,

                MEDIAN(
                    1.0
                    - import_value_fob_corrected
                      / NULLIF(import_value_cif, 0)
                ) FILTER (
                    WHERE cif_fob_correction_applied
                      AND import_value_cif > 0
                ) AS median_correction_fraction

            FROM read_parquet(?)
            """,
            [
                str(output_path),
                str(output_path),
            ],
        ).fetchone()

    finally:
        con.close()

    if row is None:
        raise RuntimeError(
            "Could not audit corrected CIF/FOB output."
        )

    (
        n_rows,
        n_duplicate_keys,
        n_prediction_eligible,
        n_corrected_rows,
        n_eligible_not_corrected,
        n_ineligible_corrected,
        n_invalid_corrected_rows,
        n_values_when_not_applied,
        n_formula_failures,
        n_direction_failures,
        n_interval_failures,
        original_import_cif_value,
        corrected_import_fob_value,
        median_factor_applied,
        factor_p05,
        factor_p95,
        median_correction_fraction,
    ) = row

    n_rows = int(n_rows or 0)
    n_corrected_rows = int(n_corrected_rows or 0)

    correction_coverage_percent = (
        100.0 * n_corrected_rows / n_rows
        if n_rows > 0
        else 0.0
    )

    total_cif_fob_reduction = None
    total_cif_fob_reduction_percent = None

    if (
        original_import_cif_value is not None
        and corrected_import_fob_value is not None
    ):
        total_cif_fob_reduction = (
            original_import_cif_value
            - corrected_import_fob_value
        )

        if original_import_cif_value > 0:
            total_cif_fob_reduction_percent = (
                100.0
                * total_cif_fob_reduction
                / original_import_cif_value
            )

    return {
        "expected_input_rows": expected_input_rows,
        "n_rows": n_rows,
        "n_duplicate_keys": int(n_duplicate_keys or 0),
        "n_prediction_eligible": int(
            n_prediction_eligible or 0
        ),
        "n_corrected_rows": n_corrected_rows,
        "n_eligible_not_corrected": int(
            n_eligible_not_corrected or 0
        ),
        "n_ineligible_corrected": int(
            n_ineligible_corrected or 0
        ),
        "n_invalid_corrected_rows": int(
            n_invalid_corrected_rows or 0
        ),
        "n_values_when_not_applied": int(
            n_values_when_not_applied or 0
        ),
        "n_formula_failures": int(
            n_formula_failures or 0
        ),
        "n_direction_failures": int(
            n_direction_failures or 0
        ),
        "n_interval_failures": int(
            n_interval_failures or 0
        ),
        "correction_coverage_percent": (
            correction_coverage_percent
        ),
        "original_import_cif_value": (
            original_import_cif_value
        ),
        "corrected_import_fob_value": (
            corrected_import_fob_value
        ),
        "total_cif_fob_reduction": (
            total_cif_fob_reduction
        ),
        "total_cif_fob_reduction_percent": (
            total_cif_fob_reduction_percent
        ),
        "median_factor_applied": median_factor_applied,
        "factor_p05": factor_p05,
        "factor_p95": factor_p95,
        "median_correction_fraction": (
            median_correction_fraction
        ),
    }


def _inspect_existing_output(
    *,
    output_path: Path,
    coverage_levels: tuple[float, ...],
    threads: int,
) -> dict[str, Any]:
    """Audit and return a previously written corrected-trade artifact."""

    audit = _audit_corrected_trade_output(
        output_path=output_path,
        expected_input_rows=None,
        coverage_levels=coverage_levels,
        threads=threads,
    )

    _validate_corrected_trade_audit(
        audit
    )

    return {
        "status": "reused",
        "audit_passed": True,
        "ready_for_trade_sync": (
            audit["n_corrected_rows"] > 0
        ),
        "output_path": output_path,
        "corrected_trade_path": output_path,
        **audit,
    }

def _validate_corrected_trade_audit(
    audit: dict[str, Any],
) -> bool:
    """Validate the corrected-trade audit and return whether it passed."""

    expected_input_rows = audit.get(
        "expected_input_rows"
    )

    if (
        expected_input_rows is not None
        and audit["n_rows"] != expected_input_rows
    ):
        raise RuntimeError(
            "Corrected-trade row count does not match its input: "
            f"output={audit['n_rows']:,}, "
            f"input={expected_input_rows:,}."
        )

    failure_metrics = {
        "duplicate bilateral keys": audit[
            "n_duplicate_keys"
        ],
        "eligible rows not corrected": audit[
            "n_eligible_not_corrected"
        ],
        "ineligible rows corrected": audit[
            "n_ineligible_corrected"
        ],
        "invalid corrected rows": audit[
            "n_invalid_corrected_rows"
        ],
        "values present when correction was not applied": audit[
            "n_values_when_not_applied"
        ],
        "correction-formula failures": audit[
            "n_formula_failures"
        ],
        "correction-direction failures": audit[
            "n_direction_failures"
        ],
        "prediction-interval failures": audit[
            "n_interval_failures"
        ],
    }

    failures = {
        name: count
        for name, count in failure_metrics.items()
        if int(count) > 0
    }

    if failures:
        details = "; ".join(
            f"{name}={count:,}"
            for name, count in failures.items()
        )

        raise RuntimeError(
            "Corrected CIF/FOB trade audit failed: "
            + details
        )

    if audit["n_corrected_rows"] <= 0:
        raise RuntimeError(
            "Corrected CIF/FOB output contains no corrected rows."
        )

    return True

def apply_cif_fob_weights(
    *,
    cfg: CifFobConfig,
    estimated_rates_path: Path,
    coverage_levels: Sequence[float] = (0.80, 0.90),
    logger,
) -> dict[str, Any]:
    """Apply the primary HGB factor and add 80%/90% prediction intervals."""
    estimated_rates_path = Path(estimated_rates_path)
    
    output_path = (
        estimated_rates_path.parent
        / cfg.product_path("corrected_trade").name
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not estimated_rates_path.exists():
        raise FileNotFoundError(f"Estimated-rates artifact not found: {estimated_rates_path}")

    levels = _normalize_coverage_levels(coverage_levels)


    if valid_output(output_path):
        if logger is not None:
            logger.info("REUSE — Corrected CIF/FOB trade already exists: %s", output_path)
        return _inspect_existing_output(
                    output_path=output_path,
                    coverage_levels=levels,
                    threads=cfg.engine.threads,
                )

    con = duckdb.connect()
    try:
        con.execute(f"SET threads = {int(cfg.engine.threads)}")
        missing = sorted(_REQUIRED_COLUMNS - _read_columns(con, estimated_rates_path))
        if missing:
            raise RuntimeError(
                "Estimated-rates artifact is missing required columns: " + ", ".join(missing)
            )
        data = con.execute("SELECT * FROM read_parquet(?)", [str(estimated_rates_path)]).df()
    finally:
        con.close()

    if data.empty:
        raise RuntimeError("Estimated-rates artifact contains no rows.")

    primary_models = (
        data["primary_model_name"].dropna().astype(str).str.strip().str.lower().unique().tolist()
    )
    if len(primary_models) != 1:
        raise RuntimeError(f"Expected one primary model; found {primary_models!r}.")
    primary_model_name = primary_models[0]
    if "hist_gradient_boosting" not in primary_model_name:
        raise RuntimeError(
            "Application expects a HistGradientBoosting primary model, but found "
            f"{primary_model_name!r}."
        )

    quantiles, calibration_rows = _calibrate_log_error_quantiles(data, levels)

    factor = pd.to_numeric(data["estimated_cif_fob_factor"], errors="coerce")
    predicted_log = pd.to_numeric(data["predicted_log_cif_fob_factor"], errors="coerce")
    import_cif = pd.to_numeric(data["import_value_cif"], errors="coerce")
    prediction_eligible = data["prediction_eligible"].fillna(False).astype(bool)

    correction_eligible = (
        prediction_eligible
        & factor.notna() & np.isfinite(factor) & factor.gt(0)
        & import_cif.notna() & np.isfinite(import_cif) & import_cif.ge(0)
    )

    data["cif_fob_factor_applied"] = factor.where(correction_eligible)
    data["import_value_fob_corrected"] = np.where(
        correction_eligible, import_cif / factor, np.nan
    )
    data["cif_fob_correction_applied"] = correction_eligible
    data["cif_fob_uncertainty_method"] = "latest_temporal_holdout_absolute_log_residual"
    data["cif_fob_uncertainty_calibration_rows"] = calibration_rows

    valid_prediction = prediction_eligible & predicted_log.notna() & np.isfinite(predicted_log)

    for level in levels:
        suffix = _coverage_suffix(level)
        q = quantiles[level]
        lower = pd.Series(np.exp(predicted_log - q), index=data.index).where(valid_prediction)
        upper = pd.Series(np.exp(predicted_log + q), index=data.index).where(valid_prediction)

        data[f"cif_fob_factor_pi{suffix}_lower"] = lower
        data[f"cif_fob_factor_pi{suffix}_upper"] = upper
        data[f"cif_fob_factor_pi{suffix}_level"] = float(level)
        data[f"cif_fob_factor_pi{suffix}_log_error_quantile"] = float(q)
        data[f"import_value_fob_corrected_pi{suffix}_lower"] = np.where(
            correction_eligible, import_cif / upper, np.nan
        )
        data[f"import_value_fob_corrected_pi{suffix}_upper"] = np.where(
            correction_eligible, import_cif / lower, np.nan
        )
        data[f"cif_fob_factor_pi{suffix}_relative_width"] = np.where(
            correction_eligible, (upper - lower) / factor, np.nan
        )

    _write_parquet(data, output_path, cfg.engine.threads)

    audit = _audit_corrected_trade_output(
        output_path=output_path,
        expected_input_rows=len(data),
        coverage_levels=levels,
        threads=cfg.engine.threads,
    )

    audit_passed = _validate_corrected_trade_audit(
        audit
    )

    ready_for_trade_sync = (
        audit_passed
        and audit["n_corrected_rows"] > 0
    )

    if logger is not None:
        logger.info(
            "Applied primary CIF/FOB factors."
        )
        logger.info(
            "  Primary model: %s",
            primary_model_name,
        )
        logger.info(
            "  Corrected rows: %s",
            audit["n_corrected_rows"],
        )
        logger.info(
            "  Correction coverage: %.2f%%",
            audit["correction_coverage_percent"],
        )
        logger.info(
            "  Calibration rows: %s",
            calibration_rows,
        )

        for level in levels:
            logger.info(
                "  PI %.0f%% absolute-log-error quantile: %.6f",
                100.0 * level,
                quantiles[level],
            )

        logger.info(
            "  Output: %s",
            output_path,
        )

    return {
        "status": "completed",
        "audit_passed": audit_passed,
        "ready_for_trade_sync": ready_for_trade_sync,
        "output_path": output_path,
        "corrected_trade_path": output_path,
        "primary_model_name": primary_model_name,
        "coverage_levels": list(levels),
        "calibration_rows": calibration_rows,
        "calibration_log_error_quantiles": {
            _coverage_suffix(level): quantiles[level]
            for level in levels
        },
        **audit,
    }