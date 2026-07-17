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


def _inspect_existing_output(output_path: Path, threads: int) -> dict[str, Any]:
    con = duckdb.connect()
    try:
        con.execute(f"SET threads = {int(threads)}")
        row = con.execute(
            """
            SELECT
                COUNT(*),
                COUNT(*) FILTER (WHERE cif_fob_correction_applied),
                SUM(import_value_cif) FILTER (WHERE cif_fob_correction_applied),
                SUM(import_value_fob_corrected) FILTER (WHERE cif_fob_correction_applied)
            FROM read_parquet(?)
            """,
            [str(output_path)],
        ).fetchone()
    finally:
        con.close()
    if row is None:
        raise RuntimeError(f"Could not inspect corrected output: {output_path}")
    return {
        "status": "reused",
        "output_path": output_path,
        "corrected_trade_path": output_path,
        "n_rows": int(row[0] or 0),
        "n_corrected_rows": int(row[1] or 0),
        "original_import_cif_value": row[2],
        "corrected_import_fob_value": row[3],
    }


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

    if valid_output(output_path):
        if logger is not None:
            logger.info("REUSE — Corrected CIF/FOB trade already exists: %s", output_path)
        return _inspect_existing_output(output_path, cfg.engine.threads)

    levels = _normalize_coverage_levels(coverage_levels)

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

    con = duckdb.connect()
    try:
        con.execute(f"SET threads = {int(cfg.engine.threads)}")
        audit = con.execute(
            """
            SELECT
                COUNT(*),
                COUNT(*) FILTER (WHERE cif_fob_correction_applied),
                COUNT(*) FILTER (
                    WHERE cif_fob_correction_applied
                      AND (
                            cif_fob_factor_applied IS NULL
                         OR cif_fob_factor_applied <= 0
                         OR import_value_fob_corrected IS NULL
                         OR import_value_fob_corrected < 0
                      )
                ),
                SUM(import_value_cif) FILTER (WHERE cif_fob_correction_applied),
                SUM(import_value_fob_corrected) FILTER (WHERE cif_fob_correction_applied)
            FROM read_parquet(?)
            """,
            [str(output_path)],
        ).fetchone()
    finally:
        con.close()

    if audit is None:
        raise RuntimeError("Could not audit corrected CIF/FOB output.")
    if int(audit[0] or 0) != len(data):
        raise RuntimeError("Corrected output row count does not match input.")
    if int(audit[2] or 0) > 0:
        raise RuntimeError(f"Corrected output contains {int(audit[2]):,} invalid rows.")

    if logger is not None:
        logger.info("Applied primary CIF/FOB factors.")
        logger.info("  Primary model: %s", primary_model_name)
        logger.info("  Corrected rows: %s", int(audit[1] or 0))
        logger.info("  Calibration rows: %s", calibration_rows)
        for level in levels:
            logger.info(
                "  PI %.0f%% absolute-log-error quantile: %.6f",
                100.0 * level,
                quantiles[level],
            )
        logger.info("  Output: %s", output_path)

    return {
        "status": "completed",
        "audit_passed": True,
        "output_path": output_path,
        "corrected_trade_path": output_path,
        "primary_model_name": primary_model_name,
        "coverage_levels": list(levels),
        "calibration_rows": calibration_rows,
        "calibration_log_error_quantiles": {
            _coverage_suffix(level): quantiles[level] for level in levels
        },
        "n_rows": int(audit[0] or 0),
        "n_corrected_rows": int(audit[1] or 0),
        "original_import_cif_value": audit[3],
        "corrected_import_fob_value": audit[4],
    }