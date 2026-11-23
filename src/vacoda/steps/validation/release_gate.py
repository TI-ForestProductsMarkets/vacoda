"""
vacoda.steps.validation.release_gate

Publication-readiness gate for the current VACODA Validation inference product.

This gate does not create corrected quantities or certify future Release outputs.
A failed readiness check is recorded and logged, not raised as an exception.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from vacoda.config.steps.validation import ValidationConfig
from vacoda.utils.sql import sql_literal
from vacoda.utils.display import get_symbols


KEY = ("year", "exporter_iso", "importer_iso", "target_cmd")
PRESERVED = (
    "trade_value_reconciled",
    "trade_value_reconciled_constant_usd",
)


def _columns(con: duckdb.DuckDBPyConnection, path: Path) -> set[str]:
    rows = con.execute(
        "DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]
    ).fetchall()
    return {str(row[0]) for row in rows}


def _one_row(con: duckdb.DuckDBPyConnection, path: Path, fields: tuple[str, ...]) -> dict:
    available = _columns(con, path)
    missing = sorted(set(fields) - available)
    if missing:
        raise ValueError(f"{path.name}: missing columns: {', '.join(missing)}")
    sql = ", ".join(f'"{field}"' for field in fields)
    rows = con.execute(
        f"SELECT {sql} FROM read_parquet(?) LIMIT 2", [str(path)]
    ).fetchall()
    if len(rows) != 1:
        raise ValueError(f"{path.name}: expected one audit row, found {len(rows)}")
    return dict(zip(fields, rows[0]))


def evaluate_release_readiness(
    *,
    cfg: ValidationConfig,
    prepared_input_path: Path,
    inference_path: Path,
    structural_audit_path: Path,
    inference_audit_path: Path,
    selected_model_path: Path,
    value_selected_model_path: Path,
    value_fitted_model_path: Path,
    plausibility_path: Path,
    run_id: str,
    logger,
    force: bool = False,
) -> dict[str, Any]:
    """Evaluate release readiness; write evidence even when readiness is false.

    `force` is accepted for consistency with workflow steps. This inexpensive
    evidence gate is recomputed on every invocation to avoid stale decisions.
    """
    del force

    output_path = Path(cfg.product_path("release_readiness"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "prepared_input": Path(prepared_input_path),
        "inference": Path(inference_path),
        "structural_audit": Path(structural_audit_path),
        "inference_audit": Path(inference_audit_path),
        "selected_model": Path(selected_model_path),
        "value_selected_model": Path(value_selected_model_path),
        "value_fitted_model": Path(value_fitted_model_path),
        "plausibility": Path(plausibility_path),
        "value_calibration": Path(plausibility_path).with_stem(
            f"{Path(plausibility_path).stem}_value_bounds"
        ),
    }

    checks: dict[str, bool] = {}
    reasons: list[str] = []
    metrics: dict[str, Any] = {
        "n_prepared_rows": None,
        "n_inference_rows": None,
        "n_missing_in_inference": None,
        "n_unexpected_in_inference": None,
        "n_value_mismatches": None,
        "n_quantity_candidate_mismatches": None,
        "n_provenance_mismatches": None,
        "n_model_input_valid_rows": None,
        "n_prediction_rows": None,
        "eligible_prediction_coverage": None,
        "population_prediction_coverage": None,
        "n_imputed_value_rows": None,
        "model_name": None,
        "model_type": None,
        "selection_value": None,
        "n_oof_predictions": None,
        "n_validation_years": None,
        "value_model_name": None,
        "value_model_type": None,
        "value_selection_value": None,
        "value_fit_training_rows": None,
        "value_fit_min_training_year": None,
        "value_fit_max_training_year": None,
        "value_fit_bundle_path": None,
        "value_calibration_year": None,
        "value_calibration_rows": None,
        "value_calibration_size_groups": None,
        "value_calibration_nominal_outside_rate": None,
        "value_residual_invalid_rows": None,
        "value_plausibility_classification_mismatches": None,
        "n_value_model_input_valid_rows": None,
        "n_value_prediction_rows": None,
        "value_eligible_prediction_coverage": None,
        "value_population_prediction_coverage": None,
        "n_plausibility_rows": None,
        "n_plausibility_missing_keys": None,
        "n_plausibility_unexpected_keys": None,
        "n_plausibility_duplicate_keys": None,
        "quantity_within_bounds": None,
        "quantity_below_bounds": None,
        "quantity_above_bounds": None,
        "quantity_not_assessed": None,
        "value_within_bounds": None,
        "value_below_bounds": None,
        "value_above_bounds": None,
        "value_not_assessed": None,
    }

    def record(name: str, passed: bool, reason: str) -> None:
        checks[name] = bool(passed)
        if not passed:
            reasons.append(reason)

    con = duckdb.connect()
    try:
        con.execute(f"SET threads = {int(cfg.engine.threads)}")

        for label, path in paths.items():
            record(
                f"{label}_exists",
                path.is_file() and path.stat().st_size > 0,
                f"Missing or empty required artifact: {label} ({path})",
            )

        # Reuse completed audits; don't reproduce their internal checks.
        for label in ("structural_audit", "inference_audit"):
            if not checks[f"{label}_exists"]:
                record(f"{label}_passed", False, f"{label} is unavailable")
                continue
            try:
                audit = _one_row(con, paths[label], ("run_id", "audit_passed", "n_rows"))
                ok = audit["audit_passed"] is True
                record(
                    f"{label}_passed", ok,
                    f"{label} did not pass",
                )
                if label == "structural_audit":
                    metrics["n_prepared_rows"] = int(audit["n_rows"])
                else:
                    metrics["n_inference_rows"] = int(audit["n_rows"])
            except (duckdb.Error, ValueError, TypeError) as exc:
                record(f"{label}_passed", False, f"Unreadable {label}: {exc}")

        # Require the value-specific invariants already computed by the
        # cross-variable inference audit; do not silently rely on the aggregate
        # audit_passed flag alone.
        value_audit_fields = (
            "n_value_prediction_availability_mismatches",
            "n_invalid_value_prediction_routing",
            "n_value_inverse_transform_mismatches",
            "n_value_residual_availability_mismatches",
            "n_value_residual_formula_mismatches",
            "n_value_absolute_residual_mismatches",
            "n_value_model_identity_mismatches",
            "n_training_rows_invalid_value_input",
            "n_joint_unit_value_identity_mismatches",
        )
        if checks["inference_audit_exists"]:
            try:
                audit_values = _one_row(con, paths["inference_audit"], value_audit_fields)
                for field, value in audit_values.items():
                    metrics[field] = int(value)
                errors = {field: n for field, n in metrics.items()
                          if field in value_audit_fields and n != 0}
                record("value_inference_audit_valid", not errors,
                       f"Value inference audit has nonzero mismatch counters: {errors}")
            except (duckdb.Error, ValueError, TypeError, OverflowError) as exc:
                record("value_inference_audit_valid", False,
                       f"Cannot verify value-specific inference audit: {exc}")
        else:
            record("value_inference_audit_valid", False, "Inference audit unavailable")

        # Validate one selected-model record and its cross-validation evidence.
        if checks["selected_model_exists"]:
            try:
                selection = _one_row(con, paths["selected_model"], (
                    "run_id", "model_name", "model_type", "selection_metric",
                    "selection_value", "n_predictions", "n_validation_years",
                ))
                metrics.update({
                    "model_name": selection["model_name"],
                    "model_type": selection["model_type"],
                    "selection_value": selection["selection_value"],
                    "n_oof_predictions": selection["n_predictions"],
                    "n_validation_years": selection["n_validation_years"],
                })
                valid = (
                    bool(selection["model_name"])
                    and bool(selection["model_type"])
                    and selection["selection_metric"] == "mae_log"
                    and selection["selection_value"] is not None
                    and math.isfinite(float(selection["selection_value"]))
                    and float(selection["selection_value"]) >= 0
                    and int(selection["n_predictions"]) > 0
                    and int(selection["n_validation_years"]) > 0
                )
                record("model_evidence_valid", valid,
                       "Selected model lacks valid out-of-fold evaluation evidence")
            except (duckdb.Error, ValueError, TypeError, OverflowError) as exc:
                record("model_evidence_valid", False, f"Invalid model evidence: {exc}")
        else:
            record("model_evidence_valid", False, "Selected-model artifact unavailable")

        # Verify data identity by canonical keys, not by row counts alone.
        if checks["prepared_input_exists"] and checks["inference_exists"]:
            try:
                prep_cols = _columns(con, paths["prepared_input"])
                infer_cols = _columns(con, paths["inference"])
                target = str(cfg.model.target.column)
                required = set(KEY) | set(PRESERVED) | {target}
                missing_p = sorted(required - prep_cols)
                missing_i = sorted(required - infer_cols)
                if missing_p or missing_i:
                    raise ValueError(
                        f"Missing preservation columns: prepared={missing_p}, "
                        f"inference={missing_i}"
                    )
                compare_provenance = "imputation_method" in prep_cols
                if compare_provenance and "imputation_method" not in infer_cols:
                    raise ValueError("Inference lost imputation_method provenance column")

                key_using = ", ".join(f'"{field}"' for field in KEY)
                value_fields = list(PRESERVED)
                value_mismatch = " OR ".join(
                    f'p."{col}" IS DISTINCT FROM i."{col}"' for col in value_fields
                )
                target_sql = '"' + target.replace('"', '""') + '"'
                provenance_sql = (
                    'p."imputation_method" IS DISTINCT FROM i."imputation_method"'
                    if compare_provenance else "FALSE"
                )
                prepared_sql = sql_literal(paths["prepared_input"])
                inference_sql = sql_literal(paths["inference"])
                row = con.execute(f"""
                    WITH p AS (
                        SELECT TRUE AS _present, * FROM read_parquet({prepared_sql})
                    ), i AS (
                        SELECT TRUE AS _present, * FROM read_parquet({inference_sql})
                    )
                    SELECT
                        COUNT(*) FILTER (WHERE p._present IS TRUE) AS n_prepared,
                        COUNT(*) FILTER (WHERE i._present IS TRUE) AS n_inference,
                        COUNT(*) FILTER (WHERE i._present IS NULL) AS n_missing,
                        COUNT(*) FILTER (WHERE p._present IS NULL) AS n_extra,
                        COUNT(*) FILTER (
                            WHERE p._present IS TRUE AND i._present IS TRUE
                              AND ({value_mismatch})
                        ) AS n_values_changed,
                        COUNT(*) FILTER (
                            WHERE p._present IS TRUE AND i._present IS TRUE
                              AND p.{target_sql} IS DISTINCT FROM i.{target_sql}
                        ) AS n_candidate_changed,
                        COUNT(*) FILTER (
                            WHERE p._present IS TRUE AND i._present IS TRUE
                              AND ({provenance_sql})
                        ) AS n_provenance_changed
                    FROM p FULL OUTER JOIN i USING ({key_using})
                """).fetchone()
                (n_p, n_i, n_missing, n_extra, n_values,
                 n_candidate, n_provenance) = map(int, row)
                metrics.update({
                    "n_prepared_rows": n_p,
                    "n_inference_rows": n_i,
                    "n_missing_in_inference": n_missing,
                    "n_unexpected_in_inference": n_extra,
                    "n_value_mismatches": n_values,
                    "n_quantity_candidate_mismatches": n_candidate,
                    "n_provenance_mismatches": n_provenance,
                })
                record("population_preserved", n_p > 0 and n_missing == 0 and n_extra == 0,
                       f"Population differs: {n_missing:,} missing and {n_extra:,} unexpected keys")
                record("monetary_values_preserved", n_values == 0,
                       f"Monetary values differ for {n_values:,} observations")
                record("quantity_candidate_preserved", n_candidate == 0,
                       f"Quantity candidate differs for {n_candidate:,} observations")
                record("imputation_provenance_preserved", n_provenance == 0,
                       f"Imputation provenance differs for {n_provenance:,} observations")
            except (duckdb.Error, ValueError, TypeError) as exc:
                for name in ("population_preserved", "monetary_values_preserved",
                             "quantity_candidate_preserved", "imputation_provenance_preserved"):
                    record(name, False, f"Cannot verify {name}: {exc}")
        else:
            for name in ("population_preserved", "monetary_values_preserved",
                         "quantity_candidate_preserved", "imputation_provenance_preserved"):
                record(name, False, f"Cannot verify {name}: input or inference artifact unavailable")

        # Coverage and descriptive findings. Missing input is not eligible.
        if checks["inference_exists"]:
            try:
                required = {"model_input_valid", "model_prediction_available"}
                missing = required - _columns(con, paths["inference"])
                if missing:
                    raise ValueError(f"Missing inference fields: {sorted(missing)}")
                has_imputation = "imputation_method" in _columns(con, paths["inference"])
                imputed_sql = (
                    "COUNT(*) FILTER (WHERE imputation_method IS NOT NULL)"
                    if has_imputation else "NULL::BIGINT"
                )
                row = con.execute(f"""
                    SELECT
                        COUNT(*) AS total,
                        COUNT(*) FILTER (WHERE model_input_valid IS TRUE) AS eligible,
                        COUNT(*) FILTER (WHERE model_prediction_available IS TRUE) AS predicted,
                        COUNT(*) FILTER (
                            WHERE model_input_valid IS TRUE
                              AND model_prediction_available IS TRUE
                        ) AS eligible_predicted,
                        {imputed_sql} AS imputed
                    FROM read_parquet({sql_literal(paths['inference'])})
                """).fetchone()
                total, eligible, predicted, eligible_predicted, imputed = row
                metrics.update({
                    "n_model_input_valid_rows": int(eligible),
                    "n_prediction_rows": int(predicted),
                    "n_imputed_value_rows": None if imputed is None else int(imputed),
                    "eligible_prediction_coverage": (
                        float(eligible_predicted) / eligible if eligible else None
                    ),
                    "population_prediction_coverage": (
                        float(predicted) / total if total else None
                    ),
                })
                coverage_ok = eligible > 0 and eligible_predicted == eligible
                record("eligible_prediction_coverage_complete", coverage_ok,
                       f"Predictions cover {eligible_predicted:,} of {eligible:,} eligible observations")
            except (duckdb.Error, ValueError, TypeError) as exc:
                record("eligible_prediction_coverage_complete", False,
                       f"Cannot evaluate model coverage: {exc}")
        else:
            record("eligible_prediction_coverage_complete", False,
                   "Cannot evaluate model coverage: inference artifact unavailable")

        # ---------------------------------------------------------------
        # Value-model selection evidence. Same distinction as quantity:
        # model-selection statistics are evidence, not a quality threshold.
        # ---------------------------------------------------------------
        if checks["value_selected_model_exists"]:
            try:
                selection_columns = _columns(con, paths["value_selected_model"])
                fields = ["model_name", "model_type", "selection_value"]
                missing = set(fields) - selection_columns
                if missing:
                    raise ValueError(f"Missing value-selection fields: {sorted(missing)}")
                selection = _one_row(con, paths["value_selected_model"], tuple(fields))
                metrics["value_model_name"] = selection["model_name"]
                metrics["value_model_type"] = selection["model_type"]
                metrics["value_selection_value"] = selection["selection_value"]
                valid = (
                    bool(selection["model_name"])
                    and bool(selection["model_type"])
                    and selection["selection_value"] is not None
                    and math.isfinite(float(selection["selection_value"]))
                    and float(selection["selection_value"]) >= 0
                )
                if "selection_metric" in selection_columns:
                    metric_name = _one_row(
                        con, paths["value_selected_model"], ("selection_metric",)
                    )["selection_metric"]
                    valid = valid and metric_name in {"mae_log", "log_mae"}
                record("value_model_evidence_valid", valid,
                       "Selected value model lacks valid hold-out comparison evidence")
            except (duckdb.Error, ValueError, TypeError, OverflowError) as exc:
                record("value_model_evidence_valid", False,
                       f"Invalid value-model selection evidence: {exc}")
        else:
            record("value_model_evidence_valid", False,
                   "Selected value-model artifact unavailable")

        # ---------------------------------------------------------------
        # Value production fit: selected identity, nonempty training evidence,
        # real persisted weights. Selection statistics are NOT fit identity.
        # ---------------------------------------------------------------
        if checks["value_fitted_model_exists"] and checks["value_selected_model_exists"]:
            try:
                fit = _one_row(con, paths["value_fitted_model"], (
                    "model_name", "model_type", "training_rows",
                    "min_training_year", "max_training_year",
                    "training_data_sha256", "model_bundle_path",
                ))
                n_fit = int(fit["training_rows"])
                first_year = int(fit["min_training_year"])
                last_year = int(fit["max_training_year"])
                bundle_string = str(fit["model_bundle_path"] or "").strip()
                bundle = Path(bundle_string) if bundle_string else None
                fingerprint = str(fit["training_data_sha256"] or "")
                matches = (
                    fit["model_name"] == metrics["value_model_name"]
                    and fit["model_type"] == metrics["value_model_type"]
                )
                fit_ok = (
                    matches and n_fit > 0 and first_year <= last_year
                    and len(fingerprint) == 64
                )
                bundle_ok = bool(bundle and bundle.is_file() and bundle.stat().st_size > 0)
                metrics.update({
                    "value_fit_training_rows": n_fit,
                    "value_fit_min_training_year": first_year,
                    "value_fit_max_training_year": last_year,
                    "value_fit_bundle_path": bundle_string,
                })
                record("value_production_fit_valid", fit_ok,
                       "Value production-fit metadata is invalid or differs from selection")
                record("value_production_bundle_available", bundle_ok,
                       f"Value production weights are missing or empty: {bundle_string}")
            except (duckdb.Error, ValueError, TypeError, OSError, OverflowError) as exc:
                record("value_production_fit_valid", False, f"Cannot verify value fit: {exc}")
                record("value_production_bundle_available", False,
                       f"Cannot verify value production weights: {exc}")
        else:
            record("value_production_fit_valid", False, "Value fitted or selected model is missing")
            record("value_production_bundle_available", False,
                   "Value fitted or selected model is missing")

        # ---------------------------------------------------------------
        # Value calibration: genuine held-out evidence, positive ratio
        # intervals, coherent predicted-value groups, and model identity.
        # A 95% observed within-bounds rate is not a pass/fail criterion.
        # ---------------------------------------------------------------
        if checks["value_calibration_exists"] and checks["value_selected_model_exists"]:
            try:
                expected = {
                    "size_group", "lower_size", "upper_size", "lower_ratio",
                    "upper_ratio", "calibration_rows", "calibration_year",
                    "nominal_outside_rate", "model_name",
                }
                absent = sorted(expected - _columns(con, paths["value_calibration"]))
                if absent:
                    raise ValueError(f"Missing value calibration fields: {absent}")
                calibration = con.execute(f"""
                    SELECT size_group, lower_size, upper_size, lower_ratio,
                           upper_ratio, calibration_rows, calibration_year,
                           nominal_outside_rate, model_name
                    FROM read_parquet({sql_literal(paths['value_calibration'])})
                    ORDER BY lower_size, upper_size
                """).fetchall()
                if not calibration:
                    raise ValueError("Empty value calibration")
                bad = False
                total_rows = 0
                common_year = set()
                common_rate = set()
                for index, (group, lo, hi, ratio_lo, ratio_hi, n, yr, rate, name) in enumerate(calibration):
                    total_rows += int(n)
                    common_year.add(int(yr))
                    common_rate.add(float(rate))
                    bad |= (
                        int(group) != index or int(n) <= 0
                        or lo is None or hi is None or ratio_lo is None or ratio_hi is None
                        or float(lo) < 0 or not math.isfinite(float(lo))
                        or float(hi) <= float(lo)
                        or not math.isfinite(float(ratio_lo))
                        or not math.isfinite(float(ratio_hi))
                        or float(ratio_lo) <= 0 or float(ratio_hi) < float(ratio_lo)
                        or name != metrics["value_model_name"]
                    )
                    if index == 0:
                        bad |= float(lo) != 0.0
                    else:
                        bad |= float(lo) != float(calibration[index - 1][2])
                bad |= (not math.isinf(float(calibration[-1][2]))
                        or len(common_year) != 1 or len(common_rate) != 1
                        or not all(0 < rate < 1 for rate in common_rate))
                metrics.update({
                    "value_calibration_year": next(iter(common_year)) if len(common_year) == 1 else None,
                    "value_calibration_rows": total_rows,
                    "value_calibration_size_groups": len(calibration),
                    "value_calibration_nominal_outside_rate": (
                        next(iter(common_rate)) if len(common_rate) == 1 else None
                    ),
                })
                record("value_calibration_evidence_valid", not bad and total_rows > 0,
                       "Value calibration groups, ratio bounds, or held-out evidence are invalid")
            except (duckdb.Error, ValueError, TypeError, OverflowError) as exc:
                record("value_calibration_evidence_valid", False,
                       f"Cannot verify value calibration: {exc}")
        else:
            record("value_calibration_evidence_valid", False,
                   "Value calibration or value selection is missing")

        # ---------------------------------------------------------------
        # Value inference: independent eligibility mask and coverage.
        # Do not penalize rows with an unavailable predictor.
        # ---------------------------------------------------------------
        if checks["inference_exists"]:
            try:
                necessary = {
                    "value_model_input_valid", "value_model_prediction_available",
                    "model_prediction_log_value", "model_prediction_value",
                    "observed_value_available", "observed_value_valid_for_log",
                    "value_log_residual", "value_abs_log_residual",
                    "value_validation_model_name", "value_validation_model_type",
                }
                missing = sorted(necessary - _columns(con, paths["inference"]))
                if missing:
                    raise ValueError(f"Missing value inference fields: {missing}")
                row = con.execute(f"""
                    SELECT COUNT(*),
                           COUNT(*) FILTER (WHERE value_model_input_valid IS TRUE),
                           COUNT(*) FILTER (WHERE value_model_prediction_available IS TRUE),
                           COUNT(*) FILTER (
                               WHERE value_model_input_valid IS TRUE
                                 AND value_model_prediction_available IS TRUE),
                           COUNT(*) FILTER (
                               WHERE value_model_input_valid IS NULL
                                  OR value_model_prediction_available IS NULL),
                           COUNT(*) FILTER (
                               WHERE value_model_prediction_available IS TRUE
                                 AND (value_validation_model_name IS DISTINCT FROM ?
                                   OR value_validation_model_type IS DISTINCT FROM ?))
                    FROM read_parquet({sql_literal(paths['inference'])})
                """, [metrics["value_model_name"], metrics["value_model_type"]]).fetchone()
                total, eligible, predicted, covered, null_flags, identity_mismatches = map(int, row)
                metrics["n_value_model_input_valid_rows"] = eligible
                metrics["n_value_prediction_rows"] = predicted
                metrics["value_eligible_prediction_coverage"] = covered / eligible if eligible else None
                metrics["value_population_prediction_coverage"] = predicted / total if total else None
                record("value_eligible_prediction_coverage_complete",
                       eligible > 0 and covered == eligible and predicted == eligible
                       and null_flags == 0,
                       f"Value predictions cover {covered:,} of {eligible:,} eligible rows"
                       f" (predicted={predicted:,}, null flags={null_flags:,})")
                record("value_model_identity_valid", identity_mismatches == 0,
                       f"Value model identity differs for {identity_mismatches:,} predictions")
                # Prediction and residual fields must agree with observed value.
                # Tolerance concerns only floating-point representation.
                residual_check = con.execute(f"""
                    SELECT COUNT(*) FILTER (
                        WHERE value_model_prediction_available IS TRUE AND (
                            model_prediction_value IS NULL
                            OR NOT isfinite(model_prediction_value)
                            OR model_prediction_value <= 0
                            OR model_prediction_log_value IS NULL
                            OR NOT isfinite(model_prediction_log_value)
                            OR ABS(LN(model_prediction_value) - model_prediction_log_value) > 1e-5
                        )
                    ) + COUNT(*) FILTER (
                        WHERE observed_value_valid_for_log IS TRUE
                          AND value_model_prediction_available IS TRUE AND (
                            value_log_residual IS NULL
                            OR value_abs_log_residual IS NULL
                            OR NOT isfinite(value_log_residual)
                            OR NOT isfinite(value_abs_log_residual)
                            OR ABS(value_log_residual -
                                (LN(trade_value_reconciled_constant_usd) -
                                 model_prediction_log_value)) > 1e-5
                            OR ABS(value_abs_log_residual - ABS(value_log_residual)) > 1e-5
                        )
                    )
                    FROM read_parquet({sql_literal(paths['inference'])})
                """).fetchone()[0]
                metrics["value_residual_invalid_rows"] = int(residual_check)
                record("value_prediction_residuals_valid", int(residual_check) == 0,
                       f"Value prediction/residual identities invalid for {int(residual_check):,} checks")
            except (duckdb.Error, ValueError, TypeError) as exc:
                record("value_eligible_prediction_coverage_complete", False,
                       f"Cannot verify value prediction coverage: {exc}")
                record("value_model_identity_valid", False,
                       f"Cannot verify value model identity: {exc}")
                record("value_prediction_residuals_valid", False,
                       f"Cannot verify value residuals: {exc}")
        else:
            record("value_eligible_prediction_coverage_complete", False,
                   "Value inference unavailable")
            record("value_model_identity_valid", False,
                   "Value inference unavailable")
            record("value_prediction_residuals_valid", False,
                   "Value inference unavailable")

        # ---------------------------------------------------------------
        # Joint plausibility artifact: same bilateral population, unique
        # keys, two complete categorical status columns. Outliers are
        # descriptive findings, NOT failures of the release gate.
        # ---------------------------------------------------------------
        if checks["plausibility_exists"] and checks["inference_exists"]:
            try:
                required = set(KEY) | {
                    "quantity_plausibility_status", "value_plausibility_status",
                    "quantity_lower_kg", "quantity_upper_kg",
                    "value_lower_constant_usd", "value_upper_constant_usd",
                }
                missing = sorted(required - _columns(con, paths["plausibility"]))
                if missing:
                    raise ValueError(f"Missing plausibility fields: {missing}")
                source = sql_literal(paths["inference"])
                scored = sql_literal(paths["plausibility"])
                key_list = ', '.join(f'"{k}"' for k in KEY)
                pop = con.execute(f"""
                    WITH i AS (SELECT {key_list} FROM read_parquet({source})),
                         p AS (SELECT {key_list} FROM read_parquet({scored}))
                    SELECT
                        (SELECT COUNT(*) FROM p),
                        (SELECT COUNT(*) - COUNT(DISTINCT ({key_list})) FROM p),
                        (SELECT COUNT(*) FROM (SELECT * FROM i EXCEPT SELECT * FROM p)),
                        (SELECT COUNT(*) FROM (SELECT * FROM p EXCEPT SELECT * FROM i))
                """).fetchone()
                n_rows, duplicate, missing_keys, extra_keys = map(int, pop)
                metrics.update({
                    "n_plausibility_rows": n_rows,
                    "n_plausibility_duplicate_keys": duplicate,
                    "n_plausibility_missing_keys": missing_keys,
                    "n_plausibility_unexpected_keys": extra_keys,
                })
                record("plausibility_population_preserved",
                       n_rows > 0 and duplicate == 0 and missing_keys == 0
                       and extra_keys == 0,
                       f"Plausibility key mismatch: {missing_keys:,} missing, "
                       f"{extra_keys:,} unexpected, {duplicate:,} duplicates")

                statuses = ("within_bounds", "below_bounds", "above_bounds", "not_assessed")
                for label, lower, upper in (
                    ("quantity", "quantity_lower_kg", "quantity_upper_kg"),
                    ("value", "value_lower_constant_usd", "value_upper_constant_usd"),
                ):
                    status_col = f"{label}_plausibility_status"
                    aggregate = ', '.join(
                        f"COUNT(*) FILTER (WHERE {status_col} = '{status}')" for status in statuses
                    )
                    row = con.execute(f"""
                        SELECT {aggregate},
                               COUNT(*) FILTER (
                                   WHERE {status_col} IS NULL OR
                                         {status_col} NOT IN
                                         ('within_bounds', 'below_bounds', 'above_bounds', 'not_assessed')),
                               COUNT(*) FILTER (
                                   WHERE {status_col} <> 'not_assessed'
                                   AND ({lower} IS NULL OR {upper} IS NULL
                                     OR NOT isfinite({lower}) OR NOT isfinite({upper})
                                     OR {lower} <= 0 OR {upper} < {lower}))
                        FROM read_parquet({scored})
                    """).fetchone()
                    counts = list(map(int, row))
                    for status, count in zip(statuses, counts[:4]):
                        metrics[f"{label}_{status}"] = count
                    bad_status, bad_bounds = counts[4:]
                    record(f"{label}_plausibility_structure_valid",
                           sum(counts[:4]) == n_rows and bad_status == 0
                           and bad_bounds == 0,
                           f"Invalid {label} plausibility: {bad_status:,} statuses, "
                           f"{bad_bounds:,} assessed bounds")
                    if label == "value":
                        mismatch = con.execute(f"""
                            SELECT COUNT(*)
                            FROM read_parquet({scored}) AS p
                            JOIN read_parquet({source}) AS i
                              USING (year, exporter_iso, importer_iso, target_cmd)
                            WHERE p.value_plausibility_status IS DISTINCT FROM
                              CASE
                                WHEN value_lower_constant_usd IS NULL
                                  OR value_upper_constant_usd IS NULL
                                  OR NOT isfinite(value_lower_constant_usd)
                                  OR NOT isfinite(value_upper_constant_usd)
                                  THEN 'not_assessed'
                                WHEN i.trade_value_reconciled_constant_usd < p.value_lower_constant_usd
                                  THEN 'below_bounds'
                                WHEN i.trade_value_reconciled_constant_usd > p.value_upper_constant_usd
                                  THEN 'above_bounds'
                                ELSE 'within_bounds'
                              END
                        """).fetchone()[0]
                        metrics["value_plausibility_classification_mismatches"] = int(mismatch)
                        record("value_plausibility_classification_valid", int(mismatch) == 0,
                               f"Value plausibility classification differs for {int(mismatch):,} rows")
            except (duckdb.Error, ValueError, TypeError) as exc:
                for check in ("plausibility_population_preserved",
                              "quantity_plausibility_structure_valid",
                              "value_plausibility_structure_valid",
                               "value_plausibility_classification_valid"):
                    record(check, False, f"Cannot verify {check}: {exc}")
        else:
            for check in ("plausibility_population_preserved",
                          "quantity_plausibility_structure_valid",
                          "value_plausibility_structure_valid",
                               "value_plausibility_classification_valid"):
                record(check, False, "Plausibility or inference artifact unavailable")

    finally:
        con.close()

    ready = bool(checks) and all(checks.values())
    audit_record = {
        "run_id": run_id,
        "ready_for_release": ready,
        **checks,
        **metrics,
        "failure_reasons": json.dumps(reasons, ensure_ascii=False),
        "release_scope": "validation_inference_and_plausibility",
    }

    # Write the decision before returning, including for failed readiness.
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    writer = duckdb.connect()
    try:
        writer.register("release_decision", pd.DataFrame([audit_record]))
        writer.execute(
            f"COPY release_decision TO {sql_literal(temporary_path)} "
            "(FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        temporary_path.replace(output_path)
    finally:
        writer.close()
        if temporary_path.exists():
            temporary_path.unlink()

    ready_symbol = get_symbols("hooray" if ready else "warning_light")
    if ready:
        logger.info("VALIDATION RELEASE READINESS: %s PASSED", ready_symbol)
    else:
        logger.warning("VALIDATION RELEASE READINESS: %s FAILED", ready_symbol)
        for reason in reasons:
            logger.warning("  - %s", reason)
    logger.info("ready_for_release = %s", ready)
    logger.info("Release-readiness artifact: %s", output_path)

    return {
        "status": "completed",
        "run_id": run_id,
        "output_path": output_path,
        "release_readiness_path": output_path,
        "ready_for_release": ready,
        "checks": checks,
        "failure_reasons": reasons,
        **metrics,
    }
