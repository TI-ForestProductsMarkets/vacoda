"""Publication-readiness gate for the current VACODA Validation inference product.

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
        "n_manual_review_rows": None,
        "n_imputed_value_rows": None,
        "model_name": None,
        "model_type": None,
        "selection_value": None,
        "n_oof_predictions": None,
        "n_validation_years": None,
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
                required = {"model_input_valid", "model_prediction_available", "manual_review_required"}
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
                        COUNT(*) FILTER (WHERE manual_review_required IS TRUE) AS review,
                        {imputed_sql} AS imputed
                    FROM read_parquet({sql_literal(paths['inference'])})
                """).fetchone()
                total, eligible, predicted, eligible_predicted, review, imputed = row
                metrics.update({
                    "n_model_input_valid_rows": int(eligible),
                    "n_prediction_rows": int(predicted),
                    "n_manual_review_rows": int(review),
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

    finally:
        con.close()

    ready = bool(checks) and all(checks.values())
    audit_record = {
        "run_id": run_id,
        "ready_for_release": ready,
        **checks,
        **metrics,
        "failure_reasons": json.dumps(reasons, ensure_ascii=False),
        "release_scope": "validation_inference",
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

    if ready:
        logger.info("VALIDATION RELEASE READINESS: PASSED")
    else:
        logger.warning("VALIDATION RELEASE READINESS: FAILED")
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
