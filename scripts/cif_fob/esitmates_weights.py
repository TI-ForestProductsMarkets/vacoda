"""
vacoda.cif_fob.estimated_rates

Estimate CIF/FOB conversion factors from the audited factor-candidate
training sample.

Method
------
1. Build a clean estimation sample using configured:
   - minimum evidence value;
   - factor bounds;
   - proxy-weight exclusion;
   - source-specific observation weights.

2. Model the logarithm of the observed CIF/FOB factor using a robust
   histogram gradient-boosting regressor with absolute-error loss.

3. Evaluate the model using the latest year as a temporal holdout.

4. Refit on all clean observations.

5. Predict a CIF/FOB factor for every bilateral row with available
   predictor information.

6. Preserve observed-factor and model-quality metadata in the output.

Important
---------
This module assumes predictor enrichment has already added the required
predictor columns to the training-sample parquet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from vacoda.config.cif_fob import CifFobConfig
from vacoda.cif_fob.prepare_input import valid_output


# ======================================================================
# Initial supported predictor specification
# ======================================================================


INITIAL_PREDICTORS = [
    "log_distance",
    "contiguity",
    "exporter_landlocked",
    "importer_landlocked",
    "log_oil_price_x_log_distance",
    "product_median_unit_value",
    "product_weight_to_value_ratio",
    "product_group",
    "year",
]

CATEGORICAL_PREDICTORS = [
    "product_group",
]

NUMERIC_PREDICTORS = [
    predictor
    for predictor in INITIAL_PREDICTORS
    if predictor not in CATEGORICAL_PREDICTORS
]


# ======================================================================
# Helpers
# ======================================================================


def _sql_literal(value: str | Path) -> str:
    """Return a safely quoted SQL string literal."""
    return "'" + str(value).replace("'", "''") + "'"


def _configured_predictors(
    cfg: CifFobConfig,
) -> list[str]:
    """Return configured predictors while preserving YAML order."""
    return cfg.method.predictors.all()


def _validate_predictor_configuration(
    cfg: CifFobConfig,
) -> None:
    """
    Verify that all predictors used by this initial estimator are present
    in the YAML configuration.
    """
    configured = set(
        _configured_predictors(cfg)
    )

    missing_from_config = [
        predictor
        for predictor in INITIAL_PREDICTORS
        if predictor not in configured
    ]

    if missing_from_config:
        raise ValueError(
            "The initial CIF/FOB estimator requires these predictors "
            "to be configured: "
            + ", ".join(missing_from_config)
        )


def _read_parquet_columns(
    *,
    con: duckdb.DuckDBPyConnection,
    path: Path,
) -> set[str]:
    """Return the column names of a parquet dataset."""
    rows = con.execute(
        """
        DESCRIBE SELECT *
        FROM read_parquet(?)
        """,
        [str(path)],
    ).fetchall()

    return {
        str(row[0])
        for row in rows
    }


def _validate_input_columns(
    *,
    con: duckdb.DuckDBPyConnection,
    input_path: Path,
) -> None:
    """Validate required training and predictor columns."""
    available = _read_parquet_columns(
        con=con,
        path=input_path,
    )

    required = {
        "year",
        "exporter_iso",
        "importer_iso",
        "target_cmd",
        "model_eligible",
        "factor_source",
        "observed_cif_fob_factor",
        "import_direct_covered_primary_value",
        "export_value_fob",
        "import_value_cif",
        "pair_uses_weight_proxy",
        "pair_uses_estimated_weight",
        "import_weight_estimated",
        *INITIAL_PREDICTORS,
    }

    missing = sorted(
        required - available
    )

    if missing:
        raise RuntimeError(
            "The CIF/FOB training sample is missing required "
            "estimation columns: "
            + ", ".join(missing)
            + ". Predictor enrichment must run before rate estimation."
        )


def _encode_categorical_predictors(
    *,
    frame: pd.DataFrame,
    category_maps: dict[str, dict[str, int]] | None = None,
) -> tuple[
    pd.DataFrame,
    dict[str, dict[str, int]],
]:
    """
    Encode categorical variables as integer category identifiers.

    HistGradientBoostingRegressor can treat these columns as categorical
    when supplied through categorical_features.
    """
    encoded = frame.copy()
    learned_maps: dict[str, dict[str, int]] = {}

    for column in CATEGORICAL_PREDICTORS:
        values = (
            encoded[column]
            .astype("string")
        )

        if category_maps is None:
            unique_values = sorted(
                value
                for value in values.dropna().unique().tolist()
            )

            mapping = {
                value: index
                for index, value in enumerate(unique_values)
            }
        else:
            mapping = category_maps[column]

        encoded[column] = (
            values
            .map(mapping)
            .astype("float64")
        )

        learned_maps[column] = mapping

    return encoded, learned_maps


def _source_weight(
    *,
    frame: pd.DataFrame,
    cfg: CifFobConfig,
) -> pd.Series:
    """Assign configured confidence weights to factor observations."""
    weights = cfg.method.estimation.source_weights

    direct_estimated = (
        frame["factor_source"].eq("direct")
        & frame["import_weight_estimated"].fillna(False)
    )

    direct_reported = (
        frame["factor_source"].eq("direct")
        & ~frame["import_weight_estimated"].fillna(False)
    )

    mirror_estimated = (
        frame["factor_source"].eq("mirror")
        & frame["pair_uses_estimated_weight"].fillna(False)
    )

    mirror_reported = (
        frame["factor_source"].eq("mirror")
        & ~frame["pair_uses_estimated_weight"].fillna(False)
    )

    result = pd.Series(
        np.nan,
        index=frame.index,
        dtype="float64",
    )

    result.loc[direct_reported] = (
        weights.direct_reported_weight
    )
    result.loc[direct_estimated] = (
        weights.direct_estimated_weight
    )
    result.loc[mirror_reported] = (
        weights.mirror_reported_weight
    )
    result.loc[mirror_estimated] = (
        weights.mirror_estimated_weight
    )

    return result


def _evidence_value(
    frame: pd.DataFrame,
) -> pd.Series:
    """Return source-appropriate monetary evidence value."""
    direct_value = frame[
        "import_direct_covered_primary_value"
    ]

    mirror_value = pd.concat(
        [
            frame["export_value_fob"],
            frame["import_value_cif"],
        ],
        axis=1,
    ).min(axis=1)

    return pd.Series(
        np.where(
            frame["factor_source"].eq("direct"),
            direct_value,
            np.where(
                frame["factor_source"].eq("mirror"),
                mirror_value,
                np.nan,
            ),
        ),
        index=frame.index,
        dtype="float64",
    )


def _prediction_error_metrics(
    *,
    observed_log: np.ndarray,
    predicted_log: np.ndarray,
) -> dict[str, float]:
    """Return interpretable temporal-holdout error metrics."""
    observed_factor = np.exp(
        observed_log
    )
    predicted_factor = np.exp(
        predicted_log
    )

    absolute_percentage_error = (
        np.abs(
            predicted_factor
            - observed_factor
        )
        /
        np.maximum(
            observed_factor,
            1e-12,
        )
    )

    return {
        "holdout_mae_log_factor": float(
            mean_absolute_error(
                observed_log,
                predicted_log,
            )
        ),
        "holdout_median_absolute_percentage_error": float(
            np.median(
                absolute_percentage_error
            )
        ),
        "holdout_p90_absolute_percentage_error": float(
            np.quantile(
                absolute_percentage_error,
                0.90,
            )
        ),
    }


def _inspect_existing_estimated_rates(
    *,
    output_path: Path,
    threads: int,
) -> dict[str, Any]:
    """Inspect an existing estimated-rate output."""
    con = duckdb.connect()

    try:
        con.execute(
            f"SET threads = {int(threads)}"
        )

        summary = con.execute(
            """
            SELECT
                COUNT(*) AS n_rows,

                COUNT(*) FILTER (
                    WHERE estimated_cif_fob_factor IS NOT NULL
                ) AS n_with_estimated_factor,

                COUNT(*) FILTER (
                    WHERE observed_cif_fob_factor IS NOT NULL
                ) AS n_with_observed_factor,

                MEDIAN(estimated_cif_fob_factor)
                    FILTER (
                        WHERE estimated_cif_fob_factor IS NOT NULL
                    )
                    AS median_estimated_factor

            FROM read_parquet(?)
            """,
            [str(output_path)],
        ).fetchone()

    finally:
        con.close()

    if summary is None:
        raise RuntimeError(
            f"Could not inspect estimated rates: {output_path}"
        )

    (
        n_rows,
        n_with_estimated_factor,
        n_with_observed_factor,
        median_estimated_factor,
    ) = summary

    return {
        "status": "reused",
        "audit_passed": True,
        "ready_for_correction": (
            int(n_with_estimated_factor or 0) > 0
        ),
        "output_path": output_path,
        "estimated_rates_path": output_path,
        "n_rows": n_rows,
        "n_with_estimated_factor": n_with_estimated_factor,
        "n_with_observed_factor": n_with_observed_factor,
        "median_estimated_factor": median_estimated_factor,
    }


# ======================================================================
# Main estimator
# ======================================================================


def estimate_cif_fob_rates(
    *,
    cfg: CifFobConfig,
    input_path: Path,
    force: bool = False,
    logger,
) -> dict[str, Any]:
    """
    Estimate bilateral/product/year CIF/FOB factors.

    Parameters
    ----------
    cfg:
        Parsed CIF/FOB configuration.
    input_path:
        Enriched training-sample parquet.
    force:
        Rebuild an existing output when True.
    logger:
        Pipeline logger.
    """
    output_path = cfg.product_path(
        "estimated_rates"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"CIF/FOB training sample not found: {input_path}"
        )

    if not force and valid_output(output_path):
        logger.info(
            "REUSE — CIF/FOB estimated rates already exist: %s",
            output_path,
        )

        return _inspect_existing_estimated_rates(
            output_path=output_path,
            threads=cfg.engine.threads,
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    _validate_predictor_configuration(
        cfg
    )

    estimation_cfg = cfg.method.estimation

    minimum_evidence_value = float(
        estimation_cfg.minimum_evidence_value_usd
    )
    minimum_factor = float(
        estimation_cfg.minimum_factor
    )
    maximum_factor = float(
        estimation_cfg.maximum_factor
    )
    exclude_proxy_weights = bool(
        estimation_cfg.exclude_proxy_weights
    )

    logger.info(
        "Estimating CIF/FOB rates."
    )
    logger.info(
        "  Input: %s",
        input_path,
    )
    logger.info(
        "  Output: %s",
        output_path,
    )
    logger.info(
        "  Predictors: %s",
        ", ".join(INITIAL_PREDICTORS),
    )
    logger.info(
        "  Minimum evidence value: %.2f USD",
        minimum_evidence_value,
    )
    logger.info(
        "  Permitted factor range: %.3f-%.3f",
        minimum_factor,
        maximum_factor,
    )
    logger.info(
        "  Exclude proxy weights: %s",
        exclude_proxy_weights,
    )
    logger.info(
        "  Model: HistGradientBoostingRegressor(loss='absolute_error')"
    )

    con = duckdb.connect()

    try:
        con.execute(
            f"SET threads = {int(cfg.engine.threads)}"
        )

        _validate_input_columns(
            con=con,
            input_path=input_path,
        )

        input_sql = _sql_literal(
            input_path
        )

        select_columns = [
            "year",
            "exporter_iso",
            "importer_iso",
            "target_cmd",
            "model_eligible",
            "factor_source",
            "observed_cif_fob_factor",
            "import_direct_covered_primary_value",
            "export_value_fob",
            "import_value_cif",
            "pair_uses_weight_proxy",
            "pair_uses_estimated_weight",
            "import_weight_estimated",
            *[
                predictor
                for predictor in INITIAL_PREDICTORS
                if predictor != "year"
            ],
        ]

        quoted_select = ",\n".join(
            f'"{column}"'
            for column in select_columns
        )

        data = con.execute(
            f"""
            SELECT
                {quoted_select}
            FROM read_parquet({input_sql})
            """
        ).df()

    finally:
        con.close()

    if data.empty:
        raise RuntimeError(
            "Rate-estimation input contains no rows."
        )

    # ------------------------------------------------------------------
    # Derive clean-estimation flags and observation weights
    # ------------------------------------------------------------------

    data["estimation_evidence_value"] = (
        _evidence_value(data)
    )

    data["estimation_source_weight"] = (
        _source_weight(
            frame=data,
            cfg=cfg,
        )
    )

    data["passes_minimum_value"] = (
        data["estimation_evidence_value"]
        >= minimum_evidence_value
    )

    data["passes_factor_lower_bound"] = (
        data["observed_cif_fob_factor"]
        >= minimum_factor
    )

    data["passes_factor_upper_bound"] = (
        data["observed_cif_fob_factor"]
        <= maximum_factor
    )

    if exclude_proxy_weights:
        data["passes_proxy_rule"] = (
            ~data[
                "pair_uses_weight_proxy"
            ].fillna(False)
        )
    else:
        data["passes_proxy_rule"] = True

    data["clean_estimation_eligible"] = (
        data["model_eligible"].fillna(False)
        & data["passes_minimum_value"]
        & data["passes_factor_lower_bound"]
        & data["passes_factor_upper_bound"]
        & data["passes_proxy_rule"]
        & data["estimation_source_weight"].gt(0)
        & data["observed_cif_fob_factor"].notna()
    )

    clean = data.loc[
        data["clean_estimation_eligible"]
    ].copy()

    if clean.empty:
        raise RuntimeError(
            "No observations passed the configured clean-estimation rules."
        )

    clean["log_observed_factor"] = np.log(
        clean["observed_cif_fob_factor"]
    )

    # ------------------------------------------------------------------
    # Encode predictors
    # ------------------------------------------------------------------

    predictor_frame = data[
        INITIAL_PREDICTORS
    ].copy()

    encoded_all, category_maps = (
        _encode_categorical_predictors(
            frame=predictor_frame
        )
    )

    encoded_clean = encoded_all.loc[
        clean.index
    ].copy()

    categorical_mask = [
        predictor in CATEGORICAL_PREDICTORS
        for predictor in INITIAL_PREDICTORS
    ]

    for predictor in NUMERIC_PREDICTORS:
        encoded_all[predictor] = pd.to_numeric(
            encoded_all[predictor],
            errors="coerce",
        )

        encoded_clean[predictor] = pd.to_numeric(
            encoded_clean[predictor],
            errors="coerce",
        )

    # HistGradientBoosting categorical features must have a manageable
    # number of categories.
    for predictor in CATEGORICAL_PREDICTORS:
        n_categories = len(
            category_maps[predictor]
        )

        if n_categories > 254:
            raise RuntimeError(
                f"Categorical predictor {predictor!r} has "
                f"{n_categories:,} levels. HistGradientBoosting supports "
                "at most 254 non-missing categories with the current "
                "max_bins setting. Use a broader product_group."
            )

    # ------------------------------------------------------------------
    # Temporal holdout: latest year
    # ------------------------------------------------------------------

    holdout_year = int(
        clean["year"].max()
    )

    train_mask = (
        clean["year"] < holdout_year
    )
    holdout_mask = (
        clean["year"] == holdout_year
    )

    if (
        train_mask.sum() == 0
        or holdout_mask.sum() == 0
    ):
        raise RuntimeError(
            "Temporal validation requires observations both before and "
            f"during holdout year {holdout_year}."
        )

    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=1.0,
        categorical_features=categorical_mask,
        random_state=42,
    )

    model.fit(
        encoded_clean.loc[
            train_mask,
            INITIAL_PREDICTORS,
        ],
        clean.loc[
            train_mask,
            "log_observed_factor",
        ],
        sample_weight=clean.loc[
            train_mask,
            "estimation_source_weight",
        ],
    )

    holdout_prediction_log = model.predict(
        encoded_clean.loc[
            holdout_mask,
            INITIAL_PREDICTORS,
        ]
    )

    holdout_metrics = _prediction_error_metrics(
        observed_log=clean.loc[
            holdout_mask,
            "log_observed_factor",
        ].to_numpy(),
        predicted_log=holdout_prediction_log,
    )

    logger.info(
        "  Temporal holdout year: %s",
        holdout_year,
    )
    logger.info(
        "  Holdout observations: %s",
        int(holdout_mask.sum()),
    )
    logger.info(
        "  Holdout MAE, log factor: %.6f",
        holdout_metrics[
            "holdout_mae_log_factor"
        ],
    )
    logger.info(
        "  Holdout median absolute percentage error: %.4f%%",
        100.0
        * holdout_metrics[
            "holdout_median_absolute_percentage_error"
        ],
    )
    logger.info(
        "  Holdout p90 absolute percentage error: %.4f%%",
        100.0
        * holdout_metrics[
            "holdout_p90_absolute_percentage_error"
        ],
    )

    # ------------------------------------------------------------------
    # Refit final model on all clean observations
    # ------------------------------------------------------------------

    final_model = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=1.0,
        categorical_features=categorical_mask,
        random_state=42,
    )

    final_model.fit(
        encoded_clean[
            INITIAL_PREDICTORS
        ],
        clean[
            "log_observed_factor"
        ],
        sample_weight=clean[
            "estimation_source_weight"
        ],
    )

    predicted_log_factor = (
        final_model.predict(
            encoded_all[
                INITIAL_PREDICTORS
            ]
        )
    )

    predicted_factor = np.exp(
        predicted_log_factor
    )

    predicted_factor = np.clip(
        predicted_factor,
        minimum_factor,
        maximum_factor,
    )

    data[
        "predicted_log_cif_fob_factor"
    ] = predicted_log_factor

    data[
        "estimated_cif_fob_factor"
    ] = predicted_factor

    data[
        "estimated_factor_method"
    ] = "hist_gradient_boosting_absolute_error"

    data[
        "estimation_holdout_year"
    ] = holdout_year

    data[
        "estimation_training_observations"
    ] = len(clean)

    # Observed direct or mirror factors remain evidence. Estimated factors
    # are always provided separately and are not overwritten.
    data[
        "factor_for_application"
    ] = data[
        "estimated_cif_fob_factor"
    ]

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------

    con = duckdb.connect()

    try:
        con.execute(
            f"SET threads = {int(cfg.engine.threads)}"
        )

        con.register(
            "estimated_rates_df",
            data,
        )

        output_sql = _sql_literal(
            output_path
        )

        con.execute(
            f"""
            COPY (
                SELECT *
                FROM estimated_rates_df
            )
            TO {output_sql}
            (
                FORMAT PARQUET,
                COMPRESSION ZSTD
            )
            """
        )

        audit_row = con.execute(
            f"""
            WITH duplicate_keys AS (
                SELECT
                    year,
                    exporter_iso,
                    importer_iso,
                    target_cmd,
                    COUNT(*) AS n_rows

                FROM read_parquet({output_sql})

                GROUP BY
                    year,
                    exporter_iso,
                    importer_iso,
                    target_cmd

                HAVING COUNT(*) > 1
            )

            SELECT
                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                ) AS n_output_rows,

                (
                    SELECT COUNT(*)
                    FROM duplicate_keys
                ) AS n_duplicate_keys,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE clean_estimation_eligible
                ) AS n_clean_estimation_observations,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE estimated_cif_fob_factor IS NOT NULL
                ) AS n_with_estimated_factor,

                (
                    SELECT COUNT(*)
                    FROM read_parquet({output_sql})
                    WHERE estimated_cif_fob_factor
                          < {minimum_factor}
                       OR estimated_cif_fob_factor
                          > {maximum_factor}
                ) AS n_prediction_bound_failures,

                (
                    SELECT MEDIAN(
                        estimated_cif_fob_factor
                    )
                    FROM read_parquet({output_sql})
                ) AS median_estimated_factor,

                (
                    SELECT QUANTILE_CONT(
                        estimated_cif_fob_factor,
                        0.05
                    )
                    FROM read_parquet({output_sql})
                ) AS estimated_factor_p05,

                (
                    SELECT QUANTILE_CONT(
                        estimated_cif_fob_factor,
                        0.95
                    )
                    FROM read_parquet({output_sql})
                ) AS estimated_factor_p95

            """
        ).fetchone()

    finally:
        con.close()

    (
        n_output_rows,
        n_duplicate_keys,
        n_clean_estimation_observations,
        n_with_estimated_factor,
        n_prediction_bound_failures,
        median_estimated_factor,
        estimated_factor_p05,
        estimated_factor_p95,
    ) = audit_row

    if n_output_rows != len(data):
        raise RuntimeError(
            "Estimated-rate output row count does not match input: "
            f"output={n_output_rows:,}, input={len(data):,}."
        )

    if n_duplicate_keys > 0:
        raise RuntimeError(
            "Estimated-rate output contains duplicate bilateral keys: "
            f"{n_duplicate_keys:,}."
        )

    if n_with_estimated_factor != n_output_rows:
        raise RuntimeError(
            "Some output rows do not have an estimated CIF/FOB factor: "
            f"estimated={n_with_estimated_factor:,}, "
            f"rows={n_output_rows:,}."
        )

    if n_prediction_bound_failures > 0:
        raise RuntimeError(
            "Estimated factors fall outside configured limits: "
            f"{n_prediction_bound_failures:,}."
        )

    audit_passed = True
    ready_for_correction = (
        n_with_estimated_factor > 0
    )

    logger.info(
        "CIF/FOB rates estimated."
    )
    logger.info(
        "  Output rows: %s",
        n_output_rows,
    )
    logger.info(
        "  Clean estimation observations: %s",
        n_clean_estimation_observations,
    )
    logger.info(
        "  Rows with estimated factor: %s",
        n_with_estimated_factor,
    )
    logger.info(
        "  Duplicate bilateral keys: %s",
        n_duplicate_keys,
    )
    logger.info(
        "  Median estimated factor: %s",
        median_estimated_factor,
    )
    logger.info(
        "  Estimated factor p05/p95: %s / %s",
        estimated_factor_p05,
        estimated_factor_p95,
    )
    logger.info(
        "  Step 4 audit passed: %s",
        audit_passed,
    )
    logger.info(
        "  Ready for trade correction: %s",
        ready_for_correction,
    )
    logger.info(
        "  Output: %s",
        output_path,
    )

    return {
        "status": "completed",
        "audit_passed": audit_passed,
        "ready_for_correction": ready_for_correction,
        "output_path": output_path,
        "estimated_rates_path": output_path,
        "n_output_rows": n_output_rows,
        "n_clean_estimation_observations": (
            n_clean_estimation_observations
        ),
        "n_with_estimated_factor": n_with_estimated_factor,
        "n_duplicate_keys": n_duplicate_keys,
        "median_estimated_factor": median_estimated_factor,
        "estimated_factor_p05": estimated_factor_p05,
        "estimated_factor_p95": estimated_factor_p95,
        "holdout_year": holdout_year,
        **holdout_metrics,
    }
