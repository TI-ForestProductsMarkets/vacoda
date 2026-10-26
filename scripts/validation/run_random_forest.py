
"""
Train the VACODA Validation Random Forest on the selected training
population and compare its predictions with the embedding model
for Release manual-review observations.

Experimental only: does not modify Validation or Release artifacts.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from time import perf_counter

import duckdb
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import TargetEncoder

from vacoda.config.pipeline import PipelineConfig
from vacoda.config.steps.validation import ValidationConfig

from vacoda.steps.validation.model_training import (
    _build_feature_frame,
    _build_target,
    _build_estimator,
)

from vacoda.utils.logger import (
    get_logger,
    parse_log_level,
    setup_logging,
)


def main() -> None:

    # ============================================================
    # CONFIGURATION
    # ============================================================

    pipeline_cfg = PipelineConfig.from_yaml()

    cfg = ValidationConfig.from_yaml(
        scope=pipeline_cfg.scope,
        engine=pipeline_cfg.engine,
    )

    training_path = Path(
        "/storage/vacoda/data/processed/validation/target_h5/"
        "training/refwin_2_0__n30__uv2p5/"
        "comtrade_target_h5_training_sample.parquet"
    )

    release_path = Path(
        "/storage/vacoda/data/release/target_h5/"
        "cif_fob__direct__hist_gradient_boosting/"
        "trade_sync__value_weighted/"
        "validation__refwin_2_0__n30__uv2p5/"
        "model__emb_mlp_bs4096_lr0p001/"
        "vacoda_target_h5_dataset_enriched.parquet"
    )

    output_dir = Path(
        "/storage/vacoda/test/experiments/random_forest_review"
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(level=parse_log_level(cfg.logging.level))
    logger = get_logger("VALIDATION.RF.EXPERIMENT")

    start = perf_counter()

    model_configs = cfg.model.models.enabled()

    if "random_forest" not in model_configs:
        raise RuntimeError(
            "Random Forest is not enabled in Validation configuration."
        )

    model_cfg = model_configs["random_forest"]

    numeric_cols = list(cfg.model.features.numeric.keys())
    categorical_cols = list(cfg.model.features.categorical)

    feature_cols = list(dict.fromkeys(
        numeric_cols + categorical_cols
    ))

    target_col = str(cfg.model.target.column)

    if target_col in feature_cols:
        raise RuntimeError(
            f"Target {target_col!r} cannot also be a model feature."
        )

    keys = [
        "year",
        "exporter_iso",
        "importer_iso",
        "target_cmd",
    ]

    logger.info("Random Forest manual-review experiment")
    logger.info("Training source: %s", training_path)
    logger.info("Release source: %s", release_path)
    logger.info("Target: %s", target_col)
    logger.info("Threads: %s", cfg.engine.threads)

    # ============================================================
    # LOAD TRAINING AND MANUAL-REVIEW DATA
    # ============================================================

    con = duckdb.connect()

    try:
        con.execute(
            f"PRAGMA threads={int(cfg.engine.threads)}"
        )

        selected_cols = list(dict.fromkeys(
            keys + feature_cols + [target_col]
        ))

        columns_sql = ", ".join(
            '"' + col.replace('"', '""') + '"'
            for col in selected_cols
        )

        train_df = con.execute(
            f"""
            SELECT {columns_sql}
            FROM read_parquet(?)
            WHERE training_selected IS TRUE
            """,
            [str(training_path)],
        ).df()

        review_df = con.execute(
            """
            SELECT *
            FROM read_parquet(?)
            WHERE manual_review_required IS TRUE
            """,
            [str(release_path)],
        ).df()

    finally:
        con.close()

    # ============================================================
    # NORMALIZE RELEASE NAMES TO VALIDATION NAMES
    # ============================================================

    # The enriched Release publishes human-readable field names.
    # Normalize once so the Validation feature builders can be reused.

    review_df = review_df.rename(columns={
        "hs6": "target_cmd",
        "trade_quantity_kg": target_col,
        "predicted_quantity_kg": "model_prediction_quantity_kg",
        "trade_value_constant_usd": "trade_value_reconciled_constant_usd",
    })

    # Derive HS4 from the already configured HS6 product column.
    if "target_hs4" in feature_cols and "target_hs4" not in review_df.columns:
        review_df["target_hs4"] = (
            review_df["target_cmd"].astype("string").str[:4]
        )

    required_review = set(keys + feature_cols) | {
        "model_prediction_quantity_kg",
        "hs6_seen",
        "hs4_seen",
        "manual_review_required",
    }

    missing = required_review - set(review_df.columns)

    # The observed quantity is useful for comparison but is not
    # required to generate Random Forest predictions.
    if missing:
        raise RuntimeError(
            f"Release review dataset missing columns: {sorted(missing)}"
        )

    if train_df.empty:
        raise RuntimeError("No selected training observations.")

    if review_df.empty:
        raise RuntimeError("No manual-review observations found.")

    logger.info(
        "Training observations: %s",
        f"{len(train_df):,}",
    )

    logger.info(
        "Manual-review observations: %s",
        f"{len(review_df):,}",
    )

    logger.info(
        "Review population by HS6:\n%s",
        review_df.groupby("target_cmd")
        .agg(
            rows=("year", "size"),
            years=("year", "nunique"),
        )
        .sort_values("rows", ascending=False)
        .to_string(),
    )

    logger.info(
        "Review support flags:\n%s",
        review_df.groupby(
            ["hs6_seen", "hs4_seen"],
            dropna=False,
        ).size().to_string(),
    )

    # ============================================================
    # VERIFY TRAINING / REVIEW SEPARATION
    # ============================================================

    overlap = train_df[keys].merge(
        review_df[keys],
        on=keys,
        how="inner",
    )

    if not overlap.empty:
        raise RuntimeError(
            f"{len(overlap)} manual-review observations "
            "also occur in the training population."
        )

    # HS4 is a derived categorical feature that may not be
    # explicitly published in the enriched Release.

    if (
        "target_hs" in feature_cols
        and "target_hs" not in review_df.columns
    ):
        review_df["target_hs"] = (
            review_df["target_cmd"]
            .astype("string")
            .str[:6]
        )

    # ============================================================
    # BUILD FEATURES AND TARGET
    # ============================================================

    x_train = _build_feature_frame(
        frame=train_df,
        cfg=cfg,
    )

    x_review = _build_feature_frame(
        frame=review_df,
        cfg=cfg,
    )

    y_train = _build_target(
        frame=train_df,
        cfg=cfg,
    )

    # ============================================================
    # CATEGORICAL TARGET ENCODING
    # ============================================================

    logger.info("Fitting categorical TargetEncoder...")

    encoder = TargetEncoder(
        target_type="continuous",
        smooth="auto",
        cv=5,
        random_state=cfg.model.random_seed,
    )

    encoded_train = encoder.fit_transform(
        x_train[categorical_cols],
        y_train,
    )

    encoded_review = encoder.transform(
        x_review[categorical_cols]
    )

    X_train = np.column_stack([
        x_train[numeric_cols].to_numpy(dtype=float),
        np.asarray(encoded_train, dtype=float),
    ])

    X_review = np.column_stack([
        x_review[numeric_cols].to_numpy(dtype=float),
        np.asarray(encoded_review, dtype=float),
    ])

    del encoded_train, encoded_review, x_train, x_review

    logger.info("Training matrix: %s", X_train.shape)
    logger.info("Review matrix: %s", X_review.shape)

    # ============================================================
    # RANDOM FOREST
    # ============================================================

    rf = _build_estimator(
        model_type=model_cfg.type,
        parameters=model_cfg.parameters,
        cfg=cfg,
    )

    logger.info("Fitting Random Forest...")
    logger.info("Parameters: %s", rf.get_params())

    rf.fit(X_train, y_train)

    # ============================================================
    # PREDICTION
    # ============================================================

    logger.info("Generating review predictions...")

    prediction_transformed = np.asarray(
        rf.predict(X_review),
        dtype=float,
    )

    if cfg.model.target.transform == "log":
        prediction_kg = np.exp(prediction_transformed)
    else:
        prediction_kg = prediction_transformed

    if not np.isfinite(prediction_kg).all():
        raise RuntimeError(
            "Random Forest generated invalid predictions."
        )

    # ============================================================
    # COMPARE RANDOM FOREST WITH EMBEDDING MODEL
    # ============================================================

    comparison_cols = keys.copy()

    if target_col in review_df.columns:
        comparison_cols.append(target_col)

    comparison_cols.append("model_prediction_quantity_kg")

    comparison = review_df[comparison_cols].copy()

    comparison["random_forest_prediction_kg"] = prediction_kg

    embedding_prediction = pd.to_numeric(
        comparison["model_prediction_quantity_kg"],
        errors="coerce",
    )

    comparison["rf_to_mlp_ratio"] = (
        comparison["random_forest_prediction_kg"]
        / embedding_prediction.where(embedding_prediction > 0)
    )

    comparison["rf_vs_mlp_abs_log_difference"] = np.abs(
        np.log(comparison["random_forest_prediction_kg"])
        - np.log(embedding_prediction.where(embedding_prediction > 0))
    )

    logger.info(
        "Comparable RF / embedding predictions: %s",
        f"{comparison['rf_to_mlp_ratio'].notna().sum():,}",
    )

    logger.info(
        "Median RF / embedding ratio: %s",
        comparison["rf_to_mlp_ratio"].median(),
    )

    logger.info(
        "Median absolute log disagreement: %s",
        comparison["rf_vs_mlp_abs_log_difference"].median(),
    )

    # ============================================================
    # SAVE EXPERIMENTAL OUTPUTS
    # ============================================================

    prediction_path = (
        output_dir / "manual_review_predictions.parquet"
    )

    comparison.to_parquet(
        prediction_path,
        index=False,
    )

    logger.info("Saving fitted Random Forest...")

    model_path = output_dir / "random_forest.joblib"

    joblib.dump(
        {
            "estimator": rf,
            "encoder": encoder,
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "target_column": target_col,
            "target_transform": cfg.model.target.transform,
            "training_path": str(training_path),
            "training_rows": len(train_df),
            "review_rows": len(review_df),
        },
        model_path,
        compress=0,
    )

    logger.info("Experiment completed.")
    logger.info("Predictions: %s", prediction_path)
    logger.info("Model: %s", model_path)
    logger.info(
        "Duration: %.2f seconds",
        perf_counter() - start,
    )


if __name__ == "__main__":
    main()
