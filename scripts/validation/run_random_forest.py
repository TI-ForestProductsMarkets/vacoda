"""
Train the VACODA Validation Random Forest on the original selected
training population and infer quantities for Release manual-review rows.

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

    pipeline_cfg = PipelineConfig.from_yaml()

    cfg = ValidationConfig.from_yaml(
        scope=pipeline_cfg.scope,
        engine=pipeline_cfg.engine,
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    setup_logging(level=parse_log_level(cfg.logging.level))
    logger = get_logger("VALIDATION.RF.EXPERIMENT")

    start = perf_counter()

    output_dir = output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    model_configs = cfg.model.models.enabled()

    if "random_forest" not in model_configs:
        raise RuntimeError("Random Forest is not enabled in Validation configuration.")

    model_cfg = model_configs["random_forest"]

    numeric_cols = list(cfg.model.features.numeric.keys())
    categorical_cols = list(cfg.model.features.categorical)

    feature_cols = list(dict.fromkeys(numeric_cols + categorical_cols))

    target_col = cfg.model.target.column

    logger.info("Random Forest experiment")
    logger.info("Training source: %s", training_path)
    logger.info("Release source: %s", release_path)
    logger.info("Threads: %s", cfg.engine.threads)

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={int(cfg.engine.threads)}")

    try:
        selected_cols = list(dict.fromkeys(
            ["year", "exporter_iso", "importer_iso", "target_cmd"]
            + feature_cols
            + [target_col]
        ))

        columns_sql = ", ".join(f'"{col}"' for col in selected_cols)

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

    logger.info("Training observations: %s", f"{len(train_df):,}")
    logger.info("Manual-review observations: %s", len(review_df))

    if train_df.empty:
        raise RuntimeError("No selected training observations.")

    if len(review_df) != 27:
        raise RuntimeError(
            f"Expected 27 manual-review rows; found {len(review_df)}."
        )

    # Verify none of the review observations occur in the training population.
    keys = ["year", "exporter_iso", "importer_iso", "target_cmd"]

    overlap = train_df[keys].merge(
        review_df[keys],
        on=keys,
        how="inner",
    )

    if not overlap.empty:
        raise RuntimeError(
            f"{len(overlap)} manual-review observations occur in training data."
        )

    # Enriched Release may not explicitly publish the derived HS4 feature.
    if "target_hs4" in feature_cols and "target_hs4" not in review_df.columns:
        review_df["target_hs4"] = (
            review_df["target_cmd"].astype("string").str[:4]
        )

    # Use VACODA's own numeric transformations and categorical preparation.
    x_train = _build_feature_frame(frame=train_df, cfg=cfg)
    x_review = _build_feature_frame(frame=review_df, cfg=cfg)

    y_train = _build_target(frame=train_df, cfg=cfg)

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

    # Same estimator factory and configured parameters as Validation.
    rf = _build_estimator(
        model_type=model_cfg.type,
        parameters=model_cfg.parameters,
        cfg=cfg,
    )

    logger.info("Fitting Random Forest...")
    logger.info("Parameters: %s", rf.get_params())

    rf.fit(X_train, y_train)

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
        raise RuntimeError("Random Forest generated invalid predictions.")

    comparison = review_df[
        [
            "year",
            "exporter_iso",
            "importer_iso",
            "target_cmd",
            "trade_synced_quantity_kg",
            "model_prediction_quantity_kg",
        ]
    ].copy()

    comparison["random_forest_prediction_kg"] = prediction_kg

    comparison["rf_to_mlp_ratio"] = (
        comparison["random_forest_prediction_kg"]
        / comparison["model_prediction_quantity_kg"]
    )

    prediction_path = output_dir / "manual_review_predictions.parquet"
    comparison.to_parquet(prediction_path, index=False)

    logger.info("Saving fitted Random Forest...")

    model_path = output_dir / "random_forest.joblib"

    joblib.dump(
        {
            "estimator": rf,
            "encoder": encoder,
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "target_transform": cfg.model.target.transform,
            "training_path": str(training_path),
            "training_rows": len(train_df),
        },
        model_path,
        compress=0,
    )

    logger.info("Experiment completed.")
    logger.info("Predictions: %s", prediction_path)
    logger.info("Model: %s", model_path)
    logger.info("Duration: %.2f seconds", perf_counter() - start)


if __name__ == "__main__":
    main()