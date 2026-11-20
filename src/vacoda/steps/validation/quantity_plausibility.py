"""
vacoda.steps.validation.quantity_plausibility

Apply held-out quantity plausibility bounds to Validation inference.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from vacoda.config.steps.validation import ValidationConfig
from vacoda.pipeline.artifact_manifest import (
    file_stat_fingerprint,
    manifest_path_for,
    validate_manifest,
    write_manifest,
)
from vacoda.utils.sql import sql_identifier, sql_literal


def _calibration_edges(
    prediction: np.ndarray,
    tail: float,
    *,
    max_size_groups: int,
    min_expected_rows_per_tail: int,
) -> np.ndarray:
    """Derive size groups from held-out predictions."""
    minimum_rows = int(np.ceil(min_expected_rows_per_tail / tail))
    max_groups = min(max_size_groups, max(1, len(prediction) // minimum_rows))

    for count in range(max_groups, 1, -1):
        interior = np.unique(np.quantile(prediction, np.arange(1, count) / count))
        edges = np.r_[0.0, interior, np.inf]
        groups = np.searchsorted(edges[1:-1], prediction, side="right")
        if np.bincount(groups, minlength=len(edges) - 1).min() >= minimum_rows:
            return edges
    return np.array([0.0, np.inf])


def _predict_calibration_fold(*, cfg, training, holdout, model, logger):
    # Use the same transforms, fold-local encoders, and estimator as model training.
    from vacoda.steps.validation.model_training import (
        _build_estimator,
        _build_feature_frame,
        _build_target,
        _encode_embedding_features,
        _encode_features,
        _fit_predict_embedding_mlp,
        _inverse_target,
    )

    x_train = _build_feature_frame(frame=training, cfg=cfg)
    x_test = _build_feature_frame(frame=holdout, cfg=cfg)
    y_train = _build_target(frame=training, cfg=cfg)
    if model.type == "embedding_mlp":
        train_num, test_num, train_cat, test_cat, sizes = _encode_embedding_features(
            x_train=x_train, x_validation=x_test, cfg=cfg,
        )
        transformed = _fit_predict_embedding_mlp(
            x_train_numeric=train_num, x_train_categorical=train_cat, y_train=y_train,
            x_validation_numeric=test_num, x_validation_categorical=test_cat,
            category_sizes=sizes, parameters=model.parameters, cfg=cfg, logger=logger,
        )
    else:
        x_fit, x_score = _encode_features(
            x_train=x_train, x_validation=x_test, y_train=y_train, cfg=cfg,
        )
        estimator = _build_estimator(model_type=model.type, parameters=model.parameters, cfg=cfg)
        estimator.fit(x_fit, y_train)
        transformed = estimator.predict(x_score)

    return _inverse_target(values=transformed, cfg=cfg)

def _load_calibration_fold(*, 
                           cfg, 
                           con, 
                           training_sample_path, 
                           key_columns=(),
                           ) -> tuple[int, pd.DataFrame, pd.DataFrame]:
    """Load selected training rows and usable rows from a later year."""
    settings = cfg.quantity_plausibility

    target = cfg.model.target.column
    numeric = cfg.model.features.numeric
    year_col = cfg.model.validation.year_column
    columns = dict.fromkeys([
        *key_columns, year_col, target, *numeric, *cfg.model.features.categorical,
    ])
    projection = ", ".join(map(sql_identifier, columns))
    year = sql_identifier(year_col)
    source = sql_literal(training_sample_path)

    # Match the feature builder and the quantity target's requirements.
    eligible = [
        "training_eligible IS TRUE",
        f"isfinite({sql_identifier(target)})",
        f"{sql_identifier(target)} > 0",
    ]
    for name, feature in numeric.items():
        column = sql_identifier(name)
        eligible.append(f"isfinite({column})")
        if feature.transform == "log":
            eligible.append(f"{column} > 0")
    usable = " AND ".join(eligible)
    years = con.execute(f"""
        SELECT {year},
               count(*) FILTER (WHERE training_selected IS TRUE),
               count(*) FILTER (WHERE {usable})
        FROM read_parquet({source})
        WHERE {year} IS NOT NULL
        GROUP BY {year}
        ORDER BY {year}
    """).fetchall()
    candidates = years[:len(years) - settings.exclude_latest_years] if (
        settings.exclude_latest_years < len(years)
    ) else []
    eligible_years = []
    earlier_training_years = 0
    for candidate_year, training_count, holdout_count in candidates:
        if holdout_count and earlier_training_years >= cfg.model.validation.minimum_training_years:
            eligible_years.append(candidate_year)
        earlier_training_years += int(training_count > 0)
    if not eligible_years:
        raise ValueError(
            "No usable quantity calibration year with enough prior selected training years "
            "after excluding the configured number of latest years"
        )

    calibration_year = eligible_years[-1]
    training = con.execute(
        f"SELECT {projection} FROM read_parquet({source}) "
        f"WHERE {year} < ? AND training_selected IS TRUE",
        [calibration_year],
    ).df()

    holdout = con.execute(
        f"SELECT {projection} FROM read_parquet({source}) "
        f"WHERE {year} = ? AND {usable}",
        [calibration_year],
    ).df()

    return calibration_year, training, holdout


def _build_or_reuse_calibration(
    *, cfg, con, training_sample_path, calibration_path, primary_model_name, force, logger,
):
    settings = cfg.quantity_plausibility
    model = cfg.model.models.enabled()[primary_model_name]
    payload = {
        "training_sample": file_stat_fingerprint(training_sample_path),
        "settings": asdict(settings),
        "target": asdict(cfg.model.target),
        "features": asdict(cfg.model.features),
        "validation": asdict(cfg.model.validation),
        "model_name": primary_model_name,
        "model": asdict(model),
        "random_seed": cfg.model.random_seed,
        "training_cutoff": cfg.model.training_sample.max_normalized_uv_distance_mad,
    }
    manifest = manifest_path_for(calibration_path)
    reusable, reason = validate_manifest(
        artifact_path=calibration_path, manifest_path=manifest, expected_payload=payload,
    )
    if reusable and not force:
        logger.info("Reusing quantity plausibility calibration: %s", calibration_path)
        return

    calibration_year, training, holdout = _load_calibration_fold(
        cfg=cfg, con=con, training_sample_path=training_sample_path,
    )
    target = cfg.model.target.column
    logger.info("Calibrating quantity bounds on %d (%s)", calibration_year, reason)

    logger.info("Quantity calibration: %s training rows, %s held-out rows",
                f"{len(training):,}", f"{len(holdout):,}")
    prediction = _predict_calibration_fold(
        cfg=cfg, training=training, holdout=holdout, model=model, logger=logger,
    )
    if len(prediction) != len(holdout) or not (np.isfinite(prediction) & (prediction > 0)).all():
        raise ValueError("Invalid predictions in the quantity calibration fold")

    ratios = holdout[target].to_numpy() / prediction
    if not (np.isfinite(ratios) & (ratios > 0)).all():
        raise ValueError("Nonpositive or nonfinite ratio in the quantity calibration fold")

    tail = settings.nominal_outside_rate / 2
    edges = _calibration_edges(
        prediction,
        tail,
        max_size_groups=settings.max_size_groups,
        min_expected_rows_per_tail=settings.min_expected_rows_per_tail,
    )

    size_group = np.searchsorted(edges[1:-1], prediction, side="right")
    grouped = pd.Series(ratios).groupby(size_group)
    bounds = grouped.quantile([tail, 1 - tail]).unstack()
    bounds.columns = ["lower_ratio", "upper_ratio"]
    bounds["calibration_rows"] = grouped.size()
    bounds.index.name = "size_group"
    bounds = bounds.reset_index().assign(
        lower_size_kg=edges[:-1],
        upper_size_kg=edges[1:],
        calibration_year=calibration_year,
        nominal_outside_rate=settings.nominal_outside_rate,
        model_name=primary_model_name,
        training_mad_cutoff=cfg.model.training_sample.max_normalized_uv_distance_mad,
    )
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    
    with TemporaryDirectory(dir=calibration_path.parent) as directory:
        temporary_path = Path(directory) / calibration_path.name
        bounds.to_parquet(temporary_path, index=False)
        temporary_path.replace(calibration_path)
    write_manifest(
        manifest_path=manifest, artifact_name="quantity_plausibility_bounds",
        semantic_payload=payload, artifact_path=calibration_path,
    )


def build_quantity_plausibility(
    *,
    cfg: ValidationConfig,
    training_sample_path: Path,
    inference_path: Path,
    primary_model_name: str,
    force: bool = False,
    logger,
) -> dict[str, Any] | None:
    settings = cfg.quantity_plausibility
    if not cfg.products.quantity_plausibility.enable:
        return None

    output_path = cfg.product_path("quantity_plausibility")
    calibration_path = output_path.with_stem(f"{output_path.stem}_bounds")
    inputs = {training_sample_path.resolve(), inference_path.resolve()}
    if (
        output_path.resolve() in inputs
        or calibration_path.resolve() in inputs
        or output_path.resolve() == calibration_path.resolve()
    ):
        raise ValueError("Quantity plausibility artifacts must use distinct paths")

    with duckdb.connect(config={"threads": cfg.engine.threads}) as con:
        _build_or_reuse_calibration(
            cfg=cfg, con=con, training_sample_path=training_sample_path,
            calibration_path=calibration_path, primary_model_name=primary_model_name,
            force=force, logger=logger,
        )
        con.execute(
            "CREATE TEMP TABLE calibration AS SELECT * FROM read_parquet(?)",
            [str(calibration_path)],
        )
        calibration_year = con.execute("SELECT min(calibration_year) FROM calibration").fetchone()[0]

        con.execute(f"""
            CREATE TEMP VIEW scored_quantity AS
            WITH eligible AS (
                SELECT i.*, c.size_group, c.lower_ratio, c.upper_ratio,
                       (
                           i.validation_model_name = {sql_literal(primary_model_name)}
                           AND i.observed_quantity_valid_for_log IS TRUE
                           AND i.model_prediction_available IS TRUE
                           AND isfinite(i.quantity_reconciled_kg)
                           AND i.quantity_reconciled_kg > 0
                           AND isfinite(i.model_prediction_quantity_kg)
                           AND i.model_prediction_quantity_kg > 0
                           AND c.size_group IS NOT NULL
                       ) AS can_assess
                FROM read_parquet({sql_literal(inference_path)}) i
                LEFT JOIN calibration c
                    ON i.model_prediction_quantity_kg >= c.lower_size_kg
                    AND i.model_prediction_quantity_kg < c.upper_size_kg
            ), bounds AS (
                SELECT year, exporter_iso, importer_iso, target_cmd, validation_model_name,
                       quantity_reconciled_kg, model_prediction_quantity_kg,
                       size_group,
                       CASE WHEN can_assess THEN model_prediction_quantity_kg * lower_ratio
                       END AS quantity_lower_kg,
                       CASE WHEN can_assess THEN model_prediction_quantity_kg * upper_ratio
                       END AS quantity_upper_kg
                FROM eligible
            )
            SELECT *,
                   CASE
                       WHEN quantity_lower_kg IS NULL OR NOT isfinite(quantity_lower_kg)
                         OR NOT isfinite(quantity_upper_kg) THEN 'not_assessed'
                       WHEN quantity_reconciled_kg < quantity_lower_kg THEN 'below_bounds'
                       WHEN quantity_reconciled_kg > quantity_upper_kg THEN 'above_bounds'
                       ELSE 'within_bounds'
                   END AS quantity_plausibility_status
            FROM bounds
        """)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=output_path.parent) as temporary_dir:
            temporary_path = Path(temporary_dir) / output_path.name
            con.execute(
                f"COPY scored_quantity TO {sql_literal(temporary_path)} (FORMAT PARQUET)"
            )
            summary = con.execute("""
                SELECT year,
                       count(*) FILTER (WHERE quantity_plausibility_status = 'within_bounds') AS within_bounds,
                       count(*) FILTER (WHERE quantity_plausibility_status = 'below_bounds') AS below_bounds,
                       count(*) FILTER (WHERE quantity_plausibility_status = 'above_bounds') AS above_bounds,
                       count(*) FILTER (WHERE quantity_plausibility_status = 'not_assessed') AS not_assessed
                FROM read_parquet(?) GROUP BY year ORDER BY year
            """, [str(temporary_path)]).df()
            temporary_path.replace(output_path)

    logger.info("Quantity plausibility output: %s", output_path)

    return {
        "quantity_plausibility_path": output_path,
        "calibration_path": calibration_path,
        "summary": summary,
        "calibration_year": calibration_year,
        "nominal_outside_rate": settings.nominal_outside_rate,
    }
