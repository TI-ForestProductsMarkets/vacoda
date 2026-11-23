"""

vacoda.steps.validation.plausibility

VACODA Validation: independently calibrate quantity and value plausibility.

Both targets share the same nominal outside rate and size-group methodology,
but are fitted, calibrated and assessed independently. No observations are altered.
"""

from __future__ import annotations

from dataclasses import asdict, replace
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
from vacoda.steps.validation.training_sample import training_definition_name
from vacoda.utils.sql import sql_identifier, sql_literal


KEYS = ("year", "exporter_iso", "importer_iso", "target_cmd")


def _calibration_edges(
    prediction: np.ndarray,
    tail: float,
    *,
    max_size_groups: int,
    min_expected_rows_per_tail: int,
) -> np.ndarray:
    """Adaptive predicted-size quantiles, each with sufficient tail evidence."""
    minimum_rows = int(np.ceil(min_expected_rows_per_tail / tail))
    max_groups = min(max_size_groups, max(1, len(prediction) // minimum_rows))
    for count in range(max_groups, 1, -1):
        interior = np.unique(np.quantile(prediction, np.arange(1, count) / count))
        edges = np.r_[0.0, interior, np.inf]
        groups = np.searchsorted(edges[1:-1], prediction, side="right")
        if np.bincount(groups, minlength=len(edges) - 1).min() >= minimum_rows:
            return edges
    return np.array([0.0, np.inf])


def _predict_calibration_fold(*, model_cfg, training, holdout, model, logger):
    """Train the selected architecture strictly on data before the held-out year."""
    from vacoda.steps.validation.model_training import (
        _build_estimator,
        _build_feature_frame,
        _build_target,
        _encode_embedding_features,
        _encode_features,
        _fit_predict_embedding_mlp,
        _inverse_target,
    )

    x_train = _build_feature_frame(frame=training, cfg=model_cfg)
    x_test = _build_feature_frame(frame=holdout, cfg=model_cfg)
    y_train = _build_target(frame=training, cfg=model_cfg)
    if model.type == "embedding_mlp":
        train_num, test_num, train_cat, test_cat, sizes = _encode_embedding_features(
            x_train=x_train, x_validation=x_test, cfg=model_cfg,
        )
        transformed = _fit_predict_embedding_mlp(
            x_train_numeric=train_num,
            x_train_categorical=train_cat,
            y_train=y_train,
            x_validation_numeric=test_num,
            x_validation_categorical=test_cat,
            category_sizes=sizes,
            parameters=model.parameters,
            cfg=model_cfg,
            logger=logger,
        )
    else:
        x_fit, x_score = _encode_features(
            x_train=x_train, x_validation=x_test, y_train=y_train, cfg=model_cfg,
        )
        estimator = _build_estimator(
            model_type=model.type, parameters=model.parameters, cfg=model_cfg,
        )
        estimator.fit(x_fit, y_train)
        transformed = estimator.predict(x_score)
    return _inverse_target(values=transformed, cfg=model_cfg)


def _load_calibration_fold(
    *, cfg, model_cfg, con, training_sample_path, label,
    selection_predictions_path: Path | None = None,
) -> tuple[int, pd.DataFrame, pd.DataFrame]:
    """Use the latest non-provisional year with enough prior training years.

    For value, exclude keys marked holdout_use='selection' in the
    value-model comparison artifact, retaining the calibration partition.
    """
    
    settings = cfg.quantity_plausibility
    model = model_cfg.model
    target = model.target.column
    numeric = model.features.numeric
    year_col = model.validation.year_column
    columns = list(dict.fromkeys([
        *KEYS, year_col, target, *numeric, *model.features.categorical,
    ]))
    projection = ", ".join(map(sql_identifier, columns))
    year = sql_identifier(year_col)
    source = sql_literal(training_sample_path)

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
               COUNT(*) FILTER (WHERE training_selected IS TRUE),
               COUNT(*) FILTER (WHERE {usable})
        FROM read_parquet({source})
        WHERE {year} IS NOT NULL
        GROUP BY {year}
        ORDER BY {year}
    """).fetchall()
    candidates = (
        years[:len(years) - settings.exclude_latest_years]
        if settings.exclude_latest_years < len(years) else []
    )
    eligible_years = []
    earlier_training_years = 0
    for candidate_year, training_count, holdout_count in candidates:
        if holdout_count and earlier_training_years >= model.validation.minimum_training_years:
            eligible_years.append(candidate_year)
        earlier_training_years += int(training_count > 0)
    if not eligible_years:
        raise ValueError(f"No usable {label} calibration year with enough earlier training years")

    calibration_year = eligible_years[-1]
    training = con.execute(
        f"SELECT {projection} FROM read_parquet({source}) "
        f"WHERE {year} < ? AND training_selected IS TRUE",
        [calibration_year],
    ).df()

    if selection_predictions_path is None:
        holdout = con.execute(
            f"SELECT {projection} FROM read_parquet({source}) "
            f"WHERE {year} = ? AND {usable}",
            [calibration_year],
        ).df()
    else:
        if not Path(selection_predictions_path).is_file():
            raise FileNotFoundError(
                f"Value-model selection predictions not found: {selection_predictions_path}"
            )
        # The selection output may contain several candidate predictions per
        # record. Distinct keys make the exclusion independent of candidate count.
        # IMPORTANT: this artifact must contain SELECTION rows only.
        selection_source = sql_literal(selection_predictions_path)
        join_terms = " AND ".join(
            f"s.{sql_identifier(key)} = t.{sql_identifier(key)}" for key in KEYS
        )
        holdout = con.execute(f"""
            WITH used_for_selection AS (
                SELECT DISTINCT {', '.join(map(sql_identifier, KEYS))}
                FROM read_parquet({selection_source})
                WHERE {year} = ?
                    AND holdout_use = 'selection'
            )
            SELECT {', '.join(f't.{sql_identifier(c)}' for c in columns)}
            FROM read_parquet({source}) AS t
            WHERE t.{year} = ? AND {usable}
              AND NOT EXISTS (
                  SELECT 1 FROM used_for_selection AS s WHERE {join_terms}
              )
        """, [calibration_year, calibration_year]).df()
        excluded_count = con.execute(f"""
            SELECT COUNT(DISTINCT (year, exporter_iso, importer_iso, target_cmd))
            FROM read_parquet({selection_source})
            WHERE {year} = ?
            AND holdout_use = 'selection'
        """, [calibration_year]).fetchone()[0]
        if not excluded_count:
            raise ValueError(
                "Value selection artifact contains no keys for the calibration year; "
                "cannot verify selection/calibration separation."
            )
        logger_count = len(holdout)
        if logger_count == 0:
            raise ValueError("No reserved value calibration rows remain after selection exclusion")
    if training.empty or holdout.empty:
        raise ValueError(f"Empty {label} calibration fit or holdout population")
    return calibration_year, training, holdout


def _calibration_path(*, output_path: Path, label: str) -> Path:
    # Retain the quantity calibration name when migrating a pre-existing artifact.
    if label == "quantity":
        return output_path.with_stem(f"{output_path.stem}_quantity_bounds")
    return output_path.with_stem(f"{output_path.stem}_value_bounds")


def _build_or_reuse_calibration(
    *, cfg, model_cfg, con, training_sample_path, calibration_path,
    selected_model_name, label, force, logger,
    selection_predictions_path: Path | None = None,
):
    settings = cfg.quantity_plausibility
    model = model_cfg.model.models.enabled()[selected_model_name]
    payload = {
        "training_sample": file_stat_fingerprint(training_sample_path),
        "settings": asdict(settings),
        "target": asdict(model_cfg.model.target),
        "features": asdict(model_cfg.model.features),
        "validation": asdict(model_cfg.model.validation),
        "model_name": selected_model_name,
        "model": asdict(model),
        "random_seed": model_cfg.model.random_seed,
        "training_cutoff": model_cfg.model.training_sample.max_normalized_uv_distance_mad,
    }
    if selection_predictions_path is not None:
        payload["selection_predictions"] = file_stat_fingerprint(selection_predictions_path)
        payload["calibration_partition"] = "exclude_selection_keys_v1"

    manifest = manifest_path_for(calibration_path)
    reusable, reason = validate_manifest(
        artifact_path=calibration_path,
        manifest_path=manifest,
        expected_payload=payload,
    )
    if reusable and not force:
        logger.info("REUSE — %s plausibility calibration: %s", label, calibration_path)
        return

    calibration_year, training, holdout = _load_calibration_fold(
        cfg=cfg,
        model_cfg=model_cfg,
        con=con,
        training_sample_path=training_sample_path,
        label=label,
        selection_predictions_path=selection_predictions_path,
    )
    logger.info("Calibrating %s bounds on %s (%s)", label, calibration_year, reason)
    logger.info("  %s calibration: %s training rows, %s held-out rows",
                label.capitalize(), f"{len(training):,}", f"{len(holdout):,}")

    prediction = _predict_calibration_fold(
        model_cfg=model_cfg,
        training=training,
        holdout=holdout,
        model=model,
        logger=logger,
    )
    if len(prediction) != len(holdout) or not (np.isfinite(prediction) & (prediction > 0)).all():
        raise ValueError(f"Invalid predictions in the {label} calibration fold")
    ratios = holdout[model_cfg.model.target.column].to_numpy(dtype=float) / prediction
    if not (np.isfinite(ratios) & (ratios > 0)).all():
        raise ValueError(f"Invalid observed/predicted ratios in the {label} calibration fold")

    tail = settings.nominal_outside_rate / 2
    edges = _calibration_edges(
        prediction, tail,
        max_size_groups=settings.max_size_groups,
        min_expected_rows_per_tail=settings.min_expected_rows_per_tail,
    )
    size_group = np.searchsorted(edges[1:-1], prediction, side="right")
    grouped = pd.Series(ratios).groupby(size_group)
    bounds = grouped.quantile([tail, 1 - tail]).unstack()
    bounds.columns = ["lower_ratio", "upper_ratio"]
    bounds["calibration_rows"] = grouped.size()
    bounds.index.name = "size_group"
    size_names = (
        ("lower_size_kg", "upper_size_kg") if label == "quantity"
        else ("lower_size", "upper_size")
    )
    bounds = bounds.reset_index().assign(
        **{size_names[0]: edges[:-1], size_names[1]: edges[1:]},
        calibration_year=calibration_year,
        nominal_outside_rate=settings.nominal_outside_rate,
        model_name=selected_model_name,
        training_mad_cutoff=model_cfg.model.training_sample.max_normalized_uv_distance_mad,
    )
    if not ((bounds.lower_ratio > 0) & (bounds.upper_ratio >= bounds.lower_ratio)).all():
        raise ValueError(f"Invalid {label} calibration interval")
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=calibration_path.parent) as directory:
        temp = Path(directory) / calibration_path.name
        bounds.to_parquet(temp, index=False)
        temp.replace(calibration_path)
    write_manifest(
        manifest_path=manifest,
        artifact_name=f"{label}_plausibility_bounds",
        semantic_payload=payload,
        artifact_path=calibration_path,
    )


def _target_assessment_sql(*, label: str, model_name: str, inference_source: str, calibration_table: str) -> str:
    if label == "quantity":
        observed = "quantity_reconciled_kg"
        predicted = "model_prediction_quantity_kg"
        input_flag = "model_input_valid"
        available = "model_prediction_available"
        valid_log = "observed_quantity_valid_for_log"
        name_field = "validation_model_name"
        lower_field, upper_field = "quantity_lower_kg", "quantity_upper_kg"
        size_field = "size_group"  # Existing quantity output contract.
        calibration_min, calibration_max = "lower_size_kg", "upper_size_kg"
        legacy_columns = "i.validation_model_name, i.quantity_reconciled_kg, i.model_prediction_quantity_kg,"
    elif label == "value":
        observed = "trade_value_reconciled_constant_usd"
        predicted = "model_prediction_value"
        input_flag = "value_model_input_valid"
        available = "value_model_prediction_available"
        valid_log = "observed_value_valid_for_log"
        name_field = "value_validation_model_name"
        lower_field, upper_field = "value_lower_constant_usd", "value_upper_constant_usd"
        size_field = "value_size_group"
        calibration_min, calibration_max = "lower_size", "upper_size"
        legacy_columns = "i.value_validation_model_name, i.model_prediction_value,"
    else:
        raise ValueError(f"Unexpected target {label!r}")

    key_projection = ", ".join(f"i.{sql_identifier(key)}" for key in KEYS)
    # There may be a very small number of non-assessable observations; do not
    # assign bounds to a different or fallback model.
    return f"""
        WITH eligible AS (
            SELECT {key_projection},
                   {legacy_columns}
                   i.{sql_identifier(observed)} AS observed,
                   i.{sql_identifier(predicted)} AS prediction,
                   c.size_group,
                   c.lower_ratio, c.upper_ratio,
                   (
                       i.{sql_identifier(name_field)} = {sql_literal(model_name)}
                       AND i.{sql_identifier(input_flag)} IS TRUE
                       AND i.{sql_identifier(available)} IS TRUE
                       AND i.{sql_identifier(valid_log)} IS TRUE
                       AND isfinite(i.{sql_identifier(observed)})
                       AND i.{sql_identifier(observed)} > 0
                       AND isfinite(i.{sql_identifier(predicted)})
                       AND i.{sql_identifier(predicted)} > 0
                       AND c.size_group IS NOT NULL
                   ) AS can_assess
            FROM read_parquet({inference_source}) AS i
            LEFT JOIN {calibration_table} AS c
              ON i.{sql_identifier(predicted)} >= c.{calibration_min}
             AND i.{sql_identifier(predicted)} < c.{calibration_max}
        ), bounds AS (
            SELECT {', '.join(map(sql_identifier, KEYS))},
                   {', '.join(c.removeprefix('i.') for c in legacy_columns.strip(',').split(', '))},
                   size_group AS {size_field},
                   CASE WHEN can_assess THEN prediction * lower_ratio END AS {lower_field},
                   CASE WHEN can_assess THEN prediction * upper_ratio END AS {upper_field},
                   observed
            FROM eligible
        )
        SELECT {', '.join(map(sql_identifier, KEYS))},
               {', '.join(c.removeprefix('i.') for c in legacy_columns.strip(',').split(', '))},
               {size_field}, {lower_field}, {upper_field},
               CASE
                   WHEN {lower_field} IS NULL OR NOT isfinite({lower_field})
                     OR {upper_field} IS NULL OR NOT isfinite({upper_field})
                     THEN 'not_assessed'
                   WHEN observed < {lower_field} THEN 'below_bounds'
                   WHEN observed > {upper_field} THEN 'above_bounds'
                   ELSE 'within_bounds'
               END AS {label}_plausibility_status
        FROM bounds
    """


def build_plausibility(
    *,
    cfg: ValidationConfig,
    training_sample_path: Path,
    inference_path: Path,
    primary_model_name: str,
    value_model_name: str,
    value_selection_predictions_path: Path,
    force: bool = False,
    logger,
) -> dict[str, Any] | None:
    """Run one step that independently calibrates and assesses both targets.

    ``value_selection_predictions_path`` MUST be the value-model selection-only
    prediction artifact (all candidate models on the selection partition).
    It is used to reserve the remaining year-matched rows for calibration.
    """
    if not cfg.products.quantity_plausibility.enable:
        return None
    training_sample_path = Path(training_sample_path)
    inference_path = Path(inference_path)
    value_selection_predictions_path = Path(value_selection_predictions_path)
    for path in (training_sample_path, inference_path, value_selection_predictions_path):
        if not path.is_file():
            raise FileNotFoundError(f"Build Plausibility missing input: {path}")

    base_output = cfg.product_path("quantity_plausibility")
    output_path = (
        base_output.parent / training_definition_name(cfg=cfg) / base_output.name
    )
    # Keep existing quantity bounds filename for checkpoint reuse.
    quantity_calibration_path = output_path.with_stem(f"{output_path.stem}_bounds")
    value_calibration_path = _calibration_path(output_path=output_path, label="value")
    paths = [training_sample_path, inference_path, value_selection_predictions_path,
             quantity_calibration_path, value_calibration_path, output_path]
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("Build Plausibility artifact paths must be distinct")

    value_cfg = replace(cfg, model=cfg.value_model)
    with duckdb.connect(config={"threads": cfg.engine.threads}) as con:
        _build_or_reuse_calibration(
            cfg=cfg, model_cfg=cfg, con=con,
            training_sample_path=training_sample_path,
            calibration_path=quantity_calibration_path,
            selected_model_name=primary_model_name,
            label="quantity", force=force, logger=logger,
        )
        _build_or_reuse_calibration(
            cfg=cfg, model_cfg=value_cfg, con=con,
            training_sample_path=training_sample_path,
            calibration_path=value_calibration_path,
            selected_model_name=value_model_name,
            label="value", force=force, logger=logger,
            selection_predictions_path=value_selection_predictions_path,
        )
        con.execute("CREATE TEMP TABLE quantity_calibration AS SELECT * FROM read_parquet(?)",
                    [str(quantity_calibration_path)])
        con.execute("CREATE TEMP TABLE value_calibration AS SELECT * FROM read_parquet(?)",
                    [str(value_calibration_path)])

        # Skip the expensive full-population join when both calibrations and
        # inference match the previously written combined output.
        semantic_payload = {
            "artifact": "validation_dual_plausibility_v1",
            "inference": file_stat_fingerprint(inference_path),
            "quantity_calibration": file_stat_fingerprint(quantity_calibration_path),
            "value_calibration": file_stat_fingerprint(value_calibration_path),
            "primary_model_name": primary_model_name,
            "value_model_name": value_model_name,
        }
        reusable, reuse_reason = validate_manifest(
            artifact_path=output_path,
            manifest_path=manifest_path_for(output_path),
            expected_payload=semantic_payload,
        )
        reusable = reusable and not force
        source = sql_literal(inference_path)
        if reusable:
            logger.info("REUSE — Combined plausibility (%s)", reuse_reason)
        else:
            logger.info("REBUILD — Combined plausibility (%s)", reuse_reason)
        if not reusable:
            qty_sql = _target_assessment_sql(
                label="quantity", model_name=primary_model_name,
                inference_source=source, calibration_table="quantity_calibration",
            )
            val_sql = _target_assessment_sql(
                label="value", model_name=value_model_name,
                inference_source=source, calibration_table="value_calibration",
            )
            con.execute(f"CREATE TEMP VIEW quantity_assessment AS {qty_sql}")
            con.execute(f"CREATE TEMP VIEW value_assessment AS {val_sql}")
            key_join = " AND ".join(f"q.{sql_identifier(k)} = v.{sql_identifier(k)}" for k in KEYS)
            con.execute(f"""
                CREATE TEMP VIEW scored_plausibility AS
                SELECT q.*, v.value_validation_model_name, v.model_prediction_value,
                       v.value_size_group,
                       v.value_lower_constant_usd, v.value_upper_constant_usd,
                       v.value_plausibility_status
                FROM quantity_assessment AS q
                INNER JOIN value_assessment AS v ON {key_join}
            """)

            expected = con.execute(f"SELECT COUNT(*) FROM read_parquet({source})").fetchone()[0]
            actual, distinct_keys = con.execute("""
                SELECT COUNT(*), COUNT(DISTINCT (year, exporter_iso, importer_iso, target_cmd))
                FROM scored_plausibility
            """).fetchone()
            if expected != actual or actual != distinct_keys:
                raise RuntimeError(
                    "Plausibility row keys do not match inference: "
                    f"inference={expected}, output={actual}, distinct={distinct_keys}"
                )

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(dir=output_path.parent) as directory:
                temp = Path(directory) / output_path.name
                con.execute(f"COPY scored_plausibility TO {sql_literal(temp)} (FORMAT PARQUET)")
                summaries = {}
                for label in ("quantity", "value"):
                    summaries[label] = con.execute(f"""
                        SELECT year,
                               COUNT(*) FILTER (WHERE {label}_plausibility_status = 'within_bounds') AS within_bounds,
                               COUNT(*) FILTER (WHERE {label}_plausibility_status = 'below_bounds') AS below_bounds,
                               COUNT(*) FILTER (WHERE {label}_plausibility_status = 'above_bounds') AS above_bounds,
                               COUNT(*) FILTER (WHERE {label}_plausibility_status = 'not_assessed') AS not_assessed
                        FROM read_parquet(?) GROUP BY year ORDER BY year
                    """, [str(temp)]).df()
                temp.replace(output_path)

        else:
            summaries = {}
            for label in ("quantity", "value"):
                summaries[label] = con.execute(f"""
                    SELECT year,
                           COUNT(*) FILTER (WHERE {label}_plausibility_status = 'within_bounds') AS within_bounds,
                           COUNT(*) FILTER (WHERE {label}_plausibility_status = 'below_bounds') AS below_bounds,
                           COUNT(*) FILTER (WHERE {label}_plausibility_status = 'above_bounds') AS above_bounds,
                           COUNT(*) FILTER (WHERE {label}_plausibility_status = 'not_assessed') AS not_assessed
                    FROM read_parquet(?) GROUP BY year ORDER BY year
                """, [str(output_path)]).df()

        quantity_year = con.execute(
            "SELECT MIN(calibration_year) FROM quantity_calibration"
        ).fetchone()[0]
        value_year = con.execute(
            "SELECT MIN(calibration_year) FROM value_calibration"
        ).fetchone()[0]

    if not reusable:
        write_manifest(
            manifest_path=manifest_path_for(output_path),
            artifact_name="validation_plausibility",
            semantic_payload=semantic_payload,
            artifact_path=output_path,
        )
    logger.info("Build Plausibility output: %s", output_path)
    return {
        "plausibility_path": output_path,
        "quantity_plausibility_path": output_path,  # Backward-compatible workflow result.
        "quantity_calibration_path": quantity_calibration_path,
        "value_calibration_path": value_calibration_path,
        "calibration_path": quantity_calibration_path,  # Backward-compatible.
        "quantity_summary": summaries["quantity"],
        "value_summary": summaries["value"],
        "summary": summaries["quantity"],  # Backward-compatible.
        "quantity_calibration_year": quantity_year,
        "value_calibration_year": value_year,
        "calibration_year": quantity_year,
        "nominal_outside_rate": cfg.quantity_plausibility.nominal_outside_rate,
        "reused": reusable,
    }
