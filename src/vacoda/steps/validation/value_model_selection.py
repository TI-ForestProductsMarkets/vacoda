"""
vacoda.steps.validation.value_model_selection

Compare value models on held-out trade flows.
"""

from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory

import duckdb
import numpy as np
import pandas as pd

from vacoda.pipeline.artifact_manifest import (
    file_stat_fingerprint,
    manifest_path_for,
    semantic_fingerprint,
    validate_manifest,
    write_manifest,
)
from vacoda.steps.validation.plausibility import (
    _load_calibration_fold,
    _predict_calibration_fold,
)
from vacoda.steps.validation.training_sample import training_definition_name

KEYS = ["year", "exporter_iso", "importer_iso", "target_cmd"]
PAIR_KEYS = ["exporter_iso", "importer_iso"]


def _save(frame, path, payload, artifact_name):
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as directory:
        temporary_path = Path(directory) / path.name
        frame.to_parquet(temporary_path, index=False)
        temporary_path.replace(path)
    write_manifest(
        manifest_path=manifest_path_for(path),
        artifact_name=artifact_name,
        semantic_payload=payload,
        artifact_path=path,
    )


def _reusable(path, payload):
    return validate_manifest(
        artifact_path=path,
        manifest_path=manifest_path_for(path),
        expected_payload=payload,
    )[0]


def compare_and_select_value_model(
    *, 
    cfg, 
    training_sample_path, 
    selected_model_path,
    force=False, 
    logger,
):
    if cfg.value_model is None:
        raise ValueError("Value model comparison requires validation.value_model.")

    value_cfg = replace(cfg, model=cfg.value_model)
    model_cfg = value_cfg.model

    ## Output directory structure
    paths = {}

    for product in ("value_model_predictions", "value_selected_model"):
        base = cfg.product_path(product)
        paths[f"{product}_path"] = (
            base.parent / selected_model_path.parent.name / base.name
        )

    predictions_path = paths["value_model_predictions_path"]
    selected_path = paths["value_selected_model_path"]
    candidates = model_cfg.models.enabled()
    common_payload = {
        "training_sample": file_stat_fingerprint(training_sample_path),
        "target": asdict(model_cfg.target),
        "features": asdict(model_cfg.features),
        "validation": asdict(model_cfg.validation),
        "exclude_latest_years": cfg.quantity_plausibility.exclude_latest_years,
        "random_seed": model_cfg.random_seed,
        "partition": "bilateral_pair_hash_parity",
    }
    selection_payload = {
        **common_payload,
        "candidates": [(name, asdict(model)) for name, model in candidates.items()],
        "selection_metric": "mae_log",
    }

    if not force and all(_reusable(path, selection_payload) for path in paths.values()):
        logger.info("Reusing value model comparison and selection.")
        return paths

    with duckdb.connect(config={"threads": cfg.engine.threads}) as con:
        year, training, holdout = _load_calibration_fold(
            cfg=value_cfg,
            model_cfg=value_cfg,
            con=con,
            training_sample_path=training_sample_path,
            label="value"
        )

    if holdout[KEYS].isna().any().any() or holdout.duplicated(KEYS).any():
        raise ValueError("Value holdout requires complete and unique trade keys.")

    selection_rows = pd.util.hash_pandas_object(
        holdout[PAIR_KEYS], index=False,
    ).mod(2).eq(0)
    if not selection_rows.any() or selection_rows.all():
        raise ValueError("The value holdout needs trade pairs in both partitions.")

    target = model_cfg.target.column
    base = holdout[KEYS].assign(
        observed_value_constant_usd=holdout[target],
        holdout_use=np.where(selection_rows, "selection", "calibration"),
    )
    frames = []
    scores = []

    for name, model in candidates.items():
        payload = {**common_payload, "model_name": name, "model": asdict(model)}
        token = semantic_fingerprint({"model_name": name})[:16]
        path = predictions_path.with_name(f"{predictions_path.stem}_{token}.parquet")

        if not force and _reusable(path, payload):
            logger.info("Reusing held-out value predictions for %s.", name)
            frame = pd.read_parquet(path)
        else:
            logger.info("Fitting value candidate %s on years before %d.", name, year)
            prediction = _predict_calibration_fold(
                        model_cfg=value_cfg, 
                        training=training, 
                        holdout=holdout, 
                        model=model, 
                        logger=logger,
                        )
            
            if len(prediction) != len(base) or not (
                np.isfinite(prediction) & (prediction > 0)
            ).all():
                raise ValueError(f"Invalid held-out value predictions from {name!r}.")

            frame = base.assign(model_name=name, predicted_value_constant_usd=prediction)
            _save(frame, path, payload, "value_candidate_predictions")

        comparison = frame.loc[frame["holdout_use"].eq("selection")]
        residual = (
            np.log(comparison["observed_value_constant_usd"])
            - np.log(comparison["predicted_value_constant_usd"])
        )
        mae = float(residual.abs().mean())
        scores.append((mae, name, model.type))
        frames.append(frame)
        logger.info("Value candidate %s: log MAE %.6f, signed log bias %.6f.",
                    name, mae, residual.mean())

    mae, name, model_type = min(scores)
    selected = pd.DataFrame([{
        "model_name": name,
        "model_type": model_type,
        "selection_metric": "mae_log",
        "selection_direction": "minimize",
        "selection_value": mae,
        "held_out_year": year,
        "selection_rows": int(selection_rows.sum()),
    }])

    _save(pd.concat(frames, ignore_index=True), predictions_path,
          selection_payload, "value_model_predictions")
    _save(selected, selected_path, selection_payload, "value_selected_model")
    logger.info("Selected value model: %s (held-out log MAE %.6f).", name, mae)
    return paths