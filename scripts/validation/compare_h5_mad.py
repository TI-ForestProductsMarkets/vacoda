"""Fit the VACODA H5 MLP at either MAD cutoff on the same held-out year."""

import argparse
import logging
import os
from pathlib import Path
from time import perf_counter

KEYS = ["year", "exporter_iso", "importer_iso", "target_cmd"]
MAD_COLUMN = "training_normalized_uv_distance"
MODEL_NAME = "emb_mlp_bs4096_lr0p001"
TRAINING_FILENAME = "comtrade_target_h5_training_sample.parquet"
DEFINITIONS = {"2.5": "refwin_2_0__n30__uv2p5", "2.0": "refwin_2_0__n30__uv2"}


def read_rows(connection, path, columns, where, year):
    projection = ", ".join('"' + name.replace('"', '""') + '"' for name in columns)
    return connection.execute(
        f"SELECT {projection} FROM read_parquet(?) WHERE {where}",
        [str(path), year],
    ).df()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--cutoff", choices=[*DEFINITIONS, "both"], default="both")
    parser.add_argument("--threads", type=int, help="Override pipeline engine.threads")
    parser.add_argument("--validation-config", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("h5_mad_comparison"))
    parser.add_argument("--check-only", action="store_true", help="Load and validate the holdout")
    args = parser.parse_args()

    import duckdb
    import numpy as np
    import pandas as pd

    from vacoda.config.pipeline import PipelineConfig
    from vacoda.config.steps.validation import ValidationConfig
    from vacoda.steps.validation.model_training import (
        _build_feature_frame,
        _build_target,
        _encode_embedding_features,
        _fit_predict_embedding_mlp,
        _inverse_target,
    )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger("vacoda.h5_mad_comparison")
    pipeline = PipelineConfig.from_yaml().with_cli_overrides(target_hs="H5")
    requested_threads = args.threads if args.threads is not None else pipeline.engine.threads
    if requested_threads < 1:
        parser.error("--threads or engine.threads must be positive")
    cpu_limits = [requested_threads]
    if hasattr(os, "sched_getaffinity"):
        cpu_limits.append(len(os.sched_getaffinity(0)))
    if os.environ.get("SLURM_CPUS_PER_TASK"):
        cpu_limits.append(int(os.environ["SLURM_CPUS_PER_TASK"]))
    pipeline = pipeline.with_cli_overrides(threads=min(cpu_limits))
    cfg = ValidationConfig.from_yaml(
        args.validation_config, scope=pipeline.scope, engine=pipeline.engine
    )
    logger.info("Engine=%s, configured/requested threads=%d, using=%d",
                cfg.engine.name, requested_threads, cfg.engine.threads)

    numeric = cfg.model.features.numeric
    target = cfg.model.target.column
    model = cfg.model.models.enabled()[MODEL_NAME]
    columns = list(dict.fromkeys([*KEYS, target, *numeric, *cfg.model.features.categorical]))
    paths = {
        cutoff: cfg.output_dir() / "training" / definition / TRAINING_FILENAME
        for cutoff, definition in DEFINITIONS.items()
    }
    cutoffs = list(DEFINITIONS) if args.cutoff == "both" else [args.cutoff]

    with duckdb.connect(config={"threads": cfg.engine.threads}) as connection:
        logger.info("Reading common %d holdout from %s", args.year, paths["2.5"])
        holdout = read_rows(
            connection, paths["2.5"], [*columns, MAD_COLUMN],
            "year = ? AND training_eligible IS TRUE", args.year,
        )
        valid = holdout[KEYS].notna().all(axis=1)
        for name, spec in numeric.items():
            values = pd.to_numeric(holdout[name], errors="coerce")
            valid &= np.isfinite(values)
            if spec.transform == "log":
                valid &= values.gt(0)

        common = holdout.loc[valid].reset_index(drop=True)
        del holdout
        if common.empty or common.duplicated(KEYS).any():
            raise ValueError("Common holdout must be nonempty with unique bilateral keys")
        logger.info("Common %d holdout: %s rows", args.year, f"{len(common):,}")
        x_common = _build_feature_frame(frame=common, cfg=cfg)
        result_keys = common[[*KEYS, target, MAD_COLUMN]].copy()
        del common
        if args.check_only:
            return

        for cutoff in cutoffs:
            start = perf_counter()
            logger.info("MAD %s: reading selected training rows from %s", cutoff, paths["2.5"])
            training_filter = "year < ? AND training_selected IS TRUE"
            if cutoff == "2.0":
                training_filter += " AND training_normalized_uv_distance <= 2.0"

            training = read_rows(
                connection, paths["2.5"], columns,
                training_filter, args.year,
            )
            logger.info("MAD %s: building features for %s training rows",
                        cutoff, f"{len(training):,}")
            x_train = _build_feature_frame(frame=training, cfg=cfg)
            y_train = _build_target(frame=training, cfg=cfg)
            del training
            logger.info("MAD %s: scaling numeric inputs and encoding categories", cutoff)
            train_num, eval_num, train_cat, eval_cat, sizes = _encode_embedding_features(
                x_train=x_train, x_validation=x_common, cfg=cfg,
            )
            del x_train
            logger.info("MAD %s: starting MLP fit (%.1f s since cutoff started)",
                        cutoff, perf_counter() - start)
            transformed = _fit_predict_embedding_mlp(
                x_train_numeric=train_num,
                x_train_categorical=train_cat,
                y_train=y_train,
                x_validation_numeric=eval_num,
                x_validation_categorical=eval_cat,
                category_sizes=sizes,
                parameters=model.parameters,
                cfg=cfg,
                logger=logger,
            )
            del train_num, eval_num, train_cat, eval_cat, y_train
            prediction = _inverse_target(values=transformed, cfg=cfg)
            if len(prediction) != len(result_keys) or not (
                np.isfinite(prediction) & (prediction > 0)
            ).all():
                raise ValueError(f"MAD {cutoff} produced invalid quantity predictions")

            output = args.output_dir / f"h5_{args.year}_mad_{cutoff.replace('.', 'p')}.parquet"
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix(f".{os.getpid()}.tmp")
            result_keys.assign(prediction_quantity_kg=prediction).to_parquet(temporary, index=False)
            temporary.replace(output)
            logger.info("MAD %s: saved %s in %.1f s", cutoff, output, perf_counter() - start)


if __name__ == "__main__":
    main()
