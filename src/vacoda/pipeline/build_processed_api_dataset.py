"""
vacoda.pipeline.build_processed_api_dataset
This module contains the main function for building the processed API dataset from raw API data.
This includes loading the raw API data, standardizing columns, building the base dataset, running diagnostics and recovery cycles, and preparing the final DataFrame for output.
"""
# pipeline/build_processed_api_dataset.py

import time
from pathlib import Path
import pandas as pd
from vacoda.io.loaders import load_data_from_config, load_config
from vacoda.preprocessing.base import standardize_columns
from vacoda.preprocessing.build_dataset import build_base_api_dataset
from vacoda.utils.diagnostics import run_multi_recovery_cycles
from vacoda.utils.diagnostics import compute_features_coverage, log_features_coverage
from vacoda.preprocessing.trade import standardize_trade_columns, REQUIRED_FIELDS
from vacoda.utils.logger import get_logger
from vacoda.utils.timing import log_step_timing
from vacoda.io.writer import write_parquet
from vacoda.io.paths import get_processed_data_file_path

logger = get_logger("PIPELINE")

def _normalize_pipeline_mode(mode: str | None) -> str:
    aliases = {
        "per_file": "per_file",
        "merged": "merged",
    }
    mode_norm = (mode or "per_file").strip().lower()
    if mode_norm not in aliases:
        valid = ", ".join(sorted(set(aliases.values()) | {"single", "all"}))
        raise ValueError(f"Unknown pipeline mode: {mode!r}. Valid values: {valid}")
    return aliases[mode_norm]

def build_processed_api_dataset(file_path: Path = None) -> pd.DataFrame:
    steps = [
        "Configure pipeline mode and source",
        "Load & standardize raw API data",
        "Build base dataset",
        "Coverage diagnostics",
        "Run multi-cycle weight recovery",
        "Run multi-cycle value_usd recovery",
        "Write processed dataset to parquet",
    ]
    total = len(steps)

    step_times = []
    def log_step(n: int, elapsed=None) -> None:
        logger.info("=" * 60)
        logger.info(f"STEP {n}/{total} — {steps[n - 1]}")
        logger.info("=" * 60)
        if elapsed is not None:
            logger.info(f"[Timing] Step {n} elapsed: {elapsed:.2f} seconds")

    LOG_STEP = 1
    t0 = time.perf_counter()

    ############################################################ Configure pipeline mode and source
    t_step = time.perf_counter()
    log_step(LOG_STEP)
    # --- Configuration code ---
    cfg = load_config()
    pipeline_cfg = cfg.get("pipeline", {})
    if not pipeline_cfg:
        logger.warning("[Config] 'pipeline' section missing in config. Using defaults: mode='per_file', source='api'.")
    if "mode" not in pipeline_cfg:
        logger.warning("[Config] 'pipeline.mode' missing. Defaulting to 'per_file'.")
    if "source" not in pipeline_cfg:
        logger.warning("[Config] 'pipeline.source' missing. Defaulting to 'api'.")
    logger.info(f"[Config] pipeline section: {pipeline_cfg}")
    mode = _normalize_pipeline_mode(pipeline_cfg.get("mode", "per_file"))
    source = pipeline_cfg.get("source", "api")
    if source not in ("api", "local"):
        logger.warning(f"[Config] Unknown pipeline.source: {source!r}. Should be 'api' or 'local'.")
    logger.info(f"Pipeline mode: {mode}, source: {source}")
    print("\n[Config] Pipeline configuration summary:")
    print(f"  Mode: {mode}")
    print(f"  Source: {source}")
    if mode == "per_file" and file_path is not None:
        print(f"  File: {file_path}")
    response = input("Proceed with these settings? [y/N]: ").strip().lower()
    if response not in ("y", "yes"):
        logger.error("Pipeline aborted by user after configuration prompt.")
        raise SystemExit("Pipeline stopped by user.")
    
    # --- End configuration code ---
    elapsed = log_step_timing(logger, LOG_STEP, steps[LOG_STEP-1], t_step)
    step_times.append(elapsed)
    LOG_STEP += 1

    ############################################################ Load & standardize raw API data
    t_step = time.perf_counter()
    log_step(LOG_STEP)

    if mode == "per_file":
        main = load_data_from_config(file_path=file_path)
        orig_cols = list(main.columns)
        main = standardize_trade_columns(main, source_type="api")
        mapped_cols = list(main.columns)
        missing = [col for col in REQUIRED_FIELDS if col not in main.columns]
        if missing:
            logger.error(
                f"\n[API Mapping Diagnostics]"
                f"\n  Original columns: {orig_cols}"
                f"\n  Columns after mapping: {mapped_cols}"
                f"\n  Missing required fields: {missing}"
                f"\nPlease check your API data and TRADE_MAP_API mapping. Pipeline will exit."
            )
            raise SystemExit(
                f"\n[API Mapping Diagnostics]"
                f"\n  Original columns: {orig_cols}"
                f"\n  Columns after mapping: {mapped_cols}"
                f"\n  Missing required fields: {missing}"
                f"\nPlease check your API data and TRADE_MAP_API mapping. Pipeline stopped."
            )
        data = {"main": standardize_columns(main)}
    elif mode == "merged":
        main = load_data_from_config(mode=source)
        orig_cols = list(main.columns)
        main = standardize_trade_columns(main, source_type="api")
        mapped_cols = list(main.columns)
        missing = [col for col in REQUIRED_FIELDS if col not in main.columns]
        if missing:
            logger.error(
                f"\n[API Mapping Diagnostics]"
                f"\n  Original columns: {orig_cols}"
                f"\n  Columns after mapping: {mapped_cols}"
                f"\n  Missing required fields: {missing}"
                f"\nPlease check your API data and TRADE_MAP_API mapping. Pipeline will exit."
            )
            raise SystemExit(
                f"\n[API Mapping Diagnostics]"
                f"\n  Original columns: {orig_cols}"
                f"\n  Columns after mapping: {mapped_cols}"
                f"\n  Missing required fields: {missing}"
                f"\nPlease check your API data and TRADE_MAP_API mapping. Pipeline stopped."
            )
        data = {"main": standardize_columns(main)}
    else:
        raise ValueError(f"Unknown pipeline mode: {mode}")
    logger.info("Done.")
    elapsed = log_step_timing(logger, LOG_STEP, steps[LOG_STEP-1], t_step)
    step_times.append(elapsed)
    LOG_STEP += 1

    ############################################################ Build base dataset
    t_step = time.perf_counter()
    log_step(LOG_STEP)

    n_input = len(data["main"])
    df = build_base_api_dataset({"main": data["main"]})
    n_output = len(df)
    logger.info(f"Base dataset built: {n_output:,} rows, {df.shape[1]} columns")
    if n_output != n_input:
        logger.warning(
            f"Pipeline row count mismatch: input had {n_input:,} rows, "
            f"base dataset has {n_output:,} rows (delta: {n_output - n_input:+,})."
        )
    else:
        logger.info(f"Pipeline reconciliation OK — {n_input:,} input rows → {n_output:,} output rows.")

    elapsed = log_step_timing(logger, LOG_STEP, steps[LOG_STEP-1], t_step)
    step_times.append(elapsed)
    LOG_STEP += 1

    ############################################################ Coverage diagnostics 
    t_step = time.perf_counter()
    log_step(LOG_STEP)
    coverage_before = compute_features_coverage(df, ["netweight_kg_raw", 'value_usd_raw'])
    log_features_coverage(coverage_before)
    elapsed = log_step_timing(logger, LOG_STEP, steps[LOG_STEP-1], t_step)
    step_times.append(elapsed)
    LOG_STEP += 1
    
    ########################################################### Run multi-cycle weight recovery
    t_step = time.perf_counter()
    log_step(LOG_STEP)
    df= run_multi_recovery_cycles(df, 
                                main_feature="netweight_kg_raw",
                                supporting_features=["quantity", "alt_quantity", "grossweight_kg"],
                                unit_features=["quantity_unit_code", "alt_quantity_unit_code", None],
                                recovery_col_prefix=None,
                                recovery_flag_prefix=None,
                                kg_unit_code=8,
                                logger=logger
                                )
    elapsed = log_step_timing(logger, LOG_STEP, steps[LOG_STEP-1], t_step)
    step_times.append(elapsed)
    LOG_STEP += 1
    

    ########################################################### Run multi-cycle value_usd recovery
    t_step = time.perf_counter()
    log_step(LOG_STEP)
    df= run_multi_recovery_cycles(df, 
                                main_feature="value_usd_raw",
                                supporting_features=["cif_value_usd", "fob_value_usd"],
                                unit_features=[None,  None],
                                recovery_col_prefix="value_usd",
                                recovery_flag_prefix=None,
                                kg_unit_code=8,
                                logger=logger
                                )
    elapsed = log_step_timing(logger, LOG_STEP, steps[LOG_STEP-1], t_step)
    step_times.append(elapsed)
    LOG_STEP += 1
    
    
    ############################################################ Write processed dataset to parquet
    t_step = time.perf_counter()
    log_step(LOG_STEP)

    output_path = get_processed_data_file_path()
    write_parquet(df, output_path)
    logger.info("Done. DataFrame written to parquet.")
    elapsed = log_step_timing(logger, LOG_STEP, steps[LOG_STEP-1], t_step)
    step_times.append(elapsed)
    LOG_STEP += 1


    ################################################################ Pipeline Summary
    total_time = time.perf_counter() - t0
    logger.info("")
    logger.info("=" * 60)
    logger.info("Pipeline complete.")
    logger.info("=" * 60)
    for i, elapsed in enumerate(step_times, 1):
        logger.info(f"[Timing] Step {i} ({steps[i-1]}): {elapsed:.2f} seconds")
    logger.info(f"[Timing] Total pipeline time: {total_time:.2f} seconds")
    return df
