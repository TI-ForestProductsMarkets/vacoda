# pipeline/build_processed_dataset.py

from pathlib import Path
import pandas as pd
from vacoda.io.loaders import (
    load_single_csv_from_config,
    load_all_csvs_from_raw_dir,
    load_metadata_from_config,
    load_config
)
from vacoda.preprocessing.base import standardize_columns
from vacoda.preprocessing.build_dataset import build_base_dataset
from vacoda.preprocessing.recovery import (
    recover_weight_from_sup_quantity,
    summarize_weight_recovery,
    log_recovery_impact,
)
from vacoda.utils.diagnostics import (
    compute_weight_coverage,
    log_weight_coverage,
    assess_weight_recoverability,
    log_weight_recoverability,
)
from vacoda.utils.logger import get_logger

logger = get_logger("pipeline")


def _normalize_pipeline_mode(mode: str | None) -> str:
    """Normalize pipeline mode to canonical names while supporting legacy aliases."""
    aliases = {
        "per_file": "per_file",
        "merged": "merged",
    }
    mode_norm = (mode or "per_file").strip().lower()
    if mode_norm not in aliases:
        valid = ", ".join(sorted(set(aliases.values()) | {"single", "all"}))
        raise ValueError(f"Unknown pipeline mode: {mode!r}. Valid values: {valid}")
    return aliases[mode_norm]

def build_processed_dataset(file_path: Path = None) -> pd.DataFrame:
    steps = [
        "Configure pipeline mode and source",
        "Load & standardize raw data and metadata",
        "Build base dataset (incl. partner source harmonization)",
        "Coverage diagnostics",
        "Assessment of weight recovery",
        "Recover weight from supplementary quantity",
        "Coverage diagnostics after weight recovery",
        "Write processed dataset to parquet",
    ]
    total = len(steps)

    def log_step(n: int) -> None:
        logger.info("=" * 60)
        logger.info(f"STEP {n}/{total} — {steps[n - 1]}")
        logger.info("=" * 60)

    LOG_STEP = 1
    
    ############################################################ CONFIG
    log_step(LOG_STEP)
    # --- Pipeline ---
    cfg = load_config()
    pipeline_cfg = cfg.get("pipeline", {})
    # Diagnostics: check for missing pipeline section
    if not pipeline_cfg:
        logger.warning("[Config] 'pipeline' section missing in config. Using defaults: mode='per_file', source='api'.")
    # Diagnostics: check for missing mode/source
    if "mode" not in pipeline_cfg:
        logger.warning("[Config] 'pipeline.mode' missing. Defaulting to 'per_file'.")
    if "source" not in pipeline_cfg:
        logger.warning("[Config] 'pipeline.source' missing. Defaulting to 'api'.")
    # Print config values for diagnostics
    logger.info(f"[Config] pipeline section: {pipeline_cfg}")
    mode = _normalize_pipeline_mode(pipeline_cfg.get("mode", "per_file"))
    source = pipeline_cfg.get("source", "api")
    # Diagnostics: check for valid source
    if source not in ("api", "local"):
        logger.warning(f"[Config] Unknown pipeline.source: {source!r}. Should be 'api' or 'local'.")
    logger.info(f"Pipeline mode: {mode}, source: {source}")
    
    # Load metadata before prompting user
    metadata = load_metadata_from_config()
    # Prompt user to confirm config before proceeding
    print("\n[Config] Pipeline configuration summary:")
    print(f"  Mode: {mode}")
    print(f"  Source: {source}")
    print(f"  Metadata files: {list(metadata.keys())}")
    response = input("Proceed with these settings? [y/N]: ").strip().lower()
    if response not in ("y", "yes"): 
        logger.error("Pipeline aborted by user after configuration prompt.")
        raise SystemExit("Pipeline stopped by user.")

    ############################################################ LOAD DATA
    LOG_STEP += 1
    log_step(LOG_STEP)
    if mode == "per_file":
        main = load_single_csv_from_config(file_path=file_path, source=source)
        data = {"main": standardize_columns(main)}
        for k, v in metadata.items():
            data[k] = standardize_columns(v)
    elif mode == "merged":
        main = load_all_csvs_from_raw_dir(mode=source)
        data = {"main": standardize_columns(main)}
        for k, v in metadata.items():
            data[k] = standardize_columns(v)
    else:
        raise ValueError(f"Unknown pipeline mode: {mode}")
    logger.info("Done.")

    ############################################################ BUILD BASE DATASET
    LOG_STEP += 1
    log_step(LOG_STEP)
    n_input = len(data["main"])
    df = build_base_dataset(data)
    n_output = len(df)
    logger.info(f"Base dataset built: {n_output:,} rows, {df.shape[1]} columns")
    if n_output != n_input:
        logger.warning(
            f"Pipeline row count mismatch: input had {n_input:,} rows, "
            f"base dataset has {n_output:,} rows (delta: {n_output - n_input:+,})."
        )
    else:
        logger.info(f"Pipeline reconciliation OK — {n_input:,} input rows → {n_output:,} output rows.")

    ############################################################ COVERAGE DIAGNOSTICS BEFORE RECOVERY
    LOG_STEP += 1
    log_step(LOG_STEP)
    coverage_before = compute_weight_coverage(df, "netweight_kg_raw")
    log_weight_coverage(coverage_before)

    ############################################################ ASSESS WEIGHT RECOVERABILITY
    LOG_STEP += 1
    log_step(LOG_STEP)
    weight_recovery_stats = assess_weight_recoverability(df,
                                                        weight_col= "netweight_kg_raw",
                                                        sup_col    = "sup_quantity",
                                                        unit_col   = "unit_code",
                                                        weight_unit_code = 8)
    log_weight_recoverability(weight_recovery_stats)

    ############################################################ RECOVER WEIGHT
    LOG_STEP += 1
    log_step(LOG_STEP)
    df = recover_weight_from_sup_quantity(
        df,
        weight_col = "netweight_kg_raw",
        sup_col    = "sup_quantity",
        unit_col   = "unit_code",
    )
    summary = summarize_weight_recovery(df)
    logger.info(f"  recovered rows: {int(summary['recovered_weight_n']):,}")

    ############################################################ COVERAGE DIAGNOSTICS AFTER RECOVERY
    LOG_STEP += 1
    log_step(LOG_STEP)
    coverage_after = compute_weight_coverage(df, "netweight_kg")
    log_weight_coverage(coverage_after)
    logger.info("")
    log_recovery_impact(int(coverage_before["missing_n"]), int(coverage_after["missing_n"]))

    ############################################################ WRITE PROCESSED DATASET
    # Writing is performed by the runner script, but we still log step 7/7
    # so pipeline step numbering is complete and consistent for users.
    LOG_STEP += 1
    log_step(LOG_STEP)
    logger.info("Done. Handing off to runner for parquet write.")

    logger.info("=" * 60)
    logger.info("Pipeline complete.")
    logger.info("=" * 60)
    return df
