
"""
This script implements the HS-sync allocation workflow for the "equal_split" strategy, which assigns equal weights to all observed needs for the target HS. It reads source code coverage and mapping tables, builds allocation weights, and writes the results to output files. The script is designed to be run as part of a larger HS-sync process, and it relies on configuration settings defined in `HsSyncConfig`.
"""

from __future__ import annotations

import pandas as pd

from vacoda.config.steps.hs_sync import HsSyncConfig
from vacoda.config.pipeline import PipelineConfig
from vacoda.config.paths import resolve_log_path
from vacoda.steps.hs_sync.allocation import (
    build_allocation_weights,
    build_observed_needs_mapping,
    write_allocation_outputs,
)
from vacoda.utils.logger import get_logger, parse_log_level, setup_logging


logger = get_logger("SCRIPT.HS_SYNC_ALLOCATION")


def main() -> None:
    pipeline_cfg = PipelineConfig.from_yaml()
    cfg = HsSyncConfig.from_yaml(
        scope=pipeline_cfg.scope,
        engine=pipeline_cfg.engine,
    )

    log_level = parse_log_level(cfg.logging.level)
    log_path = resolve_log_path(
        cfg.logging.resolve_log_file(),
        timestamp=cfg.logging.timestamp,
    )
    setup_logging(level=log_level, log_file=log_path)

    if not cfg.weights.enabled:
        logger.info("Allocation workflow is disabled in config. Nothing to do.")
        return
    else:
        logger.info("Starting HS-sync allocation workflow.")
        logger.info("  Allocation target HS: %s", cfg.weights.target_hs)
        logger.info("  Allocation strategy: %s", cfg.weights.strategy.name)
        logger.info("  Allocation mappings source: %s", cfg.weights.input.mappings)

    if cfg.weights.strategy.name != "equal_split":
        raise ValueError(
            "Unsupported allocation strategy in this script: "
            f"{cfg.weights.strategy.name!r}. "
            "This script only supports 'equal_split'."
        )

    if cfg.weights.input.mappings != "coverage.mappings":
        raise ValueError(
            "Unsupported allocation input.mappings value: "
            f"{cfg.weights.input.mappings!r}. "
            "This script currently expects 'coverage.mappings'."
        )

    source_code_coverage_path = cfg.weights_input_coverage_path()
    logger.info("Loading source code coverage from: %s", source_code_coverage_path)

    source_code_coverage = pd.read_csv(
        source_code_coverage_path,
        dtype={"source_cmd": str},
    )
    logger.info(
        "Loaded source code coverage rows: %s",
        f"{len(source_code_coverage):,}",
    )

    mapping_paths = cfg.coverage_mapping_files()
    logger.info(
        "Loading %s mapping table(s) from: %s",
        len(mapping_paths),
        cfg.coverage_mapping_dir(),
    )

    mapping_tables = {
        key: pd.read_csv(path, dtype={"code.source": str, "code.target": str})
        for key, path in mapping_paths.items()
    }
    for key, df in mapping_tables.items():
        logger.info("  Mapping %s rows: %s", key, f"{len(df):,}")

    observed_mapping = build_observed_needs_mapping(
        mapping_tables=mapping_tables,
        source_code_coverage=source_code_coverage,
    )
    logger.info(
        "Observed mapping rows requiring allocation: %s",
        f"{len(observed_mapping):,}",
    )

    result = build_allocation_weights(
        mapping=observed_mapping,
    )
    logger.info("Allocation result weight rows: %s", f"{len(result.weights):,}")
    logger.info("Allocation audit rows: %s", f"{len(result.audit):,}")

    output_dir = cfg.weights_output_dir()
    logger.info("Allocation output directory: %s", output_dir)

    weights_product = cfg.weights.products.get("allocation_weights")
    if weights_product is None:
        raise KeyError("Missing required allocation product: 'allocation_weights'.")

    prefix = weights_product.filename
    suffix = "_allocation_weights"
    if not prefix.endswith(suffix):
        raise ValueError(
            "allocation.products.allocation_weights.filename must end with "
            f"{suffix!r}; got {prefix!r}."
        )
    filename_prefix = prefix[: -len(suffix)]
    logger.info("Output filename prefix: %s", filename_prefix)
    logger.info("Overwrite enabled: %s", cfg.weights.write.overwrite)

    outputs = write_allocation_outputs(
        weights=result.weights,
        audit=result.audit,
        output_dir=output_dir,
        filename_prefix=filename_prefix,
        overwrite=cfg.weights.write.overwrite,
    )

    n_failed = int((result.audit["status"] != "pass").sum())

    logger.info("HS-sync allocation workflow completed.")
    logger.info("  Weights file: %s", outputs["weights"])
    logger.info("  Audit file: %s", outputs["audit"])
    logger.info("  Allocation audit failed rows: %s", n_failed)
    logger.info("  Log file: %s", log_path)


if __name__ == "__main__":
    main()