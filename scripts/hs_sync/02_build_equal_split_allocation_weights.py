from __future__ import annotations

import pandas as pd

from vacoda.config.hs_sync import HsSyncConfig
from vacoda.config.paths import resolve_log_path
from vacoda.hs_sync.allocation import (
    build_allocation_weights,
    build_observed_needs_mapping,
    write_allocation_outputs,
)
from vacoda.utils.logger import get_logger, parse_log_level, setup_logging


logger = get_logger("SCRIPT.HS_SYNC_ALLOCATION")


def main() -> None:
    cfg = HsSyncConfig.from_yaml()

    log_level = parse_log_level(cfg.logging.level)
    log_path = resolve_log_path(
        cfg.logging.resolve_log_file(),
        timestamp=cfg.logging.timestamp,
    )
    setup_logging(level=log_level, log_file=log_path)

    if not cfg.allocation.enabled:
        logger.info("Allocation workflow is disabled in config. Nothing to do.")
        return
    else:
        logger.info("Starting HS-sync allocation workflow.")
        logger.info("  Allocation target HS: %s", cfg.allocation.target_hs)
        logger.info("  Allocation strategy: %s", cfg.allocation.strategy.name)
        logger.info("  Allocation mappings source: %s", cfg.allocation.input.mappings)

    if cfg.allocation.strategy.name != "equal_split":
        raise ValueError(
            "Unsupported allocation strategy in this script: "
            f"{cfg.allocation.strategy.name!r}. "
            "This script only supports 'equal_split'."
        )

    if cfg.allocation.input.mappings != "coverage.mappings":
        raise ValueError(
            "Unsupported allocation input.mappings value: "
            f"{cfg.allocation.input.mappings!r}. "
            "This script currently expects 'coverage.mappings'."
        )

    target_hs_lower = cfg.allocation.target_hs.lower()

    source_code_coverage_path = cfg.allocation_input_coverage_path()
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

    output_dir = cfg.allocation_output_dir()
    logger.info("Allocation output directory: %s", output_dir)

    weights_product = cfg.allocation.products.get("allocation_weights")
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
    logger.info("Overwrite enabled: %s", cfg.allocation.write.overwrite)

    outputs = write_allocation_outputs(
        weights=result.weights,
        audit=result.audit,
        output_dir=output_dir,
        filename_prefix=filename_prefix,
        overwrite=cfg.allocation.write.overwrite,
    )

    n_failed = int((result.audit["status"] != "pass").sum())

    logger.info("HS-sync allocation workflow completed.")
    logger.info("  Weights file: %s", outputs["weights"])
    logger.info("  Audit file: %s", outputs["audit"])
    logger.info("  Allocation audit failed rows: %s", n_failed)
    logger.info("  Log file: %s", log_path)


if __name__ == "__main__":
    main()