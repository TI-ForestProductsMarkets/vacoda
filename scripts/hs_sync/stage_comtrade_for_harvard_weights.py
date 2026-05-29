"""
scripts.hs_sync.stage_comtrade_for_harvard_weights
Stage vacoda raw Comtrade CSV files into the folder layout expected by the
Harvard/Growth Lab conversion-weights generator.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from vacoda.config.hs_sync import (
    get_engine_threads,
    get_harvard_staging_input_pattern,
    get_harvard_staging_input_stage,
    get_harvard_staging_input_subdir,
    get_harvard_staging_inventory_filename,
    get_harvard_staging_output_stage,
    get_harvard_staging_output_subdir,
    get_harvard_staging_overwrite_bool,
    get_harvard_staging_preservation_filename,
    get_harvard_staging_report_output_kind,
    get_harvard_staging_report_timestamp_bool,
    get_harvard_staging_report_write_latest_bool,
    get_harvard_staging_value_abs_tolerance,
    get_harvard_staging_value_rel_tolerance,
    get_logging_file,
    get_logging_level,
    get_logging_timestamp_bool,
    load_hs_sync_config,
    get_harvard_staging_valid_cmd_regex,
    get_harvard_staging_hs_years,
    get_harvard_staging_year_col,
    get_harvard_staging_hs_col,
    get_harvard_staging_cmd_col,
    get_harvard_staging_value_col,
    get_harvard_staging_reporter_code_col,
    get_harvard_staging_partner_code_col,
    get_harvard_staging_flow_col,
    get_harvard_staging_quantity_col,
    get_harvard_staging_country_reference_stage,
    get_harvard_staging_country_reference_subdir,
    get_harvard_staging_country_reference_pattern,
    get_harvard_staging_country_reference_reporter_code_col,
    get_harvard_staging_country_reference_reporter_iso_col,
    get_harvard_staging_country_reference_partner_code_col,
    get_harvard_staging_country_reference_partner_iso_col,
    
    )

from vacoda.config.paths import get_data_dir, get_output_dir, resolve_log_path
from vacoda.utils.logger import (
    get_logger,
    make_timestamped_log_path,
    parse_log_level,
    rel_path,
    setup_logging,
)

from vacoda.hs_sync.harvard_stage import (
    audit_csv_to_partitioned_parquet_staging,
    audit_harvard_staging_partitioned_output,
    stage_comtrade_for_harvard_weights_fast,
    export_harvard_generator_compatibility_layout
)


def _resolve_data_stage_dir(stage: str, subdir: str = "") -> Path:
    base = get_data_dir(stage)
    return base / subdir if subdir else base


def main() -> None:
    cfg = load_hs_sync_config()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_level = parse_log_level(get_logging_level(cfg))
    log_path = resolve_log_path(
        get_logging_file(cfg),
        timestamp=run_id if get_logging_timestamp_bool(cfg) else None,)

    setup_logging(level=log_level, log_file=log_path)

    logger = get_logger("SCRIPT.HS_SYNC_STAGE_HARVARD")

    input_dir = _resolve_data_stage_dir(
        get_harvard_staging_input_stage(cfg),
        get_harvard_staging_input_subdir(cfg),
    )
    pattern = get_harvard_staging_input_pattern(cfg)
    valid_cmd_regex = get_harvard_staging_valid_cmd_regex(cfg)
    years_by_hs = get_harvard_staging_hs_years(cfg)
    year_col = get_harvard_staging_year_col(cfg)
    hs_col = get_harvard_staging_hs_col(cfg)
    cmd_col = get_harvard_staging_cmd_col(cfg)
    value_col = get_harvard_staging_value_col(cfg)
    reporter_code_col = get_harvard_staging_reporter_code_col(cfg)
    partner_code_col = get_harvard_staging_partner_code_col(cfg)
    flow_col = get_harvard_staging_flow_col(cfg)
    quantity_col = get_harvard_staging_quantity_col(cfg)
    
    output_root = (
        get_data_dir(get_harvard_staging_output_stage(cfg))
        / get_harvard_staging_output_subdir(cfg))
    output_dir = output_root / "as_reported" / "raw_parquet_partitioned"

    logger.info("Starting Harvard staging.")
    logger.info("Input directory: %s", rel_path(input_dir))
    logger.info("Input pattern: %s", pattern)
    logger.info("Output root: %s", rel_path(output_root))
    logger.info("Output directory: %s", rel_path(output_dir))
    logger.info("Valid commodity-code regex: %s", valid_cmd_regex)
    logger.info("HS/year filter: %s", years_by_hs)
    logger.info(
        "Harvard input columns: year=%s, hs=%s, cmd=%s, value=%s",
        year_col,
        hs_col,
        cmd_col,
        value_col)
    logger.info(
        "Harvard compatibility columns: reporter=%s, partner=%s, flow=%s, quantity=%s",
        reporter_code_col,
        partner_code_col,
        flow_col,
        quantity_col,
    )


    country_reference_dir = _resolve_data_stage_dir(
    get_harvard_staging_country_reference_stage(cfg),
    get_harvard_staging_country_reference_subdir(cfg),
    )

    country_reference_glob = str(
        country_reference_dir
        / get_harvard_staging_country_reference_pattern(cfg)
    )

    logger.info("Country reference directory: %s", rel_path(country_reference_dir))
    logger.info("Country reference glob: %s", country_reference_glob)
    
    summary = stage_comtrade_for_harvard_weights_fast(
    input_dir=input_dir,
    pattern=pattern,
    output_dir=output_dir,
    year_col=year_col,
    hs_col=hs_col,
    cmd_col=cmd_col,
    valid_cmd_regex=valid_cmd_regex,
    years_by_hs=years_by_hs,
    threads=get_engine_threads(cfg),
    overwrite=get_harvard_staging_overwrite_bool(cfg),
)
    
    inventory = audit_harvard_staging_partitioned_output(
        output_dir=output_dir,
    )

    preservation = audit_csv_to_partitioned_parquet_staging(
    input_dir=input_dir,
    input_pattern=pattern,
    output_dir=output_dir,
    year_col=year_col,
    hs_col=hs_col,
    cmd_col=cmd_col,
    value_col=value_col,
    valid_cmd_regex=valid_cmd_regex,
    years_by_hs=years_by_hs,
    threads=get_engine_threads(cfg),
    value_abs_tolerance=get_harvard_staging_value_abs_tolerance(cfg),
    value_rel_tolerance=get_harvard_staging_value_rel_tolerance(cfg),
)
    
    compatibility_summary = export_harvard_generator_compatibility_layout(
    staged_partitioned_dir=output_dir,
    output_root=output_root,
    years_by_hs=years_by_hs,
    year_col=year_col,
    hs_col=hs_col,
    cmd_col=cmd_col,
    value_col=value_col,
    reporter_code_col=reporter_code_col,
    partner_code_col=partner_code_col,
    flow_col=flow_col,
    qty_col=quantity_col,
    valid_cmd_regex=valid_cmd_regex,
    country_reference_glob=country_reference_glob,
    country_reporter_code_col=get_harvard_staging_country_reference_reporter_code_col(cfg),
    country_reporter_iso_col=get_harvard_staging_country_reference_reporter_iso_col(cfg),
    country_partner_code_col=get_harvard_staging_country_reference_partner_code_col(cfg),
    country_partner_iso_col=get_harvard_staging_country_reference_partner_iso_col(cfg),
    threads=get_engine_threads(cfg),
    overwrite=get_harvard_staging_overwrite_bool(cfg),
)

    report_kind = get_harvard_staging_report_output_kind(cfg)
    report_dir = get_output_dir(report_kind)
    
    inventory_base_path = report_dir / get_harvard_staging_inventory_filename(cfg)
    preservation_base_path = report_dir / get_harvard_staging_preservation_filename(cfg)

    if get_harvard_staging_report_timestamp_bool(cfg):
        inventory_path = make_timestamped_log_path(
                inventory_base_path,
                timestamp=run_id,
                )
        preservation_path = make_timestamped_log_path(
            preservation_base_path,
            timestamp=run_id,
            )
    else:
        inventory_path = inventory_base_path
        preservation_path = preservation_base_path

    inventory.to_csv(inventory_path, index=False)
    preservation.to_csv(preservation_path, index=False)

    logger.info("Inventory written: %s", rel_path(inventory_path))
    logger.info("Preservation written: %s", rel_path(preservation_path))


    if (get_harvard_staging_report_write_latest_bool(cfg) and get_harvard_staging_report_timestamp_bool(cfg)):
        inventory_latest_path = inventory_base_path.with_name(
                f"{inventory_base_path.stem}_latest{inventory_base_path.suffix}")
        preservation_latest_path = preservation_base_path.with_name(
                f"{preservation_base_path.stem}_latest{preservation_base_path.suffix}")

        inventory.to_csv(inventory_latest_path, index=False)
        preservation.to_csv(preservation_latest_path, index=False)

        logger.info("Latest inventory copy written: %s", rel_path(inventory_latest_path))
        logger.info("Latest preservation copy written: %s", rel_path(preservation_latest_path))

    logger.info("Compatibility export summary: %s", compatibility_summary)

    logger.info("Finished Harvard staging.")
    logger.info("Summary: %s", summary)
    logger.info("Log file: %s", log_path)

if __name__ == "__main__":
    main()