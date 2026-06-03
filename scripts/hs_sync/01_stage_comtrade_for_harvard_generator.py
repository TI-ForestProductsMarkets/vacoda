"""
scripts.hs_sync.01_stage_comtrade_for_harvard_generator
Read raw Comtrade CSV/parquet data and create Harvard-compatible staged parquet.
Also fixes/exports reporter/partner ISO compatibility if bulk data lacks ISO fields.
"""

from __future__ import annotations

from datetime import datetime
from vacoda.config.hs_sync import HsSyncConfig
from vacoda.config.paths import resolve_log_path
from vacoda.utils.logger import (
    get_logger,
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
from vacoda.io.reports import write_dataframe_report

def main() -> None:
    
    cfg =  HsSyncConfig.from_yaml()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_level = parse_log_level(cfg.logging.level)
    log_path =  resolve_log_path(
                        cfg.logging.file,
                        timestamp=run_id if cfg.logging.timestamp else None,
                        )
    if cfg.logging.timestamp:
        log_path = f"{log_path}_{run_id}"

    setup_logging(level=log_level, log_file=log_path)

    logger = get_logger("SCRIPT.HS_SYNC_STAGE_HARVARD")

    input_dir = cfg.harvard_staging.input.resolve_data_dir()
    pattern = cfg.harvard_staging.input.pattern

    valid_cmd_regex = cfg.harvard_staging.filters.valid_cmd_regex
    years_by_hs = cfg.harvard_staging.filters.hs_years

    year_col = cfg.harvard_staging.columns.year
    hs_col = cfg.harvard_staging.columns.hs
    cmd_col = cfg.harvard_staging.columns.cmd
    value_col = cfg.harvard_staging.columns.value
    reporter_code_col = cfg.harvard_staging.columns.reporter_code
    partner_code_col = cfg.harvard_staging.columns.partner_code
    flow_col = cfg.harvard_staging.columns.flow
    quantity_col = cfg.harvard_staging.columns.quantity
    
    output_root = cfg.harvard_staging.output.resolve_data_dir()
    output_dir = output_root / "as_reported" / "raw_parquet_partitioned"

    country_reference_dir = (cfg.harvard_staging.country_reference.input.resolve_data_dir())
    country_reference_glob = str(
                country_reference_dir
                / cfg.harvard_staging.country_reference.input.pattern
                )
    country_reference_cols = cfg.harvard_staging.country_reference.columns

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
                    threads=cfg.engine.threads,
                    overwrite=cfg.harvard_staging.overwrite,
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
                    threads=cfg.engine.threads,
                    value_abs_tolerance=cfg.harvard_staging.diagnostics.value_abs_tolerance,
                    value_rel_tolerance=cfg.harvard_staging.diagnostics.value_rel_tolerance,
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
                country_reporter_code_col=country_reference_cols.reporter_code,
                country_reporter_iso_col=country_reference_cols.reporter_iso,
                country_partner_code_col=country_reference_cols.partner_code,
                country_partner_iso_col=country_reference_cols.partner_iso,
                threads=cfg.engine.threads,
                overwrite=cfg.harvard_staging.overwrite,
                )


    ## Report Writing 
    report_dir = cfg.harvard_staging.report.resolve_output_dir()

    inventory_outputs = write_dataframe_report(
        inventory,
        report_dir=report_dir,
        filename=cfg.harvard_staging.report.inventory_filename,
        timestamp=cfg.harvard_staging.report.timestamp,
        write_latest=cfg.harvard_staging.report.write_latest,
        run_id=run_id,
    )

    preservation_outputs = write_dataframe_report(
        preservation,
        report_dir=report_dir,
        filename=cfg.harvard_staging.report.preservation_filename,
        timestamp=cfg.harvard_staging.report.timestamp,
        write_latest=cfg.harvard_staging.report.write_latest,
        run_id=run_id,
    )

    logger.info("Inventory written: %s", rel_path(inventory_outputs["report"]))
    logger.info("Preservation written: %s", rel_path(preservation_outputs["report"]))

    if "latest" in inventory_outputs:
        logger.info("Latest inventory copy written: %s", rel_path(inventory_outputs["latest"]))

    if "latest" in preservation_outputs:
        logger.info("Latest preservation copy written: %s", rel_path(preservation_outputs["latest"]))

    logger.info("Compatibility export summary: %s", compatibility_summary)

    logger.info("Finished Harvard staging.")
    logger.info("Summary: %s", summary)
    logger.info("Log file: %s", log_path)

if __name__ == "__main__":
    main()