"""
scripts/hs_sync/03_audit_harvard_mapping_direction.py

Audit directionality of Harvard/Growth Lab correlation-group CSV files.

This script checks whether files such as:

    from_HS2022_to_HS2017.csv

are being interpreted correctly as:

    source_hs/source_cmd -> target_hs/target_cmd

It does not use weights. It only audits mapping structure and empirical
support from the trade source summary.
"""

from __future__ import annotations

from vacoda.config.hs_sync import HsSyncConfig
from vacoda.hs_sync.mapping_direction import run_harvard_mapping_direction_audit_from_config
from vacoda.utils.logger import get_logger

logger = get_logger("HS_SYNC.MAPPING_DIRECTION")


def main() -> None:
    cfg = HsSyncConfig.from_yaml()
    outputs = run_harvard_mapping_direction_audit_from_config(
        cfg=cfg,
        logger=logger,
    )

    if outputs.get("status") == "disabled":
        print("harvard_mapping_direction is disabled in config; skipping.")
        return

    print("Harvard mapping-direction audit complete.")
    print(f"Report directory: {outputs.get('report_dir')}")
    if outputs.get("report_latest") is not None:
        print(f"Report latest: {outputs.get('report_latest')}")

    audit = outputs.get("audit")
    if audit is not None:
        print(audit.to_string(index=False))

if __name__ == "__main__":
    main()
