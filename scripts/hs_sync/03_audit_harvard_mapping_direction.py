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
from vacoda.hs_sync.mapping_direction import run_harvard_mapping_direction_audit
from vacoda.config.paths import get_reports_dir
from vacoda.utils.logger import get_logger

REPORT_NAME = "inspect_hscodes_requiring_weights.md"
REPORT_DIR = get_reports_dir() / "hs_sync"

logger = get_logger("HS_SYNC.MAPPING_DIRECTION")


def main() -> None:
    cfg = HsSyncConfig.from_yaml()
    audit = run_harvard_mapping_direction_audit(cfg=cfg,
        report_name=REPORT_NAME,
        report_dir=REPORT_DIR,
        logger=logger,
    )
    print(audit.to_string(index=False))

if __name__ == "__main__":
    main()
