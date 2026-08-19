"""
Run the Quantity workflow independently of the master pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path

from vacoda.config.pipeline import (
    PipelineEngineConfig,
    PipelineScopeConfig,
)
from vacoda.config.quantity import QuantityConfig
from vacoda.quantity.workflow import run_quantity_workflow


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    logger = logging.getLogger("quantity")

    scope = PipelineScopeConfig(
        target_hs="H5",
        min_year=2017,
        max_year=2025,
    )

    engine = PipelineEngineConfig(
        name="duckdb",
        threads=50,
    )

    cfg = QuantityConfig.from_yaml(
        scope=scope,
        engine=engine,
    )

    result = run_quantity_workflow(
        cfg=cfg,
        run_id="quantity_manual_h5",
        logger=logger,
        force=True,
        input_path=Path(
            "/storage/vacoda/data/interim/hs_sync/target_h5/applied/"
            "comtrade_target_h5_hs_synced.parquet"
        ),
    )

    print("\nFINAL RESULT")
    print(result)


if __name__ == "__main__":
    main()