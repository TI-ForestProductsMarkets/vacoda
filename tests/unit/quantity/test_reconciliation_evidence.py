from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from vacoda.quantity.reconciliation_evidence import (
    build_quantity_reconciliation_evidence,
)
from vacoda.utils.logger import get_logger


class DummyQuantityConfig:
    def __init__(self, output_dir):
        self.method = SimpleNamespace(
            kg_reconciliation=SimpleNamespace(
                quantity_unit="kg",
                agreement=SimpleNamespace(relative_tolerance=0.05),
            )
        )
        self._output_dir = output_dir

    def product_path(self, product_name: str):
        return self._output_dir / f"{product_name}.parquet"


def test_build_quantity_reconciliation_evidence_converts_grams_to_kg(tmp_path):
    input_path = tmp_path / "prepared_input.parquet"
    pd.DataFrame(
        [
            {
                "has_quantity_kg": False,
                "quantity": None,
                "alt_quantity_unit_code": 15,
                "alt_quantity_unit": "g",
                "alt_quantity": 500.0,
                "net_weight_kg": None,
                "has_net_weight_kg": False,
                "is_alt_quantity_estimated": False,
                "has_reported_quantity_kg": False,
                "has_estimated_quantity_kg": False,
                "has_reported_net_weight_kg": False,
                "has_estimated_net_weight_kg": False,
                "quantity_unit": "kg",
                "quantity_unit_code": 8,
            }
        ]
    ).to_parquet(input_path, index=False)

    result = build_quantity_reconciliation_evidence(
        cfg=DummyQuantityConfig(tmp_path),
        input_path=input_path,
        run_id="test-run",
        force=True,
        logger=get_logger("TEST"),
    )

    evidence = pd.read_parquet(result["reconciliation_evidence_path"])

    assert evidence["alt_quantity_kg_candidate"].iloc[0] == 0.5
    assert evidence["alt_quantity_kg_source"].iloc[0] == "g_converted"
