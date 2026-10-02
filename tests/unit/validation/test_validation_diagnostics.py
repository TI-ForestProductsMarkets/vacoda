import logging

import pandas as pd

from vacoda.config.pipeline import PipelineEngineConfig, PipelineScopeConfig
from vacoda.config.steps.validation import ValidationConfig, load_validation_config
from vacoda.steps.validation.diagnostics import build_diagnostics


def _make_validation_diagnostics_fixture(tmp_path):
    required = [
        "year",
        "exporter_iso",
        "importer_iso",
        "target_cmd",
        "quantity_selected_kg",
        "quantity_simple_mean_kg",
        "quantity_value_weighted_kg",
        "quantity_reliability_weighted_kg",
        "trade_value_reconciled",
        "trade_value_reconciled_constant_usd",
        "export_quantity_kg",
        "import_quantity_kg",
        "export_synchronized_quantity_kg",
        "import_synchronized_quantity_kg",
        "export_is_synchronized_quantity_estimated",
        "import_is_synchronized_quantity_estimated",
        "export_quantity_estimated",
        "import_quantity_estimated",
        "quantity_relative_difference",
        "product_year_median_unit_value",
        "unit_value_reference_observations",
        "quantity_selected_candidate",
        "quantity_selected_source",
        "candidate_simple_mean_uv_log_distance",
        "candidate_value_weighted_uv_log_distance",
        "candidate_quantity_reliability_weighted_uv_log_distance",
        "candidate_simple_mean_export_log_gap",
        "candidate_simple_mean_import_log_gap",
        "candidate_value_weighted_export_log_gap",
        "candidate_value_weighted_import_log_gap",
        "candidate_quantity_reliability_weighted_export_log_gap",
        "candidate_quantity_reliability_weighted_import_log_gap",
        "candidate_simple_mean_conditioned_uv_abs_residual",
        "candidate_value_weighted_conditioned_uv_abs_residual",
        "candidate_quantity_reliability_weighted_conditioned_uv_abs_residual",
        "quantity_conditioned_model_n_rows",
        "quantity_conditioned_log_uv_intercept",
        "quantity_conditioned_log_uv_slope",
        "quantity_conditioned_log_uv_r2",
        "selected_unit_value",
        "training_reference_eligible",
        "hs_year_uv_reference",
        "hs_year_uv_reference_log",
        "hs_year_uv_reference_n",
        "hs_year_uv_log_mad",
        "selected_uv_log_distance",
        "training_hard_eligible",
    ]

    row = {column: None for column in required}
    row.update(
        {
            "year": 2022,
            "exporter_iso": "US",
            "importer_iso": "CN",
            "target_cmd": "4401",
            "quantity_selected_kg": 100.0,
            "quantity_simple_mean_kg": 100.0,
            "quantity_value_weighted_kg": 100.0,
            "quantity_reliability_weighted_kg": 100.0,
            "trade_value_reconciled": 2000.0,
            "trade_value_reconciled_constant_usd": 2000.0,
            "export_quantity_kg": 120.0,
            "import_quantity_kg": 110.0,
            "export_synchronized_quantity_kg": 120.0,
            "import_synchronized_quantity_kg": 110.0,
            "export_is_synchronized_quantity_estimated": False,
            "import_is_synchronized_quantity_estimated": False,
            "export_quantity_estimated": False,
            "import_quantity_estimated": False,
            "quantity_relative_difference": 0.08,
            "product_year_median_unit_value": 18.0,
            "unit_value_reference_observations": 5,
            "quantity_selected_candidate": "simple_mean",
            "quantity_selected_source": "trade_sync",
            "candidate_simple_mean_uv_log_distance": 0.1,
            "candidate_value_weighted_uv_log_distance": 0.2,
            "candidate_quantity_reliability_weighted_uv_log_distance": 0.3,
            "candidate_simple_mean_export_log_gap": 0.01,
            "candidate_simple_mean_import_log_gap": 0.02,
            "candidate_value_weighted_export_log_gap": 0.03,
            "candidate_value_weighted_import_log_gap": 0.04,
            "candidate_quantity_reliability_weighted_export_log_gap": 0.05,
            "candidate_quantity_reliability_weighted_import_log_gap": 0.06,
            "candidate_simple_mean_conditioned_uv_abs_residual": 0.1,
            "candidate_value_weighted_conditioned_uv_abs_residual": 0.2,
            "candidate_quantity_reliability_weighted_conditioned_uv_abs_residual": 0.3,
            "quantity_conditioned_model_n_rows": 1,
            "quantity_conditioned_log_uv_intercept": 0.0,
            "quantity_conditioned_log_uv_slope": 1.0,
            "quantity_conditioned_log_uv_r2": 0.5,
            "selected_unit_value": 20.0,
            "training_reference_eligible": True,
            "hs_year_uv_reference": 18.0,
            "hs_year_uv_reference_log": 2.89,
            "hs_year_uv_reference_n": 5,
            "hs_year_uv_log_mad": 0.2,
            "selected_uv_log_distance": 0.1,
            "training_hard_eligible": True,
        }
    )

    rows = [
        row,
        {
            **row,
            "year": 2023,
            "quantity_selected_kg": 150.0,
            "selected_unit_value": 25.0,
            "trade_value_reconciled": 3000.0,
            "export_quantity_kg": 130.0,
            "import_quantity_kg": 120.0,
            "export_synchronized_quantity_kg": 130.0,
            "import_synchronized_quantity_kg": 120.0,
        },
    ]

    input_path = tmp_path / "validation_input.parquet"
    pd.DataFrame(rows).to_parquet(input_path)

    quantity_selection_audit_path = tmp_path / "quantity_selection_audit.parquet"
    pd.DataFrame([{"audit_passed": True}]).to_parquet(quantity_selection_audit_path)

    model_predictions_path = tmp_path / "model_predictions.parquet"
    pd.DataFrame(
        [
            {
                "model_name": "m",
                "model_type": "tree",
                "held_out_year": 2023,
                "target_transformed": 1.0,
                "prediction_transformed": 1.1,
                "target_quantity_kg": 100.0,
                "prediction_quantity_kg": 110.0,
            }
        ]
    ).to_parquet(model_predictions_path)

    structural_audit_path = tmp_path / "structural_audit.parquet"
    pd.DataFrame([{"audit_passed": True}]).to_parquet(structural_audit_path)

    cfg = ValidationConfig.from_dict(
        load_validation_config(),
        scope=PipelineScopeConfig(
            target_hs="4401",
            min_year=2022,
            max_year=2025,
        ),
        engine=PipelineEngineConfig(
            name="duckdb",
            threads=1,
        ),
    )
    object.__setattr__(
        cfg,
        "product_path",
        lambda product_name: tmp_path / f"{product_name}.parquet",
    )

    return (
        cfg,
        input_path,
        structural_audit_path,
        quantity_selection_audit_path,
        model_predictions_path,
    )


def test_build_diagnostics_creates_pre_training_reference_windows_before_use(tmp_path):
    (
        cfg,
        input_path,
        structural_audit_path,
        quantity_selection_audit_path,
        model_predictions_path,
    ) = _make_validation_diagnostics_fixture(tmp_path)

    result = build_diagnostics(
        cfg=cfg,
        input_path=input_path,
        structural_audit_path=structural_audit_path,
        quantity_selection_audit_path=quantity_selection_audit_path,
        model_predictions_path=model_predictions_path,
        run_id="test-run",
        force=True,
        logger=logging.getLogger(__name__),
    )

    assert result["status"] == "completed"
    assert (tmp_path / "diagnostics.parquet").exists()
