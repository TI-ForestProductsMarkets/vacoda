import numpy as np
import pandas as pd

from vacoda.cif_fob import estimate_rates as estimate_rates_module


def test_build_run_prediction_checkpoint_uses_canonical_keys():
    prediction_rows = pd.DataFrame(
        {
            "year": [2021, 2022],
            "exporter_iso": ["US", "CA"],
            "importer_iso": ["FR", "DE"],
            "target_cmd": [4401, 4402],
            "other": [1, 2],
        }
    )

    holdout_rows = pd.DataFrame(
        {
            "year": [2022],
            "exporter_iso": ["CA"],
            "importer_iso": ["DE"],
            "target_cmd": [4402],
        }
    )

    checkpoint = estimate_rates_module._build_run_prediction_checkpoint(
        run_name="run_demo",
        prediction_rows=prediction_rows,
        full_predicted_log=np.array([0.1, 0.2]),
        full_predicted_factor=np.array([1.1, 1.2]),
        latest_holdout_year=2022,
        latest_holdout_predictions=(
            holdout_rows,
            np.array([0.25]),
            np.array([1.28]),
        ),
    )

    assert set(estimate_rates_module.MODEL_ROW_KEY).issubset(
        checkpoint.columns
    )
    assert "_row_index" not in checkpoint.columns
    assert checkpoint.loc[checkpoint["year"].eq(2022), "holdout_predicted_factor"].iat[0] == 1.28


def test_run_temporal_backtest_returns_keyed_holdout_rows(monkeypatch):
    clean = pd.DataFrame(
        {
            "year": [2020, 2020, 2021, 2021],
            "exporter_iso": ["US", "CA", "US", "CA"],
            "importer_iso": ["FR", "DE", "FR", "DE"],
            "target_cmd": [4401, 4402, 4401, 4402],
            "observed_cif_fob_factor": [1.0, 1.2, 1.1, 1.3],
            "feature": [0.1, 0.2, 0.3, 0.4],
            "clean_estimation_eligible": [True, True, True, True],
        }
    )

    prediction_rows = pd.DataFrame(
        {
            "year": [2021, 2022],
            "exporter_iso": ["US", "CA"],
            "importer_iso": ["FR", "DE"],
            "target_cmd": [4401, 4402],
            "feature": [0.5, 0.6],
        }
    )

    cfg = type(
        "Cfg",
        (),
        {
            "method": type(
                "Method",
                (),
                {
                    "estimation": type(
                        "Estimation",
                        (),
                        {
                            "validation": type(
                                "Validation",
                                (),
                                {
                                    "temporal_backtest": type(
                                        "TemporalBacktest",
                                        (),
                                        {"upper_tail_threshold": 1.0},
                                    )()
                                },
                            )()
                        },
                    )()
                },
            )()
        },
    )()

    run_spec = type(
        "RunSpec",
        (),
        {
            "model_name": "demo_model",
            "run_name": "demo_run",
            "predictor_set_name": None,
            "model_cfg": type(
                "ModelCfg",
                (),
                {"sample_weighting": type("SampleWeighting", (), {"method": "source_only"})()},
            )(),
            "predictors": ["feature"],
            "categorical_predictors": [],
        },
    )()

    monkeypatch.setattr(
        estimate_rates_module,
        "_training_sample_weight",
        lambda **kwargs: pd.Series([1.0] * len(kwargs["train_frame"]), index=kwargs["train_frame"].index),
    )
    monkeypatch.setattr(
        estimate_rates_module,
        "_fit_model",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        estimate_rates_module,
        "_predict_model",
        lambda fitted, frame: np.linspace(0.0, 0.5, len(frame)),
    )
    monkeypatch.setattr(
        estimate_rates_module,
        "_metric_values",
        lambda **kwargs: {
            "median_ape": 0.1,
            "p90_ape": 0.2,
            "p95_ape": 0.3,
            "mae": 0.1,
            "rmse": 0.1,
            "r2": 0.9,
            "mape": 0.1,
        },
    )

    fold_results, predictions_by_year = estimate_rates_module._run_temporal_backtest(
        clean=clean,
        prediction_rows=prediction_rows,
        cfg=cfg,
        run_spec=run_spec,
        minimum_factor=0.1,
        maximum_factor=10.0,
        validation_years=[2021],
    )

    assert not fold_results.empty
    assert 2021 in predictions_by_year
    holdout_rows, predicted_log, predicted_factor = predictions_by_year[2021]
    assert set(estimate_rates_module.MODEL_ROW_KEY).issubset(holdout_rows.columns)
    assert len(holdout_rows) == len(predicted_log) == len(predicted_factor)
