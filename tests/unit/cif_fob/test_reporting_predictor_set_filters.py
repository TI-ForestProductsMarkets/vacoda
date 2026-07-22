from types import SimpleNamespace

import pandas as pd

from vacoda.cif_fob import reporting


def test_enabled_predictor_set_names_returns_only_enabled():
    cfg = SimpleNamespace(
        method=SimpleNamespace(
            estimation=SimpleNamespace(
                predictor_sets={
                    "model_1": SimpleNamespace(enabled=False),
                    "model_2": SimpleNamespace(enabled=True),
                    "model_3": SimpleNamespace(enabled=True),
                }
            )
        )
    )

    enabled = reporting._enabled_predictor_set_names(cfg)

    assert enabled == {"model_2", "model_3"}


def test_filter_to_enabled_predictor_sets_uses_predictor_set_column():
    frame = pd.DataFrame(
        {
            "model_name": [
                "hist_gradient_boosting__model_1",
                "hist_gradient_boosting__model_2",
            ],
            "predictor_set_name": ["model_1", "model_2"],
            "mean_mae_log_factor": [0.12, 0.10],
        }
    )

    filtered = reporting._filter_to_enabled_predictor_sets(
        frame,
        enabled_predictor_sets={"model_2"},
    )

    assert list(filtered["predictor_set_name"]) == ["model_2"]
    assert list(filtered["model_name"]) == [
        "hist_gradient_boosting__model_2"
    ]


def test_filter_to_enabled_predictor_sets_parses_model_name_when_missing_column():
    frame = pd.DataFrame(
        {
            "model_name": [
                "hist_gradient_boosting__model_1",
                "hist_gradient_boosting__model_2",
            ],
            "mean_mae_log_factor": [0.12, 0.10],
        }
    )

    filtered = reporting._filter_to_enabled_predictor_sets(
        frame,
        enabled_predictor_sets={"model_2"},
    )

    assert "predictor_set_name" in filtered.columns
    assert list(filtered["predictor_set_name"]) == ["model_2"]
    assert list(filtered["model_name"]) == [
        "hist_gradient_boosting__model_2"
    ]
