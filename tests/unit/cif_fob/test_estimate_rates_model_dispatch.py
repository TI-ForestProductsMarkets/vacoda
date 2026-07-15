from types import SimpleNamespace

import pandas as pd
import pytest

import vacoda.cif_fob.estimate_rates as estimate_rates


@pytest.mark.parametrize(
    "estimator_type,expected_key",
    [
        ("hgb", "hgb"),
        ("xgb", "xgb"),
        ("log_linear", "log_linear"),
        ("log_lin", "log_linear"),
    ],
)
def test_fit_model_dispatches_on_estimator_type(
    monkeypatch: pytest.MonkeyPatch,
    estimator_type: str,
    expected_key: str,
) -> None:
    sentinels = {
        "hgb": object(),
        "xgb": object(),
        "log_linear": object(),
    }

    monkeypatch.setattr(
        estimate_rates,
        "_fit_hgb",
        lambda **_: sentinels["hgb"],
    )
    monkeypatch.setattr(
        estimate_rates,
        "_fit_xgboost",
        lambda **_: sentinels["xgb"],
    )
    monkeypatch.setattr(
        estimate_rates,
        "_fit_log_linear",
        lambda **_: sentinels["log_linear"],
    )

    result = estimate_rates._fit_model(
        model_name="dummy_model",
        model_cfg=SimpleNamespace(estimator_type=estimator_type),
        train_frame=pd.DataFrame(),
        predictors=[],
        categorical_predictors=[],
        sample_weight=pd.Series(dtype="float64"),
    )

    assert result is sentinels[expected_key]


def test_fit_model_rejects_unknown_estimator_type() -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported estimator_type",
    ):
        estimate_rates._fit_model(
            model_name="dummy_model",
            model_cfg=SimpleNamespace(estimator_type="unknown"),
            train_frame=pd.DataFrame(),
            predictors=[],
            categorical_predictors=[],
            sample_weight=pd.Series(dtype="float64"),
        )


def test_safe_model_prefix_uses_configured_output_prefix() -> None:
    result = estimate_rates._safe_model_prefix(
        model_name="hist_gradient_boosting",
        model_cfg=SimpleNamespace(output_prefix="HGB_SQ"),
    )

    assert result == "hgb_sq"


def test_safe_model_prefix_falls_back_to_model_name() -> None:
    result = estimate_rates._safe_model_prefix(
        model_name="my-new model",
        model_cfg=SimpleNamespace(output_prefix=None),
    )

    assert result == "my_new_model"
