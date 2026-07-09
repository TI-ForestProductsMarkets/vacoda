"""
vacoda.visualization.labels

Shared display-label helpers for VACODA plots.
"""

from __future__ import annotations


def split_estimation_run_name(run_name: str) -> tuple[str, str]:
    """Split an algorithm × predictor-set run name."""
    if "__" not in str(run_name):
        return str(run_name), ""

    algorithm, predictor_set = str(run_name).split("__", 1)
    return algorithm, predictor_set


def short_algorithm_name(name: str) -> str:
    """Return compact display name for model algorithm."""
    mapping = {
        "hist_gradient_boosting": "HGB absolute-log",
        "hist_gradient_boosting_squared": "HGB squared-log",
        "hist_gradient_boosting_squared_level": "HGB squared-level",
        "hist_gradient_boosting_squared_regularized": "HGB squared-log regularized",
        "log_linear_regression": "OLS log-linear",
        "log_linear_regression_smearing": "OLS smearing",
        "xgboost_log_squared": "XGBoost squared-log",
    }
    return mapping.get(str(name), str(name))


def short_predictor_set_name(name: str) -> str:
    """Return compact display name for predictor set."""
    mapping = {
        "model_1_classical_gravity": "M1",
        "model_2_classical_year_oil": "M2",
        "model_3_policy_integration": "M3",
        "model_4_trade_scale_history": "M4",
        "model_5_full_history": "M5",
        "model_6_policy_plus_factor_history": "M6",
    }
    return mapping.get(str(name), str(name))


def estimation_run_display_name(run_name: str) -> str:
    """Return compact display name for one algorithm × predictor-set run."""
    algorithm, predictor_set = split_estimation_run_name(run_name)

    if predictor_set:
        return (
            short_algorithm_name(algorithm)
            + " / "
            + short_predictor_set_name(predictor_set)
        )

    return short_algorithm_name(algorithm)