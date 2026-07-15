"""
vacoda.visualization.labels

Shared display-label helpers for VACODA plots.
"""

from __future__ import annotations
import numpy as np

PRODUCT_MEDIAN_THRESHOLD_BINS: tuple[float, ...] = (
    5,
    10,
    20,
    30,
    40,
    50,
)

VALUE_BAND_BINS: tuple[float, ...] = (
    0,
    10_000,
    100_000,
    1_000_000,
    10_000_000,
    np.inf,
)

VALUE_BAND_ORDER: tuple[str, ...] = (
    "<10k",
    "10k–100k",
    "100k–1M",
    "1M–10M",
    ">10M",
)


DISTANCE_BAND_BINS: tuple[float, ...] = (
    0,
    1_000,
    3_000,
    7_000,
    12_000,
    np.inf,
)

DISTANCE_BAND_ORDER: tuple[str, ...] = (
    "<1k km",
    "1k–3k km",
    "3k–7k km",
    "7k–12k km",
    ">12k km",
)

CIFFOB_FACTOR_BINS: tuple[float, ...] = (
    1.0,
    1.05,
    1.10,
    1.20,
    1.50,
    2.00,
    3.00,
)

CIFFOB_FACTOR_LABELS: tuple[str, ...] = (
    "1.00-1.05",
    "1.05-1.10",
    "1.10-1.20",
    "1.20-1.50",
    "1.50-2.00",
    "2.00-3.00",
)

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
        "hist_gradient_boosting_weighted": "HGB weighted",
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
        "model_0_baci_replication": "M0",
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