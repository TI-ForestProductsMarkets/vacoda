from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from vacoda.config.cif_fob import CifFobTemporalBacktestConfig


def resolve_expanding_validation_years(
    *,
    clean: pd.DataFrame,
    backtest_cfg: CifFobTemporalBacktestConfig,
) -> tuple[int, ...]:
    """
    Resolve valid expanding-window holdout years from clean model data.

    A year is valid only when at least minimum_training_years distinct
    earlier years are present.
    """
    available_years = tuple(
        sorted(
            int(year)
            for year in clean["year"].dropna().unique()
        )
    )

    if not available_years:
        raise RuntimeError(
            "No years are available for temporal validation."
        )

    available_year_set = set(available_years)

    if backtest_cfg.validation_years:
        missing_years = sorted(
            set(backtest_cfg.validation_years)
            - available_year_set
        )

        if missing_years:
            raise ValueError(
                "Configured temporal-validation years are absent "
                "from the clean estimation sample: "
                + ", ".join(map(str, missing_years))
            )

        candidate_years: Iterable[int] = (
            sorted(backtest_cfg.validation_years)
        )
    else:
        candidate_years = available_years

    resolved: list[int] = []

    for validation_year in candidate_years:
        earlier_years = [
            year
            for year in available_years
            if year < validation_year
        ]

        if (
            len(earlier_years)
            >= backtest_cfg.minimum_training_years
        ):
            resolved.append(validation_year)

    if not resolved:
        raise RuntimeError(
            "No valid expanding-window folds were resolved. "
            "Reduce minimum_training_years or inspect the available "
            "estimation years."
        )

    return tuple(resolved)