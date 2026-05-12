"""
vacoda.hs_sync.coverage

Diagnostics for understanding HS classification coverage across years,
reporters, and commodity pools.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from vacoda.hs_sync.coverage_engines import (
    build_year_hs_counts_duckdb,
    build_year_hs_counts_polars,
    build_year_hs_counts_pandas,
    build_reporter_year_hs_coverage_duckdb,
)


def build_reporter_year_hs_coverage(
    data_dir: str | Path,
    *,
    pattern: str = "*.csv",
    engine: str = "duckdb",
    threads: int = 16,
    columns: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Build reporter × year × HS classification coverage.
    """
    engine = engine.strip().lower()

    if engine == "duckdb":
        return build_reporter_year_hs_coverage_duckdb(
            data_dir,
            pattern=pattern,
            threads=threads,
            columns=columns,
        )

    raise ValueError(
        f"Unknown engine for reporter-year coverage: {engine!r}. "
        "Currently supported: duckdb."
    )

    
def build_hs_system_overall_summary(
    year_hs_counts: pd.DataFrame,
    *,
    year_col: str = "RefYear",
    hs_col: str = "ClassificationCode",
    value_col: str = "n_rows",
) -> pd.DataFrame:
    """
    Build overall HS-system coverage summary across all years.

    Output columns:
        ClassificationCode
        n_rows
        share_pct
        first_year
        last_year
        n_years
    """
    total_rows = year_hs_counts[value_col].sum()

    summary = (
        year_hs_counts.groupby(hs_col, dropna=False)
        .agg(
            n_rows=(value_col, "sum"),
            first_year=(year_col, "min"),
            last_year=(year_col, "max"),
            n_years=(year_col, "nunique"),
        )
        .reset_index()
    )

    summary["share_pct"] = (summary["n_rows"] / total_rows * 100).round(2)

    summary = summary.sort_values("n_rows", ascending=False).reset_index(drop=True)

    return summary
    
def build_year_hs_counts(
    data_dir: str | Path,
    *,
    pattern: str = "*.csv",
    engine: str = "duckdb",
    year_col: str = "RefYear",
    hs_col: str = "ClassificationCode",
    threads: int = 16,
    show_progress: bool = False,
) -> pd.DataFrame:
    """
    Build year × HS-system row counts using selected backend.

    Supported engines:
    - duckdb
    - polars
    - pandas
    """
    engine = engine.strip().lower()

    if engine == "duckdb":
        return build_year_hs_counts_duckdb(
            data_dir,
            pattern=pattern,
            year_col=year_col,
            hs_col=hs_col,
            threads=threads,
        )

    if engine == "polars":
        return build_year_hs_counts_polars(
            data_dir,
            pattern=pattern,
            year_col=year_col,
            hs_col=hs_col,
        )

    if engine == "pandas":
        return build_year_hs_counts_pandas(
            data_dir,
            pattern=pattern,
            year_col=year_col,
            hs_col=hs_col,
            show_progress=show_progress,
        )

    raise ValueError(
        f"Unknown engine: {engine!r}. "
        "Valid engines are: duckdb, polars, pandas."
    )


def make_year_hs_count_pivot(
    year_hs_counts: pd.DataFrame,
    *,
    year_col: str = "RefYear",
    hs_col: str = "ClassificationCode",
    value_col: str = "n_rows",
) -> pd.DataFrame:
    """
    Convert long-format year × HS counts into a wide count pivot.
    """
    pivot = year_hs_counts.pivot_table(
        index=year_col,
        columns=hs_col,
        values=value_col,
        aggfunc="sum",
        fill_value=0,
    )

    pivot = pivot.sort_index()
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    return pivot


def make_year_hs_percentage_pivot(
    count_pivot: pd.DataFrame,
    *,
    decimals: int = 2,
    suffix: str = "_pct",
) -> pd.DataFrame:
    """
    Convert a year × HS count pivot into row-wise percentages.
    """
    totals = count_pivot.sum(axis=1)
    pct = count_pivot.div(totals.replace(0, pd.NA), axis=0) * 100

    pct.columns = [f"{col}{suffix}" for col in pct.columns]

    return pct.round(decimals)


def build_year_hs_mix_summary(
    count_pivot: pd.DataFrame,
    *,
    clean_threshold: float = 95.0,
    mostly_clean_threshold: float = 90.0,
) -> pd.DataFrame:
    """
    Build a compact summary of HS-system dominance by year.
    """
    summary = pd.DataFrame(index=count_pivot.index)

    summary["total_rows"] = count_pivot.sum(axis=1)
    summary["dominant_hs"] = count_pivot.idxmax(axis=1)
    summary["dominant_rows"] = count_pivot.max(axis=1)
    summary["n_active_hs"] = (count_pivot > 0).sum(axis=1)

    summary["dominant_share_pct"] = (
        summary["dominant_rows"] / summary["total_rows"] * 100
    ).round(2)

    def classify(share: float) -> str:
        if share >= clean_threshold:
            return "clean"
        if share >= mostly_clean_threshold:
            return "mostly_clean"
        return "mixed_transition"

    summary["mix_status"] = summary["dominant_share_pct"].apply(classify)

    return summary.reset_index()

def build_hs_sync_coverage_summary_tables(
    df: pd.DataFrame,
    *,
    dominance_metric: str = "share_rows_in_reporter_year",
) -> dict[str, pd.DataFrame]:
    """
    Build all main HS-sync coverage summary tables.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary with:

        - reporter_hs_timeline
        - reporter_year_dominant_hs
        - hs_transition_events
        - mixed_classification_reporter_years
    """

    enriched = add_reporter_year_shares(df)

    return {
        "reporter_hs_timeline": build_reporter_hs_timeline(
            enriched,
            dominance_metric=dominance_metric,
        ),
        "reporter_year_dominant_hs": build_reporter_year_dominant_hs(
            enriched,
            dominance_metric=dominance_metric,
        ),
        "hs_transition_events": build_hs_transition_events(
            enriched,
            dominance_metric=dominance_metric,
        ),
        "mixed_classification_reporter_years": build_mixed_classification_reporter_years(
            enriched,
            dominance_metric=dominance_metric,
        ),
    }