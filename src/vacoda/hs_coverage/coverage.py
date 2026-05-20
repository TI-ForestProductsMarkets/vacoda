"""
vacoda.hs_coverage.coverage

Diagnostics for understanding HS classification coverage across years,
reporters, and commodity pools.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import duckdb

from vacoda.hs_coverage.engines import (
    build_year_hs_counts_duckdb,
    build_year_hs_counts_polars,
    build_year_hs_counts_pandas,
    build_reporter_year_hs_coverage_duckdb,
)
from vacoda.hs_coverage.summaries import (
    add_reporter_year_shares,
    build_hs_transition_events,
    build_mixed_classification_reporter_years,
    build_reporter_hs_timeline,
    build_reporter_year_dominant_hs,
)


def add_cmd4(
    df: pd.DataFrame,
    *,
    cmd_col: str = "CmdCode",
    output_col: str = "cmd4",
) -> pd.DataFrame:
    """
    Add a 4-digit HS heading column from a commodity code column.
    """
    out = df.copy()

    out[output_col] = (
        out[cmd_col]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(6)
        .str[:4]
    )

    return out


def build_year_hs_cmd4_coverage_duckdb(
    data_dir: str | Path,
    *,
    pattern: str = "*.csv",
    year_col: str = "RefYear",
    hs_col: str = "ClassificationCode",
    cmd_col: str = "CmdCode",
    threads: int = 16) -> pd.DataFrame:
    """
    Build year × HS classification × HS4 commodity-family coverage using DuckDB.

    Output level:
        year × ClassificationCode × cmd4

    Notes
    -----
    CmdCode is normalized as text, left-padded to 6 characters, and truncated
    to the first 4 characters.
    """
    data_dir = Path(data_dir)
    files_glob = str(data_dir / pattern)

    con = duckdb.connect(database=":memory:")

    try:
        con.execute(f"PRAGMA threads={int(threads)}")

        query = f"""
        WITH base AS (
            SELECT
                "{year_col}" AS year,
                "{hs_col}" AS hs,
                LPAD(CAST("{cmd_col}" AS VARCHAR), 6, '0') AS cmd6
            FROM read_csv_auto(
                ?,
                union_by_name = true,
                ignore_errors = true
            )
            WHERE "{year_col}" IS NOT NULL
              AND "{hs_col}" IS NOT NULL
              AND "{cmd_col}" IS NOT NULL
        ),

        grouped AS (
            SELECT
                year,
                hs,
                SUBSTR(cmd6, 1, 4) AS cmd4,
                COUNT(*) AS n_rows,
                COUNT(DISTINCT cmd6) AS n_cmd6
            FROM base
            GROUP BY
                year,
                hs,
                cmd4
        ),

        with_totals AS (
            SELECT
                *,
                SUM(n_rows) OVER (
                    PARTITION BY year, hs
                ) AS total_rows_in_year_hs
            FROM grouped
        )

        SELECT
            year AS "{year_col}",
            hs AS "{hs_col}",
            cmd4,
            n_rows,
            n_cmd6,
            ROUND(
                CAST(n_rows AS DOUBLE)
                / NULLIF(CAST(total_rows_in_year_hs AS DOUBLE), 0),
                4
            ) AS share_rows_in_year_hs
        FROM with_totals
        ORDER BY
            "{year_col}",
            "{hs_col}",
            n_rows DESC
        """

        return con.execute(query, [files_glob]).fetchdf()

    finally:
        con.close()


def build_year_hs_cmd4_coverage_pandas(
    data_dir: str | Path,
    *,
    pattern: str = "*.csv",
    year_col: str = "RefYear",
    hs_col: str = "ClassificationCode",
    cmd_col: str = "CmdCode") -> pd.DataFrame:
    """
    Build coverage of HS4 commodity families by year and HS classification.

    Output level:
        year × HS classification × cmd4

    This helps identify which commodity families appear under each HS system
    in each year.
    """
    files = sorted(Path(data_dir).glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No files found in {data_dir} matching pattern {pattern!r}"
        )

    parts: list[pd.DataFrame] = []

    for path in files:
        df = pd.read_csv(
            path,
            usecols=[year_col, hs_col, cmd_col],
            dtype={
                year_col: "Int64",
                hs_col: "string",
                cmd_col: "string",
            },
        )

        if df.empty:
            continue

        df = add_cmd4(df, cmd_col=cmd_col)

        grouped = (
            df.groupby([year_col, hs_col, "cmd4"], dropna=False)
            .agg(
                n_rows=(cmd_col, "size"),
                n_cmd6=(cmd_col, "nunique"),
            )
            .reset_index()
        )

        parts.append(grouped)

    if not parts:
        return pd.DataFrame(
            columns=[
                year_col,
                hs_col,
                "cmd4",
                "n_rows",
                "n_cmd6",
                "share_rows_in_year_hs",
            ]
        )

    out = pd.concat(parts, ignore_index=True)

    out = (
        out.groupby([year_col, hs_col, "cmd4"], dropna=False)
        .agg(
            n_rows=("n_rows", "sum"),
            n_cmd6=("n_cmd6", "sum"),
        )
        .reset_index()
    )

    totals = out.groupby([year_col, hs_col], dropna=False)["n_rows"].transform("sum")

    out["share_rows_in_year_hs"] = (
        out["n_rows"] / totals.replace(0, pd.NA)
    ).round(4)

    return out.sort_values(
        [year_col, hs_col, "n_rows"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def build_year_hs_cmd4_coverage(
     data_dir: str | Path,
    *,
    pattern: str = "*.csv",
    engine: str = "duckdb",
    year_col: str = "RefYear",
    hs_col: str = "ClassificationCode",
    cmd_col: str = "CmdCode",
    threads: int = 16,
    show_progress: bool = False,
) -> pd.DataFrame:
    """
    Build year × HS classification × HS4 commodity-family coverage using selected backend.

    Supported engines:
    - duckdb
    - pandas
    """
    engine = engine.strip().lower()

    if engine == "duckdb":
        return build_year_hs_cmd4_coverage_duckdb(
            data_dir,
            pattern=pattern,
            year_col=year_col,
            hs_col=hs_col,
            cmd_col=cmd_col,
            threads=threads,
        )

    if engine == "pandas":
        return build_year_hs_cmd4_coverage_pandas(
            data_dir,
            pattern=pattern,
            year_col=year_col,
            hs_col=hs_col,
            cmd_col=cmd_col,
        )


    raise ValueError(
        f"Unknown engine: {engine!r}. "
        "Valid engines are: duckdb, pandas."
    )

def build_top_cmd4_by_year_hs(
    year_hs_cmd4: pd.DataFrame,
    *,
    year_col: str = "RefYear",
    hs_col: str = "ClassificationCode",
    n: int = 10,
) -> pd.DataFrame:
    """
    Return top N cmd4 families within each year × HS classification.
    """
    return (
        year_hs_cmd4.sort_values(
            [year_col, hs_col, "n_rows"],
            ascending=[True, True, False],
        )
        .groupby([year_col, hs_col], group_keys=False)
        .head(n)
        .reset_index(drop=True)
    )

def build_cmd4_hs_timeline(
    year_hs_cmd4: pd.DataFrame,
    *,
    year_col: str = "RefYear",
    hs_col: str = "ClassificationCode",
) -> pd.DataFrame:
    """
    Build cmd4 × year timeline showing row counts by HS system.
    """
    pivot = year_hs_cmd4.pivot_table(
        index=["cmd4", year_col],
        columns=hs_col,
        values="n_rows",
        aggfunc="sum",
        fill_value=0,
    )

    return pivot.reset_index()

def build_cmd4_hs_diversity_summary(
    year_hs_cmd4: pd.DataFrame,
    *,
    year_col: str = "RefYear",
    hs_col: str = "ClassificationCode",
) -> pd.DataFrame:
    """
    Summarize how many HS systems and years each cmd4 appears in.
    """
    summary = (
        year_hs_cmd4.groupby("cmd4", dropna=False)
        .agg(
            first_year=(year_col, "min"),
            last_year=(year_col, "max"),
            n_years=(year_col, "nunique"),
            n_hs_systems=(hs_col, "nunique"),
            total_rows=("n_rows", "sum"),
            total_cmd6=("n_cmd6", "sum"),
        )
        .reset_index()
    )

    return summary.sort_values(
        ["n_hs_systems", "total_rows"],
        ascending=[False, False],
    ).reset_index(drop=True)

def build_cmd4_hs_transition_events(
    year_hs_cmd4: pd.DataFrame,
    *,
    year_col: str = "RefYear",
    hs_col: str = "ClassificationCode",
) -> pd.DataFrame:
    """
    Detect years where a cmd4 heading changes dominant HS system.
    """
    yearly = (
        year_hs_cmd4.sort_values("n_rows", ascending=False)
        .groupby(["cmd4", year_col], as_index=False)
        .first()
    )

    yearly = yearly.sort_values(["cmd4", year_col])

    yearly["previous_hs"] = yearly.groupby("cmd4")[hs_col].shift(1)
    yearly["previous_year"] = yearly.groupby("cmd4")[year_col].shift(1)

    transitions = yearly[
        yearly["previous_hs"].notna()
        & (yearly[hs_col] != yearly["previous_hs"])
    ].copy()

    return transitions[
        [
            "cmd4",
            "previous_year",
            year_col,
            "previous_hs",
            hs_col,
            "n_rows",
            "n_cmd6",
            "share_rows_in_year_hs",
        ]
    ].reset_index(drop=True)

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
            progress_every=1000,
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

def build_hs_coverage_summary_tables(
    df: pd.DataFrame,
    *,
    dominance_metric: str = "share_rows_in_reporter_year",
) -> dict[str, pd.DataFrame]:
    """
    Build all main HS-coverage summary tables.

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