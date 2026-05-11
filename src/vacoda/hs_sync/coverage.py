"""
vacoda.hs_sync.coverage

Diagnostics for understanding HS classification coverage across years,
reporters, and commodity pools.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def iter_csv_files(data_dir: str | Path, pattern: str = "*.csv") -> list[Path]:
    """
    Return sorted CSV files from a directory.

    Parameters
    ----------
    data_dir:
        Directory containing downloaded Comtrade CSV files.
    pattern:
        File pattern to match. Defaults to '*.csv'.

    Returns
    -------
    list[Path]
        Sorted list of matching CSV files.
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    files = sorted(data_dir.rglob(pattern))

    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} found in {data_dir}")

    return files


def build_year_hs_counts(
    files: Iterable[str | Path],
    *,
    year_col: str = "RefYear",
    hs_col: str = "ClassificationCode",
    progress_every: int = 1000) -> pd.DataFrame:
    """
    Build long-format year × HS-classification row counts.

    Output columns:
        RefYear | ClassificationCode | n_rows

    Parameters
    ----------
    files:
        Iterable of CSV file paths.
    year_col:
        Column containing the reference year.
    hs_col:
        Column containing the HS classification code, e.g. H0, H1, H2.
    progress_every:
        Print progress every N files. Set to 0 or None to disable.

    Returns
    -------
    pd.DataFrame
        Long-format table with one row per year × HS classification.
    """
    files = [Path(p) for p in files]
    parts: list[pd.DataFrame] = []
    errors: list[tuple[str, str, str]] = []

    for i, path in enumerate(files, 1):
        try:
            df = pd.read_csv(
                path,
                usecols=[year_col, hs_col],
                dtype={
                    year_col: "int16",
                    hs_col: "string",
                },
            )

            if df.empty:
                continue

            grouped = (
                df.groupby([year_col, hs_col], dropna=False)
                .size()
                .reset_index(name="n_rows")
            )

            parts.append(grouped)

        except Exception as exc:
            errors.append((str(path), type(exc).__name__, str(exc)[:300]))

        if progress_every and i % progress_every == 0:
            print(f"processed {i}/{len(files)} files")

    if not parts:
        raise ValueError("No non-empty data found while building year-HS counts.")

    year_hs = pd.concat(parts, ignore_index=True)

    result = (
        year_hs.groupby([year_col, hs_col], dropna=False)["n_rows"]
        .sum()
        .reset_index()
        .sort_values([year_col, hs_col])
        .reset_index(drop=True)
    )

    if errors:
        print(f"WARNING: {len(errors)} file(s) failed while building counts.")
        for error in errors[:10]:
            print(error)

    return result


def make_year_hs_count_pivot(
    year_hs_counts: pd.DataFrame,
    *,
    year_col: str = "RefYear",
    hs_col: str = "ClassificationCode",
    value_col: str = "n_rows") -> pd.DataFrame:
    """
    Convert long-format year × HS counts into a wide count pivot.

    Output example:
        RefYear | H0 | H1 | H2 | H3 | H4 | H5 | H6
    """
    pivot = year_hs_counts.pivot_table(
        index=year_col,
        columns=hs_col,
        values=value_col,
        aggfunc="sum",
        fill_value=0,
    )

    pivot = pivot.sort_index()

    # Optional: sort HS columns naturally if they are H0, H1, ..., H6.
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    return pivot


def make_year_hs_percentage_pivot(
    count_pivot: pd.DataFrame,
    *,
    decimals: int = 2,
    suffix: str = "_pct",) -> pd.DataFrame:
    """
    Convert a year × HS count pivot into row-wise percentages.

    Each row sums to approximately 100.
    """
    totals = count_pivot.sum(axis=1)

    pct = count_pivot.div(totals.replace(0, pd.NA), axis=0) * 100

    pct.columns = [f"{col}{suffix}" for col in pct.columns]

    return pct.round(decimals)


def build_year_hs_mix_summary(
    count_pivot: pd.DataFrame,
    *,
    clean_threshold: float = 95.0,
    mostly_clean_threshold: float = 90.0) -> pd.DataFrame:
    """
    Build a compact summary of HS-system dominance by year.

    Output columns:
        RefYear
        total_rows
        dominant_hs
        dominant_rows
        dominant_share_pct
        n_active_hs
        mix_status
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


def build_reporter_coverage(
    files: list[Path],
    out_dir: Path,
    progress_every: int = 1000,
    preview_rows: int = 20) -> tuple[pd.DataFrame, list[tuple[str, str, str]]]:
    parts: list[pd.DataFrame] = []
    errors: list[tuple[str, str, str]] = []

    for i, p in enumerate(files, 1):
        try:
            df = pd.read_csv(
                p,
                usecols=lambda c: c in USECOLS,
                dtype=DTYPE_MAP,
            )

            if df.empty:
                continue

            g = (
                df.groupby(GROUP_KEYS, dropna=False)
                .agg(
                    n_rows=("CmdCode", "size"),
                    n_cmd_codes=("CmdCode", "nunique"),
                    n_flows=("FlowCode", "nunique"),
                    primary_value_total=("PrimaryValue", "sum"),
                    netwgt_total=("NetWgt", "sum"),
                )
                .reset_index()
            )

            parts.append(g)

        except Exception as e:
            errors.append((str(p), type(e).__name__, str(e)[:200]))

        if progress_every and i % progress_every == 0:
            print(f"processed {i}/{len(files)} files")

    if parts:
        coverage_raw = pd.concat(parts, ignore_index=True)

        coverage = (
            coverage_raw.groupby(GROUP_KEYS, dropna=False)
            .agg(
                n_rows=("n_rows", "sum"),
                n_cmd_codes=("n_cmd_codes", "sum"),
                n_flows=("n_flows", "sum"),
                primary_value_total=("primary_value_total", "sum"),
                netwgt_total=("netwgt_total", "sum"),
            )
            .reset_index()
        )

        coverage["share_rows_in_reporter_year"] = (
            coverage["n_rows"]
            / coverage.groupby(["ReporterCode", "RefYear"])["n_rows"].transform("sum")
        )
    else:
        coverage = pd.DataFrame(
            columns=GROUP_KEYS
            + [
                "n_rows",
                "n_cmd_codes",
                "n_flows",
                "primary_value_total",
                "netwgt_total",
                "share_rows_in_reporter_year",
            ]
        )

    out_parquet = out_dir / "reporter_year_hs_coverage.parquet"
    out_csv = out_dir / "reporter_year_hs_coverage.csv"

    coverage.to_parquet(out_parquet, index=False)
    coverage.to_csv(out_csv, index=False)

    print("\ncoverage shape:", coverage.shape)
    print("saved:")
    print(" ", out_parquet)
    print(" ", out_csv)

    print("\nerrors:", len(errors))
    for e in errors[:10]:
        print(e)

    print("\npreview:")
    print(coverage.head(preview_rows))

    return coverage, errors