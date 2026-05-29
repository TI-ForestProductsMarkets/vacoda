"""
Audit Harvard/Growth Lab generator input compatibility.

This script checks whether vacoda's Harvard-compatible export is likely
readable by the Harvard conversion-weight generator.

It audits:
- raw_parquet marker files
- aggregated_by_year_not_converted parquet files
- reporter/partner identifier style
- product-code coverage
- generated matrix completeness, if matrices exist
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import duckdb
import pandas as pd


def _print_header(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def audit_raw_marker_files(
    *,
    harvard_data_root: Path,
    source_hs: str,
    source_year: int,
    target_hs: str,
    target_year: int,
) -> None:
    raw_root = harvard_data_root / "as_reported" / "raw_parquet"

    checks = [
        (source_hs, source_year),
        (target_hs, target_year),
    ]

    _print_header("1. Raw marker files")

    for hs, year in checks:
        path = raw_root / hs / str(year)
        files = sorted(path.glob("*.parquet"))

        print(f"{hs}/{year}")
        print(f"  path: {path}")
        print(f"  exists: {path.exists()}")
        print(f"  n_files: {len(files):,}")

        if files:
            print(f"  first file: {files[0].name}")


def audit_aggregated_files(
    *,
    harvard_data_root: Path,
    source_hs: str,
    source_year: int,
    target_hs: str,
    target_year: int,
) -> None:
    agg_root = (
        harvard_data_root
        / "as_reported"
        / "aggregated_by_year_not_converted"
        / "parquet"
    )

    checks = [
        (source_hs, source_year),
        (target_hs, target_year),
    ]

    _print_header("2. Aggregated trade files")

    con = duckdb.connect()

    for hs, year in checks:
        path = agg_root / hs / f"{hs}_{year}.parquet"

        print(f"{hs}/{year}")
        print(f"  path: {path}")
        print(f"  exists: {path.exists()}")

        if not path.exists():
            continue

        schema = con.execute(
            f"""
            DESCRIBE SELECT *
            FROM read_parquet('{path}')
            """
        ).df()

        print("  columns:")
        print("   ", ", ".join(schema["column_name"].tolist()))

        summary = con.execute(
            f"""
            SELECT
                COUNT(*) AS n_rows,
                COUNT(DISTINCT cmdCode) AS n_cmd_codes,
                COUNT(DISTINCT reporterISO3) AS n_reporters,
                COUNT(DISTINCT partnerISO3) AS n_partners,
                MIN(cmdCode) AS min_cmd,
                MAX(cmdCode) AS max_cmd,
                SUM(primaryValue) AS value_total
            FROM read_parquet('{path}')
            """
        ).df()

        print(summary.to_string(index=False))

        samples = con.execute(
            f"""
            SELECT DISTINCT reporterISO3
            FROM read_parquet('{path}')
            WHERE reporterISO3 IS NOT NULL
            ORDER BY reporterISO3
            LIMIT 20
            """
        ).df()

        print("  reporterISO3 samples:")
        print("   ", samples["reporterISO3"].tolist())

        reporter_style = con.execute(
            f"""
            SELECT
                COUNT(*) FILTER (
                    WHERE regexp_matches(CAST(reporterISO3 AS VARCHAR), '^[0-9]+$')
                ) AS numeric_reporter_ids,
                COUNT(*) FILTER (
                    WHERE regexp_matches(CAST(reporterISO3 AS VARCHAR), '^[A-Z]{{3}}$')
                ) AS iso3_reporter_ids,
                COUNT(*) AS n_rows
            FROM read_parquet('{path}')
            """
        ).df()

        print("  reporter id style:")
        print(reporter_style.to_string(index=False))

        numeric = int(reporter_style["numeric_reporter_ids"].iloc[0])
        iso3 = int(reporter_style["iso3_reporter_ids"].iloc[0])

        if numeric > 0 and iso3 == 0:
            print(
                "  WARNING: reporterISO3 appears to contain numeric codes, "
                "not ISO3 strings. Harvard may filter these rows out."
            )


def audit_specific_product_codes(
    *,
    harvard_data_root: Path,
    target_hs: str,
    target_year: int,
    source_hs: str,
    source_year: int,
) -> None:
    _print_header("3. Check known wood transition product codes")

    agg_root = (
        harvard_data_root
        / "as_reported"
        / "aggregated_by_year_not_converted"
        / "parquet"
    )

    checks = [
        (
            "target HS2017 group 69",
            agg_root / target_hs / f"{target_hs}_{target_year}.parquet",
            ["440111", "440112"],
        ),
        (
            "source HS2012 group 69",
            agg_root / source_hs / f"{source_hs}_{source_year}.parquet",
            ["440110"],
        ),
    ]

    con = duckdb.connect()

    for label, path, codes in checks:
        print(label)
        print(f"  file: {path}")

        if not path.exists():
            print("  missing file")
            continue

        codes_sql = ", ".join(f"'{c}'" for c in codes)

        out = con.execute(
            f"""
            SELECT
                cmdCode,
                COUNT(*) AS n_rows,
                SUM(primaryValue) AS value_total,
                COUNT(DISTINCT reporterISO3) AS n_reporters
            FROM read_parquet('{path}')
            WHERE CAST(cmdCode AS VARCHAR) IN ({codes_sql})
            GROUP BY cmdCode
            ORDER BY cmdCode
            """
        ).df()

        if out.empty:
            print("  WARNING: no rows found for these product codes")
        else:
            print(out.to_string(index=False))


def audit_matrix_files(
    *,
    generator_root: Path,
    source_year: int,
    target_year: int,
) -> None:
    _print_header("4. Matrix completeness")

    matrix_dir = generator_root / "data" / "matrices"

    if not matrix_dir.exists():
        print(f"matrix directory does not exist: {matrix_dir}")
        return

    groups: dict[int, set[str]] = defaultdict(set)

    pattern = re.compile(
        rf"(source\.trade|target\.trade|conversion)"
        rf"\.matrix\.start\.{source_year}\.end\.{target_year}"
        rf"\.group\.(\d+)\.csv"
    )

    for path in matrix_dir.glob("*.csv"):
        match = pattern.search(path.name)
        if not match:
            continue

        kind, group = match.group(1), int(match.group(2))
        groups[group].add(kind)

    complete = {
        group: kinds
        for group, kinds in groups.items()
        if {"source.trade", "target.trade", "conversion"} <= kinds
    }

    partial = {
        group: kinds
        for group, kinds in groups.items()
        if group not in complete
    }

    print(f"matrix_dir: {matrix_dir}")
    print(f"groups with any matrix: {len(groups):,}")
    print(f"complete groups: {len(complete):,}")
    print(f"partial groups: {len(partial):,}")
    print(f"complete group ids: {sorted(complete)}")
    print(f"first partial examples: {sorted(partial.items())[:20]}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--harvard-data-root",
        type=Path,
        default=Path("/storage/vacoda/data/external/harvard_comtrade_bulk"),
    )
    parser.add_argument(
        "--generator-root",
        type=Path,
        default=Path(
            "/storage/vacoda/data/references/comtrade-conversion-weights/generator"
        ),
    )
    parser.add_argument("--source-hs", default="H4")
    parser.add_argument("--source-year", type=int, default=2016)
    parser.add_argument("--target-hs", default="H5")
    parser.add_argument("--target-year", type=int, default=2017)

    args = parser.parse_args()

    audit_raw_marker_files(
        harvard_data_root=args.harvard_data_root,
        source_hs=args.source_hs,
        source_year=args.source_year,
        target_hs=args.target_hs,
        target_year=args.target_year,
    )

    audit_aggregated_files(
        harvard_data_root=args.harvard_data_root,
        source_hs=args.source_hs,
        source_year=args.source_year,
        target_hs=args.target_hs,
        target_year=args.target_year,
    )

    audit_specific_product_codes(
        harvard_data_root=args.harvard_data_root,
        source_hs=args.source_hs,
        source_year=args.source_year,
        target_hs=args.target_hs,
        target_year=args.target_year,
    )

    audit_matrix_files(
        generator_root=args.generator_root,
        source_year=args.source_year,
        target_year=args.target_year,
    )


if __name__ == "__main__":
    main()