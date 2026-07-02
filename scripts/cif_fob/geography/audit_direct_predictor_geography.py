#!/usr/bin/env python3
"""
Audit geography-related predictor failures among direct CIF/FOB factors.

This standalone diagnostic writes only machine-readable artifacts under:

    /storage/vacoda/data/interim/cif_fob/target_<hs>/geography_audit/

It does not rebuild the pipeline and does not write a Markdown report.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import duckdb


DEFAULT_STORAGE_ROOT = Path("/storage/vacoda")
DEFAULT_THREADS = 16

CORE_REQUIRED_COLUMNS = {
    "year",
    "exporter_iso",
    "importer_iso",
    "target_cmd",
    "model_eligible",
    "factor_source",
    "observed_cif_fob_factor",
    "initial_predictors_complete",
    "predictor_completeness_reason",
    "pair_geography_found",
    "exporter_geography_found",
    "importer_geography_found",
    "import_direct_covered_primary_value",
}

OPTIONAL_EVIDENCE_COLUMNS = [
    "import_value_cif",
    "import_value_fob",
    "export_value_fob",
    "import_direct_covered_cif_value",
    "import_direct_covered_fob_value",
    "import_direct_value_coverage",
    "import_weight",
    "export_weight",
    "mirror_weight_concordance",
    "pair_uses_weight_proxy",
    "pair_uses_estimated_weight",
    "import_weight_estimated",
    "distance_model_source",
    "log_distance",
    "contiguity",
    "exporter_landlocked",
    "importer_landlocked",
]

DIAGNOSTIC_NAME_TOKENS = (
    "crosswalk",
    "mapping",
    "mapped",
    "proxy",
    "cepii",
    "gravity",
    "geography",
    "country_id",
    "iso3",
    "distance",
    "contig",
    "landlocked",
)

OUTPUT_FILENAMES = [
    "direct_geography_failure_rows.parquet",
    "failure_summary.csv",
    "independent_missingness_summary.csv",
    "missingness_combinations.csv",
    "missing_importer_summary.csv",
    "missing_exporter_summary.csv",
    "missing_pair_summary.csv",
    "failure_by_year.csv",
    "failure_by_product.csv",
    "audit_manifest.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit geography-related predictor failures among selected "
            "direct CIF/FOB factor rows."
        )
    )
    parser.add_argument("--target-hs", default="H5")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def configure_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("CIF_FOB_GEOGRAPHY_AUDIT")


def sql_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def resolve_paths(
    *,
    target_hs: str,
    input_path: Path | None,
    output_dir: Path | None,
) -> tuple[Path, Path]:
    target_hs_lower = target_hs.strip().lower()
    base_dir = (
        DEFAULT_STORAGE_ROOT
        / "data"
        / "interim"
        / "cif_fob"
        / f"target_{target_hs_lower}"
    )
    return (
        input_path
        or base_dir
        / f"target_{target_hs_lower}_cif_fob_training_features.parquet",
        output_dir or base_dir / "geography_audit",
    )


def read_columns(
    con: duckdb.DuckDBPyConnection,
    path: Path,
) -> list[str]:
    rows = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet({sql_literal(path)})"
    ).fetchall()
    return [str(row[0]) for row in rows]


def validate_required_columns(columns: Iterable[str]) -> None:
    missing = sorted(CORE_REQUIRED_COLUMNS - set(columns))
    if missing:
        raise RuntimeError(
            "Training-features artifact is missing required audit columns: "
            + ", ".join(missing)
        )


def select_optional_diagnostic_columns(
    columns: Iterable[str],
) -> list[str]:
    available = list(columns)
    selected: list[str] = []
    for column in OPTIONAL_EVIDENCE_COLUMNS:
        if column in available and column not in selected:
            selected.append(column)
    for column in available:
        lower = column.lower()
        if (
            column not in selected
            and column not in CORE_REQUIRED_COLUMNS
            and any(token in lower for token in DIAGNOSTIC_NAME_TOKENS)
        ):
            selected.append(column)
    return selected


def ensure_outputs(
    output_dir: Path,
    *,
    force: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [
        output_dir / name
        for name in OUTPUT_FILENAMES
        if (output_dir / name).exists()
    ]
    if existing and not force:
        raise FileExistsError(
            "Audit outputs already exist. Use --force to overwrite:\n"
            + "\n".join(f"  {path}" for path in existing)
        )
    if force:
        for path in existing:
            path.unlink()


def copy_query(
    con: duckdb.DuckDBPyConnection,
    *,
    query: str,
    output_path: Path,
    output_format: str,
) -> None:
    if output_format == "PARQUET":
        options = "FORMAT PARQUET, COMPRESSION ZSTD"
    elif output_format == "CSV":
        options = "FORMAT CSV, HEADER TRUE"
    else:
        raise ValueError(f"Unsupported output format: {output_format}")
    con.execute(
        f"COPY ({query}) TO {sql_literal(output_path)} ({options})"
    )


def build_audit(
    *,
    input_path: Path,
    output_dir: Path,
    threads: int,
    force: bool,
    logger: logging.Logger,
) -> dict[str, object]:
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Training-features Parquet not found: {input_path}"
        )

    ensure_outputs(output_dir, force=force)
    con = duckdb.connect()

    try:
        con.execute(f"SET threads = {int(threads)}")
        columns = read_columns(con, input_path)
        validate_required_columns(columns)
        optional_columns = select_optional_diagnostic_columns(columns)

        logger.info("Input: %s", input_path)
        logger.info("Output directory: %s", output_dir)
        logger.info("DuckDB threads: %d", threads)
        logger.info(
            "Additional diagnostic columns retained: %d",
            len(optional_columns),
        )

        optional_select = "".join(
            f",\n                {quote_identifier(column)}"
            for column in optional_columns
        )
        input_sql = sql_literal(input_path)

        detail_query = f"""
            SELECT
                year,
                exporter_iso,
                importer_iso,
                CAST(target_cmd AS VARCHAR) AS target_cmd,
                factor_source,
                observed_cif_fob_factor,
                import_direct_covered_primary_value
                    AS direct_evidence_value_usd,
                predictor_completeness_reason,
                pair_geography_found,
                exporter_geography_found,
                importer_geography_found,
                NOT pair_geography_found AS pair_geography_missing,
                NOT exporter_geography_found
                    AS exporter_geography_missing,
                NOT importer_geography_found
                    AS importer_geography_missing,
                CONCAT(
                    CASE WHEN NOT pair_geography_found
                        THEN 'pair_missing' ELSE 'pair_found' END,
                    '|',
                    CASE WHEN NOT exporter_geography_found
                        THEN 'exporter_missing' ELSE 'exporter_found' END,
                    '|',
                    CASE WHEN NOT importer_geography_found
                        THEN 'importer_missing' ELSE 'importer_found' END
                ) AS geography_missingness_combination
                {optional_select}
            FROM read_parquet({input_sql})
            WHERE model_eligible
              AND LOWER(CAST(factor_source AS VARCHAR)) = 'direct'
              AND NOT initial_predictors_complete
        """

        detail_path = output_dir / "direct_geography_failure_rows.parquet"
        copy_query(
            con,
            query=detail_query,
            output_path=detail_path,
            output_format="PARQUET",
        )
        detail_sql = sql_literal(detail_path)

        queries = {
            "failure_summary.csv": f"""
                WITH base AS (
                    SELECT * FROM read_parquet({detail_sql})
                ), total AS (
                    SELECT COUNT(*) AS n_total FROM base
                )
                SELECT
                    predictor_completeness_reason AS first_failure_reason,
                    COUNT(*) AS n_rows,
                    100.0 * COUNT(*) / NULLIF(MAX(n_total), 0)
                        AS share_of_excluded_direct_rows_percent,
                    COUNT(DISTINCT exporter_iso) AS n_distinct_exporters,
                    COUNT(DISTINCT importer_iso) AS n_distinct_importers,
                    COUNT(DISTINCT exporter_iso || '|' || importer_iso)
                        AS n_distinct_pairs,
                    COUNT(DISTINCT target_cmd) AS n_distinct_products,
                    COUNT(DISTINCT year) AS n_distinct_years,
                    SUM(direct_evidence_value_usd)
                        AS direct_evidence_value_usd,
                    MEDIAN(direct_evidence_value_usd)
                        AS median_direct_evidence_value_usd,
                    MEDIAN(observed_cif_fob_factor)
                        AS median_observed_factor,
                    QUANTILE_CONT(observed_cif_fob_factor, 0.95)
                        AS observed_factor_p95
                FROM base CROSS JOIN total
                GROUP BY predictor_completeness_reason
                ORDER BY n_rows DESC, first_failure_reason
            """,
            "independent_missingness_summary.csv": f"""
                WITH base AS (
                    SELECT * FROM read_parquet({detail_sql})
                ), totals AS (
                    SELECT COUNT(*) AS n_total FROM base
                ), flags AS (
                    SELECT 'pair_geography_missing' AS missing_requirement,
                           COUNT(*) FILTER (WHERE pair_geography_missing)
                               AS n_rows
                    FROM base
                    UNION ALL
                    SELECT 'exporter_geography_missing',
                           COUNT(*) FILTER (WHERE exporter_geography_missing)
                    FROM base
                    UNION ALL
                    SELECT 'importer_geography_missing',
                           COUNT(*) FILTER (WHERE importer_geography_missing)
                    FROM base
                )
                SELECT
                    missing_requirement,
                    n_rows,
                    100.0 * n_rows / NULLIF(n_total, 0)
                        AS share_of_excluded_direct_rows_percent
                FROM flags CROSS JOIN totals
                ORDER BY n_rows DESC, missing_requirement
            """,
            "missingness_combinations.csv": f"""
                WITH base AS (
                    SELECT * FROM read_parquet({detail_sql})
                ), total AS (
                    SELECT COUNT(*) AS n_total FROM base
                )
                SELECT
                    pair_geography_missing,
                    exporter_geography_missing,
                    importer_geography_missing,
                    geography_missingness_combination,
                    COUNT(*) AS n_rows,
                    100.0 * COUNT(*) / NULLIF(MAX(n_total), 0)
                        AS share_of_excluded_direct_rows_percent,
                    COUNT(DISTINCT exporter_iso) AS n_distinct_exporters,
                    COUNT(DISTINCT importer_iso) AS n_distinct_importers,
                    COUNT(DISTINCT exporter_iso || '|' || importer_iso)
                        AS n_distinct_pairs,
                    SUM(direct_evidence_value_usd)
                        AS direct_evidence_value_usd
                FROM base CROSS JOIN total
                GROUP BY
                    pair_geography_missing,
                    exporter_geography_missing,
                    importer_geography_missing,
                    geography_missingness_combination
                ORDER BY n_rows DESC
            """,
            "missing_importer_summary.csv": f"""
                WITH base AS (
                    SELECT * FROM read_parquet({detail_sql})
                    WHERE importer_geography_missing
                ), total AS (
                    SELECT COUNT(*) AS n_total FROM base
                )
                SELECT
                    importer_iso,
                    COUNT(*) AS n_rows,
                    100.0 * COUNT(*) / NULLIF(MAX(n_total), 0)
                        AS share_of_importer_missing_rows_percent,
                    COUNT(DISTINCT exporter_iso) AS n_distinct_exporters,
                    COUNT(DISTINCT target_cmd) AS n_distinct_products,
                    MIN(year) AS first_year,
                    MAX(year) AS last_year,
                    SUM(direct_evidence_value_usd)
                        AS direct_evidence_value_usd,
                    MEDIAN(direct_evidence_value_usd)
                        AS median_direct_evidence_value_usd,
                    MEDIAN(observed_cif_fob_factor)
                        AS median_observed_factor,
                    QUANTILE_CONT(observed_cif_fob_factor, 0.95)
                        AS observed_factor_p95
                FROM base CROSS JOIN total
                GROUP BY importer_iso
                ORDER BY n_rows DESC, importer_iso
            """,
            "missing_exporter_summary.csv": f"""
                WITH base AS (
                    SELECT * FROM read_parquet({detail_sql})
                    WHERE exporter_geography_missing
                ), total AS (
                    SELECT COUNT(*) AS n_total FROM base
                )
                SELECT
                    exporter_iso,
                    COUNT(*) AS n_rows,
                    100.0 * COUNT(*) / NULLIF(MAX(n_total), 0)
                        AS share_of_exporter_missing_rows_percent,
                    COUNT(DISTINCT importer_iso) AS n_distinct_importers,
                    COUNT(DISTINCT target_cmd) AS n_distinct_products,
                    MIN(year) AS first_year,
                    MAX(year) AS last_year,
                    SUM(direct_evidence_value_usd)
                        AS direct_evidence_value_usd,
                    MEDIAN(direct_evidence_value_usd)
                        AS median_direct_evidence_value_usd,
                    MEDIAN(observed_cif_fob_factor)
                        AS median_observed_factor,
                    QUANTILE_CONT(observed_cif_fob_factor, 0.95)
                        AS observed_factor_p95
                FROM base CROSS JOIN total
                GROUP BY exporter_iso
                ORDER BY n_rows DESC, exporter_iso
            """,
            "missing_pair_summary.csv": f"""
                WITH base AS (
                    SELECT * FROM read_parquet({detail_sql})
                    WHERE pair_geography_missing
                ), total AS (
                    SELECT COUNT(*) AS n_total FROM base
                )
                SELECT
                    exporter_iso,
                    importer_iso,
                    COUNT(*) AS n_rows,
                    100.0 * COUNT(*) / NULLIF(MAX(n_total), 0)
                        AS share_of_pair_missing_rows_percent,
                    COUNT(DISTINCT target_cmd) AS n_distinct_products,
                    MIN(year) AS first_year,
                    MAX(year) AS last_year,
                    SUM(direct_evidence_value_usd)
                        AS direct_evidence_value_usd,
                    MEDIAN(direct_evidence_value_usd)
                        AS median_direct_evidence_value_usd,
                    MEDIAN(observed_cif_fob_factor)
                        AS median_observed_factor,
                    QUANTILE_CONT(observed_cif_fob_factor, 0.95)
                        AS observed_factor_p95
                FROM base CROSS JOIN total
                GROUP BY exporter_iso, importer_iso
                ORDER BY n_rows DESC, exporter_iso, importer_iso
            """,
            "failure_by_year.csv": f"""
                WITH base AS (
                    SELECT * FROM read_parquet({detail_sql})
                ), total AS (
                    SELECT COUNT(*) AS n_total FROM base
                )
                SELECT
                    year,
                    predictor_completeness_reason AS first_failure_reason,
                    COUNT(*) AS n_rows,
                    100.0 * COUNT(*) / NULLIF(MAX(n_total), 0)
                        AS share_of_all_excluded_direct_rows_percent,
                    COUNT(DISTINCT exporter_iso || '|' || importer_iso)
                        AS n_distinct_pairs,
                    COUNT(DISTINCT target_cmd) AS n_distinct_products,
                    SUM(direct_evidence_value_usd)
                        AS direct_evidence_value_usd
                FROM base CROSS JOIN total
                GROUP BY year, predictor_completeness_reason
                ORDER BY year, n_rows DESC
            """,
            "failure_by_product.csv": f"""
                WITH base AS (
                    SELECT * FROM read_parquet({detail_sql})
                ), total AS (
                    SELECT COUNT(*) AS n_total FROM base
                )
                SELECT
                    target_cmd,
                    predictor_completeness_reason AS first_failure_reason,
                    COUNT(*) AS n_rows,
                    100.0 * COUNT(*) / NULLIF(MAX(n_total), 0)
                        AS share_of_all_excluded_direct_rows_percent,
                    COUNT(DISTINCT exporter_iso || '|' || importer_iso)
                        AS n_distinct_pairs,
                    COUNT(DISTINCT year) AS n_distinct_years,
                    SUM(direct_evidence_value_usd)
                        AS direct_evidence_value_usd
                FROM base CROSS JOIN total
                GROUP BY target_cmd, predictor_completeness_reason
                ORDER BY n_rows DESC, target_cmd
            """,
        }

        for filename, query in queries.items():
            copy_query(
                con,
                query=query,
                output_path=output_dir / filename,
                output_format="CSV",
            )

        audit_counts = con.execute(
            f"""
            SELECT
                COUNT(*) AS n_excluded_direct_rows,
                COUNT(*) FILTER (WHERE pair_geography_missing)
                    AS n_pair_missing,
                COUNT(*) FILTER (WHERE exporter_geography_missing)
                    AS n_exporter_missing,
                COUNT(*) FILTER (WHERE importer_geography_missing)
                    AS n_importer_missing,
                COUNT(DISTINCT exporter_iso) AS n_distinct_exporters,
                COUNT(DISTINCT importer_iso) AS n_distinct_importers,
                COUNT(DISTINCT exporter_iso || '|' || importer_iso)
                    AS n_distinct_pairs,
                COUNT(DISTINCT target_cmd) AS n_distinct_products,
                MIN(year) AS first_year,
                MAX(year) AS last_year,
                SUM(direct_evidence_value_usd)
                    AS direct_evidence_value_usd
            FROM read_parquet({detail_sql})
            """
        ).fetchone()

    finally:
        con.close()

    if audit_counts is None:
        raise RuntimeError("Audit produced no summary row.")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "threads": int(threads),
        "scope": (
            "model_eligible=true, factor_source=direct, and "
            "initial_predictors_complete=false"
        ),
        "first_failure_semantics": (
            "predictor_completeness_reason is ordered and mutually "
            "exclusive; independent missingness flags may overlap"
        ),
        "optional_diagnostic_columns_retained": optional_columns,
        "metrics": {
            "n_excluded_direct_rows": int(audit_counts[0] or 0),
            "n_pair_geography_missing": int(audit_counts[1] or 0),
            "n_exporter_geography_missing": int(audit_counts[2] or 0),
            "n_importer_geography_missing": int(audit_counts[3] or 0),
            "n_distinct_exporters": int(audit_counts[4] or 0),
            "n_distinct_importers": int(audit_counts[5] or 0),
            "n_distinct_pairs": int(audit_counts[6] or 0),
            "n_distinct_products": int(audit_counts[7] or 0),
            "first_year": int(audit_counts[8]) if audit_counts[8] is not None else None,
            "last_year": int(audit_counts[9]) if audit_counts[9] is not None else None,
            "direct_evidence_value_usd": float(audit_counts[10] or 0.0),
        },
        "artifacts": {
            name: str(output_dir / name)
            for name in OUTPUT_FILENAMES
        },
    }

    (output_dir / "audit_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    args = parse_args()
    logger = configure_logging()
    if args.threads <= 0:
        raise ValueError("--threads must be greater than zero.")

    input_path, output_dir = resolve_paths(
        target_hs=args.target_hs,
        input_path=args.input,
        output_dir=args.output_dir,
    )
    manifest = build_audit(
        input_path=input_path,
        output_dir=output_dir,
        threads=args.threads,
        force=args.force,
        logger=logger,
    )
    metrics = manifest["metrics"]

    logger.info("=" * 72)
    logger.info("Direct-factor geography audit completed.")
    logger.info("Excluded direct rows: %s", f"{metrics['n_excluded_direct_rows']:,}")
    logger.info("Pair geography missing independently: %s", f"{metrics['n_pair_geography_missing']:,}")
    logger.info("Exporter geography missing independently: %s", f"{metrics['n_exporter_geography_missing']:,}")
    logger.info("Importer geography missing independently: %s", f"{metrics['n_importer_geography_missing']:,}")
    logger.info("Output directory: %s", output_dir)
    logger.info("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
