"""
vacoda.hs_sync.harvard_stage

Stage vacoda raw Comtrade CSV files into the folder layout expected by the
Harvard/Growth Lab conversion-weights generator.

Input:
    /storage/vacoda/data/raw/api/single/*.csv

Output:
    /storage/vacoda/data/external/harvard_comtrade/as_reported/raw_parquet/
    ├── H0/1992/*.parquet
    ├── H1/1996/*.parquet
    └── ...
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

import duckdb
import pandas as pd

from vacoda.utils.logger import get_logger, rel_path
from vacoda.io.reports import write_dataframe_report

if TYPE_CHECKING:
    from vacoda.config.hs_sync import HsSyncConfig

logger = get_logger("HS_SYNC.HARVARD_STAGE")

VALID_HS_SYSTEMS = {"H0", "H1", "H2", "H3", "H4", "H5", "H6"}


@dataclass(frozen=True)
class HarvardStagingPaths:
    output_root: Path
    partitioned_root: Path
    raw_parquet_root: Path
    aggregated_root: Path
    artifact_dir: Path
    marker_path: Path
    inventory_latest: Path
    preservation_latest: Path


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _fingerprint_glob(root: Path, pattern: str) -> dict[str, Any]:
    files = [path for path in sorted(root.glob(pattern)) if path.is_file()]
    entries = [
        {
            "path": str(path.relative_to(root)),
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in files
    ]
    return {
        "root": str(root),
        "pattern": pattern,
        "n_files": len(entries),
        "hash": _json_sha256(entries),
    }


def _fingerprint_tree(root: Path, *, exclude: set[Path] | None = None) -> dict[str, Any]:
    exclude = exclude or set()
    entries: list[dict[str, Any]] = []

    if root.exists():
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            if path in exclude:
                continue
            entries.append(
                {
                    "path": str(path.relative_to(root)),
                    "size": path.stat().st_size,
                    "mtime_ns": path.stat().st_mtime_ns,
                }
            )

    return {
        "root": str(root),
        "n_files": len(entries),
        "hash": _json_sha256(entries),
    }


def _fingerprint_file(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {
            "path": str(path),
            "exists": False,
            "size": None,
            "mtime_ns": None,
        }

    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def get_harvard_staging_paths(cfg: HsSyncConfig) -> HarvardStagingPaths:
    output_root = cfg.harvard_staging.output.resolve_data_dir()
    layout = cfg.harvard_staging.layout

    artifact_dir = cfg.harvard_staging.artifact.resolve_output_dir()

    inventory_base = Path(cfg.harvard_staging.artifact.inventory_filename)
    preservation_base = Path(cfg.harvard_staging.artifact.preservation_filename)

    return HarvardStagingPaths(
        output_root=output_root,
        partitioned_root=layout.resolve_partitioned_dir(output_root),
        raw_parquet_root=layout.resolve_raw_parquet_dir(output_root),
        aggregated_root=layout.resolve_aggregated_dir(output_root),
        artifact_dir=artifact_dir,
        marker_path=artifact_dir / cfg.harvard_staging.skip.marker_filename,
        inventory_latest=artifact_dir
        / f"{inventory_base.stem}_latest{inventory_base.suffix}",
        preservation_latest=artifact_dir
        / f"{preservation_base.stem}_latest{preservation_base.suffix}",
    )

def build_harvard_staging_fingerprint(cfg: HsSyncConfig) -> dict[str, Any]:
    paths = get_harvard_staging_paths(cfg)

    config_payload = {
        "engine": {"threads": cfg.engine.threads},
        "harvard_staging": cfg.raw.get("harvard_staging", {}),
    }

    return {
        "input": {
            "primary": _fingerprint_glob(
                cfg.harvard_staging.input.resolve_data_dir(),
                cfg.harvard_staging.input.pattern or "",
            ),
            "country_reference": _fingerprint_glob(
                cfg.harvard_staging.country_reference.input.resolve_data_dir(),
                cfg.harvard_staging.country_reference.input.pattern or "",
            ),
        },
        "config": {
            "hash": _json_sha256(config_payload),
        },
        "output": {
            "staging_tree": _fingerprint_tree(paths.output_root, exclude={paths.marker_path}),
            "inventory_latest": _fingerprint_file(paths.inventory_latest),
            "preservation_latest": _fingerprint_file(paths.preservation_latest),
        },
    }

def write_harvard_staging_success_marker(
    *,
    cfg: HsSyncConfig,
    run_id: str,
    summary: dict[str, Any],
    compatibility_summary: dict[str, Any],
) -> Path:
    paths = get_harvard_staging_paths(cfg)
    paths.marker_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "status": "success",
        "run_id": run_id,
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
        "marker_version": 1,
        "paths": {
            "output_root": str(paths.output_root),
            "artifact_dir": str(paths.artifact_dir),
            "inventory_latest": str(paths.inventory_latest),
            "preservation_latest": str(paths.preservation_latest),
        },
        "summary": summary,
        "compatibility_summary": compatibility_summary,
        "fingerprint": build_harvard_staging_fingerprint(cfg),
    }

    paths.marker_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return paths.marker_path


def harvard_staging_success_marker_valid(
    *,
    cfg: HsSyncConfig,
) -> tuple[bool, list[str]]:
    paths = get_harvard_staging_paths(cfg)
    marker_path = paths.marker_path

    reasons: list[str] = []

    if not marker_path.exists():
        return False, [f"missing success marker: {marker_path}"]

    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, [f"cannot read success marker {marker_path}: {exc}"]

    if marker.get("status") != "success":
        reasons.append("success marker status is not 'success'")

    if marker.get("marker_version") != 1:
        reasons.append("unsupported success marker version")

    fingerprint = marker.get("fingerprint")
    if not isinstance(fingerprint, dict):
        reasons.append("success marker has no valid fingerprint block")
        return False, reasons

    current = build_harvard_staging_fingerprint(cfg)
    skip_cfg = cfg.harvard_staging.skip

    if skip_cfg.check_input_fingerprint and fingerprint.get("input") != current["input"]:
        reasons.append("input fingerprint changed")

    if skip_cfg.check_config_fingerprint and fingerprint.get("config") != current["config"]:
        reasons.append("config fingerprint changed")

    if skip_cfg.check_output_fingerprint and fingerprint.get("output") != current["output"]:
        reasons.append("output fingerprint changed")

    return len(reasons) == 0, reasons

def audit_csv_to_harvard_parquet_staging(
    *,
    input_dir: Path,
    input_pattern: str,
    output_root: Path,
    year_col: str = "RefYear",
    hs_col: str = "ClassificationCode",
    value_col: str = "PrimaryValue",
    threads: int = 16,
    value_abs_tolerance: float = 1.0,
    value_rel_tolerance: float = 1e-10,) -> pd.DataFrame:
    """
    Compare raw CSV input against staged Harvard-compatible Parquet output.

    The audit compares row counts and value totals by HS system and year.

    Output columns include:
        hs
        year
        n_rows_csv
        n_rows_parquet
        row_diff
        value_total_csv
        value_total_parquet
        value_diff
        status

    A status of 'ok' means row counts match and value totals are close.
    """

    input_glob = str(input_dir / input_pattern)

    parquet_glob = str(
        output_root
        / "as_reported"
        / "raw_parquet"
        / "*"
        / "*"
        / "*.parquet"
    )

    logger.info("Auditing CSV → Parquet staging.")
    logger.info("CSV input glob: %s", input_glob)
    logger.info("Parquet glob: %s", parquet_glob)

    with duckdb.connect() as con:
        con.execute(f"PRAGMA threads={int(threads)}")

        csv_summary = con.execute(
            f"""
            SELECT
                CAST("{hs_col}" AS VARCHAR) AS hs,
                CAST("{year_col}" AS INTEGER) AS year,
                COUNT(*) AS n_rows_csv,
                SUM(CAST("{value_col}" AS DOUBLE)) AS value_total_csv
            FROM read_csv_auto(
                '{input_glob}',
                union_by_name = true,
                filename = true
            )
            WHERE "{hs_col}" IS NOT NULL
              AND "{year_col}" IS NOT NULL
            GROUP BY hs, year
            """
        ).df()

        parquet_summary = con.execute(
            f"""
            SELECT
                CAST("{hs_col}" AS VARCHAR) AS hs,
                CAST("{year_col}" AS INTEGER) AS year,
                COUNT(*) AS n_rows_parquet,
                SUM(CAST("{value_col}" AS DOUBLE)) AS value_total_parquet
            FROM read_parquet(
                '{parquet_glob}',
                union_by_name = true
            )
            WHERE "{hs_col}" IS NOT NULL
              AND "{year_col}" IS NOT NULL
            GROUP BY hs, year
            """
        ).df()

    audit = csv_summary.merge(
        parquet_summary,
        on=["hs", "year"],
        how="outer",
    )

    audit["n_rows_csv"] = audit["n_rows_csv"].fillna(0).astype("int64")
    audit["n_rows_parquet"] = audit["n_rows_parquet"].fillna(0).astype("int64")

    audit["value_total_csv"] = audit["value_total_csv"].fillna(0.0)
    audit["value_total_parquet"] = audit["value_total_parquet"].fillna(0.0)

    audit["row_diff"] = audit["n_rows_parquet"] - audit["n_rows_csv"]
    audit["value_diff"] = audit["value_total_parquet"] - audit["value_total_csv"]
    audit["abs_value_diff"] = audit["value_diff"].abs()

    audit["rel_value_diff"] = audit["abs_value_diff"] / audit[
                "value_total_csv"
                ].abs().clip(lower=1.0)

    audit["value_close"] = (
        audit["abs_value_diff"] <= value_abs_tolerance
        ) | (
        audit["rel_value_diff"] <= value_rel_tolerance
        )

    audit["status"] = "ok"

    audit.loc[audit["row_diff"] != 0, "status"] = "row_mismatch"

    audit.loc[
        (audit["row_diff"] == 0)
        & (~audit["value_close"]),
        "status",
        ] = "value_mismatch"

    audit = audit.sort_values(["status", "hs", "year"]).reset_index(drop=True)

    logger.info("Audit rows: %s", audit.shape[0])
    logger.info("Audit status counts: %s", audit["status"].value_counts().to_dict())

    return audit


def stage_comtrade_for_harvard_weights_partitioned(
    *,
    input_dir: Path,
    pattern: str,
    output_root: Path,
    year_col: str = "RefYear",
    hs_col: str = "ClassificationCode",
    threads: int = 16,
    overwrite: bool = False) -> Path:
    """
    Write partitioned parquet output by HS and year.

    Output will be:
        raw_parquet/hs=H5/year=2019/*.parquet

    This is faster, but the folder names are not exactly Harvard-style.
    """

    input_glob = str(input_dir / pattern)
    output_base = output_root / "as_reported" / "raw_parquet_partitioned"

    if output_base.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory already exists: {rel_path(output_base)}"
        )

    output_base.mkdir(parents=True, exist_ok=True)

    with duckdb.connect() as con:
        con.execute(f"PRAGMA threads={int(threads)}")

        con.execute(
            f"""
            COPY (
                SELECT
                    *,
                    CAST("{hs_col}" AS VARCHAR) AS hs_partition,
                    CAST("{year_col}" AS INTEGER) AS year_partition
                FROM read_csv_auto(
                    '{input_glob}',
                    union_by_name = true,
                    filename = true
                )
                WHERE "{hs_col}" IS NOT NULL
                  AND "{year_col}" IS NOT NULL
            )
            TO '{output_base}'
            (
                FORMAT PARQUET,
                PARTITION_BY (hs_partition, year_partition),
                OVERWRITE_OR_IGNORE TRUE
            )
            """
        )

    return output_base

def stage_comtrade_for_harvard_weights(
    *,
    input_dir: Path,
    pattern: str,
    output_root: Path,
    year_col: str = "RefYear",
    hs_col: str = "ClassificationCode",
    threads: int = 16,
    overwrite: bool = False) -> dict[str, int]:
    """
    Stage raw Comtrade CSV data into Harvard-compatible parquet folders.

    Output layout:
        output_root/as_reported/raw_parquet/{HS}/{YEAR}/COMTRADE-FINALCLASSIC-CA-{HS}-{YEAR}.parquet

    Example:
        H5 / 2019 →
        output_root/as_reported/raw_parquet/H5/2019/COMTRADE-FINALCLASSIC-CA-H5-2019.parquet

    Parameters
    ----------
    input_dir:
        Directory containing raw Comtrade CSV files.

    pattern:
        Glob pattern, e.g. "*.csv".

    output_root:
        Base output directory, e.g.
        /storage/vacoda/data/external/harvard_comtrade

    year_col:
        Year column in Comtrade data.

    hs_col:
        HS classification column in Comtrade data.

    threads:
        DuckDB thread count.

    overwrite:
        If False, existing parquet files are skipped.

    Returns
    -------
    dict[str, int]
        Small summary of staged files.
    """

    input_glob = str(input_dir / pattern)
    output_base = output_root / "as_reported" / "raw_parquet"
    output_base.mkdir(parents=True, exist_ok=True)

    logger.info("Input glob: %s", input_glob)
    logger.info("Output base: %s", rel_path(output_base))
    logger.info("Threads: %s", threads)

    written = 0
    skipped = 0

    with duckdb.connect() as con:
        con.execute(f"PRAGMA threads={int(threads)}")

        logger.info("Scanning available HS/year combinations.")

        combos = con.execute(
            f"""
            SELECT DISTINCT
                CAST("{hs_col}" AS VARCHAR) AS hs,
                CAST("{year_col}" AS INTEGER) AS year
            FROM read_csv_auto(
                '{input_glob}',
                union_by_name = true,
                filename = true
            )
            WHERE "{hs_col}" IS NOT NULL
              AND "{year_col}" IS NOT NULL
            ORDER BY hs, year
            """
        ).df()

        combos["hs"] = combos["hs"].astype("string").str.strip().str.upper()
        combos = combos.loc[combos["hs"].isin(VALID_HS_SYSTEMS)].copy()

        logger.info("Found %s valid HS/year combinations.", f"{len(combos):,}")

        for i, row in enumerate(combos.itertuples(index=False), start=1):
            hs = str(row.hs)
            year = int(row.year)

            out_dir = output_base / hs / str(year)
            out_dir.mkdir(parents=True, exist_ok=True)

            out_file = out_dir / f"COMTRADE-FINALCLASSIC-CA-{hs}-{year}.parquet"

            if out_file.exists() and not overwrite:
                skipped += 1
                logger.info(
                    "[%s/%s] Skipping existing: %s",
                    f"{i:,}",
                    f"{len(combos):,}",
                    rel_path(out_file),
                )
                continue

            logger.info(
                "[%s/%s] Writing %s %s → %s",
                f"{i:,}",
                f"{len(combos):,}",
                hs,
                year,
                rel_path(out_file),
            )

            con.execute(
                f"""
                COPY (
                    SELECT *
                    FROM read_csv_auto(
                        '{input_glob}',
                        union_by_name = true,
                        filename = true
                    )
                    WHERE CAST("{hs_col}" AS VARCHAR) = '{hs}'
                      AND CAST("{year_col}" AS INTEGER) = {year}
                )
                TO '{out_file}'
                (FORMAT PARQUET)
                """
            )

            written += 1

    logger.info("Staging complete. Written=%s, skipped=%s", written, skipped)

    return {
        "written": written,
        "skipped": skipped,
        "total_combinations": int(len(combos)),
    }


def audit_harvard_staging_output(
    *,
    output_root: Path) -> pd.DataFrame:
    """
    Audit staged Harvard-compatible parquet layout.

    Returns one row per HS/year folder with file counts and total size.
    """

    output_base = output_root / "as_reported" / "raw_parquet"

    rows: list[dict] = []

    for hs_dir in sorted(output_base.glob("H*")):
        if not hs_dir.is_dir():
            continue

        for year_dir in sorted(hs_dir.glob("*")):
            if not year_dir.is_dir():
                continue

            files = sorted(year_dir.glob("*.parquet"))

            rows.append(
                {
                    "hs": hs_dir.name,
                    "year": int(year_dir.name),
                    "n_files": len(files),
                    "size_bytes": sum(p.stat().st_size for p in files),
                }
            )

    return pd.DataFrame(rows).sort_values(["hs", "year"]).reset_index(drop=True)


def summarize_staging_audit(audit: pd.DataFrame) -> dict[str, object]:
    """
    Summarize CSV → Parquet staging audit.
    """

    total_rows_csv = int(audit["n_rows_csv"].sum())
    total_rows_parquet = int(audit["n_rows_parquet"].sum())

    total_value_csv = float(audit["value_total_csv"].sum())
    total_value_parquet = float(audit["value_total_parquet"].sum())

    summary = {
        "n_groups": int(len(audit)),
        "status_counts": audit["status"].value_counts().to_dict(),
        "total_rows_csv": total_rows_csv,
        "total_rows_parquet": total_rows_parquet,
        "total_row_diff": int(total_rows_parquet - total_rows_csv),
        "total_value_csv": total_value_csv,
        "total_value_parquet": total_value_parquet,
        "total_value_diff": float(total_value_parquet - total_value_csv),
    }

    return summary


def assert_no_staging_data_loss(
    audit: pd.DataFrame,
    *,
    value_tolerance: float = 0.01) -> None:
    """
    Raise an error if CSV → Parquet staging lost rows or changed values.
    """

    total_row_diff = int(audit["row_diff"].sum())
    total_value_diff = float(audit["value_diff"].sum())

    bad = audit.loc[audit["status"] != "ok"].copy()

    if total_row_diff != 0:
        raise ValueError(
            "CSV → Parquet staging row-count mismatch. "
            f"Total row difference: {total_row_diff:,}"
        )

    if abs(total_value_diff) > value_tolerance:
        raise ValueError(
            "CSV → Parquet staging value-total mismatch. "
            f"Total value difference: {total_value_diff:,.6f}"
        )

    if not bad.empty:
        raise ValueError(
            "CSV → Parquet staging has group-level mismatches. "
            f"Bad HS/year groups: {len(bad):,}"
        )

def audit_global_csv_to_parquet_totals(
    *,
    input_dir: Path,
    input_pattern: str,
    output_root: Path,
    value_col: str = "PrimaryValue",
    threads: int = 16) -> pd.DataFrame:

    input_glob = str(input_dir / input_pattern)
    parquet_glob = str(output_root / "as_reported" / "raw_parquet" / "*" / "*" / "*.parquet")

    with duckdb.connect() as con:
        con.execute(f"PRAGMA threads={int(threads)}")

        csv = con.execute(
            f"""
            SELECT
                'csv' AS source,
                COUNT(*) AS n_rows,
                SUM(CAST("{value_col}" AS DOUBLE)) AS value_total
            FROM read_csv_auto(
                '{input_glob}',
                union_by_name = true,
                filename = true
            )
            """
        ).df()

        parquet = con.execute(
            f"""
            SELECT
                'parquet' AS source,
                COUNT(*) AS n_rows,
                SUM(CAST("{value_col}" AS DOUBLE)) AS value_total
            FROM read_parquet(
                '{parquet_glob}',
                union_by_name = true
            )
            """
        ).df()

    return pd.concat([csv, parquet], ignore_index=True)


def stage_comtrade_for_harvard_input(
    *,
    input_dir: Path,
    pattern: str,
    output_dir: Path,
    year_col: str = "RefYear",
    hs_col: str = "ClassificationCode",
    cmd_col: str = "CmdCode",
    valid_cmd_regex: str | None = None,
    years_by_hs: dict[str, list[int]] | None = None,
    threads: int = 16,
    overwrite: bool = False,
) -> Path:
    """
    Fast one-pass staging from vacoda raw/bulk Comtrade files to
    partitioned Parquet.

    This writes:
        raw_parquet_partitioned/
        ├── hs_partition=H4/
        │   └── year_partition=2016/*.parquet
        └── hs_partition=H5/
            └── year_partition=2017/*.parquet

    Optional filters:
        valid_cmd_regex:
            Keep only commodity codes matching this regex.
            Example: r"^[0-9]{6}$" keeps only 6-digit product codes.

        years_by_hs:
            Keep only selected HS/year pairs.
            Example: {"H4": [2016], "H5": [2017]}.
    """

    input_glob = str(input_dir / pattern)
    output_base = output_dir

    if output_base.exists():
        if overwrite:
            logger.info(
                "Output directory already exists and overwrite is enabled: %s",
                rel_path(output_base),
            )
        else:
            logger.info(
                "Output directory already exists; reusing it without deleting existing files: %s",
                rel_path(output_base),
            )

    output_base.mkdir(parents=True, exist_ok=True)

    where_clauses = [
        f'"{hs_col}" IS NOT NULL',
        f'"{year_col}" IS NOT NULL',
    ]

    if valid_cmd_regex:
        where_clauses.append(
            f"regexp_matches(CAST(\"{cmd_col}\" AS VARCHAR), '{valid_cmd_regex}')"
        )

    if years_by_hs:
        allowed_pairs: list[str] = []

        for hs_system, years in years_by_hs.items():
            hs = str(hs_system).strip().upper()

            for year in years:
                allowed_pairs.append(
                    "("
                    f"CAST(\"{hs_col}\" AS VARCHAR) = '{hs}' "
                    "AND "
                    f"CAST(\"{year_col}\" AS INTEGER) = {int(year)}"
                    ")"
                )

        if allowed_pairs:
            where_clauses.append("(" + " OR ".join(allowed_pairs) + ")")

    where_sql = "\n                  AND ".join(where_clauses)

    logger.info("Fast staging with DuckDB partitioned Parquet.")
    logger.info("Input glob: %s", input_glob)
    logger.info("Output base: %s", rel_path(output_base))
    logger.info("Threads: %s", threads)

    if valid_cmd_regex:
        logger.info("Commodity-code filter: %s", valid_cmd_regex)

    if years_by_hs:
        logger.info("HS/year filter: %s", years_by_hs)

    with duckdb.connect() as con:
        con.execute(f"PRAGMA threads={int(threads)}")

        con.execute(
            f"""
            COPY (
                SELECT
                    *,
                    CAST("{hs_col}" AS VARCHAR) AS hs_partition,
                    CAST("{year_col}" AS INTEGER) AS year_partition
                FROM read_csv_auto(
                    '{input_glob}',
                    union_by_name = true,
                    filename = true
                )
                WHERE {where_sql}
            )
            TO '{output_base}'
            (
                FORMAT PARQUET,
                PARTITION_BY (hs_partition, year_partition),
                OVERWRITE_OR_IGNORE TRUE
            )
            """
        )

    logger.info("Fast staging complete: %s", rel_path(output_base))

    return output_base

def audit_harvard_staging_partitioned_output(
    *,
    output_dir: Path) -> pd.DataFrame:
    """
    Audit fast DuckDB-partitioned Harvard staging output.

    Expected layout:
        output_root/as_reported/raw_parquet_partitioned/
        ├── hs_partition=H0/
        │   ├── year_partition=1988/
        │   └── ...
        └── ...

    Returns one row per HS/year partition with file counts and total size.
    """

    output_base = output_dir

    rows: list[dict] = []

    if not output_base.exists():
        return pd.DataFrame(
            columns=["hs", "year", "n_files", "size_bytes"]
        )

    for hs_dir in sorted(output_base.glob("hs_partition=*")):
        if not hs_dir.is_dir():
            continue

        hs = hs_dir.name.split("=", 1)[1]

        for year_dir in sorted(hs_dir.glob("year_partition=*")):
            if not year_dir.is_dir():
                continue

            year_text = year_dir.name.split("=", 1)[1]

            try:
                year = int(year_text)
            except ValueError:
                year = year_text

            files = sorted(year_dir.glob("*.parquet"))

            rows.append(
                {
                    "hs": hs,
                    "year": year,
                    "n_files": len(files),
                    "size_bytes": sum(p.stat().st_size for p in files),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=["hs", "year", "n_files", "size_bytes"]
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["hs", "year"])
        .reset_index(drop=True)
    )


def audit_csv_to_partitioned_parquet_staging(
    *,
    input_dir: Path,
    input_pattern: str,
    output_dir: Path,
    year_col: str = "RefYear",
    hs_col: str = "ClassificationCode",
    cmd_col: str = "CmdCode",
    value_col: str = "PrimaryValue",
    valid_cmd_regex: str | None = None,
    years_by_hs: dict[str, list[int]] | None = None,
    threads: int = 16,
    value_abs_tolerance: float = 1.0,
    value_rel_tolerance: float = 1e-10,
) -> pd.DataFrame:
    """
    Compare raw CSV/GZ input against fast DuckDB-partitioned Parquet output.

    This is the actual no-data-loss audit.

    Compares by HS system and year:
        - row counts
        - total PrimaryValue
        - number of distinct CmdCode values

    If staging was filtered by valid_cmd_regex and years_by_hs, the CSV side
    of the audit must use the same filters.
    """

    input_glob = str(input_dir / input_pattern)

    parquet_glob = str(
        output_dir
        / "hs_partition=*"
        / "year_partition=*"
        / "*.parquet"
    )

    # Build the same CSV-side filter used during staging.
    where_clauses = [
        f'"{hs_col}" IS NOT NULL',
        f'"{year_col}" IS NOT NULL',
    ]

    if valid_cmd_regex:
        where_clauses.append(
            f"regexp_matches(CAST(\"{cmd_col}\" AS VARCHAR), '{valid_cmd_regex}')"
        )

    if years_by_hs:
        allowed_pairs: list[str] = []

        for hs_system, years in years_by_hs.items():
            hs = str(hs_system).strip().upper()

            for year in years:
                allowed_pairs.append(
                    "("
                    f"CAST(\"{hs_col}\" AS VARCHAR) = '{hs}' "
                    "AND "
                    f"CAST(\"{year_col}\" AS INTEGER) = {int(year)}"
                    ")"
                )

        if allowed_pairs:
            where_clauses.append("(" + " OR ".join(allowed_pairs) + ")")

    where_sql = "\n              AND ".join(where_clauses)

    logger.info("Running CSV → partitioned Parquet preservation audit.")
    logger.info("CSV input glob: %s", input_glob)
    logger.info("Parquet glob: %s", parquet_glob)
    logger.info("CSV audit filter: %s", where_sql.replace("\n", " "))
    logger.info(
        "Value tolerances: abs=%s, rel=%s",
        value_abs_tolerance,
        value_rel_tolerance,
    )

    with duckdb.connect() as con:
        con.execute(f"PRAGMA threads={int(threads)}")

        csv_summary = con.execute(
            f"""
            SELECT
                CAST("{hs_col}" AS VARCHAR) AS hs,
                CAST("{year_col}" AS INTEGER) AS year,
                COUNT(*) AS n_rows_csv,
                COUNT(DISTINCT CAST("{cmd_col}" AS VARCHAR)) AS n_cmd_codes_csv,
                SUM(CAST("{value_col}" AS DOUBLE)) AS value_total_csv
            FROM read_csv_auto(
                '{input_glob}',
                union_by_name = true,
                filename = true
            )
            WHERE {where_sql}
            GROUP BY hs, year
            """
        ).df()

        parquet_summary = con.execute(
            f"""
            SELECT
                CAST("{hs_col}" AS VARCHAR) AS hs,
                CAST("{year_col}" AS INTEGER) AS year,
                COUNT(*) AS n_rows_parquet,
                COUNT(DISTINCT CAST("{cmd_col}" AS VARCHAR)) AS n_cmd_codes_parquet,
                SUM(CAST("{value_col}" AS DOUBLE)) AS value_total_parquet
            FROM read_parquet(
                '{parquet_glob}',
                union_by_name = true
            )
            WHERE "{hs_col}" IS NOT NULL
              AND "{year_col}" IS NOT NULL
            GROUP BY hs, year
            """
        ).df()

    audit = csv_summary.merge(
        parquet_summary,
        on=["hs", "year"],
        how="outer",
    )

    count_cols = [
        "n_rows_csv",
        "n_rows_parquet",
        "n_cmd_codes_csv",
        "n_cmd_codes_parquet",
    ]

    for col in count_cols:
        audit[col] = audit[col].fillna(0).astype("int64")

    audit["value_total_csv"] = audit["value_total_csv"].fillna(0.0)
    audit["value_total_parquet"] = audit["value_total_parquet"].fillna(0.0)

    audit["row_diff"] = audit["n_rows_parquet"] - audit["n_rows_csv"]
    audit["cmd_code_diff"] = audit["n_cmd_codes_parquet"] - audit["n_cmd_codes_csv"]
    audit["value_diff"] = audit["value_total_parquet"] - audit["value_total_csv"]
    audit["abs_value_diff"] = audit["value_diff"].abs()

    denom = audit["value_total_csv"].abs()

    audit["relative_value_diff"] = 0.0
    nonzero = denom > 0

    audit.loc[nonzero, "relative_value_diff"] = (
        audit.loc[nonzero, "abs_value_diff"] / denom.loc[nonzero]
    )

    audit["value_close"] = (
        (audit["abs_value_diff"] <= value_abs_tolerance)
        | (audit["relative_value_diff"] <= value_rel_tolerance)
    )

    audit["status"] = "ok"

    audit.loc[audit["row_diff"] != 0, "status"] = "row_mismatch"

    audit.loc[
        (audit["status"] == "ok") & (audit["cmd_code_diff"] != 0),
        "status",
    ] = "cmd_code_mismatch"

    audit.loc[
        (audit["status"] == "ok") & ~audit["value_close"],
        "status",
    ] = "value_mismatch"

    audit = audit.sort_values(["status", "hs", "year"]).reset_index(drop=True)

    logger.info("Preservation audit shape: %s", audit.shape)
    logger.info("Status counts: %s", audit["status"].value_counts().to_dict())

    return audit


def export_harvard_generator_compatibility_layout(
    *,
    staged_partitioned_dir: Path,
    output_root: Path,
    years_by_hs: dict[str, list[int]] | None = None,
    year_col: str = "RefYear",
    hs_col: str = "ClassificationCode",
    cmd_col: str = "CmdCode",
    reporter_code_col: str = "reporterCode",
    partner_code_col: str = "partnerCode",
    flow_col: str = "flowCode",
    value_col: str = "PrimaryValue",
    valid_cmd_regex: str | None = None,
    qty_col: str | None = "NetWgt",
    country_reference_glob: str | None = None,
    country_reporter_code_col: str = "ReporterCode",
    country_reporter_iso_col: str = "ReporterISO",
    country_partner_code_col: str = "PartnerCode",
    country_partner_iso_col: str = "PartnerISO",
    published_date: str = "2026-01-01",
    threads: int = 16,
    overwrite: bool = False,
) -> dict[str, int]:
    """
    Export fast partitioned vacoda staging into the layout expected by the
    Harvard/Growth Lab conversion-weight generator.

    Input:
        output_root/as_reported/raw_parquet_partitioned/
        ├── hs_partition=H4/year_partition=2016/*.parquet
        └── hs_partition=H5/year_partition=2017/*.parquet

    Output:
        output_root/as_reported/raw_parquet/H4/2016/
        └── COMTRADE-FINALCLASSIC-CA{reporter}{year}H4[{date}].parquet

        output_root/as_reported/aggregated_by_year_not_converted/parquet/H4/
        └── H4_2016.parquet

    Notes
    -----
    Harvard's MatrixBuilder uses raw_parquet mostly to identify reporter files
    and timely classification switchers. The trade values are read from
    aggregated_by_year_not_converted/parquet/{H}/{H}_{year}.parquet.

    The aggregated files must contain real ISO3 codes in reporterISO3 and
    partnerISO3. Bulk Comtrade files usually only contain numeric reporterCode
    and partnerCode, so this function builds an ISO map from single-API files
    using country_reference_glob.
    """

    if country_reference_glob is None:
        raise ValueError(
            "country_reference_glob is required. Harvard aggregated files need "
            "real ISO3 values in reporterISO3/partnerISO3, but bulk data only "
            "contains numeric reporter/partner codes."
        )

    as_reported_dir = output_root / "as_reported"

    raw_parquet_dir = as_reported_dir / "raw_parquet"
    aggregated_dir = (
        as_reported_dir
        / "aggregated_by_year_not_converted"
        / "parquet"
    )

    raw_parquet_dir.mkdir(parents=True, exist_ok=True)
    aggregated_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting Harvard generator compatibility layout.")
    logger.info("Input partitioned dir: %s", rel_path(staged_partitioned_dir))
    logger.info("Raw parquet output: %s", rel_path(raw_parquet_dir))
    logger.info("Aggregated output: %s", rel_path(aggregated_dir))
    logger.info("Country reference glob: %s", country_reference_glob)

    if years_by_hs is None:
        years_by_hs = {}

        for hs_dir in staged_partitioned_dir.glob("hs_partition=*"):
            if not hs_dir.is_dir():
                continue

            hs = hs_dir.name.split("=", 1)[1]
            years: list[int] = []

            for year_dir in hs_dir.glob("year_partition=*"):
                if not year_dir.is_dir():
                    continue
                year_text = year_dir.name.split("=", 1)[1]
                years.append(int(year_text))

            years_by_hs[hs] = sorted(years)

    written_raw_files = 0
    written_aggregated_files = 0
    skipped_raw_files = 0
    skipped_aggregated_files = 0

    with duckdb.connect() as con:
        con.execute(f"PRAGMA threads={int(threads)}")

        # ------------------------------------------------------------------
        # Build numeric Comtrade code -> ISO3 map from single-API files.
        # ------------------------------------------------------------------
        con.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW country_iso_map_raw AS
            SELECT DISTINCT
                CAST("{country_reporter_code_col}" AS INTEGER) AS code,
                UPPER(TRIM(CAST("{country_reporter_iso_col}" AS VARCHAR))) AS iso3
            FROM read_csv_auto(
                '{country_reference_glob}',
                union_by_name = true,
                filename = true
            )
            WHERE "{country_reporter_code_col}" IS NOT NULL
              AND "{country_reporter_iso_col}" IS NOT NULL
              AND TRIM(CAST("{country_reporter_iso_col}" AS VARCHAR)) <> ''

            UNION

            SELECT DISTINCT
                CAST("{country_partner_code_col}" AS INTEGER) AS code,
                UPPER(TRIM(CAST("{country_partner_iso_col}" AS VARCHAR))) AS iso3
            FROM read_csv_auto(
                '{country_reference_glob}',
                union_by_name = true,
                filename = true
            )
            WHERE "{country_partner_code_col}" IS NOT NULL
              AND "{country_partner_iso_col}" IS NOT NULL
              AND TRIM(CAST("{country_partner_iso_col}" AS VARCHAR)) <> ''
            """
        )

        # Deduplicate defensively. If a numeric code maps to multiple ISO3
        # strings in messy source files, choose the lexicographically first one
        # and expose the ambiguity through logs.
        ambiguous_mappings = con.execute(
            """
            SELECT
                code,
                COUNT(DISTINCT iso3) AS n_iso3
            FROM country_iso_map_raw
            GROUP BY code
            HAVING COUNT(DISTINCT iso3) > 1
            ORDER BY code
            """
        ).df()

        if not ambiguous_mappings.empty:
            logger.warning(
                "Country ISO map has ambiguous code→ISO3 mappings. "
                "Using MIN(iso3). First examples: %s",
                ambiguous_mappings.head(20).to_dict(orient="records"),
            )

        con.execute(
            """
            CREATE OR REPLACE TEMP VIEW country_iso_map AS
            SELECT
                code,
                MIN(iso3) AS iso3
            FROM country_iso_map_raw
            WHERE iso3 IS NOT NULL
              AND TRIM(iso3) <> ''
            GROUP BY code
            """
        )

        country_map_summary = con.execute(
            """
            SELECT
                COUNT(*) AS n_codes,
                COUNT(DISTINCT iso3) AS n_iso3
            FROM country_iso_map
            """
        ).df()

        logger.info(
            "Country ISO map summary: %s",
            country_map_summary.to_dict(orient="records")[0],
        )

        for hs, years in sorted(years_by_hs.items()):
            hs = str(hs).strip().upper()

            for year in sorted(years):
                input_glob = str(
                    staged_partitioned_dir
                    / f"hs_partition={hs}"
                    / f"year_partition={int(year)}"
                    / "*.parquet"
                )

                # ---------------------------------------------------------
                # 1. Aggregated file used by Harvard MatrixBuilder
                # ---------------------------------------------------------

                agg_out_dir = aggregated_dir / hs
                agg_out_dir.mkdir(parents=True, exist_ok=True)

                agg_out_file = agg_out_dir / f"{hs}_{int(year)}.parquet"

                if agg_out_file.exists() and not overwrite:
                    skipped_aggregated_files += 1
                    logger.info(
                        "Skipping existing aggregated file: %s",
                        rel_path(agg_out_file),
                    )
                else:
                    qty_expr = (
                        f'SUM(TRY_CAST(t."{qty_col}" AS DOUBLE)) AS qty'
                        if qty_col
                        else "CAST(NULL AS DOUBLE) AS qty"
                    )

                    logger.info(
                        "Writing aggregated Harvard file: %s",
                        rel_path(agg_out_file),
                    )

                    # Audit unmapped reporter/partner codes before writing.
                    unmapped = con.execute(
                        f"""
                        WITH trade_codes AS (
                            SELECT DISTINCT
                                TRY_CAST(t."{reporter_code_col}" AS INTEGER) AS reporter_code,
                                TRY_CAST(t."{partner_code_col}" AS INTEGER) AS partner_code
                            FROM read_parquet(
                                '{input_glob}',
                                union_by_name = true
                            ) AS t
                            WHERE t."{reporter_code_col}" IS NOT NULL
                               OR t."{partner_code_col}" IS NOT NULL
                        )
                        SELECT
                            COUNT(DISTINCT reporter_code) FILTER (
                                WHERE reporter_code IS NOT NULL
                                  AND reporter_ref.iso3 IS NULL
                            ) AS unmapped_reporter_codes,
                            COUNT(DISTINCT partner_code) FILTER (
                                WHERE partner_code IS NOT NULL
                                  AND partner_ref.iso3 IS NULL
                            ) AS unmapped_partner_codes
                        FROM trade_codes
                        LEFT JOIN country_iso_map AS reporter_ref
                            ON trade_codes.reporter_code = reporter_ref.code
                        LEFT JOIN country_iso_map AS partner_ref
                            ON trade_codes.partner_code = partner_ref.code
                        """
                    ).df()

                    logger.info(
                        "Unmapped country codes for %s/%s: %s",
                        hs,
                        year,
                        unmapped.to_dict(orient="records")[0],
                    )

                    con.execute(
                        f"""
                        COPY (
                            SELECT
                                reporter_ref.iso3 AS reporterISO3,
                                partner_ref.iso3 AS partnerISO3,
                                CAST(t."{cmd_col}" AS VARCHAR) AS cmdCode,
                                CAST(t."{flow_col}" AS VARCHAR) AS flowCode,
                                LENGTH(CAST(t."{cmd_col}" AS VARCHAR)) AS digitLevel,
                                {qty_expr},
                                CAST(NULL AS DOUBLE) AS CIFValue,
                                CAST(NULL AS DOUBLE) AS FOBValue,
                                SUM(TRY_CAST(t."{value_col}" AS DOUBLE)) AS primaryValue
                            FROM read_parquet(
                                '{input_glob}',
                                union_by_name = true
                            ) AS t
                            LEFT JOIN country_iso_map AS reporter_ref
                                ON TRY_CAST(t."{reporter_code_col}" AS INTEGER) = reporter_ref.code
                            LEFT JOIN country_iso_map AS partner_ref
                                ON TRY_CAST(t."{partner_code_col}" AS INTEGER) = partner_ref.code
                            WHERE t."{hs_col}" IS NOT NULL
                              AND t."{year_col}" IS NOT NULL
                              AND reporter_ref.iso3 IS NOT NULL
                              AND partner_ref.iso3 IS NOT NULL
                            GROUP BY
                                reporterISO3,
                                partnerISO3,
                                cmdCode,
                                flowCode,
                                digitLevel
                        )
                        TO '{agg_out_file}'
                        (FORMAT PARQUET)
                        """
                    )

                    written_aggregated_files += 1

                # ---------------------------------------------------------
                # 2. Raw parquet marker/files used for reporter switching
                # ---------------------------------------------------------

                reporters = con.execute(
                    f"""
                    SELECT DISTINCT
                        LPAD(CAST(t."{reporter_code_col}" AS VARCHAR), 3, '0') AS reporter_code
                    FROM read_parquet(
                        '{input_glob}',
                        union_by_name = true
                    ) AS t
                    INNER JOIN country_iso_map AS reporter_ref
                        ON TRY_CAST(t."{reporter_code_col}" AS INTEGER) = reporter_ref.code
                    WHERE t."{reporter_code_col}" IS NOT NULL
                      AND reporter_ref.iso3 IS NOT NULL
                    ORDER BY reporter_code
                    """
                ).fetchall()

                raw_out_dir = raw_parquet_dir / hs / str(int(year))
                raw_out_dir.mkdir(parents=True, exist_ok=True)

                for (reporter_code,) in reporters:
                    out_file = (
                        raw_out_dir
                        / f"COMTRADE-FINALCLASSIC-CA{reporter_code}{int(year)}{hs}[{published_date}].parquet"
                    )

                    if out_file.exists() and not overwrite:
                        skipped_raw_files += 1
                        continue

                    # Keep these as small marker files. Harvard parses filenames
                    # for reporter/year/classification; trade values come from
                    # aggregated_by_year_not_converted.
                    con.execute(
                        f"""
                        COPY (
                            SELECT
                                '{reporter_code}' AS ReporterCode,
                                {int(year)} AS RefYear,
                                '{hs}' AS ClassificationCode
                            LIMIT 1
                        )
                        TO '{out_file}'
                        (FORMAT PARQUET)
                        """
                    )

                    written_raw_files += 1

    summary = {
        "written_raw_files": written_raw_files,
        "skipped_raw_files": skipped_raw_files,
        "written_aggregated_files": written_aggregated_files,
        "skipped_aggregated_files": skipped_aggregated_files,
    }

    logger.info("Harvard compatibility export summary: %s", summary)

    return summary


def validate_staged_harvard_partitions(
    *,
    partitioned_dir: Path,
    years_by_hs: dict[str, list[int]],
) -> None:
    """Ensure configured HS/year partitions were actually written."""
    missing: list[tuple[str, int, Path]] = []

    for hs, years in years_by_hs.items():
        for year in years:
            partition_dir = (
                partitioned_dir
                / f"hs_partition={hs}"
                / f"year_partition={year}"
            )

            if not list(partition_dir.glob("*.parquet")):
                missing.append((hs, year, partition_dir))

    if missing:
        expected = ", ".join(f"{hs}/{year}" for hs, year, _ in missing)
        missing_dirs = "\n".join(f"  - {path}" for _, _, path in missing)

        raise FileNotFoundError(
            "Missing staged Harvard parquet partitions for configured "
            f"HS/year pairs: {expected}.\n"
            "Missing directories:\n"
            f"{missing_dirs}\n"
            "Likely cause: harvard_staging.input.pattern does not include "
            "all years requested in harvard_staging.filters.hs_years."
        )

def harvard_staging_outputs_exist(
    *,
    partitioned_root: Path,
    raw_parquet_root: Path,
    aggregated_root: Path,
    years_by_hs: dict[str, list[int]],
) -> tuple[bool, list[str]]:
    missing: list[str] = []

    for hs, years in years_by_hs.items():
        for year in years:
            partition_dir = (
                partitioned_root
                / f"hs_partition={hs}"
                / f"year_partition={year}"
            )
            raw_dir = raw_parquet_root / hs / str(year)
            aggregated_file = aggregated_root / hs / f"{hs}_{year}.parquet"

            if not any(partition_dir.glob("*.parquet")):
                missing.append(f"partitioned parquet: {partition_dir}")

            if not any(raw_dir.glob("*.parquet")):
                missing.append(f"raw_parquet layout: {raw_dir}")

            if not aggregated_file.exists():
                missing.append(f"aggregated parquet: {aggregated_file}")

    return len(missing) == 0, missing

def check_harvard_staging_complete(
    *,
    output_root: Path,
    artifact_dir: Path,
    preservation_latest_filename: str,
    years_by_hs: dict[str, list[int]],
    layout: "HarvardStagingLayoutConfig",
    threads: int = 16) -> tuple[bool, list[str]]:
    """
    Check whether Harvard staging is complete enough to skip rebuilding.

    This checks:
    - expected partitioned parquet directories
    - expected raw_parquet layout directories
    - expected aggregated parquet files
    - aggregated parquet readability/non-empty rows
    - latest preservation manifest status
    """

    missing_or_bad: list[str] = []

    partitioned_root = layout.resolve_partitioned_dir(output_root)
    raw_parquet_root = layout.resolve_raw_parquet_dir(output_root)
    aggregated_root = layout.resolve_aggregated_dir(output_root)

    with duckdb.connect() as con:
        con.execute(f"PRAGMA threads={int(threads)}")

        for hs, years in years_by_hs.items():
            for year in years:
                partition_dir = (
                    partitioned_root
                    / f"hs_partition={hs}"
                    / f"year_partition={year}"
                )
                raw_dir = raw_parquet_root / hs / str(year)
                aggregated_file = aggregated_root / hs / f"{hs}_{year}.parquet"

                if not any(partition_dir.glob("*.parquet")):
                    missing_or_bad.append(
                        f"missing partitioned parquet files: {partition_dir}"
                    )

                if not any(raw_dir.glob("*.parquet")):
                    missing_or_bad.append(
                        f"missing raw_parquet layout files: {raw_dir}"
                    )

                if not aggregated_file.exists():
                    missing_or_bad.append(
                        f"missing aggregated parquet file: {aggregated_file}"
                    )
                    continue

                try:
                    n_rows = con.execute(
                        "SELECT COUNT(*) FROM read_parquet(?)",
                        [str(aggregated_file)],
                    ).fetchone()[0]
                except Exception as exc:
                    missing_or_bad.append(
                        f"cannot read aggregated parquet {aggregated_file}: {exc}"
                    )
                    continue

                if n_rows <= 0:
                    missing_or_bad.append(
                        f"aggregated parquet has zero rows: {aggregated_file}"
                    )

    preservation_latest = artifact_dir / preservation_latest_filename

    if not preservation_latest.exists():
        missing_or_bad.append(
            f"missing preservation manifest: {preservation_latest}"
        )
    else:
        try:
            preservation = pd.read_csv(preservation_latest)
        except Exception as exc:
            missing_or_bad.append(
                f"cannot read preservation manifest {preservation_latest}: {exc}"
            )
        else:
            if "status" not in preservation.columns:
                missing_or_bad.append(
                    f"preservation manifest has no status column: {preservation_latest}"
                )
            else:
                bad = preservation[preservation["status"] != "ok"]
                if not bad.empty:
                    status_counts = bad["status"].value_counts().to_dict()
                    missing_or_bad.append(
                        f"preservation manifest has non-ok statuses: {status_counts}"
                    )

    return len(missing_or_bad) == 0, missing_or_bad


def run_harvard_staging(
    *,
    cfg: "HsSyncConfig",
    run_id: str,
) -> dict[str, Any]:
    """
    Run the full Harvard staging workflow from config to manifest artifacts.
    """

    input_dir = cfg.harvard_staging.input.resolve_data_dir()
    pattern = (cfg.harvard_staging.input.pattern or "").strip()
    if not pattern:
        raise ValueError(
            "Missing required 'harvard_staging.input.pattern' in hs_sync.yaml."
        )

    valid_cmd_regex = cfg.harvard_staging.filters.valid_cmd_regex
    years_by_hs = cfg.harvard_staging.filters.hs_years

    year_col = cfg.harvard_staging.columns.year
    hs_col = cfg.harvard_staging.columns.hs
    cmd_col = cfg.harvard_staging.columns.cmd
    value_col = cfg.harvard_staging.columns.value
    reporter_code_col = cfg.harvard_staging.columns.reporter_code
    partner_code_col = cfg.harvard_staging.columns.partner_code
    flow_col = cfg.harvard_staging.columns.flow
    quantity_col = cfg.harvard_staging.columns.quantity

    paths = get_harvard_staging_paths(cfg)

    output_root = paths.output_root
    output_dir = paths.partitioned_root
    raw_parquet_root = paths.raw_parquet_root
    aggregated_root = paths.aggregated_root
    artifact_dir = paths.artifact_dir

    if cfg.harvard_staging.skip.if_complete and not cfg.harvard_staging.overwrite:
        marker_valid, marker_reasons = harvard_staging_success_marker_valid(cfg=cfg)

        if marker_valid:
            logger.info("Harvard staging success marker is valid; skipping rebuild.")
            logger.info("Success marker: %s", rel_path(paths.marker_path))
            return {
                "status": "skipped_success_marker",
                "output_dir": paths.partitioned_root,
                "artifact_dir": paths.artifact_dir,
                "marker_path": paths.marker_path,
                "skip_reason": "valid_success_marker",
            }

        logger.info("Harvard staging success marker is missing or invalid; rebuild required.")
        for reason in marker_reasons:
            logger.info("  %s", reason)


    ## Check function parameters before starting the pipeline. This avoids doing expensive work

    is_complete, missing_outputs = harvard_staging_outputs_exist(
        partitioned_root=output_dir,
        raw_parquet_root=raw_parquet_root,
        aggregated_root=aggregated_root,
        years_by_hs=years_by_hs,
    )

    if is_complete and not cfg.harvard_staging.overwrite:
        logger.info("==================================")
        logger.info("Harvard staging outputs already exist; will perform detailed checks.")
        logger.info("Output root: %s", rel_path(output_root))


    if missing_outputs:
        logger.info("==================================")
        logger.info(
            "Harvard staging outputs are incomplete; missing %s expected outputs.",
            len(missing_outputs),)
        for item in missing_outputs[:20]:
            logger.info("  Missing: %s", item)

        if len(missing_outputs) > 20:
            logger.info("  ... and %s more missing outputs", len(missing_outputs) - 20)


    country_reference_dir = cfg.harvard_staging.country_reference.input.resolve_data_dir()
    country_reference_pattern = (
        cfg.harvard_staging.country_reference.input.pattern or ""
    ).strip()
    if not country_reference_pattern:
        raise ValueError(
            "Missing required 'harvard_staging.country_reference.pattern' in hs_sync.yaml."
        )

    country_reference_glob = str(country_reference_dir / country_reference_pattern)
    country_reference_cols = cfg.harvard_staging.country_reference.columns

    logger.info("Starting Harvard staging.")
    logger.info("Input directory: %s", rel_path(input_dir))
    logger.info("Input pattern: %s", pattern)
    logger.info("Output root: %s", rel_path(output_root))
    logger.info("Output directory: %s", rel_path(output_dir))
    logger.info("Valid commodity-code regex: %s", valid_cmd_regex)
    logger.info("HS/year filter: %s", years_by_hs)
    logger.info(
        "Harvard input columns: year=%s, hs=%s, cmd=%s, value=%s",
        year_col,
        hs_col,
        cmd_col,
        value_col,
    )
    logger.info(
        "Harvard compatibility columns: reporter=%s, partner=%s, flow=%s, quantity=%s",
        reporter_code_col,
        partner_code_col,
        flow_col,
        quantity_col,
    )
    logger.info("Country reference directory: %s", rel_path(country_reference_dir))
    logger.info("Country reference glob: %s", country_reference_glob)

    layout = cfg.harvard_staging.layout

    if is_complete and not cfg.harvard_staging.overwrite:
        detailed_ok, problems = check_harvard_staging_complete(
            output_root=output_root,
            artifact_dir=artifact_dir,
            preservation_latest_filename=preservation_latest_filename,
            years_by_hs=years_by_hs,
            layout=layout,
            threads=cfg.engine.threads,
        )

        if detailed_ok:
            marker_path = write_harvard_staging_success_marker(
                cfg=cfg,
                run_id=run_id,
                summary={"status": "reused_existing_outputs"},
                compatibility_summary={"status": "not_rerun"},
            )

            logger.info("Existing Harvard staging outputs passed detailed checks; skipping rebuild.")
            logger.info("Success marker written: %s", rel_path(marker_path))

            return {
                "status": "skipped_existing_checked",
                "output_dir": output_dir,
                "artifact_dir": artifact_dir,
                "marker_path": marker_path,
                "skip_reason": "existing_outputs_passed_detailed_check",
            }

        logger.info("Existing Harvard staging outputs failed detailed checks; rebuild required.")
        for problem in problems[:20]:
            logger.info("  %s", problem)


    staged_partitioned_dir = stage_comtrade_for_harvard_input(
        input_dir=input_dir,
        pattern=pattern,
        output_dir=output_dir,
        year_col=year_col,
        hs_col=hs_col,
        cmd_col=cmd_col,
        valid_cmd_regex=valid_cmd_regex,
        years_by_hs=years_by_hs,
        threads=cfg.engine.threads,
        overwrite=cfg.harvard_staging.overwrite,
    )

    summary: dict[str, Any] = {
        "staged_partitioned_dir": str(staged_partitioned_dir),
    }

    inventory = audit_harvard_staging_partitioned_output(
        output_dir=output_dir,
    )

    preservation = audit_csv_to_partitioned_parquet_staging(
        input_dir=input_dir,
        input_pattern=pattern,
        output_dir=output_dir,
        year_col=year_col,
        hs_col=hs_col,
        cmd_col=cmd_col,
        value_col=value_col,
        valid_cmd_regex=valid_cmd_regex,
        years_by_hs=years_by_hs,
        threads=cfg.engine.threads,
        value_abs_tolerance=cfg.harvard_staging.diagnostics.value_abs_tolerance,
        value_rel_tolerance=cfg.harvard_staging.diagnostics.value_rel_tolerance,
    )

    validate_staged_harvard_partitions(
        partitioned_dir=output_dir,
        years_by_hs=years_by_hs,
    )

    compatibility_summary = export_harvard_generator_compatibility_layout(
        staged_partitioned_dir=output_dir,
        output_root=output_root,
        years_by_hs=years_by_hs,
        year_col=year_col,
        hs_col=hs_col,
        cmd_col=cmd_col,
        value_col=value_col,
        reporter_code_col=reporter_code_col,
        partner_code_col=partner_code_col,
        flow_col=flow_col,
        qty_col=quantity_col,
        valid_cmd_regex=valid_cmd_regex,
        country_reference_glob=country_reference_glob,
        country_reporter_code_col=country_reference_cols.reporter_code,
        country_reporter_iso_col=country_reference_cols.reporter_iso,
        country_partner_code_col=country_reference_cols.partner_code,
        country_partner_iso_col=country_reference_cols.partner_iso,
        threads=cfg.engine.threads,
        overwrite=cfg.harvard_staging.overwrite,
    )

    artifact_dir = cfg.harvard_staging.artifact.resolve_output_dir()
    
    preservation_base = Path(cfg.harvard_staging.artifact.preservation_filename)
    preservation_latest_filename = (
        f"{preservation_base.stem}_latest{preservation_base.suffix}"    
    )

    inventory_outputs = write_dataframe_report(
        inventory,
        report_dir=artifact_dir,
        filename=cfg.harvard_staging.artifact.inventory_filename,
        timestamp=cfg.harvard_staging.artifact.timestamp,
        write_latest=cfg.harvard_staging.artifact.write_latest,
        run_id=run_id,
    )

    preservation_outputs = write_dataframe_report(
        preservation,
        report_dir=artifact_dir,
        filename=cfg.harvard_staging.artifact.preservation_filename,
        timestamp=cfg.harvard_staging.artifact.timestamp,
        write_latest=cfg.harvard_staging.artifact.write_latest,
        run_id=run_id,
    )

    logger.info("Inventory written: %s", rel_path(inventory_outputs["report"]))
    logger.info("Preservation written: %s", rel_path(preservation_outputs["report"]))

    if "latest" in inventory_outputs:
        logger.info("Latest inventory copy written: %s", rel_path(inventory_outputs["latest"]))

    if "latest" in preservation_outputs:
        logger.info(
            "Latest preservation copy written: %s",
            rel_path(preservation_outputs["latest"]),
        )

    is_complete, problems = check_harvard_staging_complete(
        output_root=output_root,
        artifact_dir=artifact_dir,
        preservation_latest_filename=preservation_latest_filename,
        years_by_hs=years_by_hs,
        layout=cfg.harvard_staging.layout,
        threads=cfg.engine.threads,
    )

    if not is_complete:
        problem_preview = "; ".join(problems[:10])
        raise RuntimeError(
            "Harvard staging completed but final completeness checks failed. "
            f"Problems: {problem_preview}"
        )

    marker_path = write_harvard_staging_success_marker(
            cfg=cfg,
            run_id=run_id,
            summary=summary,
            compatibility_summary=compatibility_summary,
        )
    logger.info("Success marker written: %s", rel_path(marker_path))

    logger.info("Compatibility export summary: %s", compatibility_summary)
    logger.info("Finished Harvard staging.")
    logger.info("Summary: %s", summary)

    return {
        "summary": summary,
        "output_dir": output_dir,
        "artifact_dir": artifact_dir,
        "compatibility_summary": compatibility_summary,
        "inventory_manifest": inventory_outputs.get("report"),
        "preservation_manifest": preservation_outputs.get("report"),
        "inventory_latest": inventory_outputs.get("latest"),
        "preservation_latest": preservation_outputs.get("latest"),
        "marker_path": marker_path,
    }
