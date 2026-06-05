
"""Mapping direction diagnostics for Harvard correlation-group files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from vacoda.config.paths import get_data_dir, get_repo_references_dir
from vacoda.config.hs_sync import HsSyncConfig
from vacoda.hs_sync.diagnostics import build_trade_source_pair_summary
from vacoda.io.reports import TextReport, write_text_report

HS_TO_OFFICIAL = {
    "H0": "HS1988",
    "H1": "HS1996",
    "H2": "HS2002",
    "H3": "HS2007",
    "H4": "HS2012",
    "H5": "HS2017",
    "H6": "HS2022",
}

def _normalize_code(s: pd.Series) -> pd.Series:
    out = s.astype("string").str.strip()
    out = out.str.replace(r"\.0$", "", regex=True)
    out = out.mask(out.isin(["", "nan", "None", "<NA>"]))
    is_numeric = out.str.match(r"^[0-9]+$", na=False)
    out = out.where(~is_numeric, out.str.zfill(6))
    return out


def _parse_mapping_key(key: str) -> tuple[str, str]:
    parts = key.upper().split("_TO_")
    if len(parts) != 2:
        raise ValueError(f"Mapping key must look like H4_to_H5 or H6_to_H5. Got: {key}")
    return parts[0], parts[1]


def _actual_cardinality(df: pd.DataFrame) -> pd.DataFrame:
    targets_per_source = (
        df.groupby("source_cmd")["target_cmd"]
        .nunique()
        .rename("n_targets_per_source")
        .reset_index()
    )

    sources_per_target = (
        df.groupby("target_cmd")["source_cmd"]
        .nunique()
        .rename("n_sources_per_target")
        .reset_index()
    )

    out = df.merge(targets_per_source, on="source_cmd", how="left").merge(
        sources_per_target, on="target_cmd", how="left"
    )

    out["actual_mapping_category"] = "actual_n_to_n"

    out.loc[
        (out["n_targets_per_source"] == 1) & (out["n_sources_per_target"] == 1),
        "actual_mapping_category",
    ] = "actual_1_to_1"

    out.loc[
        (out["n_targets_per_source"] == 1) & (out["n_sources_per_target"] > 1),
        "actual_mapping_category",
    ] = "actual_n_to_1"

    out.loc[
        (out["n_targets_per_source"] > 1) & (out["n_sources_per_target"] == 1),
        "actual_mapping_category",
    ] = "actual_1_to_n"

    return out


def _support_rates(
    *,
    mapping: pd.DataFrame,
    observed_source_codes: set[str],
    observed_target_codes: set[str],
) -> dict[str, float]:
    source_codes = set(mapping["source_cmd"].dropna())
    target_codes = set(mapping["target_cmd"].dropna())

    source_hit = len(source_codes & observed_source_codes)
    target_hit = len(target_codes & observed_target_codes)

    source_rate = source_hit / len(source_codes) if source_codes else 0.0
    target_rate = target_hit / len(target_codes) if target_codes else 0.0

    return {
        "n_mapping_source_codes": len(source_codes),
        "n_mapping_target_codes": len(target_codes),
        "source_code_hit_rate": source_rate,
        "target_code_hit_rate": target_rate,
        "mean_hit_rate": (source_rate + target_rate) / 2,
    }


def _resolve_mapping_files(
    *,
    cfg: HsSyncConfig,
) -> dict[str, Path]:
    mappings_cfg = cfg.coverage.mappings

    if mappings_cfg.stage.strip().lower() != "references":
        return mappings_cfg.resolve_files()

    if cfg.harvard_mapping_direction.mappings.references_root == "repo":
        root = get_repo_references_dir()
    else:
        root = get_data_dir("references")

    mapping_dir = root / mappings_cfg.subdir if mappings_cfg.subdir else root
    files = {
        key: mapping_dir / filename
        for key, filename in mappings_cfg.files.items()
    }

    missing = [path for path in files.values() if not path.exists()]
    if missing:
        missing_preview = ", ".join(str(path) for path in missing[:3])
        raise FileNotFoundError(
            "Mapping-direction audit could not find mapping files under the "
            f"configured references root '{cfg.harvard_mapping_direction.mappings.references_root}'. "
            f"Examples: {missing_preview}"
        )

    return files


def run_harvard_mapping_direction_audit(
    *,
    cfg: HsSyncConfig,
    report_name: str = "inspect_hscodes_requiring_weights.md",
    report_dir: Path | None = None,
    timestamp: bool = True,
    write_latest: bool = True,
    logger=None,
) -> pd.DataFrame:
    """Audit whether mapping files are interpreted in the correct direction."""
    report = TextReport()
  
    input_dir = cfg.coverage_input_dir()
    mapping_files = _resolve_mapping_files(cfg=cfg)

    report.section("Building observed trade source summary")

    source_summary = build_trade_source_pair_summary(
        data_dir=input_dir,
        pattern=cfg.coverage.input.pattern,
        engine=cfg.engine.name,
        year_col=cfg.columns.year,
        hs_col=cfg.columns.hs,
        cmd_col=cfg.columns.cmd,
        value_col=cfg.columns.value,
        min_year=cfg.coverage.min_year,
        threads=cfg.engine.threads,
        show_progress=cfg.progress.enabled,
    )

    source_summary = source_summary.copy()
    source_summary["source_hs"] = (
        source_summary["source_hs"].astype("string").str.strip().str.upper()
    )
    source_summary["source_cmd"] = _normalize_code(source_summary["source_cmd"])

    observed_by_hs = {
        hs: set(group["source_cmd"].dropna())
        for hs, group in source_summary.groupby("source_hs")
    }

    audit_rows: list[dict[str, object]] = []

    for mapping_key, path in mapping_files.items():
        configured_source_hs, configured_target_hs = _parse_mapping_key(mapping_key)

        raw = pd.read_csv(path, dtype=str)

        report.section(f"{mapping_key}: {path.name}")

        columns_text = f"Columns: {list(raw.columns)}"
        adjustments_text = (
            "Unique adjustment values: "
            f"{raw.get('adjustment', pd.Series(dtype=str)).dropna().unique()[:10]}"
        )
        if logger is not None:
            logger.info(columns_text)
            logger.info(adjustments_text)
        report.line(columns_text)
        report.line(adjustments_text)

        cols = cfg.coverage.mappings.columns

        if cols.source_cmd not in raw.columns or cols.target_cmd not in raw.columns:
            raise KeyError(
                f"{path} must contain {cols.source_cmd!r} and {cols.target_cmd!r}. "
                f"Available columns: {list(raw.columns)}"
            )

        mapping = raw.rename(
            columns={
                cols.source_cmd: "source_cmd",
                cols.target_cmd: "target_cmd",
                cols.relationship: "relationship_original",
                cols.adjustment: "adjustment_original",
                cols.group_id: "group_id",
            }
        ).copy()

        mapping["source_cmd"] = _normalize_code(mapping["source_cmd"])
        mapping["target_cmd"] = _normalize_code(mapping["target_cmd"])

        mapping = _actual_cardinality(mapping)

        card_summary = (
            mapping.groupby(
                ["relationship_original", "actual_mapping_category"],
                dropna=False,
            )
            .agg(
                n_rows=("source_cmd", "size"),
                n_source_codes=("source_cmd", "nunique"),
                n_target_codes=("target_cmd", "nunique"),
            )
            .reset_index()
            .sort_values(["relationship_original", "actual_mapping_category"])
        )

        card_header = "Relationship label vs actual source→target cardinality:"
        card_table = card_summary.to_string(index=False)
        if logger is not None:
            logger.info(card_header)
            logger.info("\n%s", card_table)
        report.line(card_header)
        report.dataframe(card_summary)

        observed_source_codes = observed_by_hs.get(configured_source_hs, set())
        observed_target_codes = observed_by_hs.get(configured_target_hs, set())

        configured_support = _support_rates(
            mapping=mapping,
            observed_source_codes=observed_source_codes,
            observed_target_codes=observed_target_codes,
        )

        reversed_mapping = mapping.rename(
            columns={
                "source_cmd": "target_cmd",
                "target_cmd": "source_cmd",
            }
        ).copy()

        reversed_support = _support_rates(
            mapping=reversed_mapping,
            observed_source_codes=observed_source_codes,
            observed_target_codes=observed_target_codes,
        )

        configured_lines = [
            "Configured direction support:",
            (
                f"  {configured_source_hs} ({HS_TO_OFFICIAL.get(configured_source_hs)}) "
                f"→ {configured_target_hs} ({HS_TO_OFFICIAL.get(configured_target_hs)})"
            ),
            f"  source_code_hit_rate: {configured_support['source_code_hit_rate']:.3f}",
            f"  target_code_hit_rate: {configured_support['target_code_hit_rate']:.3f}",
            f"  mean_hit_rate:        {configured_support['mean_hit_rate']:.3f}",
        ]

        reversed_lines = [
            "Reversed interpretation support:",
            f"  source_code_hit_rate: {reversed_support['source_code_hit_rate']:.3f}",
            f"  target_code_hit_rate: {reversed_support['target_code_hit_rate']:.3f}",
            f"  mean_hit_rate:        {reversed_support['mean_hit_rate']:.3f}",
        ]

        if logger is not None:
            for line in configured_lines + reversed_lines:
                logger.info(line)
        for line in configured_lines + reversed_lines:
            report.line(line)

        if configured_support["mean_hit_rate"] >= reversed_support["mean_hit_rate"]:
            conclusion = "configured_direction_supported"
        else:
            conclusion = "reversed_direction_has_higher_support"

        conclusion_line = f"Conclusion: {conclusion}"
        if logger is not None:
            logger.info(conclusion_line)
        report.line(conclusion_line)

        audit_rows.append(
            {
                "mapping_key": mapping_key,
                "filename": path.name,
                "configured_source_hs": configured_source_hs,
                "configured_target_hs": configured_target_hs,
                "configured_mean_hit_rate": configured_support["mean_hit_rate"],
                "reversed_mean_hit_rate": reversed_support["mean_hit_rate"],
                "conclusion": conclusion,
            }
        )

    audit = pd.DataFrame(audit_rows)
    audit_table = audit.to_string(index=False)

    report.section("Overall direction audit")
    if logger is not None:
        logger.info("\n%s", audit_table)
    report.dataframe(audit)

    write_text_report(
        report.render(),
        report_dir=report_dir,
        filename=report_name,
        timestamp=timestamp,
        write_latest=write_latest,
    )

    return audit


def run_harvard_mapping_direction_audit_from_config(
    *,
    cfg: HsSyncConfig,
    logger=None,
) -> dict[str, object]:
    mapping_cfg = cfg.harvard_mapping_direction

    if not mapping_cfg.enabled:
        return {
            "status": "disabled",
        }

    report_dir = mapping_cfg.artifact.resolve_output_dir()
    report_name = mapping_cfg.artifact.filename or "inspect_hscodes_requiring_weights.md"
    if not Path(report_name).suffix:
        report_name = f"{report_name}.md"

    audit = run_harvard_mapping_direction_audit(
        cfg=cfg,
        report_name=report_name,
        report_dir=report_dir,
        timestamp=mapping_cfg.artifact.timestamp,
        write_latest=mapping_cfg.artifact.write_latest,
        logger=logger,
    )

    base_path = report_dir / report_name
    latest_path = None
    if mapping_cfg.artifact.timestamp:
        latest_path = base_path.with_name(
            f"{base_path.stem}_latest{base_path.suffix}"
        )

    return {
        "status": "completed",
        "report_dir": report_dir,
        "report_latest": latest_path,
        "audit": audit,
    }