"""
vacoda.quantity.reporting

Create the final Quantity decision report and diagnostic figures.

This report is intentionally diagnostic-first: it documents the current
synchronization result while also exposing candidate evidence that is not yet
admitted by synchronize.py (notably alternate quantity in kg-equivalent units).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil
from typing import Any

import duckdb
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from vacoda.config.paths import get_figure_dir
from vacoda.config.quantity import QuantityConfig
from vacoda.utils.display import get_symbols
from vacoda.utils.markdown import markdown_image
from vacoda.utils.sql import sql_literal


# ======================================================================
# Helpers
# ======================================================================


def valid_output(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _format_integer(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{int(round(float(value))):,}"


def _format_percent(
    value: float | None,
    *,
    digits: int = 2,
    already_percent: bool = False,
) -> str:
    if value is None or pd.isna(value):
        return "NA"
    numeric = float(value)
    if not already_percent:
        numeric *= 100.0
    return f"{numeric:.{digits}f}%"


def _format_number(
    value: float | None,
    *,
    digits: int = 4,
) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):,.{digits}f}"


def _markdown_table(dataframe: pd.DataFrame, *, index: bool = False) -> str:
    if dataframe.empty:
        return "_No rows available._"
    return dataframe.to_markdown(index=index, floatfmt=".4f")


def _diagnostic_group(
    diagnostics: pd.DataFrame,
    name: str,
) -> pd.DataFrame:
    return diagnostics.loc[
        diagnostics["diagnostic_group"].eq(name)
    ].copy()


def _as_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _simple_rows_table(
    frame: pd.DataFrame,
    *,
    metric_name: str,
    percentage_name: str = "Share (%)",
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    out = frame[["metric", "value", "percentage"]].copy()
    out.columns = [metric_name, "Rows", percentage_name]
    out["Rows"] = out["Rows"].round().astype("Int64")
    return out.sort_values("Rows", ascending=False)


def _source_state_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    out = frame[["metric", "category_1", "value", "percentage"]].copy()
    out.columns = ["Candidate source", "Evidence state", "Rows", "Share (%)"]
    out["Rows"] = out["Rows"].round().astype("Int64")
    return out.sort_values(["Candidate source", "Evidence state"])


def _agreement_state_table(
    frame: pd.DataFrame,
    agreement_tolerance: float,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    out = frame[
        ["metric", "category_1", "value", "percentage", "category_2", "category_3"]
    ].copy()
    out.columns = [
        "Comparison",
        "Evidence state",
        "Comparable rows",
        f"Within {100 * agreement_tolerance:.0f}%",
        "Median ratio",
        "Median relative gap",
    ]
    out["Comparable rows"] = out["Comparable rows"].round().astype("Int64")
    out = _as_numeric(out, ["Median ratio", "Median relative gap"])
    return out.sort_values(["Comparison", "Evidence state"])


def _top_hs_table(frame: pd.DataFrame, *, n: int = 20) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    out = frame[["metric", "value", "percentage"]].copy()
    out.columns = ["HS code", "Rows", "Share (%)"]
    out["Rows"] = out["Rows"].round().astype("Int64")
    return out.sort_values("Rows", ascending=False).head(n)


# ======================================================================
# Figures
# ======================================================================


def _save_figure(figure, output_path: Path) -> None:
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_unit_distribution(
    *,
    diagnostics: pd.DataFrame,
    diagnostic_group: str,
    title: str,
    output_path: Path,
) -> None:
    frame = _diagnostic_group(diagnostics, diagnostic_group)
    if frame.empty:
        return
    frame = (
        frame.groupby("metric", as_index=False)
        .agg(rows=("value", "sum"))
        .sort_values("rows", ascending=False)
    )
    total = frame["rows"].sum()
    if total <= 0:
        return
    frame["share"] = 100.0 * frame["rows"] / total
    figure, axis = plt.subplots(figsize=(10, 6))
    bars = axis.bar(frame["metric"].astype(str), frame["share"])
    axis.bar_label(
        bars,
        labels=[f"{value:.1f}%" for value in frame["share"]],
        padding=3,
        fontsize=8,
    )
    axis.set_title(title)
    axis.set_xlabel("Quantity unit")
    axis.set_ylabel("Share of observations (%)")
    axis.tick_params(axis="x", labelrotation=35)
    axis.grid(axis="y", linestyle="--", alpha=0.4)
    axis.set_axisbelow(True)
    _save_figure(figure, output_path)


def _plot_unit_share_by_year(
    *,
    diagnostics: pd.DataFrame,
    diagnostic_group: str,
    title: str,
    output_path: Path,
) -> None:
    frame = _diagnostic_group(diagnostics, diagnostic_group)
    frame = frame.dropna(subset=["year", "metric", "percentage"])
    if frame.empty:
        return
    pivot = frame.pivot_table(
        index="year",
        columns="metric",
        values="percentage",
        aggfunc="sum",
        fill_value=0.0,
    )
    figure, axis = plt.subplots(figsize=(11, 6))
    pivot.plot(kind="bar", stacked=True, ax=axis)
    axis.set_title(title)
    axis.set_xlabel("Year")
    axis.set_ylabel("Share of observations (%)")
    axis.set_ylim(0, 100)
    axis.grid(axis="y", linestyle="--", alpha=0.4)
    axis.legend(title="Unit", bbox_to_anchor=(1.02, 1), loc="upper left")
    _save_figure(figure, output_path)


def _plot_agreement_by_year(
    *,
    diagnostics: pd.DataFrame,
    output_path: Path,
) -> None:
    frame = _diagnostic_group(diagnostics, "quantity_net_weight_by_year")
    frame = frame.dropna(subset=["year", "percentage"]).sort_values("year")
    if frame.empty:
        return
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(frame["year"], frame["percentage"], marker="o")
    axis.set_title("Quantity and net-weight agreement by year")
    axis.set_xlabel("Year")
    axis.set_ylabel("Rows within configured tolerance (%)")
    axis.set_ylim(0, 100)
    axis.grid(axis="y", linestyle="--", alpha=0.4)
    _save_figure(figure, output_path)


def _plot_relative_gap_distribution(
    *,
    con: duckdb.DuckDBPyConnection,
    synchronized_sql: str,
    output_path: Path,
) -> None:
    frame = con.execute(
        f"""
        SELECT quantity_net_weight_relative_gap AS relative_gap
        FROM read_parquet({synchronized_sql})
        WHERE quantity_net_weight_relative_gap IS NOT NULL
          AND isfinite(quantity_net_weight_relative_gap)
        """
    ).fetchdf()
    if frame.empty:
        return
    values = frame["relative_gap"].dropna()
    if values.empty:
        return
    clipped = values.clip(lower=0, upper=0.50)
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.hist(100.0 * clipped, bins=50)
    axis.set_title("Distribution of quantity–net-weight relative gaps")
    axis.set_xlabel("Symmetric relative difference (%)")
    axis.set_ylabel("Number of observations")
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    _save_figure(figure, output_path)


def _plot_alt_agreement_by_evidence_state(
    *,
    diagnostics: pd.DataFrame,
    output_path: Path,
) -> None:
    frame = _diagnostic_group(diagnostics, "alt_quantity_kg_agreement")
    frame = frame.dropna(subset=["metric", "category_1", "percentage"])
    if frame.empty:
        return
    pivot = frame.pivot_table(
        index="category_1",
        columns="metric",
        values="percentage",
        aggfunc="first",
    ).sort_index()
    figure, axis = plt.subplots(figsize=(11, 6))
    pivot.plot(kind="bar", ax=axis)
    axis.set_title("Alternate-quantity kg agreement by evidence state")
    axis.set_xlabel("Evidence state")
    axis.set_ylabel("Rows within configured tolerance (%)")
    axis.set_ylim(0, 100)
    axis.grid(axis="y", linestyle="--", alpha=0.4)
    axis.legend(title="Comparison")
    _save_figure(figure, output_path)


def _plot_net_weight_coverage_by_year(
    *,
    diagnostics: pd.DataFrame,
    output_path: Path,
) -> None:
    frame = _diagnostic_group(diagnostics, "net_weight_by_year")
    frame = frame.dropna(subset=["year", "metric", "percentage"])
    if frame.empty:
        return
    pivot = frame.pivot_table(
        index="year",
        columns="metric",
        values="percentage",
        aggfunc="first",
        fill_value=0.0,
    )
    figure, axis = plt.subplots(figsize=(10, 6))
    for column in pivot.columns:
        axis.plot(pivot.index, pivot[column], marker="o", label=str(column))
    axis.set_title("Positive net-weight coverage by year")
    axis.set_xlabel("Year")
    axis.set_ylabel("Share of annual observations (%)")
    axis.set_ylim(0, 100)
    axis.grid(axis="y", linestyle="--", alpha=0.4)
    axis.legend(title="Evidence state")
    _save_figure(figure, output_path)


def _plot_net_weight_log_distribution_by_year(
    *,
    diagnostics: pd.DataFrame,
    output_path: Path,
) -> None:
    frame = _diagnostic_group(diagnostics, "net_weight_distribution_by_year")
    if frame.empty:
        return
    frame = _as_numeric(frame, ["category_1", "category_2", "category_3"])
    frame = frame.dropna(subset=["year", "category_1", "category_2", "category_3"])
    if frame.empty:
        return
    frame = frame.sort_values("year")
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(frame["year"], frame["category_1"], marker="o", label="p10")
    axis.plot(frame["year"], frame["category_2"], marker="o", label="median")
    axis.plot(frame["year"], frame["category_3"], marker="o", label="p90")
    axis.set_title("Net-weight distribution by year")
    axis.set_xlabel("Year")
    axis.set_ylabel("log10(net_weight_kg)")
    axis.grid(axis="y", linestyle="--", alpha=0.4)
    axis.legend()
    _save_figure(figure, output_path)


def _plot_gross_net_ratio_distribution(
    *,
    con: duckdb.DuckDBPyConnection,
    prepared_sql: str,
    output_path: Path,
) -> None:
    frame = con.execute(
        f"""
        SELECT gross_weight_kg / net_weight_kg AS ratio
        FROM read_parquet({prepared_sql})
        WHERE gross_weight_kg > 0
          AND net_weight_kg > 0
          AND isfinite(gross_weight_kg / net_weight_kg)
        """
    ).fetchdf()
    if frame.empty:
        return
    values = frame["ratio"].dropna()
    if values.empty:
        return
    # Plotting is clipped only for readability; report statistics remain full.
    clipped = values.clip(lower=0, upper=5.0)
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.hist(clipped, bins=60)
    axis.axvline(1.0, linestyle="--")
    axis.set_title("Gross-weight / net-weight ratio distribution")
    axis.set_xlabel("gross_weight_kg / net_weight_kg (clipped at 5)")
    axis.set_ylabel("Number of observations")
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    _save_figure(figure, output_path)


def _plot_hierarchy_coverage(
    *,
    diagnostics: pd.DataFrame,
    output_path: Path,
) -> None:
    frame = _diagnostic_group(diagnostics, "hierarchy_coverage")
    if frame.empty:
        return
    frame["step_order"] = pd.to_numeric(frame["category_1"], errors="coerce")
    frame = frame.sort_values("step_order")
    figure, axis = plt.subplots(figsize=(10, 6))
    bars = axis.bar(frame["metric"], frame["percentage"])
    axis.bar_label(
        bars,
        labels=[f"{value:.2f}%" for value in frame["percentage"]],
        padding=3,
        fontsize=8,
    )
    axis.set_title("Cumulative kg coverage by current synchronization step")
    axis.set_xlabel("Current synchronization outcome")
    axis.set_ylabel("Cumulative population coverage (%)")
    axis.set_ylim(0, 100)
    axis.tick_params(axis="x", labelrotation=25)
    axis.grid(axis="y", linestyle="--", alpha=0.4)
    _save_figure(figure, output_path)


def _plot_synchronization_status_by_year(
    *,
    diagnostics: pd.DataFrame,
    output_path: Path,
) -> None:
    frame = _diagnostic_group(diagnostics, "synchronization_status_by_year")
    frame = frame.dropna(subset=["year", "metric", "percentage"])
    if frame.empty:
        return
    pivot = frame.pivot_table(
        index="year",
        columns="metric",
        values="percentage",
        aggfunc="sum",
        fill_value=0.0,
    )
    figure, axis = plt.subplots(figsize=(11, 6))
    pivot.plot(kind="bar", stacked=True, ax=axis)
    axis.set_title("Current Quantity synchronization outcomes by year")
    axis.set_xlabel("Year")
    axis.set_ylabel("Share of rows (%)")
    axis.set_ylim(0, 100)
    axis.grid(axis="y", linestyle="--", alpha=0.4)
    axis.legend(title="Synchronization status", bbox_to_anchor=(1.02, 1), loc="upper left")
    _save_figure(figure, output_path)


def _plot_m3_by_hs(
    *,
    diagnostics: pd.DataFrame,
    diagnostic_group: str,
    title: str,
    output_path: Path,
    maximum_codes: int = 20,
) -> None:
    frame = _diagnostic_group(diagnostics, diagnostic_group)
    if frame.empty:
        return
    frame = (
        frame.groupby("metric", as_index=False)
        .agg(rows=("value", "sum"))
        .nlargest(maximum_codes, "rows")
        .sort_values("rows", ascending=True)
    )
    if frame.empty:
        return
    figure_height = max(6.0, 0.32 * len(frame) + 2.0)
    figure, axis = plt.subplots(figsize=(10, figure_height))
    bars = axis.barh(frame["metric"].astype(str), frame["rows"])
    axis.bar_label(
        bars,
        labels=[f"{int(value):,}" for value in frame["rows"]],
        padding=3,
        fontsize=8,
    )
    axis.set_title(title)
    axis.set_xlabel("Number of observations")
    axis.set_ylabel("HS code")
    axis.grid(axis="x", linestyle="--", alpha=0.4)
    _save_figure(figure, output_path)


# ======================================================================
# Main reporting function
# ======================================================================


def build_quantity_report(
    *,
    cfg: QuantityConfig,
    prepared_input_path: Path,
    reconciliation_evidence_path: Path,
    synchronized_quantity_path: Path,
    diagnostics_path: Path,
    run_id: str,
    force: bool = False,
    logger,
) -> dict[str, Any]:
    """Create the final Quantity Markdown decision report."""

    prepared_input_path = Path(prepared_input_path)
    reconciliation_evidence_path = Path(reconciliation_evidence_path)
    synchronized_quantity_path = Path(synchronized_quantity_path)
    diagnostics_path = Path(diagnostics_path)

    for label, path in (
        ("prepared Quantity input", prepared_input_path),
        ("Quantity reconciliation evidence", reconciliation_evidence_path),
        ("synchronized Quantity output", synchronized_quantity_path),
        ("Quantity diagnostics", diagnostics_path),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    if not cfg.artifacts.enable:
        return {
            "status": "disabled",
            "report_path": None,
            "latest_report_path": None,
            "ready_for_cif_fob": False,
        }

    report_path = cfg.artifacts_path(run_id)
    latest_path = cfg.artifacts_latest_path()
    figure_dir = (
        get_figure_dir()
        / "quantity"
        / f"target_{cfg.scope.target_hs.lower()}"
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    if not force and valid_output(report_path):
        logger.info("REUSE — Quantity report already exists: %s", report_path)
        return {
            "status": "reused",
            "output_path": report_path,
            "report_path": report_path,
            "latest_report_path": latest_path if latest_path.exists() else None,
            "ready_for_cif_fob": True,
        }

    prepared_sql = sql_literal(prepared_input_path)
    synchronized_sql = sql_literal(synchronized_quantity_path)
    diagnostics_sql = sql_literal(diagnostics_path)

    agreement_tolerance = float(
        cfg.method.kg_reconciliation.agreement.relative_tolerance
    )

    info = get_symbols("info")
    ok = get_symbols("check")

    logger.info("%s Building Quantity decision report.", info)
    logger.info("  Prepared input: %s", prepared_input_path)
    logger.info("  Reconciliation evidence: %s", reconciliation_evidence_path)
    logger.info("  Synchronized quantity: %s", synchronized_quantity_path)
    logger.info("  Diagnostics: %s", diagnostics_path)
    logger.info("  Report: %s", report_path)

    con = duckdb.connect()

    try:
        con.execute(f"SET threads = {int(cfg.engine.threads)}")
        diagnostics = con.execute(
            f"SELECT * FROM read_parquet({diagnostics_sql})"
        ).fetchdf()

        # --------------------------------------------------------------
        # Population
        # --------------------------------------------------------------
        population_row = con.execute(
            f"""
            SELECT
                COUNT(*) AS n_rows,
                COUNT(*) FILTER (
                    WHERE synchronized_quantity_kg > 0
                ) AS n_synchronized,
                COUNT(*) FILTER (
                    WHERE quantity_sync_status = 'unresolved_conflict'
                ) AS n_conflicts,
                COUNT(*) FILTER (
                    WHERE quantity_sync_status = 'no_kg_evidence'
                ) AS n_no_kg,
                COUNT(*) FILTER (
                    WHERE synchronized_quantity_source = 'gross_weight_proxy'
                ) AS n_gross_weight_proxy,
                COUNT(*) FILTER (
                    WHERE NOT COALESCE(has_quantity_kg, FALSE)
                      AND NOT COALESCE(has_net_weight_kg, FALSE)
                      AND COALESCE(has_gross_weight_kg, FALSE)
                ) AS n_potential_gross_weight_fallback
            FROM read_parquet({synchronized_sql})
            """
        ).fetchone()

        (
            n_rows,
            n_synchronized,
            n_conflicts,
            n_no_kg,
            n_gross_weight_proxy,
            n_potential_gross_weight_fallback,
        ) = population_row

        synchronized_share = n_synchronized / n_rows if n_rows else None
        conflict_share = n_conflicts / n_rows if n_rows else None
        no_kg_share = n_no_kg / n_rows if n_rows else None
        gross_weight_proxy_share = (
            n_gross_weight_proxy / n_rows if n_rows else None
        )

        # --------------------------------------------------------------
        # Inventory
        # --------------------------------------------------------------
        inventory = _diagnostic_group(diagnostics, "field_inventory")
        inventory_rows = []
        for field_name in sorted(inventory["metric"].unique()) if not inventory.empty else []:
            field_frame = inventory.loc[inventory["metric"].eq(field_name)]

            def get_stat(statistic: str, column: str):
                match = field_frame.loc[field_frame["category_1"].eq(statistic)]
                return None if match.empty else match.iloc[0][column]

            inventory_rows.append(
                {
                    "Field": field_name,
                    "Total rows": _format_integer(get_stat("total_rows", "value")),
                    "Positive rows": _format_integer(get_stat("positive_rows", "value")),
                    "Positive share": _format_percent(
                        get_stat("positive_rows", "percentage"),
                        already_percent=True,
                    ),
                    "Zero rows": _format_integer(get_stat("zero_rows", "value")),
                    "Zero share": _format_percent(
                        get_stat("zero_rows", "percentage"),
                        already_percent=True,
                    ),
                    "Negative rows": _format_integer(get_stat("negative_rows", "value")),
                    "Negative share": _format_percent(
                        get_stat("negative_rows", "percentage"),
                        already_percent=True,
                    ),
                    "Missing rows": _format_integer(get_stat("missing_rows", "value")),
                    "Missing share": _format_percent(
                        get_stat("missing_rows", "percentage"),
                        already_percent=True,
                    ),
                }
            )
        inventory_display = pd.DataFrame(inventory_rows)

        # --------------------------------------------------------------
        # Unit distributions
        # --------------------------------------------------------------
        def unit_table(group: str) -> pd.DataFrame:
            frame = _diagnostic_group(diagnostics, group)
            if frame.empty:
                return pd.DataFrame()
            # Diagnostics may retain more than one provenance row under the
            # same canonical unit; collapse for the visible unit table.
            frame = (
                frame.groupby(["metric", "category_1"], dropna=False, as_index=False)
                .agg(Rows=("value", "sum"))
            )
            total = frame["Rows"].sum()
            frame["Share (%)"] = 100.0 * frame["Rows"] / total if total else 0.0
            frame.columns = ["Unit", "Unit code", "Rows", "Share (%)"]
            frame["Rows"] = frame["Rows"].round().astype("Int64")
            return frame.sort_values("Rows", ascending=False)

        quantity_units = unit_table("quantity_unit_distribution")
        alt_quantity_units = unit_table("alt_quantity_unit_distribution")

        quantity_m3 = _top_hs_table(
            _diagnostic_group(diagnostics, "quantity_m3_by_hs"), n=30
        )
        alt_quantity_m3 = _top_hs_table(
            _diagnostic_group(diagnostics, "alt_quantity_m3_by_hs"), n=30
        )

        # --------------------------------------------------------------
        # Existing quantity/net evidence
        # --------------------------------------------------------------
        evidence_state = _diagnostic_group(
            diagnostics,
            "quantity_net_weight_by_evidence_state",
        )
        if evidence_state.empty:
            evidence_state_display = pd.DataFrame()
        else:
            evidence_state_display = evidence_state[
                ["metric", "value", "percentage", "category_1", "category_2"]
            ].copy()
            evidence_state_display.columns = [
                "Evidence state",
                "Comparable rows",
                f"Within {100 * agreement_tolerance:.0f}%",
                "Median q/net-weight ratio",
                "Median relative gap",
            ]
            evidence_state_display["Comparable rows"] = (
                evidence_state_display["Comparable rows"].round().astype("Int64")
            )
            evidence_state_display = _as_numeric(
                evidence_state_display,
                ["Median q/net-weight ratio", "Median relative gap"],
            )

        agreement_stats = _diagnostic_group(
            diagnostics,
            "quantity_net_weight_statistics",
        )
        agreement_lookup = {
            row.metric: row.value for row in agreement_stats.itertuples()
        }

        # --------------------------------------------------------------
        # Alt kg candidate diagnostics
        # --------------------------------------------------------------
        alt_candidate_display = _source_state_table(
            _diagnostic_group(diagnostics, "alt_quantity_kg_candidate_coverage")
        )
        alt_incremental_display = _source_state_table(
            _diagnostic_group(diagnostics, "alt_quantity_kg_incremental_coverage")
        )
        alt_agreement_display = _agreement_state_table(
            _diagnostic_group(diagnostics, "alt_quantity_kg_agreement"),
            agreement_tolerance,
        )

        alt_stats = _diagnostic_group(
            diagnostics,
            "alt_quantity_kg_statistics",
        )
        if alt_stats.empty:
            alt_stats_display = pd.DataFrame()
        else:
            alt_stats_display = alt_stats[
                ["metric", "value", "category_1", "category_2", "category_3"]
            ].copy()
            alt_stats_display.columns = [
                "Comparison",
                "Comparable rows",
                "Median ratio",
                "Within 1%",
                "Within 5%",
            ]
            alt_stats_display["Comparable rows"] = (
                alt_stats_display["Comparable rows"].round().astype("Int64")
            )
            alt_stats_display = _as_numeric(
                alt_stats_display,
                ["Median ratio", "Within 1%", "Within 5%"],
            )

        # --------------------------------------------------------------
        # Supplementary dimensions alongside net weight
        # --------------------------------------------------------------
        quantity_supplementary_display = _simple_rows_table(
            _diagnostic_group(
                diagnostics,
                "net_weight_quantity_supplementary_overlap",
            ),
            metric_name="Quantity unit",
            percentage_name="Share of overlap (%)",
        )
        alt_supplementary_display = _simple_rows_table(
            _diagnostic_group(
                diagnostics,
                "net_weight_alt_quantity_supplementary_overlap",
            ),
            metric_name="Alt-quantity unit",
            percentage_name="Share of overlap (%)",
        )

        supplementary_hs = _diagnostic_group(
            diagnostics,
            "net_weight_supplementary_by_hs",
        )
        supplementary_hs_tables: dict[str, pd.DataFrame] = {}
        if not supplementary_hs.empty:
            for unit_code, group in supplementary_hs.groupby("category_1"):
                table = group[
                    ["metric", "category_2", "value", "percentage"]
                ].copy()
                table.columns = ["HS code", "Unit", "Rows", "Share within unit (%)"]
                table["Rows"] = table["Rows"].round().astype("Int64")
                supplementary_hs_tables[str(unit_code)] = table.sort_values(
                    "Rows", ascending=False
                ).head(15)

        # --------------------------------------------------------------
        # Net-weight diagnostics
        # --------------------------------------------------------------
        net_weight_year = _diagnostic_group(diagnostics, "net_weight_by_year")
        if net_weight_year.empty:
            net_weight_year_display = pd.DataFrame()
        else:
            net_weight_year_display = net_weight_year[
                ["year", "metric", "value", "percentage"]
            ].copy()
            net_weight_year_display.columns = [
                "Year",
                "Evidence state",
                "Positive rows",
                "Positive coverage within evidence-state rows (%)",
            ]
            net_weight_year_display["Positive rows"] = (
                net_weight_year_display["Positive rows"].round().astype("Int64")
            )
            net_weight_year_display = net_weight_year_display.sort_values(
                ["Year", "Evidence state"]
            )

        net_weight_distribution = _diagnostic_group(
            diagnostics,
            "net_weight_distribution_by_year",
        )
        if net_weight_distribution.empty:
            net_weight_distribution_display = pd.DataFrame()
        else:
            net_weight_distribution_display = net_weight_distribution[
                ["year", "value", "category_1", "category_2", "category_3"]
            ].copy()
            net_weight_distribution_display.columns = [
                "Year",
                "Positive rows",
                "p10 log10(kg)",
                "Median log10(kg)",
                "p90 log10(kg)",
            ]
            net_weight_distribution_display["Positive rows"] = (
                net_weight_distribution_display["Positive rows"].round().astype("Int64")
            )
            net_weight_distribution_display = _as_numeric(
                net_weight_distribution_display,
                ["p10 log10(kg)", "Median log10(kg)", "p90 log10(kg)"],
            ).sort_values("Year")

        # --------------------------------------------------------------
        # Gross-weight diagnostics
        # --------------------------------------------------------------
        gross_weight = _diagnostic_group(diagnostics, "gross_weight_intervention")
        gross_lookup = {
            row.metric: {"value": row.value, "percentage": row.percentage}
            for row in gross_weight.itertuples()
        }
        gross_available = gross_lookup.get("gross_weight_available", {}).get("value")
        gross_after_other_missing = gross_lookup.get(
            "gross_weight_available_after_other_sources_missing", {}
        ).get("value")
        gross_used = gross_lookup.get("gross_weight_proxy_used", {}).get("value")

        gross_relationship = _diagnostic_group(
            diagnostics,
            "gross_net_weight_relationship",
        )
        gross_relationship_lookup = {
            row.metric: row.value for row in gross_relationship.itertuples()
        }
        gross_comparable = gross_relationship_lookup.get("comparable_rows")
        gross_below_net = gross_relationship_lookup.get("gross_below_net_rows")
        gross_below_net_share = (
            gross_below_net / gross_comparable
            if gross_comparable and gross_below_net is not None
            else None
        )

        gross_evidence = _diagnostic_group(
            diagnostics,
            "gross_net_weight_by_evidence_state",
        )
        if gross_evidence.empty:
            gross_evidence_display = pd.DataFrame()
        else:
            gross_evidence_display = gross_evidence[
                ["metric", "value", "percentage", "category_1", "category_2", "category_3"]
            ].copy()
            gross_evidence_display.columns = [
                "Evidence state",
                "Comparable rows",
                "Gross < net (%)",
                "Median gross/net ratio",
                "p10 ratio",
                "p90 ratio",
            ]
            gross_evidence_display["Comparable rows"] = (
                gross_evidence_display["Comparable rows"].round().astype("Int64")
            )
            gross_evidence_display = _as_numeric(
                gross_evidence_display,
                ["Median gross/net ratio", "p10 ratio", "p90 ratio"],
            )

        gross_without_net_year = _diagnostic_group(
            diagnostics,
            "gross_weight_without_net_by_year",
        )
        if gross_without_net_year.empty:
            gross_without_net_year_display = pd.DataFrame()
        else:
            gross_without_net_year_display = gross_without_net_year[
                ["year", "value", "percentage"]
            ].copy()
            gross_without_net_year_display.columns = [
                "Year",
                "Gross-positive / net-unavailable rows",
                "Share of such rows (%)",
            ]
            gross_without_net_year_display[
                "Gross-positive / net-unavailable rows"
            ] = gross_without_net_year_display[
                "Gross-positive / net-unavailable rows"
            ].round().astype("Int64")
            gross_without_net_year_display = gross_without_net_year_display.sort_values("Year")

        gross_without_net_hs_display = _top_hs_table(
            _diagnostic_group(diagnostics, "gross_weight_without_net_by_hs"),
            n=20,
        )

        # --------------------------------------------------------------
        # Current synchronization diagnostics
        # --------------------------------------------------------------
        hierarchy = _diagnostic_group(diagnostics, "hierarchy_coverage")
        if hierarchy.empty:
            hierarchy_display = pd.DataFrame()
        else:
            hierarchy = hierarchy.copy()
            hierarchy["Step"] = pd.to_numeric(hierarchy["category_1"], errors="coerce")
            hierarchy["Cumulative rows"] = pd.to_numeric(
                hierarchy["category_2"], errors="coerce"
            )
            hierarchy["Incremental coverage (%)"] = pd.to_numeric(
                hierarchy["category_3"], errors="coerce"
            )
            hierarchy_display = hierarchy[
                [
                    "Step",
                    "metric",
                    "value",
                    "Cumulative rows",
                    "percentage",
                    "Incremental coverage (%)",
                ]
            ].copy()
            hierarchy_display.columns = [
                "Step",
                "Current outcome",
                "New rows",
                "Cumulative rows",
                "Cumulative coverage (%)",
                "Incremental coverage (%)",
            ]
            hierarchy_display["New rows"] = hierarchy_display["New rows"].round().astype("Int64")
            hierarchy_display["Cumulative rows"] = hierarchy_display["Cumulative rows"].round().astype("Int64")
            hierarchy_display = hierarchy_display.sort_values("Step")

        sync_status = _diagnostic_group(diagnostics, "synchronization_status")
        if sync_status.empty:
            sync_status_display = pd.DataFrame()
        else:
            sync_status_display = sync_status[
                ["metric", "category_1", "value", "percentage"]
            ].copy()
            sync_status_display.columns = ["Status", "Final source", "Rows", "Share (%)"]
            sync_status_display["Rows"] = sync_status_display["Rows"].round().astype("Int64")
            sync_status_display = sync_status_display.sort_values("Rows", ascending=False)

        remaining_gap_display = _simple_rows_table(
            _diagnostic_group(diagnostics, "remaining_kg_gap"),
            metric_name="Outcome",
            percentage_name="Share (%)",
        )

        conflict_stats = _diagnostic_group(
            diagnostics,
            "unresolved_conflict_statistics",
        )
        conflict_lookup = {
            row.metric: row.value for row in conflict_stats.itertuples()
        }
        conflict_hs_display = _top_hs_table(
            _diagnostic_group(diagnostics, "unresolved_conflicts_by_hs"),
            n=20,
        )
        missing_hs_display = _top_hs_table(
            _diagnostic_group(diagnostics, "missing_kg_by_hs"),
            n=20,
        )

        # --------------------------------------------------------------
        # Figures
        # --------------------------------------------------------------
        figure_paths = {
            "quantity_units": figure_dir / "quantity_unit_distribution.png",
            "alt_quantity_units": figure_dir / "alt_quantity_unit_distribution.png",
            "quantity_units_year": figure_dir / "quantity_unit_share_by_year.png",
            "alt_quantity_units_year": figure_dir / "alt_quantity_unit_share_by_year.png",
            "agreement_year": figure_dir / "quantity_net_weight_agreement_by_year.png",
            "gap_distribution": figure_dir / "quantity_net_weight_relative_gap_distribution.png",
            "alt_agreement_state": figure_dir / "alt_quantity_kg_agreement_by_evidence_state.png",
            "net_weight_year": figure_dir / "net_weight_coverage_by_year.png",
            "net_weight_distribution": figure_dir / "net_weight_log_distribution_by_year.png",
            "gross_net_ratio": figure_dir / "gross_net_weight_ratio_distribution.png",
            "hierarchy": figure_dir / "kg_coverage_by_reconciliation_step.png",
            "sync_status_year": figure_dir / "quantity_sync_status_by_year.png",
            "quantity_m3_hs": figure_dir / "quantity_m3_by_hs_code.png",
            "alt_quantity_m3_hs": figure_dir / "alt_quantity_m3_by_hs_code.png",
        }

        _plot_unit_distribution(
            diagnostics=diagnostics,
            diagnostic_group="quantity_unit_distribution",
            title="Quantity unit distribution",
            output_path=figure_paths["quantity_units"],
        )
        _plot_unit_distribution(
            diagnostics=diagnostics,
            diagnostic_group="alt_quantity_unit_distribution",
            title="Canonical alternate-quantity unit distribution",
            output_path=figure_paths["alt_quantity_units"],
        )
        _plot_unit_share_by_year(
            diagnostics=diagnostics,
            diagnostic_group="quantity_unit_by_year",
            title="Quantity unit composition by year",
            output_path=figure_paths["quantity_units_year"],
        )
        _plot_unit_share_by_year(
            diagnostics=diagnostics,
            diagnostic_group="alt_quantity_unit_by_year",
            title="Alternate quantity unit composition by year",
            output_path=figure_paths["alt_quantity_units_year"],
        )
        _plot_agreement_by_year(
            diagnostics=diagnostics,
            output_path=figure_paths["agreement_year"],
        )
        _plot_relative_gap_distribution(
            con=con,
            synchronized_sql=synchronized_sql,
            output_path=figure_paths["gap_distribution"],
        )
        _plot_alt_agreement_by_evidence_state(
            diagnostics=diagnostics,
            output_path=figure_paths["alt_agreement_state"],
        )
        _plot_net_weight_coverage_by_year(
            diagnostics=diagnostics,
            output_path=figure_paths["net_weight_year"],
        )
        _plot_net_weight_log_distribution_by_year(
            diagnostics=diagnostics,
            output_path=figure_paths["net_weight_distribution"],
        )
        _plot_gross_net_ratio_distribution(
            con=con,
            prepared_sql=prepared_sql,
            output_path=figure_paths["gross_net_ratio"],
        )
        _plot_hierarchy_coverage(
            diagnostics=diagnostics,
            output_path=figure_paths["hierarchy"],
        )
        _plot_synchronization_status_by_year(
            diagnostics=diagnostics,
            output_path=figure_paths["sync_status_year"],
        )
        _plot_m3_by_hs(
            diagnostics=diagnostics,
            diagnostic_group="quantity_m3_by_hs",
            title="HS codes with quantity reported in cubic metres",
            output_path=figure_paths["quantity_m3_hs"],
        )
        _plot_m3_by_hs(
            diagnostics=diagnostics,
            diagnostic_group="alt_quantity_m3_by_hs",
            title="HS codes with alternate quantity reported in cubic metres",
            output_path=figure_paths["alt_quantity_m3_hs"],
        )

        # --------------------------------------------------------------
        # Decision logic: still evaluates CURRENT synchronize.py output.
        # Candidate alt-quantity diagnostics do not silently change it.
        # --------------------------------------------------------------
        population_partition_passed = bool(
            n_rows == n_synchronized + n_conflicts + n_no_kg
        )
        synchronized_output_valid = bool(
            synchronized_quantity_path.is_file()
            and synchronized_quantity_path.stat().st_size > 0
        )
        ready_for_cif_fob = bool(
            synchronized_output_valid and population_partition_passed
        )
        decision = "APPROVED" if ready_for_cif_fob else "REVIEW REQUIRED"

        if n_gross_weight_proxy > 0:
            gross_weight_statement = (
                f"Gross weight was used as a proxy for `{n_gross_weight_proxy:,}` "
                f"rows ({_format_percent(gross_weight_proxy_share)} of the full "
                "input). These rows remain explicitly identified by provenance."
            )
        else:
            gross_weight_statement = (
                "Gross weight was not used in the synchronized kg quantity."
            )

        if n_potential_gross_weight_fallback > 0 and n_gross_weight_proxy == 0:
            gross_weight_statement += (
                f" Gross weight nevertheless exists for "
                f"`{n_potential_gross_weight_fallback:,}` rows where the "
                "currently admitted quantity-in-kg and net-weight sources are "
                "unavailable."
            )

        # --------------------------------------------------------------
        # Markdown
        # --------------------------------------------------------------
        lines: list[str] = [
            "# Quantity Decision Report",
            "",
            f"**Decision:** {decision}  ",
            f"**Run ID:** {run_id}  ",
            f"**Target HS:** {cfg.scope.target_hs}  ",
            f"**Input period:** {cfg.scope.min_year}–{cfg.scope.max_year}  ",
            f"**Generated:** {datetime.now().isoformat(timespec='seconds')}",
            "",
            "## 1. Purpose of Quantity synchronization",
            "",
            (
                "The Quantity module receives the HS-synchronized Comtrade "
                "observations and evaluates the available physical-quantity "
                "information before downstream value processing."
            ),
            "",
            (
                "The objective is to construct a transparent kg-based physical "
                "quantity while preserving supplementary physical dimensions "
                "and source provenance. Gross weight is evaluated separately "
                "because it is not equivalent to net physical weight."
            ),
            "",
            "## 2. Input data and observation population",
            "",
            f"Quantity received `{n_rows:,}` observations from the HS Sync output.",
            "",
            "| Property | Value |",
            "|---|---|",
            "| Upstream module | HS Sync |",
            f"| Target HS classification | {cfg.scope.target_hs} |",
            f"| Covered years | {cfg.scope.min_year}–{cfg.scope.max_year} |",
            "| Canonical synchronized field | `synchronized_quantity_kg` |",
            f"| Agreement tolerance | {100 * agreement_tolerance:.2f}% |",
            (
                "| Source-selection method | "
                f"`{cfg.method.kg_reconciliation.source_selection.method}` |"
            ),
            "",
            "### 2.1 Quantity-field inventory",
            "",
            _markdown_table(inventory_display),
            "",
            "## 3. Physical-unit structure",
            "",
            "### 3.1 `quantity` unit distribution",
            "",
            _markdown_table(quantity_units),
            "",
        ]

        if figure_paths["quantity_units"].exists():
            lines.extend([
                markdown_image(
                    "Quantity unit distribution",
                    figure_paths["quantity_units"],
                    document_path=report_path,
                    width=750,
                ),
                "",
            ])

        lines.extend([
            "### 3.2 Canonical `alt_quantity` unit distribution",
            "",
            (
                "The visible unit distribution reflects the Quantity module's "
                "canonical physical interpretation. Alternate quantity reported "
                "as `1000 KG` is kg-equivalent after the exact ×1000 conversion "
                "performed in reconciliation evidence; it is therefore counted "
                "within the canonical kg class rather than shown as a separate "
                "physical unit."
            ),
            "",
            _markdown_table(alt_quantity_units),
            "",
        ])

        if figure_paths["alt_quantity_units"].exists():
            lines.extend([
                markdown_image(
                    "Canonical alternate quantity unit distribution",
                    figure_paths["alt_quantity_units"],
                    document_path=report_path,
                    width=750,
                ),
                "",
            ])

        lines.extend([
            "### 3.3 Unit composition over time",
            "",
            "Unit composition by year is retained to reveal changes in reporting practice.",
            "",
        ])
        for key, caption in (
            ("quantity_units_year", "Quantity units by year"),
            ("alt_quantity_units_year", "Alternate quantity units by year"),
        ):
            if figure_paths[key].exists():
                lines.extend([
                    markdown_image(
                        caption,
                        figure_paths[key],
                        document_path=report_path,
                        width=800,
                    ),
                    "",
                ])

        lines.extend([
            "### 3.4 Cubic-metre observations by HS code",
            "",
            "**`quantity` reported in m³**",
            "",
            _markdown_table(quantity_m3),
            "",
        ])
        if figure_paths["quantity_m3_hs"].exists():
            lines.extend([
                markdown_image(
                    "Quantity m³ observations by HS code",
                    figure_paths["quantity_m3_hs"],
                    document_path=report_path,
                    width=750,
                ),
                "",
            ])
        lines.extend([
            "**`alt_quantity` reported in m³**",
            "",
            _markdown_table(alt_quantity_m3),
            "",
        ])
        if figure_paths["alt_quantity_m3_hs"].exists():
            lines.extend([
                markdown_image(
                    "Alternate quantity m³ observations by HS code",
                    figure_paths["alt_quantity_m3_hs"],
                    document_path=report_path,
                    width=750,
                ),
                "",
            ])

        # Canonical kg evidence + alt candidate diagnostics
        lines.extend([
            "## 4. Canonical kg evidence",
            "",
            (
                "Three kg-equivalent evidence families are evaluated: quantity "
                "reported directly in kg, `net_weight_kg`, and alternate quantity "
                "that is either directly expressed in kg or exactly convertible "
                "to kg (for example `1000 KG`). Special mass bases such as "
                "`kg 90% sdt` are not treated as ordinary net kg."
            ),
            "",
            "### 4.1 Quantity and net-weight reported/estimated evidence",
            "",
            _markdown_table(evidence_state_display),
            "",
            "### 4.2 Alternate quantity as kg-equivalent candidate evidence",
            "",
            (
                "Alternate quantity is evaluated here as candidate evidence only. "
                "These diagnostics do not yet imply that synchronize.py has admitted "
                "the source into the final hierarchy."
            ),
            "",
            _markdown_table(alt_candidate_display),
            "",
            "**Incremental coverage if quantity kg and net weight are unavailable**",
            "",
            _markdown_table(alt_incremental_display),
            "",
            "### 4.3 Alternate-quantity agreement with existing kg evidence",
            "",
            _markdown_table(alt_agreement_display),
            "",
        ])
        if not alt_stats_display.empty:
            lines.extend([
                "**Whole-population comparison summary**",
                "",
                _markdown_table(alt_stats_display),
                "",
            ])
        if figure_paths["alt_agreement_state"].exists():
            lines.extend([
                markdown_image(
                    "Alternate-quantity kg agreement by evidence state",
                    figure_paths["alt_agreement_state"],
                    document_path=report_path,
                    width=800,
                ),
                "",
            ])

        # Existing q/net comparison
        lines.extend([
            "## 5. Quantity versus net-weight consistency",
            "",
            (
                "For observations where quantity in kg and net weight are both "
                "usable, the symmetric relative gap is"
            ),
            "",
            "$$",
            "d = \\frac{|Q-W|}{(|Q|+|W|)/2}",
            "$$",
            "",
            "| Statistic | Value |",
            "|---|---:|",
            f"| Comparable observations | {_format_integer(agreement_lookup.get('n_comparable'))} |",
            f"| Median quantity / net-weight ratio | {_format_number(agreement_lookup.get('median_ratio'))} |",
            f"| 10th percentile ratio | {_format_number(agreement_lookup.get('p10_ratio'))} |",
            f"| 90th percentile ratio | {_format_number(agreement_lookup.get('p90_ratio'))} |",
            f"| Median relative gap | {_format_percent(agreement_lookup.get('median_relative_gap'))} |",
            f"| 90th percentile relative gap | {_format_percent(agreement_lookup.get('p90_relative_gap'))} |",
            f"| 95th percentile relative gap | {_format_percent(agreement_lookup.get('p95_relative_gap'))} |",
            f"| Within 1% | {_format_percent(agreement_lookup.get('within_1pct_share'))} |",
            f"| Within 5% | {_format_percent(agreement_lookup.get('within_5pct_share'))} |",
            f"| Within 10% | {_format_percent(agreement_lookup.get('within_10pct_share'))} |",
            "",
        ])
        for key, caption in (
            ("agreement_year", "Quantity and net-weight agreement by year"),
            ("gap_distribution", "Quantity and net-weight relative-gap distribution"),
        ):
            if figure_paths[key].exists():
                lines.extend([
                    markdown_image(
                        caption,
                        figure_paths[key],
                        document_path=report_path,
                        width=800,
                    ),
                    "",
                ])

        # Supplementary dimensions
        lines.extend([
            "## 6. Net weight alongside supplementary physical quantities",
            "",
            (
                "A positive net weight can coexist with a non-kg quantity because "
                "the fields may represent different physical dimensions (for "
                "example mass together with cubic metres or number of units). "
                "These overlaps are therefore diagnosed rather than treated as "
                "dimensional conflicts."
            ),
            "",
            "### 6.1 Non-kg `quantity` with positive net weight",
            "",
            _markdown_table(quantity_supplementary_display),
            "",
            "### 6.2 Non-kg `alt_quantity` with positive net weight",
            "",
            _markdown_table(alt_supplementary_display),
            "",
            "### 6.3 HS structure of major supplementary units",
            "",
        ])
        unit_names = {"2": "m²", "5": "u", "12": "m³"}
        for unit_code in ("2", "5", "12"):
            table = supplementary_hs_tables.get(unit_code, pd.DataFrame())
            lines.extend([
                f"**Unit {unit_names.get(unit_code, unit_code)} (code {unit_code})**",
                "",
                _markdown_table(table),
                "",
            ])

        # Net weight
        lines.extend([
            "## 7. Net-weight diagnostics",
            "",
            "### 7.1 Positive net-weight coverage by year and estimation state",
            "",
            _markdown_table(net_weight_year_display),
            "",
        ])
        if figure_paths["net_weight_year"].exists():
            lines.extend([
                markdown_image(
                    "Net-weight coverage by year",
                    figure_paths["net_weight_year"],
                    document_path=report_path,
                    width=800,
                ),
                "",
            ])
        lines.extend([
            "### 7.2 Net-weight magnitude distribution",
            "",
            (
                "Because trade weights span many orders of magnitude, distribution "
                "statistics are shown on the log10(kg) scale rather than with a "
                "raw linear histogram."
            ),
            "",
            _markdown_table(net_weight_distribution_display),
            "",
        ])
        if figure_paths["net_weight_distribution"].exists():
            lines.extend([
                markdown_image(
                    "Net-weight log distribution by year",
                    figure_paths["net_weight_distribution"],
                    document_path=report_path,
                    width=800,
                ),
                "",
            ])

        # Gross weight
        lines.extend([
            "## 8. Gross-weight diagnostics",
            "",
            gross_weight_statement,
            "",
            "### 8.1 Availability and possible intervention",
            "",
            "| Gross-weight diagnostic | Rows |",
            "|---|---:|",
            f"| Gross weight available | {_format_integer(gross_available)} |",
            (
                "| Gross weight available where currently admitted quantity kg "
                "and net weight are unavailable | "
                f"{_format_integer(gross_after_other_missing)} |"
            ),
            f"| Gross-weight proxy actually used | {_format_integer(gross_used)} |",
            "",
            f"Configuration setting `use_as_quantity_proxy = {cfg.method.gross_weight.use_as_quantity_proxy}`.",
            "",
            "### 8.2 Gross-weight versus net-weight relationship",
            "",
            "| Statistic | Value |",
            "|---|---:|",
            f"| Comparable positive rows | {_format_integer(gross_comparable)} |",
            f"| Rows with gross weight < net weight | {_format_integer(gross_below_net)} |",
            f"| Share with gross weight < net weight | {_format_percent(gross_below_net_share)} |",
            f"| Median gross/net ratio | {_format_number(gross_relationship_lookup.get('median_ratio'))} |",
            f"| p10 ratio | {_format_number(gross_relationship_lookup.get('p10_ratio'))} |",
            f"| p90 ratio | {_format_number(gross_relationship_lookup.get('p90_ratio'))} |",
            "",
            "### 8.3 Gross/net relationship by estimation state",
            "",
            _markdown_table(gross_evidence_display),
            "",
        ])
        if figure_paths["gross_net_ratio"].exists():
            lines.extend([
                markdown_image(
                    "Gross-weight / net-weight ratio distribution",
                    figure_paths["gross_net_ratio"],
                    document_path=report_path,
                    width=800,
                ),
                "",
            ])
        lines.extend([
            "### 8.4 Positive gross weight where net weight is unavailable",
            "",
            _markdown_table(gross_without_net_year_display),
            "",
            "**Top HS codes**",
            "",
            _markdown_table(gross_without_net_hs_display),
            "",
            (
                "Gross weight remains diagnostic-only unless explicitly enabled. "
                "Rows where gross weight is smaller than net weight are especially "
                "important evidence against silently treating gross weight as a "
                "general net-mass substitute."
            ),
            "",
        ])

        # Current synchronization
        lines.extend([
            "## 9. Current synchronization outcome",
            "",
            (
                "The following tables describe the hierarchy currently implemented "
                "by `synchronize.py`. Candidate alt-quantity evidence shown above "
                "has not been silently inserted into this hierarchy."
            ),
            "",
            "### 9.1 Current contribution and cumulative kg coverage",
            "",
            _markdown_table(hierarchy_display),
            "",
        ])
        if figure_paths["hierarchy"].exists():
            lines.extend([
                markdown_image(
                    "Cumulative kg coverage by current synchronization step",
                    figure_paths["hierarchy"],
                    document_path=report_path,
                    width=800,
                ),
                "",
            ])
        lines.extend([
            "### 9.2 Synchronization statuses",
            "",
            _markdown_table(sync_status_display),
            "",
        ])
        if figure_paths["sync_status_year"].exists():
            lines.extend([
                markdown_image(
                    "Quantity synchronization outcomes by year",
                    figure_paths["sync_status_year"],
                    document_path=report_path,
                    width=800,
                ),
                "",
            ])
        lines.extend([
            "### 9.3 Final current coverage",
            "",
            (
                f"`{n_synchronized:,}` of `{n_rows:,}` rows "
                f"({_format_percent(synchronized_share)}) receive a positive "
                "synchronized kg quantity under the current synchronization logic."
            ),
            "",
            "## 10. Remaining gaps and conflicts",
            "",
            _markdown_table(remaining_gap_display),
            "",
            (
                f"`{n_conflicts:,}` rows ({_format_percent(conflict_share)}) are "
                f"currently unresolved conflicts; `{n_no_kg:,}` rows "
                f"({_format_percent(no_kg_share)}) currently have no resolved kg value."
            ),
            "",
            "### 10.1 Conflict severity",
            "",
            "| Statistic | Value |",
            "|---|---:|",
            f"| Conflict rows | {_format_integer(conflict_lookup.get('n_conflicts'))} |",
            f"| Median relative gap | {_format_percent(conflict_lookup.get('median_relative_gap'))} |",
            f"| p90 relative gap | {_format_percent(conflict_lookup.get('p90_relative_gap'))} |",
            f"| p95 relative gap | {_format_percent(conflict_lookup.get('p95_relative_gap'))} |",
            "",
            "### 10.2 HS codes with unresolved conflicts",
            "",
            _markdown_table(conflict_hs_display),
            "",
            "### 10.3 HS codes with missing synchronized kg",
            "",
            _markdown_table(missing_hs_display),
            "",
            "## 11. Methodological interpretation",
            "",
            (
                "This report distinguishes measurement interpretation from source "
                "selection. Exact unit conversions create canonical kg candidates; "
                "reported/estimated provenance and cross-source agreement are then "
                "used to evaluate whether those candidates should participate in "
                "future synchronization. Supplementary non-kg quantities remain "
                "separate physical dimensions."
            ),
            "",
            (
                "The present run still reports the existing two-source "
                "synchronization outcome. The new alternate-quantity diagnostics "
                "provide the evidence required for a subsequent explicit decision "
                "on admission and precedence."
            ),
            "",
            "## 12. Validation and decision",
            "",
            "| Validation | Result |",
            "|---|---|",
            f"| Synchronized output exists and is non-empty | {'PASS' if synchronized_output_valid else 'FAIL'} |",
            f"| Population accounting is complete | {'PASS' if population_partition_passed else 'FAIL'} |",
            f"| Final current Quantity decision | **{decision}** |",
            "",
            "## 13. Downstream recommendation",
            "",
        ])

        if ready_for_cif_fob:
            lines.extend([
                (
                    f"{ok} The current Quantity output is structurally ready for "
                    "downstream CIF/FOB processing. This structural approval does "
                    "not pre-empt the pending methodological decision on admitting "
                    "alternate-quantity kg candidates."
                ),
                "",
            ])
        else:
            lines.extend([
                "Quantity synchronization requires review before downstream use.",
                "",
            ])

        lines.extend([
            "## 14. Output provenance",
            "",
            "| Artifact | Path |",
            "|---|---|",
            f"| Prepared input | `{prepared_input_path}` |",
            f"| Reconciliation evidence | `{reconciliation_evidence_path}` |",
            f"| Synchronized quantity | `{synchronized_quantity_path}` |",
            f"| Diagnostics | `{diagnostics_path}` |",
            "",
        ])

        report_text = "\n".join(lines)
        report_path.write_text(report_text, encoding="utf-8")

        if cfg.artifacts.write_latest:
            shutil.copy2(report_path, latest_path)

        logger.info("%s Quantity decision report written: %s", ok, report_path)
        if cfg.artifacts.write_latest:
            logger.info("Latest Quantity report: %s", latest_path)

    finally:
        con.close()

    return {
        "status": "completed",
        "run_id": run_id,
        "output_path": report_path,
        "report_path": report_path,
        "latest_report_path": latest_path if cfg.artifacts.write_latest else None,
        "decision": decision,
        "ready_for_cif_fob": ready_for_cif_fob,
        "n_rows": int(n_rows),
        "n_synchronized": int(n_synchronized),
        "n_unresolved_conflict": int(n_conflicts),
        "n_no_kg_evidence": int(n_no_kg),
        "n_gross_weight_proxy": int(n_gross_weight_proxy),
    }
