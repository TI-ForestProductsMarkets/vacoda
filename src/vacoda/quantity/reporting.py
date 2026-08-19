"""
vacoda.quantity.reporting

Create the final Quantity decision report and diagnostic figures.

Outputs
-------
- Timestamped Markdown decision report
- Optional latest Markdown copy
- Quantity and alternate-quantity unit-distribution figures
- Quantity/net-weight agreement figures
- Cumulative kg-coverage figure
- Synchronization-status figure
- m³-by-HS-code figures where applicable
- Gross-weight intervention summary
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


# ======================================================================
# Helpers
# ======================================================================


def _sql_literal(value: str | Path) -> str:
    """Return a safely quoted SQL literal."""
    return "'" + str(value).replace("'", "''") + "'"


def valid_output(path: Path) -> bool:
    """Return True when an existing output appears usable."""
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


def _markdown_table(
    dataframe: pd.DataFrame,
    *,
    index: bool = False,
) -> str:
    """Convert a DataFrame to Markdown."""

    if dataframe.empty:
        return "_No rows available._"

    return dataframe.to_markdown(
        index=index,
        floatfmt=".4f",
    )


def _diagnostic_group(
    diagnostics: pd.DataFrame,
    name: str,
) -> pd.DataFrame:
    """Return one diagnostics group."""

    return diagnostics.loc[
        diagnostics["diagnostic_group"].eq(name)
    ].copy()


def _diagnostic_value(
    diagnostics: pd.DataFrame,
    *,
    group: str,
    metric: str,
    category_1: str | None = None,
) -> float | None:
    """Retrieve one numeric value from long-format diagnostics."""

    frame = diagnostics.loc[
        diagnostics["diagnostic_group"].eq(group)
        & diagnostics["metric"].eq(metric)
    ]

    if category_1 is not None:
        frame = frame.loc[
            frame["category_1"].eq(category_1)
        ]

    if frame.empty:
        return None

    value = frame.iloc[0]["value"]

    if pd.isna(value):
        return None

    return float(value)


def _diagnostic_percentage(
    diagnostics: pd.DataFrame,
    *,
    group: str,
    metric: str,
) -> float | None:
    """Retrieve one percentage stored on the 0-100 scale."""

    frame = diagnostics.loc[
        diagnostics["diagnostic_group"].eq(group)
        & diagnostics["metric"].eq(metric)
    ]

    if frame.empty:
        return None

    value = frame.iloc[0]["percentage"]

    if pd.isna(value):
        return None

    return float(value)


# ======================================================================
# Figures
# ======================================================================


def _plot_unit_distribution(
    *,
    diagnostics: pd.DataFrame,
    diagnostic_group: str,
    title: str,
    output_path: Path,
) -> None:
    """Plot physical-unit composition."""

    frame = _diagnostic_group(
        diagnostics,
        diagnostic_group,
    )

    if frame.empty:
        return

    frame = (
        frame.groupby(
            "metric",
            as_index=False,
        )
        .agg(
            rows=("value", "sum"),
        )
        .sort_values(
            "rows",
            ascending=False,
        )
    )

    total = frame["rows"].sum()

    if total <= 0:
        return

    frame["share"] = 100.0 * frame["rows"] / total

    figure, axis = plt.subplots(figsize=(10, 6))

    bars = axis.bar(
        frame["metric"].astype(str),
        frame["share"],
    )

    axis.bar_label(
        bars,
        labels=[
            f"{value:.1f}%"
            for value in frame["share"]
        ],
        padding=3,
        fontsize=8,
    )

    axis.set_title(title)
    axis.set_xlabel("Quantity unit")
    axis.set_ylabel("Share of observations (%)")
    axis.tick_params(
        axis="x",
        labelrotation=35,
    )
    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
    )
    axis.set_axisbelow(True)

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def _plot_unit_share_by_year(
    *,
    diagnostics: pd.DataFrame,
    diagnostic_group: str,
    title: str,
    output_path: Path,
) -> None:
    """Plot unit composition by year."""

    frame = _diagnostic_group(
        diagnostics,
        diagnostic_group,
    )

    if frame.empty:
        return

    frame = frame.dropna(
        subset=["year", "metric", "percentage"]
    )

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

    pivot.plot(
        kind="bar",
        stacked=True,
        ax=axis,
    )

    axis.set_title(title)
    axis.set_xlabel("Year")
    axis.set_ylabel("Share of observations (%)")
    axis.set_ylim(0, 100)
    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
    )
    axis.legend(
        title="Unit",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def _plot_agreement_by_year(
    *,
    diagnostics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot quantity/net-weight agreement by year."""

    frame = _diagnostic_group(
        diagnostics,
        "quantity_net_weight_by_year",
    )

    if frame.empty:
        return

    frame = frame.dropna(
        subset=["year", "percentage"]
    ).sort_values("year")

    if frame.empty:
        return

    figure, axis = plt.subplots(figsize=(10, 6))

    axis.plot(
        frame["year"],
        frame["percentage"],
        marker="o",
    )

    axis.set_title(
        "Quantity and net-weight agreement by year"
    )
    axis.set_xlabel("Year")
    axis.set_ylabel(
        "Rows within configured tolerance (%)"
    )
    axis.set_ylim(
        0,
        min(
            100.0,
            max(
                10.0,
                frame["percentage"].max() * 1.08,
            ),
        ),
    )
    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
    )

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def _plot_relative_gap_distribution(
    *,
    con: duckdb.DuckDBPyConnection,
    synchronized_sql: str,
    output_path: Path,
) -> None:
    """Plot the quantity/net-weight relative-gap distribution."""

    frame = con.execute(
        f"""
        SELECT
            quantity_net_weight_relative_gap AS relative_gap
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

    clipped = values.clip(
        lower=0,
        upper=0.50,
    )

    figure, axis = plt.subplots(figsize=(10, 6))

    axis.hist(
        100.0 * clipped,
        bins=50,
    )

    axis.set_title(
        "Distribution of quantity–net-weight relative gaps"
    )
    axis.set_xlabel(
        "Symmetric relative difference (%)"
    )
    axis.set_ylabel("Number of observations")
    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.35,
    )

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def _plot_hierarchy_coverage(
    *,
    diagnostics: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot cumulative coverage and incremental contribution of each
    synchronization source.
    """

    frame = _diagnostic_group(
        diagnostics,
        "hierarchy_coverage",
    )

    if frame.empty:
        return

    frame["step_order"] = pd.to_numeric(
        frame["category_1"],
        errors="coerce",
    )

    frame["incremental_percent"] = pd.to_numeric(
        frame["category_3"],
        errors="coerce",
    )

    frame = frame.sort_values(
        "step_order"
    )

    figure, axis = plt.subplots(figsize=(10, 6))

    bars = axis.bar(
        frame["metric"],
        frame["percentage"],
    )

    axis.bar_label(
        bars,
        labels=[
            f"{value:.2f}%"
            for value in frame["percentage"]
        ],
        padding=3,
        fontsize=8,
    )

    axis.set_title(
        "Cumulative kg coverage by synchronization step"
    )
    axis.set_xlabel("Synchronization source")
    axis.set_ylabel("Cumulative population coverage (%)")
    axis.set_ylim(
        0,
        min(
            100.0,
            max(
                10.0,
                frame["percentage"].max() * 1.08,
            ),
        ),
    )
    axis.tick_params(
        axis="x",
        labelrotation=25,
    )
    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
    )

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def _plot_synchronization_status_by_year(
    *,
    diagnostics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot Quantity synchronization status by year."""

    frame = _diagnostic_group(
        diagnostics,
        "synchronization_status_by_year",
    )

    if frame.empty:
        return

    frame = frame.dropna(
        subset=["year", "metric", "percentage"]
    )

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

    pivot.plot(
        kind="bar",
        stacked=True,
        ax=axis,
    )

    axis.set_title(
        "Quantity synchronization outcomes by year"
    )
    axis.set_xlabel("Year")
    axis.set_ylabel("Share of rows (%)")
    axis.set_ylim(0, 100)
    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
    )
    axis.legend(
        title="Synchronization status",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def _plot_m3_by_hs(
    *,
    diagnostics: pd.DataFrame,
    diagnostic_group: str,
    title: str,
    output_path: Path,
    maximum_codes: int = 20,
) -> None:
    """Plot HS codes responsible for cubic-metre observations."""

    frame = _diagnostic_group(
        diagnostics,
        diagnostic_group,
    )

    if frame.empty:
        return

    frame = (
        frame.groupby(
            "metric",
            as_index=False,
        )
        .agg(rows=("value", "sum"))
        .nlargest(
            maximum_codes,
            "rows",
        )
        .sort_values(
            "rows",
            ascending=True,
        )
    )

    if frame.empty:
        return

    figure_height = max(
        6.0,
        0.32 * len(frame) + 2.0,
    )

    figure, axis = plt.subplots(
        figsize=(10, figure_height)
    )

    bars = axis.barh(
        frame["metric"].astype(str),
        frame["rows"],
    )

    axis.bar_label(
        bars,
        labels=[
            f"{int(value):,}"
            for value in frame["rows"]
        ],
        padding=3,
        fontsize=8,
    )

    axis.set_title(title)
    axis.set_xlabel("Number of observations")
    axis.set_ylabel("HS code")
    axis.grid(
        axis="x",
        linestyle="--",
        alpha=0.4,
    )

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


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

    prepared_input_path = Path(
        prepared_input_path
    )
    reconciliation_evidence_path = Path(
        reconciliation_evidence_path
    )
    synchronized_quantity_path = Path(
        synchronized_quantity_path
    )
    diagnostics_path = Path(
        diagnostics_path
    )

    for label, path in (
        (
            "prepared Quantity input",
            prepared_input_path,
        ),
        (
            "Quantity reconciliation evidence",
            reconciliation_evidence_path,
        ),
        (
            "synchronized Quantity output",
            synchronized_quantity_path,
        ),
        (
            "Quantity diagnostics",
            diagnostics_path,
        ),
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"{label} not found: {path}"
            )

    if not cfg.artifacts.enable:
        return {
            "status": "disabled",
            "report_path": None,
            "latest_report_path": None,
            "ready_for_cif_fob": False,
        }

    report_path = cfg.artifacts_path(
        run_id
    )
    latest_path = cfg.artifacts_latest_path()

    figure_dir = (
        get_figure_dir()
        / "quantity"
        / f"target_{cfg.scope.target_hs.lower()}"
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not force and valid_output(report_path):
        logger.info(
            "REUSE — Quantity report already exists: %s",
            report_path,
        )

        return {
            "status": "reused",
            "output_path": report_path,
            "report_path": report_path,
            "latest_report_path": (
                latest_path
                if latest_path.exists()
                else None
            ),
            "ready_for_cif_fob": True,
        }

    prepared_sql = _sql_literal(
        prepared_input_path
    )
    evidence_sql = _sql_literal(
        reconciliation_evidence_path
    )
    synchronized_sql = _sql_literal(
        synchronized_quantity_path
    )
    diagnostics_sql = _sql_literal(
        diagnostics_path
    )

    agreement_tolerance = float(
        cfg.method
        .kg_reconciliation
        .agreement
        .relative_tolerance
    )

    info = get_symbols("info")
    ok = get_symbols("check")

    logger.info(
        "%s Building Quantity decision report.",
        info,
    )
    logger.info(
        "  Prepared input: %s",
        prepared_input_path,
    )
    logger.info(
        "  Synchronized quantity: %s",
        synchronized_quantity_path,
    )
    logger.info(
        "  Diagnostics: %s",
        diagnostics_path,
    )
    logger.info(
        "  Report: %s",
        report_path,
    )
    logger.info(
        "  Figure directory: %s",
        figure_dir,
    )

    con = duckdb.connect()

    try:
        con.execute(
            f"SET threads = {int(cfg.engine.threads)}"
        )

        diagnostics = con.execute(
            f"""
            SELECT *
            FROM read_parquet({diagnostics_sql})
            """
        ).fetchdf()

        # ==============================================================
        # Core population statistics
        # ==============================================================

        population_row = con.execute(
            f"""
            SELECT
                COUNT(*) AS n_rows,

                COUNT(*) FILTER (
                    WHERE synchronized_quantity_kg > 0
                ) AS n_synchronized,

                COUNT(*) FILTER (
                    WHERE quantity_sync_status =
                        'unresolved_conflict'
                ) AS n_conflicts,

                COUNT(*) FILTER (
                    WHERE quantity_sync_status =
                        'no_kg_evidence'
                ) AS n_no_kg,

                COUNT(*) FILTER (
                    WHERE synchronized_quantity_source =
                        'gross_weight_proxy'
                ) AS n_gross_weight_proxy,

                COUNT(*) FILTER (
                    WHERE NOT has_quantity_kg
                      AND NOT has_net_weight_kg
                      AND has_gross_weight_kg
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

        synchronized_share = (
            n_synchronized / n_rows
            if n_rows
            else None
        )

        conflict_share = (
            n_conflicts / n_rows
            if n_rows
            else None
        )

        no_kg_share = (
            n_no_kg / n_rows
            if n_rows
            else None
        )

        gross_weight_proxy_share = (
            n_gross_weight_proxy / n_rows
            if n_rows
            else None
        )

        # ==============================================================
        # Quantity-field inventory
        # ==============================================================

        inventory = _diagnostic_group(
            diagnostics,
            "field_inventory",
        )

        if not inventory.empty:
            inventory_table = (
                inventory.pivot_table(
                    index="metric",
                    columns="category_1",
                    values=[
                        "value",
                        "percentage",
                    ],
                    aggfunc="first",
                )
            )

            rows = []

            for field_name in sorted(
                inventory["metric"].unique()
            ):
                field_frame = inventory.loc[
                    inventory["metric"].eq(
                        field_name
                    )
                ]

                def get_stat(
                    statistic: str,
                    column: str,
                ):
                    match = field_frame.loc[
                        field_frame[
                            "category_1"
                        ].eq(statistic)
                    ]

                    if match.empty:
                        return None

                    return match.iloc[0][column]

                rows.append(
                    {
                        "Field": field_name,
                        "Total rows": _format_integer(
                            get_stat(
                                "non_null_rows",
                                "value",
                            )
                        ),
                        "Total rows share (%)": _format_percent(
                            get_stat(
                                "non_null_rows",
                                "percentage",
                            ),
                            already_percent=True,
                        ),
                        "Positive rows": _format_integer(
                            get_stat(
                                "positive_rows",
                                "value",
                            )
                        ),
                        "Positive share": _format_percent(
                            get_stat(
                                "positive_rows",
                                "percentage",
                            ),
                            already_percent=True,
                        ),
                        "Zero rows": _format_integer(
                            get_stat(
                                "zero_rows",
                                "value",
                            )
                        ),
                        "Missing rows": _format_integer(
                            get_stat(
                                "missing_rows",
                                "value",
                            )
                        ),
                    }
                )

            inventory_display = pd.DataFrame(
                rows
            )

        else:
            inventory_display = pd.DataFrame()

        # ==============================================================
        # Unit-distribution tables
        # ==============================================================

        quantity_units = _diagnostic_group(
            diagnostics,
            "quantity_unit_distribution",
        )

        if not quantity_units.empty:
            quantity_units = quantity_units[
                [
                    "metric",
                    "category_1",
                    "value",
                    "percentage",
                ]
            ].copy()

            quantity_units.columns = [
                "Unit",
                "Unit code",
                "Rows",
                "Share (%)",
            ]

            quantity_units["Rows"] = (
                quantity_units["Rows"]
                .round()
                .astype(int)
            )

            quantity_units = (
                quantity_units.sort_values(
                    "Rows",
                    ascending=False,
                )
            )

        alt_quantity_units = _diagnostic_group(
            diagnostics,
            "alt_quantity_unit_distribution",
        )

        if not alt_quantity_units.empty:
            alt_quantity_units = (
                alt_quantity_units[
                    [
                        "metric",
                        "category_1",
                        "value",
                        "percentage",
                    ]
                ].copy()
            )

            alt_quantity_units.columns = [
                "Unit",
                "Unit code",
                "Rows",
                "Share (%)",
            ]

            alt_quantity_units["Rows"] = (
                alt_quantity_units["Rows"]
                .round()
                .astype(int)
            )

            alt_quantity_units = (
                alt_quantity_units.sort_values(
                    "Rows",
                    ascending=False,
                )
            )

        # ==============================================================
        # m³ HS-code tables
        # ==============================================================

        quantity_m3 = _diagnostic_group(
            diagnostics,
            "quantity_m3_by_hs",
        )

        if not quantity_m3.empty:
            quantity_m3 = (
                quantity_m3[
                    [
                        "metric",
                        "value",
                        "percentage",
                    ]
                ]
                .copy()
                .sort_values(
                    "value",
                    ascending=False,
                )
            )

            quantity_m3.columns = [
                "HS code",
                "Rows",
                "Share of m³ rows (%)",
            ]

            quantity_m3["Rows"] = (
                quantity_m3["Rows"]
                .round()
                .astype(int)
            )

        alt_quantity_m3 = _diagnostic_group(
            diagnostics,
            "alt_quantity_m3_by_hs",
        )

        if not alt_quantity_m3.empty:
            alt_quantity_m3 = (
                alt_quantity_m3[
                    [
                        "metric",
                        "value",
                        "percentage",
                    ]
                ]
                .copy()
                .sort_values(
                    "value",
                    ascending=False,
                )
            )

            alt_quantity_m3.columns = [
                "HS code",
                "Rows",
                "Share of m³ rows (%)",
            ]

            alt_quantity_m3["Rows"] = (
                alt_quantity_m3["Rows"]
                .round()
                .astype(int)
            )

        # ==============================================================
        # Reported / estimated overlap
        # ==============================================================

        evidence_state = _diagnostic_group(
            diagnostics,
            "quantity_net_weight_by_evidence_state",
        )

        if not evidence_state.empty:
            evidence_state_display = (
                evidence_state[
                    [
                        "metric",
                        "value",
                        "percentage",
                        "category_1",
                        "category_2",
                    ]
                ].copy()
            )

            evidence_state_display.columns = [
                "Evidence state",
                "Comparable rows",
                (
                    f"Within "
                    f"{100 * agreement_tolerance:.0f}%"
                ),
                "Median q/net-weight ratio",
                "Median relative gap",
            ]

            evidence_state_display[
                "Comparable rows"
            ] = (
                evidence_state_display[
                    "Comparable rows"
                ]
                .round()
                .astype(int)
            )

            evidence_state_display[
                "Median q/net-weight ratio"
            ] = pd.to_numeric(
                evidence_state_display[
                    "Median q/net-weight ratio"
                ],
                errors="coerce",
            )

            evidence_state_display[
                "Median relative gap"
            ] = pd.to_numeric(
                evidence_state_display[
                    "Median relative gap"
                ],
                errors="coerce",
            )

        else:
            evidence_state_display = pd.DataFrame()

        # ==============================================================
        # Agreement statistics
        # ==============================================================

        agreement_stats = _diagnostic_group(
            diagnostics,
            "quantity_net_weight_statistics",
        )

        agreement_lookup = {
            row.metric: row.value
            for row in agreement_stats.itertuples()
        }

        # ==============================================================
        # Hierarchy coverage
        # ==============================================================

        hierarchy = _diagnostic_group(
            diagnostics,
            "hierarchy_coverage",
        )

        if not hierarchy.empty:
            hierarchy["Step"] = pd.to_numeric(
                hierarchy["category_1"],
                errors="coerce",
            )
            hierarchy[
                "Cumulative rows"
            ] = pd.to_numeric(
                hierarchy["category_2"],
                errors="coerce",
            )
            hierarchy[
                "Incremental coverage (%)"
            ] = pd.to_numeric(
                hierarchy["category_3"],
                errors="coerce",
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
                "Source admitted",
                "New rows",
                "Cumulative rows",
                "Cumulative coverage (%)",
                "Incremental coverage (%)",
            ]

            hierarchy_display[
                "New rows"
            ] = (
                hierarchy_display[
                    "New rows"
                ]
                .round()
                .astype(int)
            )

            hierarchy_display[
                "Cumulative rows"
            ] = (
                hierarchy_display[
                    "Cumulative rows"
                ]
                .round()
                .astype(int)
            )

            hierarchy_display = (
                hierarchy_display.sort_values(
                    "Step"
                )
            )

        else:
            hierarchy_display = pd.DataFrame()

        # ==============================================================
        # Synchronization-status table
        # ==============================================================

        sync_status = _diagnostic_group(
            diagnostics,
            "synchronization_status",
        )

        if not sync_status.empty:
            sync_status_display = (
                sync_status[
                    [
                        "metric",
                        "category_1",
                        "value",
                        "percentage",
                    ]
                ].copy()
            )

            sync_status_display.columns = [
                "Status",
                "Final source",
                "Rows",
                "Share (%)",
            ]

            sync_status_display["Rows"] = (
                sync_status_display["Rows"]
                .round()
                .astype(int)
            )

            sync_status_display = (
                sync_status_display.sort_values(
                    "Rows",
                    ascending=False,
                )
            )

        else:
            sync_status_display = pd.DataFrame()

        # ==============================================================
        # Remaining kg gap
        # ==============================================================

        remaining_gap = _diagnostic_group(
            diagnostics,
            "remaining_kg_gap",
        )

        if not remaining_gap.empty:
            remaining_gap_display = (
                remaining_gap[
                    [
                        "metric",
                        "value",
                        "percentage",
                    ]
                ].copy()
            )

            remaining_gap_display.columns = [
                "Outcome",
                "Rows",
                "Share (%)",
            ]

            remaining_gap_display[
                "Rows"
            ] = (
                remaining_gap_display[
                    "Rows"
                ]
                .round()
                .astype(int)
            )

        else:
            remaining_gap_display = pd.DataFrame()

        # ==============================================================
        # Gross-weight diagnostics
        # ==============================================================

        gross_weight = _diagnostic_group(
            diagnostics,
            "gross_weight_intervention",
        )

        gross_lookup = {
            row.metric: {
                "value": row.value,
                "percentage": row.percentage,
            }
            for row in gross_weight.itertuples()
        }

        gross_available = gross_lookup.get(
            "gross_weight_available",
            {},
        ).get("value")

        gross_after_other_missing = gross_lookup.get(
            "gross_weight_available_after_other_sources_missing",
            {},
        ).get("value")

        gross_used = gross_lookup.get(
            "gross_weight_proxy_used",
            {},
        ).get("value")

        # ==============================================================
        # Figures
        # ==============================================================

        figure_paths = {
            "quantity_units": (
                figure_dir
                / "quantity_unit_distribution.png"
            ),
            "alt_quantity_units": (
                figure_dir
                / "alt_quantity_unit_distribution.png"
            ),
            "quantity_units_year": (
                figure_dir
                / "quantity_unit_share_by_year.png"
            ),
            "alt_quantity_units_year": (
                figure_dir
                / "alt_quantity_unit_share_by_year.png"
            ),
            "agreement_year": (
                figure_dir
                / "quantity_net_weight_agreement_by_year.png"
            ),
            "gap_distribution": (
                figure_dir
                / "quantity_net_weight_relative_gap_distribution.png"
            ),
            "hierarchy": (
                figure_dir
                / "kg_coverage_by_reconciliation_step.png"
            ),
            "sync_status_year": (
                figure_dir
                / "quantity_sync_status_by_year.png"
            ),
            "quantity_m3_hs": (
                figure_dir
                / "quantity_m3_by_hs_code.png"
            ),
            "alt_quantity_m3_hs": (
                figure_dir
                / "alt_quantity_m3_by_hs_code.png"
            ),
        }

        _plot_unit_distribution(
            diagnostics=diagnostics,
            diagnostic_group=(
                "quantity_unit_distribution"
            ),
            title="Quantity unit distribution",
            output_path=figure_paths[
                "quantity_units"
            ],
        )

        _plot_unit_distribution(
            diagnostics=diagnostics,
            diagnostic_group=(
                "alt_quantity_unit_distribution"
            ),
            title="Alternate quantity unit distribution",
            output_path=figure_paths[
                "alt_quantity_units"
            ],
        )

        _plot_unit_share_by_year(
            diagnostics=diagnostics,
            diagnostic_group=(
                "quantity_unit_by_year"
            ),
            title="Quantity unit composition by year",
            output_path=figure_paths[
                "quantity_units_year"
            ],
        )

        _plot_unit_share_by_year(
            diagnostics=diagnostics,
            diagnostic_group=(
                "alt_quantity_unit_by_year"
            ),
            title=(
                "Alternate quantity unit composition "
                "by year"
            ),
            output_path=figure_paths[
                "alt_quantity_units_year"
            ],
        )

        _plot_agreement_by_year(
            diagnostics=diagnostics,
            output_path=figure_paths[
                "agreement_year"
            ],
        )

        _plot_relative_gap_distribution(
            con=con,
            synchronized_sql=synchronized_sql,
            output_path=figure_paths[
                "gap_distribution"
            ],
        )

        _plot_hierarchy_coverage(
            diagnostics=diagnostics,
            output_path=figure_paths[
                "hierarchy"
            ],
        )

        _plot_synchronization_status_by_year(
            diagnostics=diagnostics,
            output_path=figure_paths[
                "sync_status_year"
            ],
        )

        _plot_m3_by_hs(
            diagnostics=diagnostics,
            diagnostic_group="quantity_m3_by_hs",
            title=(
                "HS codes with quantity reported "
                "in cubic metres"
            ),
            output_path=figure_paths[
                "quantity_m3_hs"
            ],
        )

        _plot_m3_by_hs(
            diagnostics=diagnostics,
            diagnostic_group=(
                "alt_quantity_m3_by_hs"
            ),
            title=(
                "HS codes with alternate quantity "
                "reported in cubic metres"
            ),
            output_path=figure_paths[
                "alt_quantity_m3_hs"
            ],
        )

        # ==============================================================
        # Decision logic
        # ==============================================================

        population_partition_passed = bool(
            n_rows
            == (
                n_synchronized
                + n_conflicts
                + n_no_kg
            )
        )

        synchronized_output_valid = bool(
            synchronized_quantity_path.is_file()
            and synchronized_quantity_path.stat().st_size > 0
        )

        ready_for_cif_fob = bool(
            synchronized_output_valid
            and population_partition_passed
        )

        decision = (
            "APPROVED"
            if ready_for_cif_fob
            else "REVIEW REQUIRED"
        )

        # ==============================================================
        # Interpretation text
        # ==============================================================

        if n_gross_weight_proxy > 0:
            gross_weight_statement = (
                f"Gross weight was used as a proxy for "
                f"`{n_gross_weight_proxy:,}` rows "
                f"({_format_percent(gross_weight_proxy_share)} "
                "of the full input). These rows remain explicitly "
                "identified through "
                "`synchronized_quantity_source = "
                "'gross_weight_proxy'`."
            )
        else:
            gross_weight_statement = (
                "Gross weight was not used in the synchronized "
                "kg quantity."
            )

        if (
            n_potential_gross_weight_fallback > 0
            and n_gross_weight_proxy == 0
        ):
            gross_weight_statement += (
                f" Gross weight nevertheless exists for "
                f"`{n_potential_gross_weight_fallback:,}` rows "
                "where neither quantity-in-kg nor net weight is "
                "available. These observations therefore quantify "
                "the maximum additional coverage that a future "
                "gross-weight fallback could provide."
            )

        # ==============================================================
        # Markdown report
        # ==============================================================

        lines = [
            "# Quantity Decision Report",
            "",
            f"**Decision:** {decision}  ",
            f"**Run ID:** {run_id}  ",
            f"**Target HS:** {cfg.scope.target_hs}  ",
            (
                f"**Input period:** "
                f"{cfg.scope.min_year}–"
                f"{cfg.scope.max_year}  "
            ),
            (
                f"**Generated:** "
                f"{datetime.now().isoformat(timespec='seconds')}"
            ),
            "",
            "## 1. Purpose of Quantity synchronization",
            "",
            (
                "The Quantity module receives the HS-synchronized "
                "Comtrade observations and evaluates the available "
                "physical-quantity information before downstream "
                "value processing."
            ),
            "",
            (
                "The principal objective is to create one transparent "
                "kg-based physical quantity while preserving the "
                "original quantity, alternate-quantity, net-weight, "
                "gross-weight, unit, and estimated-status fields."
            ),
            "",
            (
                "The module does not assume that all quantity fields "
                "are interchangeable. Quantity observations expressed "
                "in supplementary units such as cubic metres remain "
                "separate physical information. Gross weight is also "
                "treated separately because it includes packing and "
                "is not equivalent to net physical weight."
            ),
            "",
            "## 2. Input data and observation population",
            "",
            (
                f"Quantity received `{n_rows:,}` observations from "
                "the HS Sync output."
            ),
            "",
            "| Property | Value |",
            "|---|---|",
            "| Upstream module | HS Sync |",
            (
                f"| Target HS classification | "
                f"{cfg.scope.target_hs} |"
            ),
            (
                f"| Covered years | "
                f"{cfg.scope.min_year}–"
                f"{cfg.scope.max_year} |"
            ),
            (
                "| Canonical synchronized field | "
                "`synchronized_quantity_kg` |"
            ),
            (
                "| Agreement tolerance | "
                f"{100 * agreement_tolerance:.2f}% |"
            ),
            (
                "| Source-selection method | "
                f"`{cfg.method.kg_reconciliation.source_selection.method}` |"
            ),
            "",
            "### 2.1 Quantity-field inventory",
            "",
            _markdown_table(
                inventory_display
            ),
            "",
            (
                "This table describes the quantity information "
                "before synchronization. No original physical "
                "quantity field is overwritten by the Quantity module."
            ),
            "",
            "## 3. Physical-unit structure",
            "",
            "### 3.1 `quantity` unit distribution",
            "",
            _markdown_table(
                quantity_units
            ),
            "",
        ]

        if figure_paths[
            "quantity_units"
        ].exists():
            lines.extend(
                [
                    markdown_image(
                        "Quantity unit distribution",
                        figure_paths[
                            "quantity_units"
                        ],
                        document_path=report_path,
                        width=750,
                    ),
                    "",
                ]
            )

        lines.extend(
            [
                "### 3.2 `alt_quantity` unit distribution",
                "",
                _markdown_table(
                    alt_quantity_units
                ),
                "",
            ]
        )

        if figure_paths[
            "alt_quantity_units"
        ].exists():
            lines.extend(
                [
                    markdown_image(
                        "Alternate quantity unit distribution",
                        figure_paths[
                            "alt_quantity_units"
                        ],
                        document_path=report_path,
                        width=750,
                    ),
                    "",
                ]
            )

        lines.extend(
            [
                "### 3.3 Unit composition over time",
                "",
                (
                    "Unit composition is also evaluated by year so "
                    "that structural changes in reporting practice "
                    "are not hidden by whole-period totals."
                ),
                "",
            ]
        )

        for key, caption in (
            (
                "quantity_units_year",
                "Quantity units by year",
            ),
            (
                "alt_quantity_units_year",
                "Alternate quantity units by year",
            ),
        ):
            if figure_paths[key].exists():
                lines.extend(
                    [
                        markdown_image(
                            caption,
                            figure_paths[key],
                            document_path=report_path,
                            width=800,
                        ),
                        "",
                    ]
                )

        lines.extend(
            [
                "### 3.4 Cubic-metre observations by HS code",
                "",
                (
                    "Cubic-metre observations are shown by HS code "
                    "to establish which products generate volume-based "
                    "supplementary quantities."
                ),
                "",
                "**`quantity` reported in m³**",
                "",
                _markdown_table(
                    quantity_m3.head(30)
                    if not quantity_m3.empty
                    else quantity_m3
                ),
                "",
            ]
        )

        if figure_paths[
            "quantity_m3_hs"
        ].exists():
            lines.extend(
                [
                    markdown_image(
                        "Quantity m³ observations by HS code",
                        figure_paths[
                            "quantity_m3_hs"
                        ],
                        document_path=report_path,
                        width=750,
                    ),
                    "",
                ]
            )

        lines.extend(
            [
                "**`alt_quantity` reported in m³**",
                "",
                _markdown_table(
                    alt_quantity_m3.head(30)
                    if not alt_quantity_m3.empty
                    else alt_quantity_m3
                ),
                "",
            ]
        )

        if figure_paths[
            "alt_quantity_m3_hs"
        ].exists():
            lines.extend(
                [
                    markdown_image(
                        "Alternate quantity m³ observations by HS code",
                        figure_paths[
                            "alt_quantity_m3_hs"
                        ],
                        document_path=report_path,
                        width=750,
                    ),
                    "",
                ]
            )

        lines.extend(
            [
                "## 4. kg evidence",
                "",
                (
                    "Only quantity observations whose reported unit "
                    f"is `{cfg.method.kg_reconciliation.quantity_unit}` "
                    "participate directly in kg reconciliation. "
                    "`net_weight_kg` forms the second independent "
                    "kg-based source."
                ),
                "",
                "### 4.1 Reported and estimated evidence",
                "",
                _markdown_table(
                    evidence_state_display
                ),
                "",
                (
                    "Reported and estimated observations are retained "
                    "as separate evidence states. This makes it "
                    "possible to determine whether agreement differs "
                    "between genuinely reported and statistically "
                    "estimated physical quantities."
                ),
                "",
                "## 5. Quantity versus net-weight consistency",
                "",
                (
                    "For observations where both kg sources are usable, "
                    "the Quantity module compares `quantity` with "
                    "`net_weight_kg`."
                ),
                "",
                (
                    "The reconciliation evidence uses a symmetric "
                    "relative gap:"
                ),
                "",
                "$$",
                (
                    "d = "
                    "\\frac{|Q-W|}"
                    "{(|Q|+|W|)/2}"
                ),
                "$$",
                "",
                (
                    "where `Q` is quantity reported in kilograms and "
                    "`W` is net weight in kilograms. The symmetric "
                    "denominator avoids declaring either field to be "
                    "the reference truth before source selection."
                ),
                "",
                "| Statistic | Value |",
                "|---|---:|",
                (
                    "| Comparable observations | "
                    f"{_format_integer(agreement_lookup.get('n_comparable'))} |"
                ),
                (
                    "| Median quantity / net-weight ratio | "
                    f"{_format_number(agreement_lookup.get('median_ratio'))} |"
                ),
                (
                    "| 10th percentile ratio | "
                    f"{_format_number(agreement_lookup.get('p10_ratio'))} |"
                ),
                (
                    "| 90th percentile ratio | "
                    f"{_format_number(agreement_lookup.get('p90_ratio'))} |"
                ),
                (
                    "| Median relative gap | "
                    f"{_format_percent(agreement_lookup.get('median_relative_gap'))} |"
                ),
                (
                    "| 90th percentile relative gap | "
                    f"{_format_percent(agreement_lookup.get('p90_relative_gap'))} |"
                ),
                (
                    "| 95th percentile relative gap | "
                    f"{_format_percent(agreement_lookup.get('p95_relative_gap'))} |"
                ),
                (
                    "| Within 1% | "
                    f"{_format_percent(agreement_lookup.get('within_1pct_share'))} |"
                ),
                (
                    "| Within 5% | "
                    f"{_format_percent(agreement_lookup.get('within_5pct_share'))} |"
                ),
                (
                    "| Within 10% | "
                    f"{_format_percent(agreement_lookup.get('within_10pct_share'))} |"
                ),
                "",
            ]
        )

        for key, caption in (
            (
                "agreement_year",
                "Quantity and net-weight agreement by year",
            ),
            (
                "gap_distribution",
                "Quantity and net-weight relative-gap distribution",
            ),
        ):
            if figure_paths[key].exists():
                lines.extend(
                    [
                        markdown_image(
                            caption,
                            figure_paths[key],
                            document_path=report_path,
                            width=800,
                        ),
                        "",
                    ]
                )

        lines.extend(
            [
                "## 6. Source contribution and cumulative kg coverage",
                "",
                (
                    "The following table shows how many observations "
                    "are added by each synchronization outcome and how "
                    "overall kg coverage accumulates."
                ),
                "",
                _markdown_table(
                    hierarchy_display
                ),
                "",
            ]
        )

        if figure_paths["hierarchy"].exists():
            lines.extend(
                [
                    markdown_image(
                        "Cumulative kg coverage by synchronization step",
                        figure_paths["hierarchy"],
                        document_path=report_path,
                        width=800,
                    ),
                    "",
                ]
            )

        lines.extend(
            [
                "## 7. Gross-weight intervention",
                "",
                gross_weight_statement,
                "",
                "| Gross-weight diagnostic | Rows |",
                "|---|---:|",
                (
                    "| Gross weight available | "
                    f"{_format_integer(gross_available)} |"
                ),
                (
                    "| Gross weight available where quantity kg "
                    "and net weight are unavailable | "
                    f"{_format_integer(gross_after_other_missing)} |"
                ),
                (
                    "| Gross-weight proxy actually used | "
                    f"{_format_integer(gross_used)} |"
                ),
                "",
                (
                    f"Configuration setting "
                    f"`use_as_quantity_proxy = "
                    f"{cfg.method.gross_weight.use_as_quantity_proxy}`."
                ),
                "",
                (
                    "Any use of gross weight remains explicitly "
                    "identified through source provenance and is never "
                    "silently presented as a net physical quantity."
                ),
                "",
                "## 8. Synchronization outcome",
                "",
                _markdown_table(
                    sync_status_display
                ),
                "",
            ]
        )

        if figure_paths[
            "sync_status_year"
        ].exists():
            lines.extend(
                [
                    markdown_image(
                        "Quantity synchronization outcomes by year",
                        figure_paths[
                            "sync_status_year"
                        ],
                        document_path=report_path,
                        width=800,
                    ),
                    "",
                ]
            )

        lines.extend(
            [
                "### 8.1 Final coverage",
                "",
                (
                    f"`{n_synchronized:,}` of `{n_rows:,}` rows "
                    f"({_format_percent(synchronized_share)}) receive "
                    "a positive synchronized kg quantity."
                ),
                "",
                "## 9. Unresolved and non-kg observations",
                "",
                _markdown_table(
                    remaining_gap_display
                ),
                "",
                (
                    f"`{n_conflicts:,}` rows "
                    f"({_format_percent(conflict_share)}) contain "
                    "usable quantity and net-weight evidence that "
                    "disagree beyond the configured tolerance and "
                    "are therefore left unresolved."
                ),
                "",
                (
                    f"`{n_no_kg:,}` rows "
                    f"({_format_percent(no_kg_share)}) have no "
                    "resolved kg-based physical quantity."
                ),
                "",
                (
                    "A row without kg evidence is not automatically "
                    "a row without physical information. Supplementary "
                    "quantities such as cubic metres remain preserved "
                    "and are reported separately."
                ),
                "",
                "## 10. Methodological treatment",
                "",
                "| Input situation | Quantity treatment | Final status |",
                "|---|---|---|",
                (
                    "| Quantity kg and net weight both available and "
                    "within tolerance | Reconcile both sources | "
                    "`reconciled_agreement` |"
                ),
                (
                    "| Quantity kg only | Retain quantity kg | "
                    "`quantity_only` |"
                ),
                (
                    "| Net weight only | Retain net weight | "
                    "`net_weight_only` |"
                ),
                (
                    "| Both kg sources available but outside tolerance | "
                    "Leave unresolved | `unresolved_conflict` |"
                ),
                (
                    "| Neither kg source available | Preserve row and "
                    "supplementary quantities; no canonical kg value | "
                    "`no_kg_evidence` |"
                ),
                (
                    "| Gross weight only and gross-weight proxy enabled | "
                    "Use explicitly flagged gross-weight proxy | "
                    "`gross_weight_proxy` |"
                ),
                "",
                "## 11. Validation and decision",
                "",
                "| Validation | Result |",
                "|---|---|",
                (
                    "| Synchronized output exists and is non-empty | "
                    f"{'PASS' if synchronized_output_valid else 'FAIL'} |"
                ),
                (
                    "| Population accounting is complete | "
                    f"{'PASS' if population_partition_passed else 'FAIL'} |"
                ),
                (
                    "| Final Quantity decision | "
                    f"**{decision}** |"
                ),
                "",
                (
                    "Population accounting requires every input row "
                    "to fall into exactly one of three broad outcomes: "
                    "resolved kg quantity, unresolved kg conflict, or "
                    "no usable kg evidence."
                ),
                "",
                "## 12. Downstream recommendation",
                "",
            ]
        )

        if ready_for_cif_fob:
            lines.extend(
                [
                    (
                        f"{ok} The Quantity output is structurally "
                        "ready for downstream CIF/FOB processing."
                    ),
                    "",
                    (
                        "The downstream module should consume "
                        "`synchronized_quantity_kg` where a harmonized "
                        "kg-based quantity is required while retaining "
                        "the original physical-quantity fields and "
                        "provenance columns."
                    ),
                ]
            )
        else:
            lines.extend(
                [
                    (
                        "Quantity synchronization requires review before "
                        "being introduced into the downstream pipeline."
                    ),
                ]
            )

        lines.extend(
            [
                "",
                "## 13. Output provenance",
                "",
                "| Artifact | Path |",
                "|---|---|",
                (
                    "| Prepared input | "
                    f"`{prepared_input_path}` |"
                ),
                (
                    "| Reconciliation evidence | "
                    f"`{reconciliation_evidence_path}` |"
                ),
                (
                    "| Synchronized quantity | "
                    f"`{synchronized_quantity_path}` |"
                ),
                (
                    "| Diagnostics | "
                    f"`{diagnostics_path}` |"
                ),
                "",
            ]
        )

        report_text = "\n".join(lines)

        report_path.write_text(
            report_text,
            encoding="utf-8",
        )

        if cfg.artifacts.write_latest:
            shutil.copy2(
                report_path,
                latest_path,
            )

        logger.info(
            "%s Quantity decision report written: %s",
            ok,
            report_path,
        )

        if cfg.artifacts.write_latest:
            logger.info(
                "Latest Quantity report: %s",
                latest_path,
            )

    finally:
        con.close()

    return {
        "status": "completed",
        "run_id": run_id,
        "output_path": report_path,
        "report_path": report_path,
        "latest_report_path": (
            latest_path
            if cfg.artifacts.write_latest
            else None
        ),
        "decision": decision,
        "ready_for_cif_fob": ready_for_cif_fob,

        "n_rows": int(n_rows),
        "n_synchronized": int(
            n_synchronized
        ),
        "n_unresolved_conflict": int(
            n_conflicts
        ),
        "n_no_kg_evidence": int(
            n_no_kg
        ),
        "n_gross_weight_proxy": int(
            n_gross_weight_proxy
        ),
    }