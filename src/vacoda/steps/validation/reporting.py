"""
vacoda.steps.validation.reporting

Create the VACODA Validation decision and model-development report.

The report summarizes the evidence produced by the Validation workflow,
including:

- structural-integrity results;
- candidate training population;
- training-threshold sensitivity;
- joint threshold-grid evidence;
- temporal and product coverage;
- distributional evidence relevant to model development.

Scientific calculations belong in the producing Validation steps.
This module consumes their audit and diagnostics artifacts and turns them
into human-readable tables, figures, and methodological interpretation.
"""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any

import duckdb
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from vacoda.config.paths import get_figure_dir
from vacoda.config.steps.validation import ValidationConfig
from vacoda.utils.display import get_symbols
from vacoda.utils.markdown import (
    insert_markdown_toc,
    markdown_image,
)
from vacoda.utils.sql import sql_literal


# ======================================================================
# Helpers
# ======================================================================


def valid_output(
    path: Path,
) -> bool:
    """Return True when an existing output appears usable."""

    return (
        path.is_file()
        and path.stat().st_size > 0
    )


def _format_integer(
    value: float | int | None,
) -> str:
    """Format an integer-like value for the report."""

    if value is None or pd.isna(value):
        return "NA"

    return f"{int(round(float(value))):,}"


def _format_percent(
    value: float | None,
    *,
    digits: int = 2,
) -> str:
    """Format a fractional share as a percentage."""

    if value is None or pd.isna(value):
        return "NA"

    return (
        f"{100.0 * float(value):.{digits}f}%"
    )


def _format_number(
    value: float | int | None,
    *,
    digits: int = 4,
) -> str:
    """Format a general numeric value."""

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


def _scope_frame(
    diagnostics: pd.DataFrame,
    scope: str,
) -> pd.DataFrame:
    """Return one Validation diagnostic scope."""

    return diagnostics[
        diagnostics["scope"].eq(scope)
    ].copy()


# ======================================================================
# Figures
# ======================================================================


def _plot_training_population_funnel(
    *,
    diagnostics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot row retention through the candidate-training funnel."""

    frame = _scope_frame(
        diagnostics,
        "population_funnel",
    )

    if frame.empty:
        raise RuntimeError(
            "No population-funnel diagnostics are available."
        )

    order = [
        "all_rows",
        "trade_synced_quantity_available",
        "both_quantities_available",
        "both_quantities_non_estimated",
        "both_observed_with_disagreement",
        "both_observed_with_uv_distance",
        "complete_training_evidence",
    ]

    labels = {
        "all_rows": "All rows",
        "trade_synced_quantity_available": (
            "Trade-Synced quantity"
        ),
        "both_quantities_available": (
            "Bilateral quantity observations"
        ),
        "both_quantities_non_estimated": (
            "Direct bilateral observations"
        ),
        "both_observed_with_disagreement": (
            "Mirror disagreement available"
        ),
        "both_observed_with_uv_distance": (
            "UV distance available"
        ),
        "complete_training_evidence": (
            "All candidate-selection diagnostics"
        ),
    }

    frame["order"] = frame[
        "criterion"
    ].map(
        {
            criterion: index
            for index, criterion in enumerate(order)
        }
    )

    frame = (
        frame[
            frame["criterion"].isin(order)
        ]
        .sort_values("order")
        .reset_index(drop=True)
    )

    if frame.empty:
        raise RuntimeError(
            "Population-funnel diagnostics do not contain "
            "the expected criteria."
        )

    display_labels = [
        labels.get(
            criterion,
            criterion,
        )
        for criterion in frame["criterion"]
    ]

    figure, axis = plt.subplots(
        figsize=(11, 6)
    )

    bars = axis.bar(
        display_labels,
        frame["n_rows"],
    )

    axis.bar_label(
        bars,
        labels=[
            f"{int(value):,}"
            for value in frame["n_rows"]
        ],
        padding=3,
        fontsize=8,
    )

    axis.set_title(
        "Candidate quantity-training population"
    )
    axis.set_xlabel(
        "Evidence requirement"
    )
    axis.set_ylabel(
        "Number of observations"
    )
    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.5,
    )
    axis.set_axisbelow(True)

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def _plot_training_evidence_by_year(
    *,
    diagnostics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot directly observed bilateral quantity coverage by year."""

    frame = _scope_frame(
        diagnostics,
        "year_coverage",
    )

    if frame.empty:
        raise RuntimeError(
            "No year-coverage model diagnostics are available."
        )

    frame = frame.sort_values(
        "year"
    ).copy()

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    axis.plot(
        frame["year"],
        100.0 * frame["share_of_all"],
        marker="o",
    )

    axis.set_title(
        "Directly observed bilateral quantity evidence by year"
    )
    axis.set_xlabel(
        "Year"
    )
    axis.set_ylabel(
        "Share of annual Validation rows (%)"
    )
    axis.set_xticks(
        frame["year"].astype(int)
    )
    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.5,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def _plot_training_evidence_distributions(
    *,
    diagnostics: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot selected quantiles for the main pseudo-truth evidence measures.

    Values are displayed on their native scales. This figure is intended
    to reveal the shape of candidate-training evidence before final
    thresholds are selected.
    """

    frame = _scope_frame(
        diagnostics,
        "distribution",
    )

    if frame.empty:
        raise RuntimeError(
            "No distribution diagnostics are available."
        )

    variables = {
        "quantity_relative_difference": (
            "Mirror quantity disagreement"
        ),
        "synced_unit_value_log_distance": (
            "Trade-Synced UV distance"
        ),
    }

    frame = frame[
        frame["variable"].isin(
            variables
        )
    ].copy()

    if frame.empty:
        raise RuntimeError(
            "Expected model-development distribution variables "
            "are unavailable."
        )

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    for variable, group in frame.groupby(
        "variable",
        sort=False,
    ):
        group = group.sort_values(
            "quantile"
        )

        axis.plot(
            100.0 * group["quantile"],
            group["value"],
            marker="o",
            label=variables.get(
                variable,
                variable,
            ),
        )

    axis.set_title(
        "Distributional evidence in directly observed bilateral rows"
    )
    axis.set_xlabel(
        "Quantile (%)"
    )
    axis.set_ylabel(
        "Diagnostic value"
    )
    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.5,
    )
    axis.legend()

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# ======================================================================
# Main reporting function
# ======================================================================


def build_validation_report(
    *,
    cfg: ValidationConfig,
    prepared_input_path: Path,
    structural_audit_path: Path,
    diagnostics_path: Path,
    run_id: str,
    force: bool = False,
    logger,
) -> dict[str, Any]:
    """
    Create the current VACODA Validation Markdown decision report.

    At the present model-development stage, the report documents:

    - the canonical Validation input;
    - structural-integrity results;
    - candidate training population evidence;
    - sensitivity to possible training-sample thresholds;
    - coverage across years and products;
    - the current training-population decision status.

    The report will be extended as later Validation stages become active.
    """

    prepared_input_path = Path(prepared_input_path)
    structural_audit_path = Path(structural_audit_path)
    diagnostics_path = Path(diagnostics_path)

    for label, path in (
        (
            "prepared Validation input",
            prepared_input_path,
        ),
        (
            "Validation structural audit",
            structural_audit_path,
        ),
        (
            "Validation model diagnostics",
            diagnostics_path,
        ),
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"{label} not found: {path}"
            )

    # ------------------------------------------------------------------
    # Report configuration
    # ------------------------------------------------------------------

    if not cfg.artifacts.enable:
        return {
            "status": "disabled",
            "output_path": None,
            "report_path": None,
            "latest_report_path": None,
            "ready_for_training_decision": False,
        }

    report_path = cfg.artifacts_path(
        run_id
    )

    latest_path = (
        cfg.artifacts_latest_path()
    )

    figure_dir = (
        get_figure_dir()
        / "validation"
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

    # if (
    #     not force
    #     and valid_output(report_path)
    # ):
    #     logger.info(
    #         "REUSE — Validation report already exists: %s",
    #         report_path,
    #     )

    #     return {
    #         "status": "reused",
    #         "output_path": report_path,
    #         "report_path": report_path,
    #         "latest_report_path": (
    #             latest_path
    #             if latest_path.exists()
    #             else None
    #         ),
    #         "ready_for_training_decision": True,
    #     }

    structural_sql = sql_literal(
        structural_audit_path
    )

    diagnostics_sql = sql_literal(
        diagnostics_path
    )

    info = get_symbols(
        "info"
    )

    ok = get_symbols(
        "check"
    )

    logger.info(
        "%s Building Validation decision report.",
        info,
    )

    logger.info(
        "  Prepared input: %s",
        prepared_input_path,
    )

    logger.info(
        "  Structural audit: %s",
        structural_audit_path,
    )

    logger.info(
        "  Model diagnostics: %s",
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

        # ==============================================================
        # Read produced artifacts
        # ==============================================================

        structural_audit = con.execute(
            f"""
            SELECT *
            FROM read_parquet({structural_sql})
            """
        ).fetchdf()

        diagnostics = con.execute(
            f"""
            SELECT *
            FROM read_parquet({diagnostics_sql})
            """
        ).fetchdf()

        if len(structural_audit) != 1:
            raise RuntimeError(
                "Validation structural audit must contain "
                "exactly one row."
            )

        if diagnostics.empty:
            raise RuntimeError(
                "Validation model diagnostics contain no rows."
            )

        required_diagnostic_columns = {
            "scope",
            "criterion",
            "n_rows",
            "share_of_all",
            "share_of_parent",
        }

        missing_diagnostic_columns = (
            required_diagnostic_columns
            - set(diagnostics.columns)
        )

        if missing_diagnostic_columns:
            raise RuntimeError(
                "Validation model diagnostics are missing "
                "required columns: "
                + ", ".join(
                    sorted(
                        missing_diagnostic_columns
                    )
                )
            )

        # ==============================================================
        # Structural audit
        # ==============================================================

        audit_row = structural_audit.iloc[
            0
        ]

        audit_passed = bool(
            audit_row["audit_passed"]
        )

        n_rows = int(
            audit_row["n_rows"]
        )

        n_duplicate_rows = int(
            audit_row[
                "n_duplicate_rows"
            ]
        )

        n_null_key_rows = int(
            audit_row[
                "n_null_key_rows"
            ]
        )

        n_invalid_year_rows = int(
            audit_row[
                "n_invalid_year_rows"
            ]
        )

        n_invalid_hs_rows = int(
            audit_row[
                "n_invalid_hs_rows"
            ]
        )

        n_invalid_nominal_value_rows = int(
            audit_row[
                "n_invalid_nominal_value_rows"
            ]
        )

        n_invalid_constant_value_rows = int(
            audit_row[
                "n_invalid_constant_value_rows"
            ]
        )

        n_invalid_quantity_rows = int(
            audit_row[
                "n_invalid_quantity_rows"
            ]
        )

        n_missing_quantity_rows = int(
            audit_row[
                "n_missing_quantity_rows"
            ]
        )

        n_available_quantity_rows = int(
            audit_row[
                "n_available_quantity_rows"
            ]
        )

        structural_summary = pd.DataFrame(
            [
                {
                    "Check": "Rows",
                    "Result": _format_integer(
                        n_rows
                    ),
                },
                {
                    "Check": (
                        "Duplicate rows"
                    ),
                    "Result": _format_integer(
                        n_duplicate_rows
                    ),
                },
                {
                    "Check": "Null rows",
                    "Result": _format_integer(
                        n_null_key_rows
                    ),
                },
                {
                    "Check": "Invalid year rows",
                    "Result": _format_integer(
                        n_invalid_year_rows
                    ),
                },
                {
                    "Check": "Invalid HS rows",
                    "Result": _format_integer(
                        n_invalid_hs_rows
                    ),
                },
                {
                    "Check": "Invalid nominal values",
                    "Result": _format_integer(
                        n_invalid_nominal_value_rows
                    ),
                },
                {
                    "Check": (
                        "Invalid constant-USD values"
                    ),
                    "Result": _format_integer(
                        n_invalid_constant_value_rows
                    ),
                },
                {
                    "Check": (
                        "Invalid synchronized quantities"
                    ),
                    "Result": _format_integer(
                        n_invalid_quantity_rows
                    ),
                },
                {
                    "Check": (
                        "Missing synchronized quantities"
                    ),
                    "Result": _format_integer(
                        n_missing_quantity_rows
                    ),
                },
                {
                    "Check": "Structural audit passed",
                    "Result": str(
                        audit_passed
                    ),
                },
            ]
        )

        # ==============================================================
        # Training-population funnel
        # ==============================================================

        funnel = _scope_frame(
            diagnostics,
            "population_funnel",
        )

        funnel_order = [
            "all_rows",
            "trade_synced_quantity_available",
            "both_quantities_available",
            "both_quantities_non_estimated",
            "both_observed_with_disagreement",
            "both_observed_with_uv_distance",
            "complete_training_evidence",
        ]

        funnel_labels = {
            "all_rows": "All Validation rows",
            "trade_synced_quantity_available": (
                "Trade-Synced quantity available"
            ),
            "both_quantities_available": (
                "Bilateral quantity observation"
            ),
            "both_quantities_non_estimated": (
                "Direct bilateral quantity observations"
            ),
            "both_observed_with_disagreement": (
                "Direct bilateral observations with exporter–importer quantity difference available"
            ),
            "both_observed_with_uv_distance": (
                "Direct bilateral observations with unit-value distance available"
            ),
            "complete_training_evidence": (
                "Direct bilateral observations with all candidate-selection diagnostics available"
            ),
        }

        funnel["order"] = funnel[
            "criterion"
        ].map(
            {
                criterion: index
                for index, criterion
                in enumerate(funnel_order)
            }
        )

        funnel = (
            funnel[
                funnel["criterion"].isin(
                    funnel_order
                )
            ]
            .sort_values("order")
            .reset_index(drop=True)
        )

        funnel_display = pd.DataFrame(
            {
                "Population": funnel[
                    "criterion"
                ].map(
                    funnel_labels
                ),
                "Rows": funnel[
                    "n_rows"
                ].map(
                    _format_integer
                ),
                "Share of all": funnel[
                    "share_of_all"
                ].map(
                    _format_percent
                ),
                "Share of parent (previous group)": funnel[
                    "share_of_parent"
                ].map(
                    _format_percent
                ),
            }
        )

        # ==============================================================
        # Threshold sensitivity
        # ==============================================================

        sensitivity = _scope_frame(
            diagnostics,
            "threshold_sensitivity",
        )

        sensitivity_rows: list[
            dict[str, Any]
        ] = []

        for row in sensitivity.itertuples(
            index=False
        ):

            criterion = str(
                row.criterion
            )

            if (
                criterion
                == "quantity_relative_difference"
            ):
                threshold_label = (
                    f"≤ "
                    f"{100.0 * float(row.disagreement_threshold):.0f}%"
                )

                evidence_label = (
                    "Mirror quantity disagreement"
                )

            elif (
                criterion
                == "unit_value_reference_observations"
            ):
                threshold_label = (
                    f"≥ "
                    f"{int(row.uv_support_threshold):,}"
                )

                evidence_label = (
                    "Product-year UV support"
                )

            elif (
                criterion
                == "synced_unit_value_log_distance"
            ):
                threshold_label = (
                    f"≤ "
                    f"{float(row.uv_distance_threshold):g}"
                )

                evidence_label = (
                    "Trade-Synced UV distance"
                )

            else:
                threshold_label = "—"
                evidence_label = criterion

            sensitivity_rows.append(
                {
                    "Evidence": evidence_label,
                    "Threshold": threshold_label,
                    "Rows": _format_integer(
                        row.n_rows
                    ),
                    "Share of direct bilateral observations": (
                        _format_percent(
                            row.share_of_parent
                        )
                    ),
                }
            )

        sensitivity_display = pd.DataFrame(
            sensitivity_rows
        )

        # ==============================================================
        # Joint threshold grid
        # ==============================================================

        threshold_grid = _scope_frame(
            diagnostics,
            "threshold_grid",
        )

        threshold_grid = threshold_grid[
            [
                "disagreement_threshold",
                "uv_support_threshold",
                "uv_distance_threshold",
                "n_rows",
                "share_of_parent",
            ]
        ].copy()

        threshold_grid = threshold_grid.sort_values(
            [
                "disagreement_threshold",
                "uv_support_threshold",
                "uv_distance_threshold",
            ]
        )

        threshold_grid_display = pd.DataFrame(
            {
                "Maximum Mirror quantity disagreement": (
                    threshold_grid[
                        "disagreement_threshold"
                    ].map(
                        lambda value: (
                            f"≤ {100.0 * float(value):.0f}%"
                        )
                    )
                ),
                "Minimum Product-year UV support": (
                    threshold_grid[
                        "uv_support_threshold"
                    ].map(
                        lambda value: (
                            f"{int(value):,}"
                        )
                    )
                ),
                "Maximum Trade-Synced UV distance": (
                    threshold_grid[
                        "uv_distance_threshold"
                    ].map(
                        lambda value: (
                            f"{float(value):g}"
                        )
                    )
                ),
                "Candidate rows": (
                    threshold_grid[
                        "n_rows"
                    ].map(
                        _format_integer
                    )
                ),
                "Share of direct bilateral observations": (
                    threshold_grid[
                        "share_of_parent"
                    ].map(
                        _format_percent
                    )
                ),
            }
        )

        evidence_matrix = _scope_frame(
            diagnostics,
            "evidence_matrix",
        )

        evidence_matrix = evidence_matrix[
            [
                "disagreement_band",
                "uv_distance_band",
                "n_rows",
                "share_of_parent",
            ]
        ].copy()

        evidence_matrix = evidence_matrix.sort_values(
            [
                "disagreement_band",
                "uv_distance_band",
            ]
        )

        evidence_matrix_display = pd.DataFrame(
            {
                "Mirror agreement": (
                    evidence_matrix[
                        "disagreement_band"
                    ]
                ),
                "Bilateral UV plausibility": (
                    evidence_matrix[
                        "uv_distance_band"
                    ]
                ),
                "Rows": (
                    evidence_matrix[
                        "n_rows"
                    ].map(
                        _format_integer
                    )
                ),
                "Share of direct bilateral observations": (
                    evidence_matrix[
                        "share_of_parent"
                    ].map(
                        _format_percent
                    )
                ),
            }
        )

        # ==============================================================
        # Year coverage
        # ==============================================================

        year_coverage = _scope_frame(
            diagnostics,
            "year_coverage",
        ).sort_values(
            "year"
        )

        for column in (
            "median_quantity_difference_vs_import",
            "median_absolute_quantity_difference_vs_import",
            "exporter_higher_share",
        ):
            if column not in year_coverage.columns:
                year_coverage[column] = None

        year_display = pd.DataFrame(
            {
                "Year": (
                    year_coverage[
                        "year"
                    ].astype(
                        "Int64"
                    )
                ),
                "All rows": (
                    year_coverage[
                        "total_rows"
                    ].map(
                        _format_integer
                    )
                ),
                "Direct bilateral quantity rows": (
                    year_coverage[
                        "n_rows"
                    ].map(
                        _format_integer
                    )
                ),
                "Share": (
                    year_coverage[
                        "share_of_all"
                    ].map(
                        _format_percent
                    )
                ),
                "Median Mirror quantity disagreement": (
                    year_coverage[
                        "median_quantity_relative_difference"
                    ].map(
                        _format_percent
                    )
                ),
                "Median Trade-Synced UV distance": (
                    year_coverage[
                        "median_synced_uv_log_distance"
                    ].map(
                        lambda value: (
                            _format_number(
                                value,
                                digits=3,
                            )
                        )
                    )
                ),
                "Median Product-year UV support": (
                    year_coverage[
                        "median_uv_reference_observations"
                    ].map(
                        _format_integer
                    )
                ),

                "Median exporter difference vs importer": (
                    year_coverage[
                        "median_quantity_difference_vs_import"
                    ].map(
                        _format_percent
                    )
                ),

                "Median absolute difference vs importer": (
                    year_coverage[
                        "median_absolute_quantity_difference_vs_import"
                    ].map(
                        _format_percent
                    )
                ),

                "Exporter quantity higher": (
                    year_coverage[
                        "exporter_higher_share"
                    ].map(
                        _format_percent
                    )
                ),
            }
        )

        # ==============================================================
        # Product coverage summary
        # ==============================================================

        product_coverage = _scope_frame(
            diagnostics,
            "product_coverage",
        )

        n_products = int(
            len(
                product_coverage
            )
        )

        n_products_with_direct_evidence = int(
            (
                product_coverage["n_rows"]
                > 0
            ).sum()
        )

        median_product_direct_share = (
            float(
                product_coverage[
                    "share_of_all"
                ].median()
            )
            if not product_coverage.empty
            else None
        )

        product_rows = product_coverage[
            [
                "target_cmd",
                "total_rows",
                "n_rows",
                "share_of_all",
                "median_quantity_relative_difference",
                "median_synced_uv_log_distance",
            ]
        ].copy()

        product_rows = product_rows.sort_values(
            [
                "n_rows",
                "target_cmd",
            ],
            ascending=[
                False,
                True,
            ],
        )

        product_display = product_rows.head(
            20
        ).copy()

        product_display["total_rows"] = (
            product_display[
                "total_rows"
            ].map(
                _format_integer
            )
        )

        product_display["n_rows"] = (
            product_display[
                "n_rows"
            ].map(
                _format_integer
            )
        )

        product_display["share_of_all"] = (
            product_display[
                "share_of_all"
            ].map(
                _format_percent
            )
        )

        product_display[
            "median_quantity_relative_difference"
        ] = (
            product_display[
                "median_quantity_relative_difference"
            ].map(
                _format_percent
            )
        )

        product_display[
            "median_synced_uv_log_distance"
        ] = (
            product_display[
                "median_synced_uv_log_distance"
            ].map(
                lambda value: (
                    _format_number(
                        value,
                        digits=3,
                    )
                )
            )
        )

        product_display = product_display.rename(
            columns={
                "target_cmd": "HS code",
                "total_rows": "All rows",
                "n_rows": (
                    "Direct bilateral quantity rows"
                ),
                "share_of_all": "Direct share",
                "median_quantity_relative_difference": (
                    "Median Mirror quantity disagreement"
                ),
                "median_synced_uv_log_distance": (
                    "Median Trade-Synced UV distance"
                ),
            }
        )

        # ==============================================================
        # Distribution diagnostics
        # ==============================================================

        distribution = _scope_frame(
            diagnostics,
            "distribution",
        )

        distribution_labels = {
            "quantity_relative_difference": (
                "Mirror quantity disagreement"
            ),
            "quantity_difference_vs_import": (
                "Exporter quantity relative to importer"
            ),
            "quantity_absolute_difference_vs_import": (
                "Absolute exporter quantity difference relative to importer"
            ),
            "synced_unit_value_log_distance": (
                "Trade-Synced UV distance"
            ),
            "unit_value_reference_observations": (
                "Product-year UV support"
            ),
        }

        distribution_display = (
            distribution[
                [
                    "variable",
                    "quantile",
                    "value",
                ]
            ]
            .copy()
        )

        distribution_display[
            "variable"
        ] = (
            distribution_display[
                "variable"
            ].map(
                distribution_labels
            )
        )

        distribution_display[
            "quantile"
        ] = (
            distribution_display[
                "quantile"
            ].map(
                lambda value: (
                    f"P{int(round(100 * float(value)))}"
                )
            )
        )

        percentage_distribution_variables = {
            "Mirror quantity disagreement",
            "Exporter quantity relative to importer",
            "Absolute exporter quantity difference relative to importer",
        }

        distribution_display[
            "value"
        ] = (
            distribution_display.apply(
                lambda row: (
                    _format_percent(
                        row["value"]
                    )
                    if row["variable"]
                    in percentage_distribution_variables
                    else _format_number(
                        row["value"],
                        digits=3,
                    )
                ),
                axis=1,
            )
        )

        distribution_display = (
            distribution_display
            .pivot(
                index="variable",
                columns="quantile",
                values="value",
            )
            .reset_index()
            .rename(
                columns={
                    "variable": "Evidence"
                }
            )
        )

        # ==============================================================
        # Current population totals
        # ==============================================================

        funnel_lookup = {
            row.criterion: int(
                row.n_rows
            )
            for row in funnel.itertuples(
                index=False
            )
        }

        n_synced_quantity = funnel_lookup.get(
            "trade_synced_quantity_available",
            0,
        )

        n_both_quantities = funnel_lookup.get(
            "both_quantities_available",
            0,
        )

        n_both_observed = funnel_lookup.get(
            "both_quantities_non_estimated",
            0,
        )

        n_complete_training_evidence = (
            funnel_lookup.get(
                "complete_training_evidence",
                0,
            )
        )

        # ==============================================================
        # Figures
        # ==============================================================

        figure_paths = {
            "training_population_funnel": (
                figure_dir
                / (
                    f"target_{cfg.scope.target_hs.lower()}"
                    "_validation_training_population_funnel.png"
                )
            ),
            "training_evidence_by_year": (
                figure_dir
                / (
                    f"target_{cfg.scope.target_hs.lower()}"
                    "_validation_training_evidence_by_year.png"
                )
            ),
            "training_evidence_distributions": (
                figure_dir
                / (
                    f"target_{cfg.scope.target_hs.lower()}"
                    "_validation_training_evidence_distributions.png"
                )
            ),
        }

        _plot_training_population_funnel(
            diagnostics=diagnostics,
            output_path=figure_paths[
                "training_population_funnel"
            ],
        )

        _plot_training_evidence_by_year(
            diagnostics=diagnostics,
            output_path=figure_paths[
                "training_evidence_by_year"
            ],
        )

        _plot_training_evidence_distributions(
            diagnostics=diagnostics,
            output_path=figure_paths[
                "training_evidence_distributions"
            ],
        )

        # ==============================================================
        # Current methodological status
        # ==============================================================

        training_selection_enabled = bool(
            cfg.diagnostics.selection.enabled
        )

        if training_selection_enabled:
            training_status = (
                "A final candidate-training selection is configured."
            )
        else:
            training_status = (
                "No final training population has "
                "been selected yet. The current diagnostics are intended "
                "to support that decision before model fitting."
            )


        # ==============================================================
        # Mirror agreement × bilateral UV plausibility
        # ==============================================================

        evidence_matrix = _scope_frame(
            diagnostics,
            "evidence_matrix",
        )

        if evidence_matrix.empty:
            raise RuntimeError(
                "Validation evidence-matrix diagnostics are unavailable."
            )

        matrix_display = (
            evidence_matrix
            .pivot_table(
                index="disagreement_band",
                columns="uv_distance_band",
                values="n_rows",
                aggfunc="sum",
                fill_value=0,
            )
        )

        matrix_display = matrix_display.map(
            _format_integer
        )

        matrix_display.index.name = (
            "Mirror quantity disagreement"
        )


        # ==============================================================
        # Side-specific unit-value plausibility
        # ==============================================================

        side_plausibility = _scope_frame(
            diagnostics,
            "side_plausibility",
        )

        if side_plausibility.empty:
            raise RuntimeError(
                "Validation side-plausibility diagnostics are unavailable."
            )

        side_plausibility_display = (
            side_plausibility
            .pivot_table(
                index="disagreement_band",
                columns="variable",
                values="value",
                aggfunc="first",
            )
            .reset_index()
        )

        side_plausibility_display = (
            side_plausibility_display.rename(
                columns={
                    "disagreement_band": (
                        "Mirror quantity disagreement"
                    ),

                    "median_export_uv_distance": (
                        "Median Exporter UV distance"
                    ),

                    "median_import_uv_distance": (
                        "Median Importer UV distance"
                    ),

                    "median_bilateral_max_uv_distance": (
                        "Median Worst-side UV distance"
                    ),

                    "both_within_025_share": (
                        "Both within 0.25"
                    ),

                    "both_within_050_share": (
                        "Both within 0.50"
                    ),

                    "both_within_100_share": (
                        "Both within 1.00"
                    ),
                }
            )
        )

        for column in (
            "Both within 0.25",
            "Both within 0.50",
            "Both within 1.00",
        ):
            side_plausibility_display[
                column
            ] = side_plausibility_display[
                column
            ].map(
                _format_percent
            )

        # ==============================================================
        # Importer-referenced quantity comparison
        # ==============================================================

        importer_reference = _scope_frame(
            diagnostics,
            "importer_reference",
        )

        if importer_reference.empty:
            raise RuntimeError(
                "Importer-reference quantity diagnostics are unavailable."
            )

        importer_reference_lookup = {
            str(row.variable): (
                float(row.value)
                if pd.notna(row.value)
                else None
            )
            for row in importer_reference.itertuples(
                index=False
            )
        }

        n_importer_reference_rows = int(
            importer_reference["n_rows"].max()
        )

        mean_difference_vs_import = (
            importer_reference_lookup.get(
                "mean_quantity_difference_vs_import"
            )
        )

        median_difference_vs_import = (
            importer_reference_lookup.get(
                "median_quantity_difference_vs_import"
            )
        )

        median_absolute_difference_vs_import = (
            importer_reference_lookup.get(
                "median_absolute_quantity_difference_vs_import"
            )
        )

        exporter_higher_share = (
            importer_reference_lookup.get(
                "exporter_quantity_higher_share"
            )
        )

        importer_higher_share = (
            importer_reference_lookup.get(
                "importer_quantity_higher_share"
            )
        )

        exact_agreement_share = (
            importer_reference_lookup.get(
                "exact_quantity_agreement_share"
            )
        )

        importer_reference_display = pd.DataFrame(
            [
                {
                    "Measure": (
                        "Comparable directly observed bilateral rows"
                    ),
                    "Result": _format_integer(
                        n_importer_reference_rows
                    ),
                },
                {
                    "Measure": (
                        "Mean exporter difference relative to importer"
                    ),
                    "Result": _format_percent(
                        mean_difference_vs_import
                    ),
                },
                {
                    "Measure": (
                        "Median exporter difference relative to importer"
                    ),
                    "Result": _format_percent(
                        median_difference_vs_import
                    ),
                },
                {
                    "Measure": (
                        "Median absolute difference relative to importer"
                    ),
                    "Result": _format_percent(
                        median_absolute_difference_vs_import
                    ),
                },
                {
                    "Measure": (
                        "Exporter quantity larger than importer"
                    ),
                    "Result": _format_percent(
                        exporter_higher_share
                    ),
                },
                {
                    "Measure": (
                        "Importer quantity larger than exporter"
                    ),
                    "Result": _format_percent(
                        importer_higher_share
                    ),
                },
                {
                    "Measure": (
                        "Exact exporter-importer agreement"
                    ),
                    "Result": _format_percent(
                        exact_agreement_share
                    ),
                },
            ]
        )

        # ==============================================================
        # Report
        # ==============================================================

        lines = [
            (
                "# VACODA Validation Decision Report — "
                f"{cfg.scope.target_hs.upper()}"
            ),
            "",
            f"**Run ID:** `{run_id}`  ",
            (
                "**Pipeline scope:** "
                f"`{cfg.scope.min_year}–{cfg.scope.max_year}`  "
            ),
            (
                "**Structural audit:** "
                f"`{'PASS' if audit_passed else 'FAIL'}`  "
            ),
            (
                "**Model fitted:** `False`  "
            ),
            (
                "**Training population selected:** "
                f"`{training_selection_enabled}`"
            ),
            "",
            (
                "This report documents the current Validation state and "
                "the evidence used to develop the quantity-validation "
                "model. At this stage no machine-learning model has been "
                "fitted and no quantity observations have been corrected "
                "or imputed."
            ),
            "",
            "__TOC__",
            "",           
            "## 1. Validation objective",
            "",
            (
                "The final VACODA Validation workflow assesses whether "
                "the synchronized bilateral trade quantity is credible, "
                "identifies observations that require further attention, "
                "and ultimately determines whether missing or implausible "
                "quantities can be reconstructed with sufficient evidence."
            ),
            "",
            "",
            "## 2. Structural integrity audit",
            "",
            (
                f"- The prepared Validation dataset contains "
                f"`{_format_integer(n_rows)}` observations across "
                f"`{cfg.scope.min_year}–{cfg.scope.max_year}`."
            ),
            "",
            (
                f"- `{_format_integer(n_available_quantity_rows)}` rows "
                "contain a Trade-Synced quantity and "
                f"`{_format_integer(n_missing_quantity_rows)}` rows "
                "enter Validation without a usable synchronized quantity."
            ),
            "",
            _markdown_table(
                structural_summary
            ),
            "",
            (
                "The structural audit is deliberately limited to technical "
                "integrity: canonical grain, required fields, year and HS "
                "scope, numeric validity, and synchronization-method/status "
                "consistency. Economic plausibility is not decided here."
            ),
            "",
            "## 3. Quantity diagnostics",
            "",
            (
                "Quantity diagnostics examine the available quantity "
                "information before model development. They provide evidence "
                "on agreement between reporting partners, consistency with "
                "product-year unit values, and the amount of usable evidence "
                "available for candidate training selection."
            ),
            "",
            (
                "The following terminology is used throughout this section:"
            ),
            "",
            (
                "- **Mirror quantity disagreement** —  `|exporter quantity - importer quantity| / max(exporter quantity, importer quantity)`<br>"
                "&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; symmetric relative difference between exporter- and importer-reported quantities:<br>"
            ),
            (
                "&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; A value of `0%` means exact agreement. Larger values indicate "
                "greater disagreement. Neither reporter is treated as the "
                "reference."
            ),
            "",
            (
                "- **Unit value (UV)** — `trade value / quantity`, trade value per kilogram:<br>"
            ),
            "",
            (
                "- **Product-year UV reference** — a robust reference unit "
                "value for the same HS product and year."
            ),
            "",
            (
                "- **Product-year UV support** — the number of observations "
                "supporting the product-year UV reference."
            ),
            "",
            (
                "- **UV distance** — `|log(implied UV) − log(product-year UV reference)|`<br>"
                "&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; the distance between an observation's implied unit value and the product-year UV reference:<br>"
                "&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; A value close to `0` means that the implied unit value is "
                "close to the reference. Larger values indicate greater separation."
            ),
            "",
            (
                "- **Trade-Synced UV distance** — the UV distance calculated "
                "using the Trade-Synced quantity."
            ),
            "",
            (
                "- **Exporter UV distance** and **Importer UV distance** — "
                "the corresponding UV distances calculated separately using "
                "the exporter- and importer-reported quantities."
            ),
            "",
            (
                "- **Worst-side UV distance** — the larger of the exporter "
                "and importer UV distances. A small value therefore requires "
                "both reporting sides to be close to the product-year UV "
                "reference."
            ),
            "",
            "",
            "### 3.1. Candidate training population",
            "",
            (
                "The first diagnostic identifies the subset of Validation "
                "rows for which quantity information from both reporting "
                "sides is available and the diagnostics needed for later "
                "candidate selection can be calculated."
            ),
            "",
            (
                "The population is narrowed using the following terms and "
                "data requirements:"
            ),
            "",
            (
                "- **Trade-Synced quantity available** — a synchronized "
                "quantity produced by the earlier Trade Sync stage is "
                "available for the observation."
            ),
            "",
            (
                "- **Bilateral quantity observations** — both an exporter "
                "quantity and an importer quantity are available for the "
                "same bilateral trade observation. Either quantity may still "
                "be marked as estimated."
            ),
            "",
            (
                "- **Direct bilateral quantity observations** — both the "
                "exporter and importer quantities are directly reported "
                "rather than marked as estimated."
            ),
            "",
            (
                "- **Candidate-selection diagnostics** — the Mirror quantity "
                "disagreement and UV diagnostics required for later candidate "
                "training selection."
            ),
            "",
            "",
            (
                "The table below shows how the Validation population narrows "
                "as these quantities and diagnostics become available."
            ),
            "",
            _markdown_table(
                funnel_display
            ),
            "",
            markdown_image(
                "Candidate quantity-training population",
                figure_paths[
                    "training_population_funnel"
                ],
                document_path=report_path,
            ),
            "",
            (
                f"Of all Validation rows, "
                f"`{_format_percent(n_both_quantities / n_rows)}` contain "
                "both exporter and importer quantities. Among these bilateral "
                f"quantity observations, "
                f"`{_format_percent(n_both_observed / n_both_quantities)}` "
                "have both quantities directly reported rather than estimated."
            ),
            "",
            (
                f"This leaves `{_format_integer(n_both_observed)}` direct "
                "bilateral quantity observations, representing "
                f"`{_format_percent(n_both_observed / n_rows)}` of all "
                "Validation rows."
            ),
            "",
            (
                "The final three rows of the table do not yet apply quality "
                "thresholds. They only indicate whether the required "
                "candidate-selection diagnostics can be calculated. In the "
                f"current dataset, all `{_format_integer(n_both_observed)}` "
                "direct bilateral quantity observations have Mirror quantity "
                "disagreement and UV distance available."
            ),
            "",
            "### 3.2. Exporter quantity relative to importer",
            "",
            (
                "For bilateral rows where both exporter and importer "
                "quantities are directly observed, Validation also measures "
                "the exporter-reported quantity relative to the importer-"
                "reported quantity."
            ),
            "",
            (
                "- The importer quantity is used as the reference denominator:"
            ),
            "",
            (
                "&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;`(exporter quantity − importer quantity) / importer quantity`"
            ),
            "",
            (
                "- This measure retains direction. A positive value means "
                "the exporter reports a larger quantity than the importer; "
                "a negative value means the exporter reports a smaller "
                "quantity. For example, if the importer reports `100 kg` "
                "and the exporter reports `120 kg`, the result is `+20%`. "
                "If the exporter reports `80 kg`, the result is `−20%`."
            ),
            "",
            (
                "- This importer-referenced measure complements the symmetric "
                "exporter–importer quantity-difference measure used elsewhere "
                "in the diagnostics. The symmetric measure describes the "
                "magnitude of disagreement without choosing either reporter "
                "as the reference. The importer-referenced measure additionally "
                "shows the direction of that disagreement."
            ),
            "",
            _markdown_table(
                importer_reference_display
            ),
            "",
            (
                f"- The mean exporter difference relative to the importer is "
                f"`{_format_percent(mean_difference_vs_import)}`. This very "
                "large value indicates an extremely right-skewed distribution "
                "rather than a typical exporter quantity being this much "
                "larger than the importer quantity."
            ),
            "",
            (
                "- Because the importer quantity is the denominator, rows with "
                "a very small importer-reported quantity and a much larger "
                "exporter-reported quantity can generate extremely large "
                "positive ratios. A relatively small number of such observations "
                "can therefore have a very large influence on the arithmetic mean."
            ),
            "",
            (
                f"- The median signed difference is "
                f"`{_format_percent(median_difference_vs_import)}`. "
                "This indicates that the typical bilateral observation shows "
                "very little overall directional difference between exporter "
                "and importer quantities."
            ),
            "",
            (
                f"- However, the median absolute difference is "
                f"`{_format_percent(median_absolute_difference_vs_import)}`. "
                "This shows that exporter and importer quantities frequently "
                "differ substantially in magnitude even though the direction "
                "of those differences is broadly balanced."
            ),
            "",
            (
                f"- Exporter quantities are larger than importer quantities in "
                f"`{_format_percent(exporter_higher_share)}` of comparable "
                "rows, while importer quantities are larger in "
                f"`{_format_percent(importer_higher_share)}`. Exact agreement "
                f"occurs in `{_format_percent(exact_agreement_share)}` of rows."
            ),
            "",
            (
                "- The importer is used here as the comparison base, not as an "
                "assumption that every importer-reported quantity is correct. "
                "The purpose of this diagnostic is to distinguish three "
                "different properties of the bilateral quantity discrepancy: "
                "its direction, its typical magnitude, and the presence of "
                "extreme asymmetric observations."
            ),
            "",
            "### 3.3. Threshold sensitivity",
            "",
            (
                "No scientific threshold has yet been selected as the "
                "definition of a clean training observation. Instead, "
                "thresholds for Mirror quantity disagreement, Product-year "
                "UV support, and Trade-Synced UV distance are evaluated "
                "independently."
            ),
            "",
            _markdown_table(
                sensitivity_display
            ),
            "",
            (
                "These results show how restrictive each individual "
                "criterion would be within the directly observed bilateral "
                "quantity population. They should not yet be interpreted "
                "as alternative final training samples."
            ),
            "",
            "### 3.4. Joint candidate-training populations",
            "",
            (
                "The following grid applies the three candidate-selection "
                "diagnostics jointly. Each row represents one possible "
                "candidate training population:"
            ),
            "",
            (
                "`direct bilateral quantities` + "
                "`maximum Mirror quantity disagreement` + "
                "`minimum Product-year UV support` + "
                "`maximum Trade-Synced UV distance`."
            ),
            "",
            _markdown_table(
                threshold_grid_display
            ),
            "",
            (
                "The purpose of this grid is to expose the trade-off "
                "between purity and sample size. No combination becomes "
                "the production training definition merely because it has "
                "a large or small population."
            ),
            ""
            "### 3.5. Mirror quantity disagreement and bilateral UV plausibility"
            "",
            (
                "- Mirror quantity disagreement and bilateral UV distance "
                "provide complementary evidence about quantity credibility. "
                "The following matrix groups direct bilateral observations "
                "simultaneously by Mirror quantity disagreement and "
                "Worst-side UV distance."
            ),
            "",
            (
                "- Worst-side UV distance is the larger of the exporter and "
                "importer UV distances. A row therefore enters a low "
                "Worst-side UV-distance category only when both reporting "
                "sides imply unit values close to the Product-year UV "
                "reference."
            ),
            "",
            _markdown_table(
                matrix_display,
                index=True,
            ),
            "",
            (
                "The low-disagreement, low-unit-value-distance cells are "
                "the strongest current candidates for training population. "
                "Rows with strong mirror agreement but weak unit-value "
                "plausibility remain questionable because both reporters "
                "can agree on a quantity that is economically unusual."
            ),
            "",
            "### 3.6. Side-specific unit-value plausibility",
            "",
            (
                "The following table evaluates Exporter UV distance and "
                "Importer UV distance separately within each Mirror quantity "
                "disagreement band. Worst-side UV distance shows the less "
                "plausible of the two reporting sides."
            ),
            "",
            _markdown_table(
                side_plausibility_display
            ),
            "",
            "### 3.7. Distributional evidence",
            "",
            (
                "The following diagnostics describe the distribution of "
                "`Mirror quantity disagreement`, `Product-year UV support`, "
                "and `Trade-Synced UV distance` among directly observed "
                "bilateral quantity rows."
            ),
            "",
            _markdown_table(
                distribution_display
            ),
            "",
            markdown_image(
                "Quantity-model training evidence distributions",
                figure_paths[
                    "training_evidence_distributions"
                ],
                document_path=report_path,
            ),
            "",
            (
                "The synchronized unit-value distance is calculated using "
                "nominal reconciled trade value because the product-year "
                "unit-value reference is defined on the same nominal "
                "within-year basis. Constant-price trade value remains "
                "available for later cross-year predictive modelling."
            ),
            "",
            "## 4. Diagnostic Coverage",
            "",
            (
                "The following diagnostics describe the temporal and "
                "product coverage of the available quantity evidence. "
                "They are intended to support the selection of a "
                "defensible training population that is large enough, "
                "sufficiently clean, and adequately represented across "
                "years and products."
            ),
            "",
            "### 4.1. Temporal coverage",
            "",
            (
                "A candidate training population must provide "
                "useful evidence across the full time horizon rather than "
                "being concentrated in only a few years."
            ),
            "",
            _markdown_table(
                year_display
            ),
            "",
            markdown_image(
                "Direct bilateral quantity evidence by year",
                figure_paths[
                    "training_evidence_by_year"
                ],
                document_path=report_path,
            ),
            "",
            "### 4.2. Product coverage",
            "",
            (
                f"- The current diagnostics contain `{_format_integer(n_products)}` "
                "HS products. "
                f"`{_format_integer(n_products_with_direct_evidence)}` "
                "have at least one directly observed bilateral quantity "
                "observation."
            ),
            "",
            (
                "- The median within-product share represented by directly "
                "observed bilateral quantity evidence is "
                f"`{_format_percent(median_product_direct_share)}`."
            ),
            "",
            (
                "- The following table shows the 20 products with the largest "
                "numbers of directly observed bilateral quantity rows. "
                "This is a coverage diagnostic, not a ranking of product "
                "quality."
            ),
            "",
            _markdown_table(
                product_display
            ),
            "",
            "## 5. Current methodological assessment",
            "",
            training_status,
            "",
            (
                "- The next scientific decision is therefore not yet the "
                "choice of machine-learning algorithm. It is the selection "
                "of a defensible training population that is "
                "large enough, sufficiently clean, and adequately "
                "represented across years and products."
            ),
            "",
            (
                "- Once that population is selected, model performance should "
                "be tested using controlled missingness and corruption of "
                "high-confidence observations rather than by assuming that "
                "all synchronized quantities are true."
            ),
            "",
            "## 6. Output artifacts",
            "",
            (
                f"- Prepared Validation input: "
                f"`{prepared_input_path}`"
            ),
            (
                f"- Structural audit: "
                f"`{structural_audit_path}`"
            ),
            (
                f"- Diagnostics: "
                f"`{diagnostics_path}`"
            ),
            f"- Report: `{report_path}`",
            "",
        ]

        insert_markdown_toc(
            lines,
            min_level=2,
            max_level=3,
            flat_sublevels=True,
        )

        report_path.write_text(
            "\n".join(
                lines
            ),
            encoding="utf-8",
        )

        if not valid_output(
            report_path
        ):
            raise RuntimeError(
                "Validation decision report was not written correctly."
            )

        latest_report_path: Path | None = None

        if cfg.artifacts.write_latest:
            latest_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            shutil.copyfile(
                report_path,
                latest_path,
            )

            latest_report_path = (
                latest_path
            )

        logger.info(
            "%s Validation decision report created.",
            ok,
        )

        logger.info(
            "  Structural audit passed: %s",
            audit_passed,
        )

        logger.info(
            "  Training population selected: %s",
            training_selection_enabled,
        )

        logger.info(
            "  Report: %s",
            report_path,
        )

        if latest_report_path is not None:
            logger.info(
                "  Latest report: %s",
                latest_report_path,
            )

        logger.info(
            "  Figures written: %d",
            len(
                figure_paths
            ),
        )

        return {
            "status": "completed",
            "run_id": run_id,

            "output_path": report_path,
            "report_path": report_path,

            "latest_report_path": (
                latest_report_path
            ),

            "figure_paths": (
                figure_paths
            ),

            "audit_passed": (
                audit_passed
            ),

            "training_population_selected": (
                training_selection_enabled
            ),

            "ready_for_training_decision": True,

            "duration_seconds": None,
        }

    finally:
        con.close()