"""
vacoda.steps.validation.reporting

Create the VACODA Validation model-development report.

The report follows the canonical bilateral Validation methodology:

1. structural integrity of the Validation input;
2. selected canonical bilateral quantity methodology;
3. source-reporter diagnostics, shown descriptively only;
4. HS × year unit-value reference evidence;
5. configured model-training population;
6. expanding-window model evaluation;
7. placeholder for later production inference diagnostics.

Scientific calculations belong in the producing Validation steps.  This module
consumes structural-audit and diagnostics artifacts and turns them into concise
human-readable tables, figures, and methodological interpretation.
"""

from __future__ import annotations

from pathlib import Path
import math
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
from vacoda.utils.markdown import insert_markdown_toc, markdown_image
from vacoda.utils.sql import sql_literal


# ======================================================================
# Helpers
# ======================================================================


def valid_output(path: Path) -> bool:
    """Return True when an existing output appears usable."""

    return path.is_file() and path.stat().st_size > 0



def _format_integer(value: float | int | None) -> str:
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

    return f"{100.0 * float(value):.{digits}f}%"



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



def _candidate_label(value: str) -> str:
    """Return a readable label for the selected quantity methodology."""

    labels = {
        "simple_mean": "Simple mean",
        "value_weighted": "Value-weighted",
        "quantity_reliability_weighted": "Quantity-reliability-weighted",
    }

    return labels.get(str(value), str(value))



def _require_diagnostic_scope(
    diagnostics: pd.DataFrame,
    scope: str,
) -> pd.DataFrame:
    """Return a required scope or raise a clear error."""

    frame = _scope_frame(
        diagnostics,
        scope,
    )

    if frame.empty:
        raise RuntimeError(
            f"Validation diagnostics do not contain required scope: {scope}"
        )

    return frame


# ======================================================================
# Figures
# ======================================================================


def _plot_source_quantity_availability(
    *,
    diagnostics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot source-reporter quantity availability."""

    frame = _require_diagnostic_scope(
        diagnostics,
        "source_quantity_availability",
    )

    order = (
        "both",
        "export_only",
        "import_only",
        "neither",
    )

    labels = {
        "both": "Both sides",
        "export_only": "Exporter only",
        "import_only": "Importer only",
        "neither": "Neither side",
    }

    frame = frame[
        frame["criterion"].isin(order)
    ].copy()

    frame["order"] = frame["criterion"].map(
        {value: index for index, value in enumerate(order)}
    )

    frame = frame.sort_values("order")

    figure, axis = plt.subplots(
        figsize=(9, 5.5)
    )

    bars = axis.bar(
        [labels[value] for value in frame["criterion"]],
        100.0 * pd.to_numeric(frame["share_of_all"], errors="coerce"),
    )

    axis.bar_label(
        bars,
        labels=[
            _format_percent(value)
            for value in frame["share_of_all"]
        ],
        padding=3,
        fontsize=8,
    )

    axis.set_title(
        "Source-reporter quantity availability"
    )
    axis.set_ylabel(
        "Share of canonical bilateral rows (%)"
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
    plt.close(figure)



def _plot_source_disagreement_bands(
    *,
    diagnostics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot descriptive exporter/importer quantity disagreement bands."""

    frame = _require_diagnostic_scope(
        diagnostics,
        "source_quantity_disagreement_band",
    )

    order = (
        "0-5%",
        "5-10%",
        "10-25%",
        "25-50%",
        "50-100%",
    )

    frame = frame[
        frame["disagreement_band"].isin(order)
    ].copy()

    frame["order"] = frame["disagreement_band"].map(
        {value: index for index, value in enumerate(order)}
    )

    frame = frame.sort_values("order")

    figure, axis = plt.subplots(
        figsize=(9, 5.5)
    )

    bars = axis.bar(
        frame["disagreement_band"],
        100.0 * pd.to_numeric(frame["share_of_parent"], errors="coerce"),
    )

    axis.bar_label(
        bars,
        labels=[
            _format_percent(value)
            for value in frame["share_of_parent"]
        ],
        padding=3,
        fontsize=8,
    )

    axis.set_title(
        "Exporter/importer quantity disagreement — source diagnostic"
    )
    axis.set_xlabel(
        "Relative disagreement band"
    )
    axis.set_ylabel(
        "Share of rows with both source quantities (%)"
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
    plt.close(figure)



def _plot_training_population_funnel(
    *,
    diagnostics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot row retention through the configured training rule."""

    frame = _require_diagnostic_scope(
        diagnostics,
        "training_population_funnel",
    )

    order = (
        "all_rows",
        "training_eligible",
        "support",
        "positive_uv_mad",
        "uv_distance_available",
        "training_selected",
    )

    labels = {
        "all_rows": "All canonical rows",
        "training_eligible": "Usable quantity + value",
        "support": "HS×year support threshold",
        "positive_uv_mad": "Positive UV MAD",
        "uv_distance_available": "Normalized UV evaluable",
        "training_selected": "Final training sample",
    }

    frame = frame[
        frame["criterion"].isin(order)
    ].copy()

    frame["order"] = frame["criterion"].map(
        {value: index for index, value in enumerate(order)}
    )

    frame = frame.sort_values("order")

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    bars = axis.barh(
        [labels[value] for value in frame["criterion"]],
        pd.to_numeric(frame["n_rows"], errors="coerce"),
    )

    axis.bar_label(
        bars,
        labels=[
            _format_integer(value)
            for value in frame["n_rows"]
        ],
        padding=4,
        fontsize=8,
    )

    axis.invert_yaxis()
    axis.set_title(
        "Configured model-training population"
    )
    axis.set_xlabel(
        "Number of canonical bilateral observations"
    )
    axis.grid(
        axis="x",
        linestyle="--",
        alpha=0.5,
    )
    axis.set_axisbelow(True)
    axis.margins(x=0.12)
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)



def _plot_normalized_uv_cutoff_sensitivity(
    *,
    diagnostics: pd.DataFrame,
    output_path: Path,
    configured_cutoff: float,
) -> None:
    """Plot training retention across normalized-UV cutoffs."""

    frame = _require_diagnostic_scope(
        diagnostics,
        "training_normalized_uv_cutoff_sensitivity",
    ).sort_values(
        "uv_distance_threshold"
    )

    x = pd.to_numeric(
        frame["uv_distance_threshold"],
        errors="coerce",
    )

    y = 100.0 * pd.to_numeric(
        frame["share_of_parent"],
        errors="coerce",
    )

    figure, axis = plt.subplots(
        figsize=(9, 5.5)
    )

    axis.plot(
        x,
        y,
        marker="o",
    )

    axis.axvline(
        float(configured_cutoff),
        linestyle="--",
        linewidth=1.4,
        label=f"Configured cutoff = {configured_cutoff:g} MAD",
    )

    axis.set_title(
        "Normalized unit-value cutoff sensitivity"
    )
    axis.set_xlabel(
        "Maximum normalized UV distance (MAD units)"
    )
    axis.set_ylabel(
        "Retained share of UV-evaluable rows (%)"
    )
    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.5,
    )
    axis.set_axisbelow(True)
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)



def _plot_training_coverage_by_year(
    *,
    diagnostics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot final training-sample coverage by year."""

    frame = _require_diagnostic_scope(
        diagnostics,
        "training_coverage_by_year",
    ).sort_values("year")

    figure, axis = plt.subplots(
        figsize=(10, 5.5)
    )

    axis.plot(
        pd.to_numeric(frame["year"], errors="coerce"),
        100.0 * pd.to_numeric(frame["share_of_all"], errors="coerce"),
        marker="o",
    )

    axis.set_title(
        "Final training-sample coverage by year"
    )
    axis.set_xlabel("Year")
    axis.set_ylabel(
        "Training-selected share of annual Validation rows (%)"
    )
    axis.set_xticks(
        sorted(
            frame["year"].dropna().astype(int).unique()
        )
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
    plt.close(figure)



def _plot_model_performance_by_year(
    *,
    diagnostics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot expanding-window MAE by candidate model and validation year."""

    frame = _require_diagnostic_scope(
        diagnostics,
        "model_performance_by_year",
    )

    frame = frame[
        frame["variable"].eq("mae_log")
    ].copy()

    if frame.empty:
        raise RuntimeError(
            "Model performance-by-year diagnostics contain no mae_log rows."
        )

    figure, axis = plt.subplots(
        figsize=(10, 5.5)
    )

    for model_name, group in frame.groupby(
        "model_name",
        sort=False,
    ):
        group = group.sort_values("year")
        axis.plot(
            pd.to_numeric(group["year"], errors="coerce"),
            pd.to_numeric(group["value"], errors="coerce"),
            marker="o",
            label=str(model_name),
        )

    axis.set_title(
        "Expanding-window model performance by validation year"
    )
    axis.set_xlabel(
        "Validation year"
    )
    axis.set_ylabel(
        "MAE on log quantity"
    )
    axis.set_xticks(
        sorted(
            frame["year"].dropna().astype(int).unique()
        )
    )
    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.5,
    )
    axis.set_axisbelow(True)
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)



def _build_validation_figure_paths(
    *,
    figure_dir: Path,
    target_hs: str,
) -> dict[str, Path]:
    """Return figure paths used by the streamlined Validation report."""

    target_hs = str(target_hs).lower()
    prefix = f"target_{target_hs}_validation_"

    return {
        "source_quantity_availability": (
            figure_dir / f"{prefix}source_quantity_availability.png"
        ),
        "source_quantity_disagreement": (
            figure_dir / f"{prefix}source_quantity_disagreement.png"
        ),
        "training_population_funnel": (
            figure_dir / f"{prefix}training_population_funnel.png"
        ),
        "normalized_uv_cutoff_sensitivity": (
            figure_dir / f"{prefix}normalized_uv_cutoff_sensitivity.png"
        ),
        "training_coverage_by_year": (
            figure_dir / f"{prefix}training_coverage_by_year.png"
        ),
        "model_performance_by_year": (
            figure_dir / f"{prefix}model_performance_by_year.png"
        ),
    }


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
    Create the VACODA Validation Markdown model-development report.

    The report documents the canonical bilateral quantity methodology,
    descriptive source-reporter diagnostics, the configured HS × year UV
    training-sample rule, training coverage, and expanding-window model
    evaluation.  Production inference remains a later Validation stage.
    """

    prepared_input_path = Path(prepared_input_path)
    structural_audit_path = Path(structural_audit_path)
    diagnostics_path = Path(diagnostics_path)

    for label, path in (
        ("prepared Validation input", prepared_input_path),
        ("Validation structural audit", structural_audit_path),
        ("Validation diagnostics", diagnostics_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"{label} not found: {path}"
            )

    if not cfg.artifacts.enable:
        return {
            "status": "disabled",
            "output_path": None,
            "report_path": None,
            "latest_report_path": None,
            "ready_for_training_decision": False,
        }

    report_path = cfg.artifacts_path(run_id)
    latest_path = cfg.artifacts_latest_path()

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

    structural_sql = sql_literal(
        structural_audit_path
    )
    diagnostics_sql = sql_literal(
        diagnostics_path
    )

    info = get_symbols("info")
    ok = get_symbols("check")

    logger.info(
        "%s Building Validation model-development report.",
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
        "  Diagnostics: %s",
        diagnostics_path,
    )

    con = duckdb.connect()

    try:
        con.execute(
            f"SET threads = {int(cfg.engine.threads)}"
        )

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
                "Validation structural audit must contain exactly one row."
            )

        if diagnostics.empty:
            raise RuntimeError(
                "Validation diagnostics contain no rows."
            )

        required_diagnostic_columns = {
            "scope",
            "criterion",
            "n_rows",
            "share_of_all",
            "share_of_parent",
            "value",
        }

        missing = (
            required_diagnostic_columns
            - set(diagnostics.columns)
        )

        if missing:
            raise RuntimeError(
                "Validation diagnostics are missing required columns: "
                + ", ".join(sorted(missing))
            )

        # ============================================================== 
        # Structural integrity
        # ============================================================== 
        audit_row = structural_audit.iloc[0]

        if "audit_passed" not in structural_audit.columns:
            raise RuntimeError(
                "Validation structural audit is missing audit_passed."
            )

        audit_passed = bool(
            audit_row["audit_passed"]
        )

        structural_fields = (
            ("n_rows", "Rows"),
            ("n_duplicate_rows", "Duplicate rows"),
            ("n_null_key_rows", "Null canonical-key rows"),
            ("n_invalid_year_rows", "Invalid year rows"),
            ("n_invalid_hs_rows", "Invalid HS rows"),
            ("n_invalid_nominal_value_rows", "Invalid nominal-value rows"),
            (
                "n_invalid_constant_value_rows",
                "Invalid constant-USD value rows",
            ),
            (
                "n_invalid_value_weighted_rows",
                "Invalid value-weighted quantity rows",
            ),
            ("n_missing_quantity_rows", "Rows without quantity"),
            ("n_available_quantity_rows", "Rows with quantity"),
        )

        structural_rows = []

        for column, label in structural_fields:
            if column in structural_audit.columns:
                structural_rows.append(
                    {
                        "Check": label,
                        "Result": _format_integer(
                            audit_row[column]
                        ),
                    }
                )

        structural_rows.append(
            {
                "Check": "Structural audit passed",
                "Result": str(audit_passed),
            }
        )

        structural_summary = pd.DataFrame(
            structural_rows
        )

        # ============================================================== 
        # Canonical quantity methodology
        # ============================================================== 
        methodology = _require_diagnostic_scope(
            diagnostics,
            "quantity_methodology",
        )

        selected_candidates = (
            methodology["candidate"]
            .dropna()
            .astype(str)
            .unique()
        )

        if len(selected_candidates) != 1:
            raise RuntimeError(
                "Validation report requires exactly one canonical "
                "quantity methodology."
            )

        selected_candidate = str(
            selected_candidates[0]
        )
        selected_candidate_label = _candidate_label(
            selected_candidate
        )

        # ============================================================== 
        # Source diagnostics
        # ============================================================== 
        source_availability = _require_diagnostic_scope(
            diagnostics,
            "source_quantity_availability",
        )

        source_labels = {
            "both": "Both exporter and importer quantities available",
            "export_only": "Exporter quantity only",
            "import_only": "Importer quantity only",
            "neither": "Neither source quantity available",
            "both_non_estimated": "Both source quantities non-estimated",
        }

        source_availability_display = pd.DataFrame(
            [
                {
                    "Source evidence": source_labels.get(
                        str(row.criterion),
                        str(row.criterion),
                    ),
                    "Rows": _format_integer(row.n_rows),
                    "Share of all rows": _format_percent(
                        row.share_of_all
                    ),
                }
                for row in source_availability.itertuples(
                    index=False
                )
            ]
        )

        source_disagreement = _require_diagnostic_scope(
            diagnostics,
            "source_quantity_disagreement_distribution",
        ).sort_values("quantile")

        source_disagreement_display = pd.DataFrame(
            {
                "Quantile": [
                    f"P{100.0 * float(value):g}"
                    for value in source_disagreement["quantile"]
                ],
                "Relative exporter/importer disagreement": [
                    _format_percent(value)
                    for value in source_disagreement["value"]
                ],
            }
        )

        # ============================================================== 
        # Training rule and coverage
        # ============================================================== 
        training_funnel = _require_diagnostic_scope(
            diagnostics,
            "training_population_funnel",
        )

        funnel_labels = {
            "all_rows": "All canonical rows",
            "training_eligible": "Technically usable quantity/value",
            "support": "HS×year UV support threshold satisfied",
            "positive_uv_mad": "Positive finite HS×year UV MAD",
            "uv_distance_available": "Normalized UV distance evaluable",
            "training_selected": "Final training sample",
        }

        training_funnel_display = pd.DataFrame(
            [
                {
                    "Stage": funnel_labels.get(
                        str(row.criterion),
                        str(row.criterion),
                    ),
                    "Rows": _format_integer(row.n_rows),
                    "Share of all rows": _format_percent(
                        row.share_of_all
                    ),
                    "Retention from previous stage": _format_percent(
                        row.share_of_parent
                    ),
                }
                for row in training_funnel.itertuples(
                    index=False
                )
            ]
        )

        training_definition = _require_diagnostic_scope(
            diagnostics,
            "training_definition",
        )

        if len(training_definition) != 1:
            raise RuntimeError(
                "Validation diagnostics must contain exactly one "
                "training_definition row."
            )

        training_definition_row = training_definition.iloc[0]
        min_reference_observations = int(
            round(
                float(
                    training_definition_row[
                        "uv_support_threshold"
                    ]
                )
            )
        )
        max_normalized_uv_distance_mad = float(
            training_definition_row[
                "uv_distance_threshold"
            ]
        )
        n_training_selected = int(
            training_definition_row["n_rows"]
        )
        training_selected_share = float(
            training_definition_row["share_of_all"]
        )

        training_rule_display = pd.DataFrame(
            [
                {
                    "Criterion": "Minimum HS×year UV reference support",
                    "Configured value": (
                        f"≥ {min_reference_observations:,} observations"
                    ),
                },
                {
                    "Criterion": "HS×year log-UV dispersion",
                    "Configured value": "> 0 MAD",
                },
                {
                    "Criterion": "Maximum normalized UV distance",
                    "Configured value": (
                        f"≤ {max_normalized_uv_distance_mad:g} MAD"
                    ),
                },
            ]
        )

        training_by_year = _require_diagnostic_scope(
            diagnostics,
            "training_coverage_by_year",
        ).sort_values("year")

        training_year_display = pd.DataFrame(
            {
                "Year": training_by_year["year"].astype(int),
                "Training rows": [
                    _format_integer(value)
                    for value in training_by_year["n_rows"]
                ],
                "Share of annual rows": [
                    _format_percent(value)
                    for value in training_by_year["share_of_all"]
                ],
                "Share of technically eligible rows": [
                    _format_percent(value)
                    for value in training_by_year["share_of_parent"]
                ],
                "Products represented": [
                    _format_integer(value)
                    for value in training_by_year["value"]
                ],
            }
        )

        training_by_product = _require_diagnostic_scope(
            diagnostics,
            "training_coverage_by_product",
        )

        n_products = int(
            training_by_product["target_cmd"]
            .dropna()
            .astype(str)
            .nunique()
        )
        n_unseen_products = int(
            training_by_product["criterion"]
            .eq("unseen")
            .sum()
        )

        product_display = (
            training_by_product
            .sort_values(
                ["n_rows", "target_cmd"],
                ascending=[True, True],
            )
            .head(20)
            .copy()
        )

        product_display = pd.DataFrame(
            {
                "HS product": product_display["target_cmd"].astype(str),
                "Training rows": [
                    _format_integer(value)
                    for value in product_display["n_rows"]
                ],
                "Technically eligible rows": [
                    _format_integer(value)
                    for value in product_display["eligible_rows"]
                ],
                "Training share": [
                    _format_percent(value)
                    for value in product_display["share_of_all"]
                ],
                "Training-year coverage": [
                    _format_percent(value)
                    for value in product_display[
                        "training_year_coverage"
                    ]
                ],
            }
        )

        rejection = _scope_frame(
            diagnostics,
            "training_rejection_by_product",
        )

        if rejection.empty:
            rejection_display = pd.DataFrame()
        else:
            rejection_display = (
                rejection.pivot_table(
                    index="target_cmd",
                    columns="criterion",
                    values="n_rows",
                    aggfunc="first",
                )
                .reset_index()
            )

            rejection_columns = [
                column
                for column in (
                    "target_cmd",
                    "training_eligible",
                    "reference_support",
                    "positive_uv_mad",
                    "uv_distance_available",
                    "normalized_uv_distance",
                )
                if column in rejection_display.columns
            ]

            rejection_display = rejection_display[
                rejection_columns
            ].rename(
                columns={
                    "target_cmd": "HS product",
                    "training_eligible": "Eligible",
                    "reference_support": "After support",
                    "positive_uv_mad": "After positive MAD",
                    "uv_distance_available": "UV evaluable",
                    "normalized_uv_distance": "After ≤ cutoff",
                }
            )

        # ============================================================== 
        # Reference-window and UV evidence
        # ============================================================== 
        reference_window = _require_diagnostic_scope(
            diagnostics,
            "reference_window_summary",
        )

        support_rows = reference_window[
            reference_window["variable"].eq(
                "groups_with_support"
            )
        ].copy()
        mad_rows = reference_window[
            reference_window["variable"].eq(
                "median_log_mad"
            )
        ].copy()

        reference_display_rows = []

        for criterion in (
            "0_0",
            "1_0",
            "2_0",
        ):
            support = support_rows[
                support_rows["criterion"].eq(criterion)
            ]
            mad = mad_rows[
                mad_rows["criterion"].eq(criterion)
            ]

            if support.empty:
                continue

            support_row = support.iloc[0]
            mad_row = (
                mad.iloc[0]
                if not mad.empty
                else None
            )

            reference_display_rows.append(
                {
                    "Window": criterion,
                    "Groups meeting support": _format_integer(
                        support_row["n_rows"]
                    ),
                    "Share of HS×year groups": _format_percent(
                        support_row["share_of_parent"]
                    ),
                    "Median reference support": _format_number(
                        support_row["value"],
                        digits=1,
                    ),
                    "Median log-UV MAD": _format_number(
                        mad_row["value"]
                        if mad_row is not None
                        else None,
                        digits=4,
                    ),
                }
            )

        reference_window_display = pd.DataFrame(
            reference_display_rows
        )

        support_distribution = _require_diagnostic_scope(
            diagnostics,
            "training_reference_support_distribution",
        ).sort_values("quantile")

        support_distribution_display = pd.DataFrame(
            {
                "Quantile": [
                    f"P{100.0 * float(value):g}"
                    for value in support_distribution["quantile"]
                ],
                "HS×year reference observations": [
                    _format_number(value, digits=1)
                    for value in support_distribution["value"]
                ],
            }
        )

        mad_distribution = _require_diagnostic_scope(
            diagnostics,
            "training_uv_mad_distribution",
        ).sort_values("quantile")

        normalized_distribution = _require_diagnostic_scope(
            diagnostics,
            "training_normalized_uv_distance_distribution",
        ).sort_values("quantile")

        uv_distribution_display = pd.DataFrame(
            {
                "Quantile": [
                    f"P{100.0 * float(value):g}"
                    for value in normalized_distribution["quantile"]
                ],
                "Normalized UV distance (MAD)": [
                    _format_number(value, digits=3)
                    for value in normalized_distribution["value"]
                ],
                "HS×year log-UV MAD": [
                    _format_number(value, digits=4)
                    for value in mad_distribution["value"]
                ],
            }
        )

        cutoff_sensitivity = _require_diagnostic_scope(
            diagnostics,
            "training_normalized_uv_cutoff_sensitivity",
        ).sort_values("uv_distance_threshold")

        cutoff_sensitivity_display = pd.DataFrame(
            {
                "Maximum normalized UV distance": [
                    f"≤ {float(value):g} MAD"
                    for value in cutoff_sensitivity[
                        "uv_distance_threshold"
                    ]
                ],
                "Rows retained": [
                    _format_integer(value)
                    for value in cutoff_sensitivity["n_rows"]
                ],
                "Share of UV-evaluable rows": [
                    _format_percent(value)
                    for value in cutoff_sensitivity[
                        "share_of_parent"
                    ]
                ],
                "Share of all Validation rows": [
                    _format_percent(value)
                    for value in cutoff_sensitivity[
                        "share_of_all"
                    ]
                ],
            }
        )

        # ============================================================== 
        # Model evaluation
        # ============================================================== 
        model_overall = _require_diagnostic_scope(
            diagnostics,
            "model_performance_overall",
        )

        model_pivot = model_overall.pivot_table(
            index="model_name",
            columns="variable",
            values="value",
            aggfunc="first",
        ).reset_index()

        for required_metric in (
            "mae_log",
            "rmse_log",
            "median_absolute_log_error",
        ):
            if required_metric not in model_pivot.columns:
                model_pivot[required_metric] = math.nan

        model_performance_display = pd.DataFrame(
            {
                "Model": model_pivot["model_name"].astype(str),
                "MAE log": model_pivot["mae_log"],
                "Median absolute log error": (
                    model_pivot["median_absolute_log_error"]
                ),
                "RMSE log": model_pivot["rmse_log"],
            }
        ).sort_values("MAE log")

        # ============================================================== 
        # Figures
        # ============================================================== 
        figure_paths = _build_validation_figure_paths(
            figure_dir=figure_dir,
            target_hs=cfg.scope.target_hs,
        )

        _plot_source_quantity_availability(
            diagnostics=diagnostics,
            output_path=figure_paths[
                "source_quantity_availability"
            ],
        )
        _plot_source_disagreement_bands(
            diagnostics=diagnostics,
            output_path=figure_paths[
                "source_quantity_disagreement"
            ],
        )
        _plot_training_population_funnel(
            diagnostics=diagnostics,
            output_path=figure_paths[
                "training_population_funnel"
            ],
        )
        _plot_normalized_uv_cutoff_sensitivity(
            diagnostics=diagnostics,
            output_path=figure_paths[
                "normalized_uv_cutoff_sensitivity"
            ],
            configured_cutoff=max_normalized_uv_distance_mad,
        )
        _plot_training_coverage_by_year(
            diagnostics=diagnostics,
            output_path=figure_paths[
                "training_coverage_by_year"
            ],
        )
        _plot_model_performance_by_year(
            diagnostics=diagnostics,
            output_path=figure_paths[
                "model_performance_by_year"
            ],
        )

        # ============================================================== 
        # Report narrative
        # ============================================================== 
        validation_strategy = str(
            cfg.model.validation.strategy
        )
        minimum_training_years = int(
            cfg.model.validation.minimum_training_years
        )

        if validation_strategy == "expanding_window":
            validation_description = (
                "Candidate models are evaluated using expanding-window "
                "validation. Each validation year is predicted using only "
                "earlier training years, with at least "
                f"{minimum_training_years} preceding training years required "
                "before the first validation fold."
            )
            performance_image_alt = (
                "Expanding-window model performance"
            )
        else:
            validation_description = (
                "Candidate models are evaluated using the configured "
                f"`{validation_strategy}` validation strategy."
            )
            performance_image_alt = (
                "Model performance by validation year"
            )

        lines = [
            (
                "# VACODA Validation Model-Development Report — "
                f"{str(cfg.scope.target_hs).upper()}"
            ),
            "",
            "## 1. Run overview",
            "",
            (
                "This report summarizes the Validation evidence at the canonical "
                "bilateral grain `(exporter, importer, year, HS product)`. "
                "Reporter-side exporter/importer quantities are retained as "
                "descriptive source diagnostics only; they do not define model-"
                "training eligibility."
            ),
            "",
            f"- Run ID: `{run_id}`",
            f"- Prepared input: `{prepared_input_path}`",
            f"- Structural audit: `{structural_audit_path}`",
            f"- Diagnostics: `{diagnostics_path}`",
            "",
            "## 2. Structural integrity",
            "",
            _markdown_table(structural_summary),
            "",
            "## 3. Canonical bilateral quantity methodology",
            "",
            (
                "The quantity-selection stage has fixed one canonical bilateral "
                "quantity methodology for downstream Validation. The current "
                f"methodology is **{selected_candidate_label}**."
            ),
            "",
            (
                "All subsequent unit-value, training-sample, and model-evaluation "
                "diagnostics refer to this canonical selected quantity rather than "
                "treating exporter or importer quantities as separate truth labels."
            ),
            "",
            "## 4. Source-reporter diagnostics",
            "",
            (
                "The following diagnostics describe the quantity evidence supplied "
                "by the two reporting sides before reconciliation. They are retained "
                "to characterize source-data quality and provenance only. **No "
                "reporter-availability or mirror-disagreement threshold is used in "
                "the model-training sample.**"
            ),
            "",
            "### 4.1. Quantity availability by reporting side",
            "",
            _markdown_table(source_availability_display),
            "",
            markdown_image(
                "Source-reporter quantity availability",
                figure_paths["source_quantity_availability"],
                document_path=report_path,
                width=800,
            ),
            "",
            "### 4.2. Exporter/importer disagreement",
            "",
            (
                "Where both source quantities are available, the distribution below "
                "shows how strongly they disagree. This is a diagnostic of the raw "
                "reporting evidence, not a training-selection criterion."
            ),
            "",
            _markdown_table(source_disagreement_display),
            "",
            markdown_image(
                "Exporter/importer quantity disagreement bands",
                figure_paths["source_quantity_disagreement"],
                document_path=report_path,
                width=800,
            ),
            "",
            "## 5. HS × year unit-value reference",
            "",
            (
                "For technically usable canonical bilateral observations, Validation "
                "forms a robust HS × focal-year unit-value reference in log space. "
                "The configured reference window pools the focal year with preceding "
                "years only; the reference centre is the median log unit value and "
                "dispersion is the median absolute deviation (MAD)."
            ),
            "",
            "### 5.1. Reference-window comparison",
            "",
            (
                "The table compares same-year, one-year-back, and two-year-back "
                "causal reference windows. It is diagnostic evidence for the chosen "
                "reference construction, not a separate training criterion."
            ),
            "",
            _markdown_table(reference_window_display),
            "",
            "### 5.2. Current reference support",
            "",
            _markdown_table(support_distribution_display),
            "",
            "### 5.3. UV dispersion and normalized distance",
            "",
            (
                "The selected unit-value distance is normalized by the HS × year "
                "log-UV MAD so that deviations are interpreted relative to the "
                "natural dispersion of each product/time group."
            ),
            "",
            _markdown_table(uv_distribution_display),
            "",
            "## 6. Model-training sample",
            "",
            (
                "The model-training sample is defined only from canonical bilateral "
                "unit-value evidence. Reporter-side quantity agreement is not part "
                "of this rule."
            ),
            "",
            "### 6.1. Configured training rule",
            "",
            _markdown_table(training_rule_display),
            "",
            (
                f"The resulting training sample contains **{_format_integer(n_training_selected)}** "
                f"rows, or **{_format_percent(training_selected_share)}** of all "
                "Validation rows."
            ),
            "",
            "### 6.2. Population funnel",
            "",
            _markdown_table(training_funnel_display),
            "",
            markdown_image(
                "Configured model-training population",
                figure_paths["training_population_funnel"],
                document_path=report_path,
                width=850,
            ),
            "",
            "### 6.3. Normalized-UV cutoff sensitivity",
            "",
            (
                "The cutoff sensitivity is retained to show how the final tail "
                "restriction changes sample size. It does not reopen reporter-side "
                "truth assumptions."
            ),
            "",
            _markdown_table(cutoff_sensitivity_display),
            "",
            markdown_image(
                "Normalized unit-value cutoff sensitivity",
                figure_paths["normalized_uv_cutoff_sensitivity"],
                document_path=report_path,
                width=800,
            ),
            "",
            "### 6.4. Training coverage by year",
            "",
            _markdown_table(training_year_display),
            "",
            markdown_image(
                "Final training-sample coverage by year",
                figure_paths["training_coverage_by_year"],
                document_path=report_path,
                width=850,
            ),
            "",
            "### 6.5. Product coverage",
            "",
            (
                f"The Validation population contains **{_format_integer(n_products)}** "
                f"HS products; **{_format_integer(n_unseen_products)}** have no "
                "training-selected observations under the configured UV-based rule."
            ),
            "",
            (
                "The table below shows the 20 products with the smallest final "
                "training populations."
            ),
            "",
            _markdown_table(product_display),
            "",
            "### 6.6. Rejection path for products unseen in training",
            "",
            (
                "For products with no final training observations, the cumulative "
                "table shows whether rows disappear because of insufficient "
                "reference support, non-positive/undefined MAD, unavailable UV "
                "distance, or the configured normalized-UV tail cutoff."
            ),
            "",
            _markdown_table(rejection_display),
            "",
            "## 7. Model evaluation",
            "",
            validation_description,
            "",
            "### 7.1. Overall out-of-fold performance",
            "",
            _markdown_table(model_performance_display),
            "",
            "### 7.2. Performance by validation year",
            "",
            markdown_image(
                performance_image_alt,
                figure_paths["model_performance_by_year"],
                document_path=report_path,
                width=850,
            ),
            "",
            "## 8. Current methodological status",
            "",
            (
                f"- Canonical bilateral quantity methodology: **{selected_candidate_label}**."
            ),
            (
                "- Training sample: fixed by HS×year support, positive MAD, and "
                f"normalized UV distance ≤ {max_normalized_uv_distance_mad:g} MAD."
            ),
            (
                "- Exporter/importer disagreement is retained only as descriptive "
                "source-data evidence and does not determine training eligibility."
            ),
            (
                f"- Model validation strategy: `{validation_strategy}`."
            ),
            (
                "- Production inference remains a subsequent Validation stage."
            ),
            "",
            "## 9. Inference",
            "",
            (
                "Production inference is not yet reported by the current Validation "
                "workflow. This section is reserved for the final selected model, "
                "the canonical bilateral rows requiring reconstruction, prediction "
                "coverage, post-inference unit-value plausibility, and the completed "
                "quantity output."
            ),
            "",
        ]

        insert_markdown_toc(
            lines,
            min_level=2,
            max_level=4,
            flat_sublevels=True,
        )

        report_path.write_text(
            "\n".join(lines),
            encoding="utf-8",
        )

        if not valid_output(report_path):
            raise RuntimeError(
                "Validation report was not written correctly."
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
            latest_report_path = latest_path

        logger.info(
            "%s Validation model-development report created.",
            ok,
        )
        logger.info(
            "  Structural audit passed: %s",
            audit_passed,
        )
        logger.info(
            "  Selected training rows: %s",
            f"{n_training_selected:,}",
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
            len(figure_paths),
        )

        return {
            "status": "completed",
            "run_id": run_id,
            "output_path": report_path,
            "report_path": report_path,
            "latest_report_path": latest_report_path,
            "figure_paths": figure_paths,
            "audit_passed": audit_passed,
            "training_population_selected": True,
            "ready_for_training_decision": True,
            "duration_seconds": None,
        }

    finally:
        con.close()
