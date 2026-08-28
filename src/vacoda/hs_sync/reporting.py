"""
vacoda.hs_sync.reporting

Create the final HS Sync decision report and diagnostic figures.

Outputs
-------
- Timestamped Markdown decision report
- Optional latest Markdown copy
- Overall source-HS coverage figure
- Annual HS-classification mix figure
- Reporter HS-transition figure
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from vacoda.config.hs_sync import HsSyncConfig
from vacoda.config.paths import get_figure_dir
from vacoda.utils.display import get_symbols
from vacoda.utils.markdown import (
    insert_markdown_toc,
    markdown_image,
)


# ======================================================================
# Helpers
# ======================================================================


def valid_output(path: Path) -> bool:
    """Return True when an existing report appears usable."""
    return path.is_file() and path.stat().st_size > 0


def _format_integer(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NA"

    try:
        return f"{int(round(float(value))):,}"
    except (TypeError, ValueError):
        return str(value)


def _markdown_table(
    dataframe: pd.DataFrame | None,
    *,
    index: bool = False,
    max_rows: int | None = None,
) -> str:
    """Convert a DataFrame to compact Markdown."""
    if dataframe is None or dataframe.empty:
        return "_No rows available._"

    frame = dataframe.copy()

    if max_rows is not None:
        frame = frame.head(max_rows)

    return frame.to_markdown(
        index=index,
        floatfmt=".4f",
    )


def _coverage_table(
    coverage_tables: dict[str, pd.DataFrame],
    name: str,
) -> pd.DataFrame:
    """Return one configured coverage product or an empty DataFrame."""
    table = coverage_tables.get(name)

    if table is None:
        return pd.DataFrame()

    return table.copy()


def _result_value(
    result: dict[str, Any] | None,
    *keys: str,
) -> Any:
    """Return the first available non-null result value."""
    if not result:
        return None

    for key in keys:
        if key in result and result[key] is not None:
            return result[key]

    return None


def _result_summary(
    result: dict[str, Any] | None,
    *,
    preferred_keys: tuple[str, ...],
) -> pd.DataFrame:
    """
    Build a compact metric/value summary from a workflow result.

    Only scalar values explicitly returned by the producing workflow step
    are included.
    """
    if not result:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []

    for key in preferred_keys:
        if key not in result:
            continue

        value = result[key]

        if isinstance(
            value,
            (
                pd.DataFrame,
                dict,
                list,
                tuple,
                set,
            ),
        ):
            continue

        rows.append(
            {
                "Metric": key.replace("_", " ").capitalize(),
                "Value": value,
            }
        )

    return pd.DataFrame(rows)



def _save_figure(
    figure,
    output_path: Path,
) -> None:
    """Write one report figure."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)


# ======================================================================
# Figures
# ======================================================================


def _plot_overall_hs_coverage(
    *,
    overall_summary: pd.DataFrame,
    hs_col: str,
    output_path: Path,
) -> bool:
    """
    Plot overall source-row coverage by source HS classification.

    Expected product
    ----------------
    hs_system_overall_summary
    """
    if overall_summary.empty:
        return False

    required = {
        hs_col,
        "share_pct",
    }

    if not required.issubset(
        overall_summary.columns
    ):
        return False

    frame = (
        overall_summary[
            [
                hs_col,
                "share_pct",
            ]
        ]
        .dropna()
        .sort_values(
            "share_pct",
            ascending=True,
        )
    )

    if frame.empty:
        return False

    figure, axis = plt.subplots(
        figsize=(
            9,
            max(
                4.5,
                0.45 * len(frame) + 1.5,
            ),
        )
    )

    bars = axis.barh(
        frame[hs_col].astype(str),
        frame["share_pct"].astype(float),
    )

    axis.bar_label(
        bars,
        labels=[
            f"{value:.2f}%"
            for value in frame["share_pct"]
        ],
        padding=3,
        fontsize=8,
    )

    axis.set_title(
        "Overall source HS-classification coverage"
    )
    axis.set_xlabel(
        "Share of source rows (%)"
    )
    axis.set_ylabel(
        "HS classification"
    )

    axis.grid(
        axis="x",
        linestyle="--",
        alpha=0.4,
    )
    axis.set_axisbelow(True)

    _save_figure(
        figure,
        output_path,
    )

    return True


def _plot_year_hs_mix(
    *,
    percentage_pivot: pd.DataFrame,
    year_col: str,
    output_path: Path,
) -> bool:
    """
    Plot source HS-classification shares by year.

    Expected product
    ----------------
    year_hs_percentage_pivot
    """
    if percentage_pivot.empty:
        return False

    frame = percentage_pivot.copy()

    # The workflow may retain the configured year as either
    # an ordinary column or the pivot index.
    if year_col in frame.columns:
        frame = frame.set_index(year_col)

    elif frame.index.name != year_col:
        return False

    numeric = frame.apply(
        pd.to_numeric,
        errors="coerce",
    )

    numeric = numeric.dropna(
        axis=1,
        how="all",
    )

    if numeric.empty:
        return False

    numeric.columns = [
        str(column).removesuffix("_pct")
        for column in numeric.columns
    ]

    figure, axis = plt.subplots(
        figsize=(11, 6)
    )

    numeric.plot(
        kind="bar",
        stacked=True,
        ax=axis,
    )

    axis.set_title(
        "Source HS-classification mix by year"
    )
    axis.set_xlabel("Year")
    axis.set_ylabel(
        "Share of source rows (%)"
    )
    axis.set_ylim(0, 100)

    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
    )

    axis.legend(
        title="HS classification",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    _save_figure(
        figure,
        output_path,
    )

    return True


def _plot_hs_transitions(
    *,
    transitions: pd.DataFrame,
    output_path: Path,
) -> bool:
    """
    Plot reporter HS-classification transitions by year.

    build_hs_transition_events() defines transition_year as part of its
    output contract.
    """
    if (
        transitions.empty
        or "transition_year"
        not in transitions.columns
    ):
        return False

    frame = (
        transitions["transition_year"]
        .dropna()
        .value_counts()
        .sort_index()
        .rename_axis("year")
        .reset_index(
            name="n_transitions"
        )
    )

    if frame.empty:
        return False

    figure, axis = plt.subplots(
        figsize=(10, 5.5)
    )

    axis.bar(
        frame["year"].astype(str),
        frame["n_transitions"],
    )

    axis.set_title(
        "Reporter HS-classification transitions by year"
    )
    axis.set_xlabel(
        "Transition year"
    )
    axis.set_ylabel(
        "Reporter transitions"
    )

    axis.tick_params(
        axis="x",
        labelrotation=45,
    )

    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
    )

    axis.set_axisbelow(True)

    _save_figure(
        figure,
        output_path,
    )

    return True


# ======================================================================
# Main reporting function
# ======================================================================


def build_hs_sync_report(
    *,
    cfg: HsSyncConfig,
    run_id: str,
    coverage_tables: dict[str, pd.DataFrame],
    coverage_result: dict[str, Any] | None,
    mapping_result: dict[str, Any] | None,
    mapping_audit_result: dict[str, Any] | None,
    equal_split_result: dict[str, Any] | None,
    complete_weights_result: dict[str, Any] | None,
    apply_result: dict[str, Any] | None,
    applied_audit_result: dict[str, Any] | None,
    report_path: Path,
    latest_report_path: Path | None = None,
    force: bool = False,
    logger,
) -> dict[str, Any]:
    """
    Create the final HS Sync Markdown decision report.

    Reporting consumes evidence already produced by the HS Sync workflow.
    It does not rebuild source coverage, mappings, allocation weights,
    synchronized trade, or audits.
    """

    # ==================================================================
    # Final workflow decision
    # ==================================================================

    ready_for_quantity = bool(
        _result_value(
            applied_audit_result,
            "ready_for_quantity",
            "ready_for_next_step",
        )
    )

    final_audit_passed = bool(
        _result_value(
            applied_audit_result,
            "audit_passed",
            "ready_for_quantity",
        )
    )

    n_failed_audit_rows = int(
        _result_value(
            applied_audit_result,
            "n_failed_audit_rows",
        )
        or 0
    )

    decision = (
        "APPROVED"
        if ready_for_quantity
        else "REVIEW REQUIRED"
    )

    # ==================================================================
    # Artifact configuration
    # ==================================================================

    if not cfg.artifacts.enable:
        return {
            "status": "disabled",
            "output_path": None,
            "report_path": None,
            "latest_report_path": None,
            "figure_paths": [],
            "figures": {},
            "decision": decision,
            "audit_passed": final_audit_passed,
            "ready_for_quantity": ready_for_quantity,
            "n_failed_audit_rows": (
                n_failed_audit_rows
            ),
        }

    report_path = Path(report_path)

    if latest_report_path is not None:
        latest_report_path = Path(
            latest_report_path
        )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure_dir = (
        get_figure_dir()
        / "hs_sync"
        / (
            f"target_"
            f"{cfg.scope.target_hs.lower()}"
        )
    )

    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ==================================================================
    # Reuse
    # ==================================================================

    if (
        not force
        and valid_output(report_path)
    ):
        logger.info(
            "REUSE — HS Sync decision report already exists: %s",
            report_path,
        )

        latest_existing = (
            latest_report_path
            if (
                latest_report_path is not None
                and latest_report_path.exists()
            )
            else None
        )

        return {
            "status": "reused",
            "output_path": report_path,
            "report_path": report_path,
            "latest_report_path": (
                latest_existing
            ),
            "figure_paths": [],
            "figures": {},
            "decision": decision,
            "audit_passed": final_audit_passed,
            "ready_for_quantity": (
                ready_for_quantity
            ),
            "n_failed_audit_rows": (
                n_failed_audit_rows
            ),
        }

    info = get_symbols("info")
    ok = get_symbols("check")

    logger.info(
        "%s Building HS Sync decision report.",
        info,
    )
    logger.info(
        "  Target HS: %s",
        cfg.scope.target_hs,
    )
    logger.info(
        "  Year range: %s-%s",
        cfg.scope.min_year,
        cfg.scope.max_year,
    )
    logger.info(
        "  Report: %s",
        report_path,
    )

    # ==================================================================
    # 1. Run overview
    # ==================================================================

    input_path = _result_value(
        apply_result,
        "input_path",
        "source_path",
    )

    if input_path is None:
        input_path = cfg.coverage_input_dir()

    output_path = _result_value(
        apply_result,
        "output_path",
        "final_parquet",
    )

    run_overview = pd.DataFrame(
        [
            {
                "Item": "Run ID",
                "Value": run_id,
            },
            {
                "Item": "Target HS",
                "Value": cfg.scope.target_hs,
            },
            {
                "Item": "Year range",
                "Value": (
                    f"{cfg.scope.min_year}"
                    f"–{cfg.scope.max_year}"
                ),
            },
            {
                "Item": "Source input",
                "Value": input_path,
            },
            {
                "Item": "Final HS-sync output",
                "Value": (
                    output_path
                    if output_path is not None
                    else "NA"
                ),
            },
        ]
    )

    # ==================================================================
    # 2. Source HS coverage
    # ==================================================================

    overall_summary = _coverage_table(
        coverage_tables,
        "hs_system_overall_summary",
    )

    year_hs_mix = _coverage_table(
        coverage_tables,
        "year_hs_mix_summary",
    )

    percentage_pivot = _coverage_table(
        coverage_tables,
        "year_hs_percentage_pivot",
    )

    reporter_year_coverage = _coverage_table(
        coverage_tables,
        "reporter_year_hs_coverage",
    )

    reporter_timeline = _coverage_table(
        coverage_tables,
        "reporter_hs_timeline",
    )

    transitions = _coverage_table(
        coverage_tables,
        "hs_transition_events",
    )

    mixed_reporter_years = _coverage_table(
        coverage_tables,
        "mixed_classification_reporter_years",
    )

    # ==================================================================
    # 3. Mapping quality
    # ==================================================================

    mapping_coverage = _coverage_table(
        coverage_tables,
        "mapping_coverage",
    )

    mapping_coverage_summary = _coverage_table(
        coverage_tables,
        "mapping_summary",
    )

    missing_mappings = _coverage_table(
        coverage_tables,
        "missing_mappings",
    )

    mapping_preparation_summary = _result_summary(
        mapping_result,
        preferred_keys=(
            "status",
            "n_rows",
            "n_mappings",
            "output_path",
            "mapping_path",
            "audit_passed",
        ),
    )

    mapping_audit_summary = _result_summary(
        mapping_audit_result,
        preferred_keys=(
            "status",
            "audit_passed",
            "n_audit_rows",
            "n_failed_audit_rows",
            "n_missing_mappings",
            "missing_mapping_count",
            "n_invalid_mappings",
            "n_normalized_mappings",
            "ready_for_allocation",
            "audit_path",
            "report_path",
        ),
    )

    # ==================================================================
    # 4. Allocation
    # ==================================================================

    needs_weights = _coverage_table(
        coverage_tables,
        "needs_weights",
    )

    easy_without_weights = _coverage_table(
        coverage_tables,
        "easy_without_weights",
    )

    equal_split_summary = _result_summary(
        equal_split_result,
        preferred_keys=(
            "status",
            "n_rows",
            "n_mapping_rows",
            "n_weight_rows",
            "n_allocated_rows",
            "n_one_to_many",
            "audit_passed",
            "output_path",
            "weights_path",
            "audit_path",
        ),
    )

    complete_weights_summary = _result_summary(
        complete_weights_result,
        preferred_keys=(
            "status",
            "n_rows",
            "n_weight_rows",
            "n_complete_mappings",
            "n_missing_weights",
            "n_invalid_weights",
            "audit_passed",
            "output_path",
            "weights_path",
            "audit_path",
        ),
    )

    # ==================================================================
    # 5. Applied HS-sync output
    # ==================================================================

    apply_summary = _result_summary(
        apply_result,
        preferred_keys=(
            "status",
            "n_input_rows",
            "n_output_rows",
            "n_allocated_rows",
            "n_unchanged_rows",
            "n_expanded_rows",
            "n_target_rows",
            "allocation_weight_sum",
            "value_preserved",
            "weight_preserved",
            "audit_passed",
            "output_path",
        ),
    )

    # ==================================================================
    # 6. Final audit
    # ==================================================================

    final_audit_summary = _result_summary(
        applied_audit_result,
        preferred_keys=(
            "status",
            "audit_passed",
            "n_audit_rows",
            "n_failed_audit_rows",
            "n_duplicate_keys",
            "n_missing_required_columns",
            "n_invalid_weights",
            "n_invalid_values",
            "ready_for_quantity",
            "audit_path",
        ),
    )

    # ==================================================================
    # 7. Diagnostic figures
    # ==================================================================

    figure_paths = {
        "overall_hs_coverage": (
            figure_dir
            / "overall_hs_coverage.png"
        ),
        "year_hs_mix": (
            figure_dir
            / "year_hs_mix.png"
        ),
        "hs_transitions": (
            figure_dir
            / "hs_transitions.png"
        ),
    }

    written_figures: dict[str, Path] = {}

    if _plot_overall_hs_coverage(
        overall_summary=overall_summary,
        hs_col=cfg.columns.hs,
        output_path=figure_paths[
            "overall_hs_coverage"
        ],
    ):
        written_figures[
            "overall_hs_coverage"
        ] = figure_paths[
            "overall_hs_coverage"
        ]

    if _plot_year_hs_mix(
        percentage_pivot=percentage_pivot,
        year_col=cfg.columns.year,
        output_path=figure_paths[
            "year_hs_mix"
        ],
    ):
        written_figures[
            "year_hs_mix"
        ] = figure_paths[
            "year_hs_mix"
        ]

    if _plot_hs_transitions(
        transitions=transitions,
        output_path=figure_paths[
            "hs_transitions"
        ],
    ):
        written_figures[
            "hs_transitions"
        ] = figure_paths[
            "hs_transitions"
        ]

    # ==================================================================
    # Build Markdown
    # ==================================================================

    lines: list[str] = [
        "# HS Sync Decision Report",
        "",
        f"**Decision:** {decision}  ",
        f"**Run ID:** {run_id}  ",
        (
            f"**Target HS:** "
            f"{cfg.scope.target_hs}  "
        ),
        (
            f"**Input period:** "
            f"{cfg.scope.min_year}"
            f"–{cfg.scope.max_year}  "
        ),
        (
            f"**Generated:** "
            f"{datetime.now().isoformat(timespec='seconds')}"
        ),
        "",
        "",
        "__TOC__",
        "",
        # --------------------------------------------------------------
        # 1. Run overview
        # --------------------------------------------------------------

        "## 1. Run overview",
        "",
        (
            "This report summarizes the HS synchronization workflow "
            "from source-classification coverage through mapping, "
            "allocation, application, and final downstream-readiness "
            "validation."
        ),
        "",
        _markdown_table(
            run_overview
        ),
        "",

        # --------------------------------------------------------------
        # 2. Source HS coverage
        # --------------------------------------------------------------

        "## 2. Source HS coverage",
        "",
        (
            "These diagnostics describe the HS classifications observed "
            "in the source trade data before conversion to the configured "
            f"target classification `{cfg.scope.target_hs}`."
        ),
        "",

        "### 2.1 Overall HS-system coverage",
        "",
        _markdown_table(
            overall_summary,
            max_rows=30,
        ),
        "",

        "### 2.2 Year-by-year HS mix",
        "",
        _markdown_table(
            year_hs_mix,
            max_rows=50,
        ),
        "",

        "### 2.3 Reporter coverage",
        "",
        (
            f"Reporter × year × HS coverage rows: "
            f"`{len(reporter_year_coverage):,}`."
        ),
        "",
        _markdown_table(
            reporter_year_coverage,
            max_rows=20,
        ),
        "",

        "### 2.4 Reporter HS timelines",
        "",
        (
            f"Continuous reporter HS-classification periods: "
            f"`{len(reporter_timeline):,}`."
        ),
        "",
        _markdown_table(
            reporter_timeline,
            max_rows=30,
        ),
        "",

        "### 2.5 Classification transitions and anomalies",
        "",
        (
            f"Reporter classification transitions identified: "
            f"`{len(transitions):,}`."
        ),
        "",
        _markdown_table(
            transitions,
            max_rows=30,
        ),
        "",
        (
            f"Mixed-classification reporter-years identified: "
            f"`{len(mixed_reporter_years):,}`."
        ),
        "",
        _markdown_table(
            mixed_reporter_years,
            max_rows=30,
        ),
        "",

        # --------------------------------------------------------------
        # 3. Mapping quality
        # --------------------------------------------------------------

        "## 3. Mapping quality",
        "",
        (
            "This section summarizes preparation and validation of the "
            "HS concordances used to translate observed source codes into "
            f"the target `{cfg.scope.target_hs}` classification."
        ),
        "",

        "### 3.1 Mapping preparation",
        "",
        _markdown_table(
            mapping_preparation_summary
        ),
        "",

        "### 3.2 Mapping coverage",
        "",
        _markdown_table(
            mapping_coverage_summary,
            max_rows=50,
        ),
        "",

        "### 3.3 Missing mappings",
        "",
        (
            f"Missing-mapping rows identified: "
            f"`{len(missing_mappings):,}`."
        ),
        "",
        _markdown_table(
            missing_mappings,
            max_rows=50,
        ),
        "",

        "### 3.4 Mapping normalization audit",
        "",
        _markdown_table(
            mapping_audit_summary
        ),
        "",

        # --------------------------------------------------------------
        # 4. Allocation
        # --------------------------------------------------------------

        "## 4. Allocation",
        "",
        (
            "One-to-many and many-to-many HS concordance relationships "
            "require allocation weights before the synchronized "
            "target-HS output can be constructed."
        ),
        "",

        "### 4.1 Coverage requiring weights",
        "",
        (
            f"Coverage rows requiring allocation weights: "
            f"`{len(needs_weights):,}`."
        ),
        "",
        _markdown_table(
            needs_weights,
            max_rows=30,
        ),
        "",

        "### 4.2 Coverage not requiring weights",
        "",
        (
            f"Coverage rows directly resolvable without allocation "
            f"weights: `{len(easy_without_weights):,}`."
        ),
        "",
        _markdown_table(
            easy_without_weights,
            max_rows=30,
        ),
        "",

        "### 4.3 Equal-split allocation",
        "",
        _markdown_table(
            equal_split_summary
        ),
        "",

        "### 4.4 Complete mapping weights",
        "",
        _markdown_table(
            complete_weights_summary
        ),
        "",

        # --------------------------------------------------------------
        # 5. Applied HS-sync output
        # --------------------------------------------------------------

        "## 5. Applied HS-sync output",
        "",
        (
            "The normalized mappings and completed allocation weights "
            "are applied to the source observations to construct the "
            f"canonical `{cfg.scope.target_hs}` trade output."
        ),
        "",
        _markdown_table(
            apply_summary
        ),
        "",

        # --------------------------------------------------------------
        # 6. Final audit
        # --------------------------------------------------------------

        "## 6. Final audit",
        "",
        (
            f"**Audit status:** "
            f"{'PASS' if final_audit_passed else 'FAIL'}  "
        ),
        (
            f"**Failed audit rows:** "
            f"{_format_integer(n_failed_audit_rows)}  "
        ),
        (
            f"**Ready for Quantity:** "
            f"{'YES' if ready_for_quantity else 'NO'}"
        ),
        "",
        _markdown_table(
            final_audit_summary
        ),
        "",

        "### 6.1 Decision",
        "",
        (
            f"{ok} The HS Sync output passed the final applied-output "
            "audit and is approved for Quantity processing."
            if ready_for_quantity
            else (
                "The HS Sync output requires review before it can be "
                "used by the Quantity workflow."
            )
        ),
        "",

        # --------------------------------------------------------------
        # 7. Diagnostic figures
        # --------------------------------------------------------------

        "## 7. Diagnostic figures",
        "",
    ]

    # ==================================================================
    # Figure references
    # ==================================================================

    if "overall_hs_coverage" in written_figures:
        lines.extend(
            [
                "### 7.1 Overall source HS coverage",
                "",
                (
                    "The figure shows the overall distribution of source "
                    "observations across HS classification systems."
                ),
                "",
                markdown_image(
                    "Overall source HS coverage",
                    written_figures["overall_hs_coverage"],
                    document_path=report_path,
                ),
                "",
            ]
        )

    if "year_hs_mix" in written_figures:
        lines.extend(
            [
                "### 7.2 HS classification mix by year",
                "",
                (
                    "The annual composition highlights transitions, "
                    "overlap periods, and years in which multiple HS "
                    "systems coexist in the source data."
                ),
                "",
                markdown_image(
                    "HS classification mix by year",
                    written_figures["year_hs_mix"],
                    document_path=report_path,
                ),
                "",
            ]
        )

    if "hs_transitions" in written_figures:
        lines.extend(
            [
                "### 7.3 Reporter classification transitions",
                "",
                (
                    "The figure counts detected changes in dominant "
                    "reporter HS classification by transition year."
                ),
                "",
                markdown_image(
                    "Reporter HS classification transitions",
                    written_figures["hs_transitions"],
                    document_path=report_path,
                ),
                "",
            ]
        )

    if not written_figures:
        lines.extend(
            [
                "_No diagnostic figures were generated._",
                "",
            ]
        )

    # ==================================================================
    # 8. Output provenance
    # ==================================================================

    lines.extend(
        [
            "## 8. Output provenance",
            "",
            "| Artifact | Path |",
            "|---|---|",
            (
                f"| Source trade input | "
                f"`{input_path}` |"
            ),
            (
                f"| Coverage outputs | "
                f"`{_result_value(coverage_result, 'output_dir') or cfg.coverage_output_dir()}` |"
            ),
            (
                f"| HS-synced trade | "
                f"`{output_path or 'NA'}` |"
            ),
            (
                f"| Final applied-output audit | "
                f"`{_result_value(applied_audit_result, 'audit_path') or 'NA'}` |"
            ),
            (
                f"| Decision report | "
                f"`{report_path}` |"
            ),
            "",
        ]
    )

    # ==================================================================
    # Table of contents
    # ==================================================================

    insert_markdown_toc(
        lines,
        min_level=2,
        max_level=3,
    )

    # ==================================================================
    # Write report
    # ==================================================================

    report_text = "\n".join(lines)

    report_path.write_text(
        report_text,
        encoding="utf-8",
    )

    if not valid_output(report_path):
        raise RuntimeError(
            "HS Sync decision report was not written correctly."
        )

    written_latest_path: Path | None = None

    if (
        cfg.artifacts.write_latest
        and latest_report_path is not None
    ):
        latest_report_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copyfile(
            report_path,
            latest_report_path,
        )

        written_latest_path = (
            latest_report_path
        )

    logger.info(
        "%s HS Sync decision report created.",
        ok,
    )
    logger.info(
        "  Decision: %s",
        decision,
    )
    logger.info(
        "  Ready for Quantity: %s",
        ready_for_quantity,
    )
    logger.info(
        "  Report: %s",
        report_path,
    )

    if written_latest_path is not None:
        logger.info(
            "  Latest report: %s",
            written_latest_path,
        )

    logger.info(
        "  Figures written: %d",
        len(written_figures),
    )

    return {
        "status": "completed",
        "run_id": run_id,
        "output_path": report_path,
        "report_path": report_path,
        "latest_report_path": (
            written_latest_path
        ),
        "figure_paths": list(
            written_figures.values()
        ),
        "figures": written_figures,
        "decision": decision,
        "audit_passed": final_audit_passed,
        "ready_for_quantity": ready_for_quantity,
        "n_failed_audit_rows": (
            n_failed_audit_rows
        ),
    }