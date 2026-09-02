"""
vacoda.value.reporting

Create the final Value decision report and diagnostic figures.

Outputs
-------
- Timestamped Markdown decision report
- Optional latest Markdown copy
- Annual nominal-versus-constant-value figure
- Annual GDP-deflation-factor figure
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
from vacoda.config.value import ValueConfig
from vacoda.utils.display import get_symbols
from vacoda.utils.markdown import (
    insert_markdown_toc,
    markdown_image,
)
from vacoda.utils.sql import sql_literal
from vacoda.value.deflate import (
    CONSTANT_VALUE_COLUMN,
    DEFLATION_FACTOR_COLUMN,
    DEFLATOR_COLUMN,
)


# ======================================================================
# Helpers
# ======================================================================


def valid_output(path: Path) -> bool:
    """Return True when an existing output appears usable."""
    return path.is_file() and path.stat().st_size > 0


def _format_integer(
    value: float | int | None,
) -> str:
    if value is None or pd.isna(value):
        return "NA"

    return f"{int(round(float(value))):,}"


def _format_usd(value: Any) -> str:
    if value is None or pd.isna(value):
        return "—"

    numeric = float(value)

    if abs(numeric) >= 1e12:
        return (
            f"${numeric / 1e12:,.3f} trillion"
        )

    if abs(numeric) >= 1e9:
        return (
            f"${numeric / 1e9:,.3f} billion"
        )

    if abs(numeric) >= 1e6:
        return (
            f"${numeric / 1e6:,.3f} million"
        )

    return f"${numeric:,.2f}"


def _format_factor(
    value: float | None,
) -> str:
    if value is None or pd.isna(value):
        return "NA"

    return f"{float(value):.6f}"


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


def _audit_passed_value(value: Any) -> bool:
    """Interpret the CSV representation of an audit boolean."""
    if isinstance(value, bool):
        return value

    if pd.isna(value):
        return False

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
    }


# ======================================================================
# Figures
# ======================================================================


def _plot_annual_nominal_vs_constant(
    *,
    annual: pd.DataFrame,
    output_path: Path,
    reference_year: int,
) -> None:
    """Plot annual nominal and constant-price trade totals."""
    if annual.empty:
        raise RuntimeError(
            "Annual Value summary is empty."
        )

    frame = annual.copy()

    frame["nominal_billions"] = (
        frame["nominal_total"] / 1e9
    )

    frame["constant_billions"] = (
        frame["constant_total"] / 1e9
    )

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    axis.plot(
        frame["year"],
        frame["nominal_billions"],
        marker="o",
        label="Nominal USD",
    )

    axis.plot(
        frame["year"],
        frame["constant_billions"],
        marker="o",
        label=f"Constant {reference_year} USD",
    )

    axis.set_title(
        "Annual Trade Sync value before and after deflation"
    )
    axis.set_xlabel("Year")
    axis.set_ylabel("Trade value, USD billions")
    axis.set_xticks(
        frame["year"].astype(int).tolist()
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

    plt.close(figure)


def _plot_deflation_factor(
    *,
    annual: pd.DataFrame,
    output_path: Path,
    reference_year: int,
) -> None:
    """Plot the annual multiplier used to express values in constant USD."""
    if annual.empty:
        raise RuntimeError(
            "Annual Value summary is empty."
        )

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    axis.plot(
        annual["year"],
        annual["deflation_factor"],
        marker="o",
    )

    axis.axhline(
        1.0,
        linestyle="--",
        linewidth=1.2,
    )

    axis.set_title(
        f"USD deflation factor relative to {reference_year}"
    )
    axis.set_xlabel("Year")
    axis.set_ylabel("Deflation multiplier")
    axis.set_xticks(
        annual["year"].astype(int).tolist()
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

    plt.close(figure)


# ======================================================================
# Main reporting function
# ======================================================================


def build_value_report(
    *,
    cfg: ValueConfig,
    input_path: Path,
    deflated_trade_path: Path,
    deflator_path: Path,
    audit_path: Path,
    run_id: str,
    force: bool = False,
    logger,
) -> dict[str, Any]:
    """Create the final Value Markdown decision report."""

    input_path = Path(input_path)
    deflated_trade_path = Path(
        deflated_trade_path
    )
    deflator_path = Path(deflator_path)
    audit_path = Path(audit_path)

    for label, path in (
        ("Trade Sync input", input_path),
        (
            "deflated Value output",
            deflated_trade_path,
        ),
        (
            "prepared GDP deflator",
            deflator_path,
        ),
        ("Value audit", audit_path),
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
            "decision": None,
            "ready_for_validation": False,
        }

    report_path = cfg.artifacts_path(
        run_id
    )
    latest_path = (
        cfg.artifacts_latest_path()
    )

    figure_dir = (
        get_figure_dir()
        / "value"
        / (
            f"target_"
            f"{cfg.scope.target_hs.lower()}"
        )
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        not force
        and valid_output(report_path)
    ):
        logger.info(
            "REUSE — Value decision report already exists: %s",
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
            "decision": "APPROVED",
            "ready_for_validation": True,
        }

    output_sql = sql_literal(
        deflated_trade_path
    )

    deflator_sql = sql_literal(
        deflator_path
    )

    year_column = (
        cfg.input.columns.year
    )

    nominal_value_column = (
        cfg.input.columns.nominal_value
    )

    info = get_symbols("info")
    ok = get_symbols("check")

    logger.info(
        "%s Building Value decision report.",
        info,
    )
    logger.info(
        "  Deflated trade: %s",
        deflated_trade_path,
    )
    logger.info(
        "  Audit: %s",
        audit_path,
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
            f"SET threads = "
            f"{int(cfg.engine.threads)}"
        )

        # ==============================================================
        # Audit
        # ==============================================================

        audit = pd.read_csv(
            audit_path
        )

        required_audit_columns = {
            "check",
            "passed",
            "value",
            "expected",
            "details",
        }

        missing_audit_columns = (
            required_audit_columns
            - set(audit.columns)
        )

        if missing_audit_columns:
            raise RuntimeError(
                "Value audit is missing required columns: "
                + ", ".join(
                    sorted(
                        missing_audit_columns
                    )
                )
            )

        audit["passed_bool"] = (
            audit["passed"].map(
                _audit_passed_value
            )
        )

        n_audit_checks = int(
            len(audit)
        )

        n_failed_audit_checks = int(
            (~audit["passed_bool"]).sum()
        )

        audit_passed = bool(
            n_audit_checks > 0
            and n_failed_audit_checks == 0
        )

        # ==============================================================
        # Overall output statistics
        # ==============================================================

        overall = con.execute(
            f"""
            SELECT
                COUNT(*) AS n_rows,

                COUNT(*) FILTER (
                    WHERE
                        {nominal_value_column}
                        IS NOT NULL
                ) AS n_nominal_rows,

                COUNT(*) FILTER (
                    WHERE
                        {CONSTANT_VALUE_COLUMN}
                        IS NOT NULL
                ) AS n_constant_rows,

                SUM(
                    {nominal_value_column}
                ) FILTER (
                    WHERE
                        {nominal_value_column}
                        IS NOT NULL
                ) AS nominal_total,

                SUM(
                    {CONSTANT_VALUE_COLUMN}
                ) FILTER (
                    WHERE
                        {CONSTANT_VALUE_COLUMN}
                        IS NOT NULL
                ) AS constant_total,

                MIN(
                    {year_column}
                ) AS minimum_year,

                MAX(
                    {year_column}
                ) AS maximum_year

            FROM read_parquet(
                {output_sql}
            )
            """
        ).fetchone()

        if overall is None:
            raise RuntimeError(
                "Could not calculate Value output statistics."
            )

        (
            n_rows,
            n_nominal_rows,
            n_constant_rows,
            nominal_total,
            constant_total,
            minimum_year,
            maximum_year,
        ) = overall

        n_rows = int(
            n_rows or 0
        )

        n_nominal_rows = int(
            n_nominal_rows or 0
        )

        n_constant_rows = int(
            n_constant_rows or 0
        )

        # ==============================================================
        # Annual summary
        # ==============================================================

        annual = con.execute(
            f"""
            SELECT
                CAST(
                    {year_column}
                    AS INTEGER
                ) AS year,

                COUNT(*) AS rows,

                COUNT(*) FILTER (
                    WHERE
                        {nominal_value_column}
                        IS NOT NULL
                ) AS rows_with_value,

                MAX(
                    {DEFLATOR_COLUMN}
                ) AS gdp_deflator,

                MAX(
                    {DEFLATION_FACTOR_COLUMN}
                ) AS deflation_factor,

                SUM(
                    {nominal_value_column}
                ) FILTER (
                    WHERE
                        {nominal_value_column}
                        IS NOT NULL
                ) AS nominal_total,

                SUM(
                    {CONSTANT_VALUE_COLUMN}
                ) FILTER (
                    WHERE
                        {CONSTANT_VALUE_COLUMN}
                        IS NOT NULL
                ) AS constant_total

            FROM read_parquet(
                {output_sql}
            )

            GROUP BY
                CAST(
                    {year_column}
                    AS INTEGER
                )

            ORDER BY year
            """
        ).fetchdf()

        if annual.empty:
            raise RuntimeError(
                "Value output contains no annual observations."
            )

        # ==============================================================
        # Reference series
        # ==============================================================

        deflator_series = con.execute(
            f"""
            SELECT
                year,
                gdp_deflator
            FROM read_parquet(
                {deflator_sql}
            )
            WHERE year BETWEEN ? AND ?
            ORDER BY year
            """,
            [
                cfg.scope.min_year,
                cfg.scope.max_year,
            ],
        ).fetchdf()

        reference_rows = (
            deflator_series.loc[
                deflator_series[
                    "year"
                ].eq(
                    cfg.method.reference_year
                )
            ]
        )

        if len(reference_rows) != 1:
            raise RuntimeError(
                "Could not resolve exactly one "
                "reference-year GDP deflator "
                "for Value reporting."
            )

        reference_deflator = float(
            reference_rows.iloc[0][
                "gdp_deflator"
            ]
        )

        # ==============================================================
        # Decision
        # ==============================================================

        population_preserved = bool(
            n_rows > 0
            and n_nominal_rows
            == n_constant_rows
        )

        ready_for_validation = bool(
            audit_passed
            and population_preserved
        )

        decision = (
            "APPROVED"
            if ready_for_validation
            else "REVIEW REQUIRED"
        )

        # ==============================================================
        # Tables
        # ==============================================================

        annual_table = (
            annual.copy()
        )

        annual_table[
            "gdp_deflator"
        ] = annual_table[
            "gdp_deflator"
        ].map(
            lambda value: (
                f"{float(value):.6f}"
                if not pd.isna(value)
                else "NA"
            )
        )

        annual_table[
            "deflation_factor"
        ] = annual_table[
            "deflation_factor"
        ].map(
            _format_factor
        )

        annual_table[
            "nominal_total"
        ] = annual_table[
            "nominal_total"
        ].map(
            _format_usd
        )

        annual_table[
            "constant_total"
        ] = annual_table[
            "constant_total"
        ].map(
            _format_usd
        )

        annual_table = (
            annual_table.rename(
                columns={
                    "year": "Year",
                    "rows": "Rows",
                    "rows_with_value": (
                        "Rows with trade value"
                    ),
                    "gdp_deflator": (
                        "US GDP deflator"
                    ),
                    "deflation_factor": (
                        "Deflation factor"
                    ),
                    "nominal_total": (
                        "Nominal USD total"
                    ),
                    "constant_total": (
                        f"Constant "
                        f"{cfg.method.reference_year} "
                        "USD total"
                    ),
                }
            )
        )

        audit_table = (
            audit[
                [
                    "check",
                    "passed_bool",
                    "value",
                    "expected",
                ]
            ]
            .rename(
                columns={
                    "check": "Audit check",
                    "passed_bool": "Passed",
                    "value": "Observed",
                    "expected": "Expected",
                }
            )
        )

        # ==============================================================
        # Figures
        # ==============================================================

        figure_paths = {
            "annual_values": (
                figure_dir
                / (
                    "annual_nominal_vs_"
                    "constant_trade_value.png"
                )
            ),
            "deflation_factor": (
                figure_dir
                / (
                    "annual_usd_"
                    "deflation_factor.png"
                )
            ),
        }

        _plot_annual_nominal_vs_constant(
            annual=annual,
            output_path=(
                figure_paths[
                    "annual_values"
                ]
            ),
            reference_year=(
                cfg.method.reference_year
            ),
        )

        _plot_deflation_factor(
            annual=annual,
            output_path=(
                figure_paths[
                    "deflation_factor"
                ]
            ),
            reference_year=(
                cfg.method.reference_year
            ),
        )

        # ==============================================================
        # Markdown
        # ==============================================================

        lines = [
            "# Value Deflation Decision Report",
            "",
            f"**Decision:** {decision}  ",
            f"**Run ID:** {run_id}  ",
            (
                f"**Target HS:** "
                f"{cfg.scope.target_hs}  "
            ),
            (
                f"**Input period:** "
                f"{cfg.scope.min_year}–"
                f"{cfg.scope.max_year}  "
            ),
            (
                f"**Reference year:** "
                f"{cfg.method.reference_year}  "
            ),
            (
                "**Generated:** "
                f"{datetime.now().isoformat(timespec='seconds')}"
            ),
            "",
            "__TOC__",
            "",
            "",
            "## 1. Purpose",
            "",
            (
                "The Value stage converts the final synchronized "
                "nominal USD trade values produced by Trade Sync into "
                f"constant-{cfg.method.reference_year} USD."
            ),
            "",
            (
                "The transformation adjusts the changes in the general "
                "US price level so that trade values from different "
                "years can be compared on a common monetary basis."
            ),
            "",
            (
                "The original nominal Trade Sync value is preserved. "
                "The Value stage adds the annual GDP deflator, the "
                "corresponding deflation factor, and the resulting "
                "constant-price trade value."
            ),
            "",
            "## 2. Input and output",
            "",
            "| Property | Value |",
            "|---|---|",
            "| Upstream module | Trade Sync |",
            (
                "| Nominal value column | "
                f"`{nominal_value_column}` |"
            ),
            (
                "| Constant-price column | "
                f"`{CONSTANT_VALUE_COLUMN}` |"
            ),
            (
                "| GDP-deflator column | "
                f"`{DEFLATOR_COLUMN}` |"
            ),
            (
                "| Deflation-factor column | "
                f"`{DEFLATION_FACTOR_COLUMN}` |"
            ),
            (
                f"| Covered years | "
                f"{minimum_year}–{maximum_year} |"
            ),
            (
                f"| Output rows | "
                f"{_format_integer(n_rows)} |"
            ),
            (
                f"| Rows with nominal value | "
                f"{_format_integer(n_nominal_rows)} |"
            ),
            (
                f"| Rows with constant value | "
                f"{_format_integer(n_constant_rows)} |"
            ),
            "",
            "## 3. Reference series",
            "",
            (
                "The workflow uses the World Bank US GDP deflator "
                "series `NY.GDP.DEFL.ZS` for the United States."
            ),
            "",
            (
                f"The configured reference year is "
                f"`{cfg.method.reference_year}`, for which the "
                f"deflator is `{reference_deflator:.6f}`."
            ),
            "",
            (
                "A US price index is used because the synchronized monetary values"
                "and are denominated in US dollars and the objective is to express them "
                "and in constant US-dollar purchasing-power terms.” The GDP deflator measures the broad US "
                "price level rather than a trade-specific price index."
            ),
            "",
            "## 4. Deflation method",
            "",
            (
                "For observation year $t$, the constant-price value "
                "is calculated as:"
            ),
            "",
            "$$",
            (
                "V^{constant}_t = "
                "V^{nominal}_t "
                "\\times "
                "\\frac{D_{ref}}{D_t}"
            ),
            "$$",
            "",
            (
                "where $D_t$ is the US GDP deflator for year $t$ "
                f"and $D_{{ref}}$ is the deflator for "
                f"{cfg.method.reference_year}."
            ),
            "",
            (
                f"For {cfg.method.reference_year}, the ratio is "
                "exactly 1, so nominal and constant-price values "
                "are identical in the reference year."
            ),
            "",
            markdown_image(
                "Annual USD deflation factor",
                figure_paths[
                    "deflation_factor"
                ],
                document_path=report_path,
            ),
            "",
            "## 5. Annual results",
            "",
            _markdown_table(
                annual_table
            ),
            "",
            markdown_image(
                (
                    "Annual nominal and constant-price "
                    "trade values"
                ),
                figure_paths[
                    "annual_values"
                ],
                document_path=report_path,
            ),
            "",
            (
                "For years before the reference year, a deflation "
                "factor greater than 1 expresses earlier nominal "
                "dollar values in the purchasing-power terms of "
                f"{cfg.method.reference_year} dollars."
            ),
            "",
            "## 6. Aggregate values",
            "",
            "| Measure | Value |",
            "|---|---:|",
            (
                f"| Nominal synchronized total | "
                f"{_format_usd(nominal_total)} |"
            ),
            (
                f"| Constant "
                f"{cfg.method.reference_year} USD total | "
                f"{_format_usd(constant_total)} |"
            ),
            "",
            (
                "These aggregate totals cover multiple calendar "
                "years. Their difference therefore reflects the "
                "year-specific conversion to the common reference-"
                "year price level and should not be interpreted as "
                "an additional trade flow."
            ),
            "",
            "## 7. Integrity audit",
            "",
            (
                f"The final Value audit executed "
                f"`{n_audit_checks}` checks. "
                f"`{n_audit_checks - n_failed_audit_checks}` passed "
                f"and `{n_failed_audit_checks}` failed."
            ),
            "",
            _markdown_table(
                audit_table
            ),
            "",
            "## 8. Interpretation and limitations",
            "",
            (
                "- Constant-price values are intended for temporal "
                "comparison of synchronized USD trade values."
            ),
            (
                "- The US GDP deflator is a broad economy-wide "
                "price index; it is not a commodity-specific or "
                "international-trade-specific price index."
            ),
            (
                "- Deflation changes the monetary price basis only. "
                "It does not alter the Trade Sync reconciliation, "
                "physical quantities, bilateral identities, or "
                "provenance fields."
            ),
            (
                "- The original nominal synchronized value remains "
                "available alongside the constant-price value."
            ),
            "",
            "## 9. Final decision",
            "",
            (
                f"**{decision}**"
            ),
            "",
            (
                (
                    "✅ The Value output passed the final integrity "
                    "audit and is ready for downstream analysis."
                )
                if ready_for_validation
                else (
                    "⚠️ The Value output did not satisfy all final "
                    "readiness conditions and requires review."
                )
            ),
            "",
            "## 10. Output artifacts",
            "",
            (
                f"- Trade Sync input: "
                f"`{input_path}`"
            ),
            (
                f"- Prepared GDP deflator: "
                f"`{deflator_path}`"
            ),
            (
                f"- Deflated trade: "
                f"`{deflated_trade_path}`"
            ),
            (
                f"- Value audit: "
                f"`{audit_path}`"
            ),
            (
                f"- Report: "
                f"`{report_path}`"
            ),
            "",
        ]

        insert_markdown_toc(
            lines,
            min_level=2,
            max_level=3,
        )

        report_path.write_text(
            "\n".join(lines),
            encoding="utf-8",
        )

        if not valid_output(
            report_path
        ):
            raise RuntimeError(
                "Value decision report was not written correctly."
            )

        latest_report_path: (
            Path | None
        ) = None

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
            "%s Value decision report created.",
            ok,
        )
        logger.info(
            "  Decision: %s",
            decision,
        )
        logger.info(
            "  Ready for validation: %s",
            ready_for_validation,
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
            "output_path": report_path,
            "report_path": report_path,
            "latest_report_path": (
                latest_report_path
            ),
            "figure_paths": list(
                figure_paths.values()
            ),
            "figures": figure_paths,
            "decision": decision,
            "audit_passed": audit_passed,
            "ready_for_validation": (
                ready_for_validation
            ),
            "n_rows": n_rows,
            "n_nominal_value_rows": (
                n_nominal_rows
            ),
            "n_constant_value_rows": (
                n_constant_rows
            ),
            "n_audit_checks": (
                n_audit_checks
            ),
            "n_failed_audit_rows": (
                n_failed_audit_checks
            ),
            "nominal_total": (
                None
                if nominal_total is None
                else float(nominal_total)
            ),
            "constant_total": (
                None
                if constant_total is None
                else float(constant_total)
            ),
            "reference_year": int(
                cfg.method.reference_year
            ),
            "reference_year_deflator": (
                reference_deflator
            ),
        }

    finally:
        con.close()