"""
vacoda.hs_sync.coverage
Functions for calculating and summarizing the coverage of the HS concordance.
"""
from __future__ import annotations

import pandas as pd


def _dataframe_to_markdown_table(df: pd.DataFrame) -> str:
    """Serialize a DataFrame into an aligned GitHub-flavored markdown table."""
    def _sanitize(value: object) -> str:
        text = str(value)
        return text.replace("\n", " ").replace("|", r"\|")

    headers = [str(col) for col in df.columns]

    if not headers:
        return ""

    rows_data = [
        [_sanitize(value) for value in row]
        for row in df.itertuples(index=False, name=None)
    ]

    col_widths = [len(header) for header in headers]
    for row in rows_data:
        for idx, cell in enumerate(row):
            if idx < len(col_widths):
                col_widths[idx] = max(col_widths[idx], len(cell))

    header_row = "| " + " | ".join(
        header.ljust(col_widths[idx]) for idx, header in enumerate(headers)
    ) + " |"
    divider_row = "| " + " | ".join(
        "-" * col_widths[idx] for idx in range(len(headers))
    ) + " |"

    if not rows_data:
        return "\n".join([header_row, divider_row])

    rows: list[str] = []
    for row in rows_data:
        rows.append(
            "| " + " | ".join(
                cell.ljust(col_widths[idx]) for idx, cell in enumerate(row)
            ) + " |"
        )

    return "\n".join([header_row, divider_row, *rows])

def _format_compact_number(value: float) -> str:
    """Format large numbers compactly for human-readable reports."""
    value = float(value)

    abs_value = abs(value)

    if abs_value >= 1_000_000_000_000:
        return f"{value / 1_000_000_000_000:.2f}T"

    if abs_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"

    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"

    if abs_value >= 1_000:
        return f"{value / 1_000:.2f}K"

    return f"{value:.2f}"


def build_target_h5_mapping_interpretation_display_report(
    report: pd.DataFrame,
    *,
    output: str = "dataframe",
) -> pd.DataFrame | str:
    """
    Convert the raw target-H5 interpretation report into a compact
    human-readable display table.

    This is meant for reports, slides, and quick inspection.
    """

    required = {
        "result_class",
        "n_source_hs_code_pairs",
        "share_source_hs_code_pairs_percent",
        "n_rows",
        "value_total",
        "share_rows_percent",
        "share_value_percent",
    }

    missing = required - set(report.columns)
    if missing:
        raise KeyError(
            f"Interpretation report is missing required columns: {sorted(missing)}. "
            f"Available columns: {list(report.columns)}"
        )

    meaning = {
        "Easy without weights": (
            "Already H5, 1:1, or deterministic aggregation"
        ),
        "Needs weights": (
            "Source code may split across multiple H5 targets"
        ),
        "Unsupported source HS": (
            "H2/H3 not covered by direct H4/H6 → H5 mappings"
        ),
        "Missing mapping": (
            "Supported HS system, but no mapping found"
        ),
    }

    display = report.copy()

    display["Rows"] = display["n_rows"].map(lambda x: f"{int(x):,}")
    display["Source HS-code pairs"] = display["n_source_hs_code_pairs"].map(lambda x: f"{int(x):,}")
    display["Source-code share"] = display["share_source_hs_code_pairs_percent"].map(lambda x: f"{x:.2f}%")
    display["Row share"] = display["share_rows_percent"].map(lambda x: f"{x:.2f}%")
    display["Value"] = display["value_total"].map(_format_compact_number)
    display["Value share"] = display["share_value_percent"].map(lambda x: f"{x:.2f}%")
    display["Meaning"] = display["result_class"].map(meaning).fillna(
        display.get("interpretation", "")
    )

    display = display.rename(columns={"result_class": "Result class"})

    display_df = display[
        [
            "Result class",
            "Source HS-code pairs",
            "Source-code share",
            "Rows",
            "Row share",
            "Value",
            "Value share",
            "Meaning",
        ]
    ].reset_index(drop=True)

    output_normalized = output.strip().lower()
    if output_normalized == "dataframe":
        return display_df

    if output_normalized == "markdown":
        return _dataframe_to_markdown_table(display_df)

    raise ValueError(
        "Invalid output for build_target_h5_mapping_interpretation_display_report: "
        f"{output!r}. Expected 'dataframe' or 'markdown'."
    )

def _find_first_existing_column(
    df: pd.DataFrame,
    candidates: list[str],
    *,
    label: str,
) -> str:
    """
    Return the first existing column from candidates.

    This keeps the coverage functions robust against small differences in
    source-summary column naming.
    """
    for col in candidates:
        if col in df.columns:
            return col

    raise KeyError(
        f"Could not find {label} column. Tried: {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


def _normalize_cmd_code_series(series: pd.Series) -> pd.Series:
    """
    Normalize HS commodity codes to six-character strings where possible.

    Handles values read as int/float/string, e.g.:
        440110      -> '440110'
        440110.0    -> '440110'
        '010121'    -> '010121'
    """
    out = series.astype("string").str.strip()

    # Remove trailing .0 from numeric-looking strings.
    out = out.str.replace(r"\.0$", "", regex=True)

    # Keep missing values as missing.
    out = out.mask(out.isin(["", "nan", "None", "<NA>"]))

    # Pad purely numeric codes up to 6 digits.
    is_numeric = out.str.match(r"^[0-9]+$", na=False)
    out = out.where(~is_numeric, out.str.zfill(6))

    return out


def _prepare_mapping_table(
    mapping_df: pd.DataFrame,
    *,
    source_hs: str,
    target_hs: str,
    mapping_columns: dict[str, str],
) -> pd.DataFrame:
    """
    Normalize one Harvard correlation-group mapping file.

    Expected logical columns:
        source_cmd
        target_cmd
        relationship
        adjustment
        group_id
    """

    source_cmd_col = mapping_columns["source_cmd"]
    target_cmd_col = mapping_columns["target_cmd"]
    relationship_col = mapping_columns["relationship"]
    adjustment_col = mapping_columns["adjustment"]
    group_id_col = mapping_columns["group_id"]

    required = [
        source_cmd_col,
        target_cmd_col,
        relationship_col,
        adjustment_col,
        group_id_col,
    ]

    missing = [col for col in required if col not in mapping_df.columns]
    if missing:
        raise KeyError(
            "Mapping table is missing required columns: "
            f"{missing}. Available columns: {list(mapping_df.columns)}"
        )

    out = mapping_df[
        [
            source_cmd_col,
            target_cmd_col,
            relationship_col,
            adjustment_col,
            group_id_col,
        ]
    ].copy()

    out = out.rename(
        columns={
            source_cmd_col: "source_cmd",
            target_cmd_col: "target_cmd",
            relationship_col: "relationship",
            adjustment_col: "adjustment",
            group_id_col: "group_id",
        }
    )

    out["source_hs"] = source_hs
    out["target_hs"] = target_hs
    out["source_cmd"] = _normalize_cmd_code_series(out["source_cmd"])
    out["target_cmd"] = _normalize_cmd_code_series(out["target_cmd"])
    out["relationship"] = out["relationship"].astype("string").str.strip()

    return out


def build_target_h5_source_code_coverage(
    coverage: pd.DataFrame,
) -> pd.DataFrame:
    """
    Collapse detailed source→target mapping coverage to one row per
    source HS/code pair.

    This is the decision-level table for the target-H5 audit.

    It avoids double-counting n_rows/value_total when one source code maps to
    multiple target codes.
    """

    required = {
        "source_hs",
        "source_cmd",
        "target_hs",
        "target_cmd",
        "relationship",
        "mapping_category",
        "conversion_need",
        "first_year",
        "last_year",
        "n_rows",
        "value_total",
    }

    missing = required - set(coverage.columns)
    if missing:
        raise KeyError(
            f"Coverage table is missing required columns: {sorted(missing)}. "
            f"Available columns: {list(coverage.columns)}"
        )

    base_cols = [
        "source_hs",
        "source_cmd",
    ]

    # One source_hs/source_cmd has the same n_rows/value_total repeated
    # across target candidates after the source→target merge.
    # Therefore use first() for row/value metrics, not sum().
    out = (
        coverage.groupby(base_cols, dropna=False)
        .agg(
            first_year=("first_year", "min"),
            last_year=("last_year", "max"),
            target_hs=("target_hs", "first"),
            n_rows=("n_rows", "first"),
            value_total=("value_total", "first"),
            n_target_codes=("target_cmd", lambda x: x.dropna().nunique()),
            target_codes=(
                "target_cmd",
                lambda x: sorted(set(x.dropna().astype(str))),
            ),
            relationship_original=(
                "relationship",
                lambda x: sorted(set(x.dropna().astype(str))),
            ),
            group_ids=(
                "group_id",
                lambda x: sorted(set(x.dropna().astype(str)))
                if "group_id" in coverage.columns
                else [],
            ),
        )
        .reset_index()
    )

    out["target_codes"] = out["target_codes"].map(
        lambda values: ";".join(values)
    )

    out["relationship_original"] = out["relationship_original"].map(
        lambda values: ";".join(values)
    )

    out["group_ids"] = out["group_ids"].map(
        lambda values: ";".join(values)
    )

    # Compute actual mapping category in our configured direction:
    # source_cmd → target_cmd.
    out["actual_mapping_category"] = "missing_mapping"

    already_target = out["source_hs"].eq(out["target_hs"])

    unsupported = coverage.groupby(base_cols, dropna=False)["mapping_category"].first()
    unsupported = unsupported.reset_index()

    out = out.merge(
        unsupported.rename(columns={"mapping_category": "_raw_mapping_category"}),
        on=base_cols,
        how="left",
    )

    out.loc[already_target, "actual_mapping_category"] = "already_target"

    out.loc[
        out["_raw_mapping_category"].eq("unsupported_source_hs"),
        "actual_mapping_category",
    ] = "unsupported_source_hs"

    out.loc[
        out["_raw_mapping_category"].eq("missing_mapping"),
        "actual_mapping_category",
    ] = "missing_mapping"

    mapped = (
        ~already_target
        & ~out["_raw_mapping_category"].isin(
            ["unsupported_source_hs", "missing_mapping"]
        )
    )

    out.loc[
        mapped & out["n_target_codes"].eq(1),
        "actual_mapping_category"] = "actual_to_single_target"

    out.loc[
    mapped & out["n_target_codes"].gt(1),
    "actual_mapping_category"] = "actual_to_multiple_targets"

    # Cleaner category names for interpretation.
    # We cannot fully distinguish actual_1_to_1 from actual_n_to_1 at this
    # source-code level without sources-per-target. For the decision question,
    # both are deterministic and easy. The important distinction is whether
    # one source code maps to one or multiple target codes.
    out["conversion_need"] = "other"

    out.loc[
        out["actual_mapping_category"].isin(
            ["already_target", "actual_to_single_target"]
        ),
        "conversion_need",
    ] = "easy_without_weights"

    out.loc[
        out["actual_mapping_category"].eq("actual_to_multiple_targets"),
        "conversion_need",
    ] = "needs_weights"

    out.loc[
        out["actual_mapping_category"].eq("missing_mapping"),
        "conversion_need",
    ] = "missing"

    out.loc[
        out["actual_mapping_category"].eq("unsupported_source_hs"),
        "conversion_need",
    ] = "unsupported_source_hs"

    out["is_simple_without_weights"] = out["conversion_need"].eq(
        "easy_without_weights"
    )
    out["requires_weights"] = out["conversion_need"].eq("needs_weights")

    out = out.drop(columns=["_raw_mapping_category"])

    ordered_cols = [
        "first_year",
        "last_year",
        "source_hs",
        "source_cmd",
        "target_hs",
        "n_target_codes",
        "target_codes",
        "relationship_original",
        "actual_mapping_category",
        "conversion_need",
        "is_simple_without_weights",
        "requires_weights",
        "group_ids",
        "n_rows",
        "value_total",
    ]

    existing = [col for col in ordered_cols if col in out.columns]
    other = [col for col in out.columns if col not in existing]

    return out[existing + other].sort_values(
        ["conversion_need", "source_hs", "value_total"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

def build_target_h5_mapping_coverage(
    *,
    source_summary: pd.DataFrame,
    mappings: dict[str, pd.DataFrame],
    target_hs: str,
    mapping_columns: dict[str, str],
    easy_without_weights: list[str],
    needs_weights: list[str],
    missing_categories: list[str],
) -> pd.DataFrame:
    """
    Build target-H5 mapping coverage from source trade summary and mapping files.

    This answers:
        How many records / how much value can be converted to target H5
        without optimized conversion weights?

    Classification logic:
        - source_hs == target_hs:
            already_target, easy_without_weights

        - source_hs has direct mapping file and mapping exists:
            relationship from mapping table, e.g. 1:1, n:1, 1:n, n:n

        - source_hs has direct mapping file but source_cmd not found:
            missing_mapping

        - source_hs has no configured mapping file and is not target_hs:
            unsupported_source_hs

    For target H5, expected mapping keys are typically:
        H4_to_H5
        H6_to_H5
    """

    target_hs = target_hs.strip().upper()

    first_year_col = _find_first_existing_column(
    source_summary,
    ["first_year", "year", "RefYear", "refYear"],
    label="first year",
    )

    last_year_col = _find_first_existing_column(
    source_summary,
    ["last_year", "year", "RefYear", "refYear"],
    label="last year",
    )

    hs_col = _find_first_existing_column(
        source_summary,
        ["source_hs", "hs", "ClassificationCode", "classificationCode"],
        label="HS/source HS",
    )

    cmd_col = _find_first_existing_column(
        source_summary,
        ["source_cmd", "cmd", "CmdCode", "cmdCode"],
        label="commodity/source command code",
    )

    rows_col = _find_first_existing_column(
        source_summary,
        ["n_trade_rows", "n_rows", "rows", "row_count", "count"],
        label="row count",
    )

    value_col = _find_first_existing_column(
        source_summary,
        [
        "trade_value_total",
        "value_total",
        "primary_value_total",
        "PrimaryValue",
        "primaryValue",
    ],
    label="value total",
    )
    
    trade = source_summary.copy()

    trade = trade.rename(
        columns={
            first_year_col: "first_year",
            last_year_col: "last_year",
            hs_col: "source_hs",
            cmd_col: "source_cmd",
            rows_col: "n_rows",
            value_col: "value_total",
        }
    )

    trade["source_hs"] = trade["source_hs"].astype("string").str.strip().str.upper()
    trade["source_cmd"] = _normalize_cmd_code_series(trade["source_cmd"])
    trade["first_year"] = pd.to_numeric(trade["first_year"], errors="coerce",).astype("Int64")
    trade["last_year"] = pd.to_numeric(trade["last_year"], errors="coerce",).astype("Int64")
    trade["n_rows"] = pd.to_numeric(trade["n_rows"], errors="coerce").fillna(0).astype("int64")
    trade["value_total"] = pd.to_numeric(
        trade["value_total"],
        errors="coerce",
    ).fillna(0.0)

    prepared_mappings: list[pd.DataFrame] = []

    for mapping_name, mapping_df in mappings.items():
        # Expected names like H4_to_H5, H6_to_H5.
        parts = mapping_name.upper().split("_TO_")

        if len(parts) != 2:
            raise ValueError(
                "Mapping file key must look like 'H4_to_H5' or 'H6_to_H5'. "
                f"Got: {mapping_name}"
            )

        source_hs, mapping_target_hs = parts

        if mapping_target_hs != target_hs:
            raise ValueError(
                f"Mapping {mapping_name} points to {mapping_target_hs}, "
                f"but target_hs is {target_hs}."
            )

        prepared_mappings.append(
            _prepare_mapping_table(
                mapping_df,
                source_hs=source_hs,
                target_hs=mapping_target_hs,
                mapping_columns=mapping_columns,
            )
        )

    if prepared_mappings:
        mapping_all = pd.concat(prepared_mappings, ignore_index=True)
    else:
        mapping_all = pd.DataFrame(
            columns=[
                "source_cmd",
                "target_cmd",
                "relationship",
                "adjustment",
                "group_id",
                "source_hs",
                "target_hs",
            ]
        )

    # One source code can map to multiple target codes.
    # For coverage classification, keep relationship-level rows.
    coverage = trade.merge(
        mapping_all,
        on=["source_hs", "source_cmd"],
        how="left",
        validate="many_to_many",
    )

    configured_source_hs = set(mapping_all["source_hs"].dropna().unique())

    coverage["mapping_category"] = coverage["relationship"].astype("string")
    coverage["target_hs"] = target_hs

    already_target = coverage["source_hs"] == target_hs
    has_mapping = coverage["relationship"].notna()
    source_supported = coverage["source_hs"].isin(configured_source_hs)

    coverage.loc[already_target, "mapping_category"] = "already_target"

    coverage.loc[
        (~already_target) & source_supported & (~has_mapping),
        "mapping_category",
    ] = "missing_mapping"

    coverage.loc[
        (~already_target) & (~source_supported),
        "mapping_category",
    ] = "unsupported_source_hs"

    easy_set = set(easy_without_weights)
    needs_set = set(needs_weights)
    missing_set = set(missing_categories)

    coverage["conversion_need"] = "other"

    coverage.loc[
        coverage["mapping_category"].isin(easy_set),
        "conversion_need",
    ] = "easy_without_weights"

    coverage.loc[
        coverage["mapping_category"].isin(needs_set),
        "conversion_need",
    ] = "needs_weights"

    coverage.loc[
        coverage["mapping_category"].isin(missing_set),
        "conversion_need",
    ] = "missing"

    coverage.loc[
        coverage["mapping_category"] == "unsupported_source_hs",
        "conversion_need",
    ] = "unsupported_source_hs"

    # For already target rows, target_cmd is the source code.
    coverage.loc[already_target, "target_cmd"] = coverage.loc[
        already_target,
        "source_cmd",
    ]

    coverage["target_cmd"] = _normalize_cmd_code_series(coverage["target_cmd"])

    # Indicator useful for summaries.
    coverage["is_simple_without_weights"] = (
        coverage["conversion_need"] == "easy_without_weights"
    )

    coverage["requires_weights"] = coverage["conversion_need"] == "needs_weights"

    ordered_cols = [
    "first_year",
    "last_year",
    "source_hs",
    "source_cmd",
    "target_hs",
    "target_cmd",
    "relationship",
    "mapping_category",
    "conversion_need",
    "is_simple_without_weights",
    "requires_weights",
    "group_id",
    "adjustment",
    "n_rows",
    "value_total",
]

    existing_cols = [col for col in ordered_cols if col in coverage.columns]
    other_cols = [col for col in coverage.columns if col not in existing_cols]

    return coverage[existing_cols + other_cols].reset_index(drop=True)


def summarize_target_h5_mapping_coverage(
    coverage: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize target-H5 mapping coverage by conversion need and category.
    """

    total_rows = coverage["n_rows"].sum()
    total_value = coverage["value_total"].sum()

    group_cols = [
        "conversion_need",
        "mapping_category",
        "source_hs",
    ]

    summary = (
        coverage.groupby(group_cols, dropna=False)
        .agg(
            n_source_codes=("source_cmd", "nunique"),
            n_target_codes=("target_cmd", "nunique"),
            n_rows=("n_rows", "sum"),
            value_total=("value_total", "sum"),
        )
        .reset_index()
    )

    summary["share_rows"] = 0.0
    summary["share_value"] = 0.0

    if total_rows:
        summary["share_rows"] = summary["n_rows"] / total_rows

    if total_value:
        summary["share_value"] = summary["value_total"] / total_value

    summary = summary.sort_values(
        ["conversion_need", "mapping_category", "source_hs"],
    ).reset_index(drop=True)

    return summary


def get_target_h5_missing_mappings(
    coverage: pd.DataFrame,
    *,
    top_n: int | None = 50,
) -> pd.DataFrame:
    """
    Return missing/unsupported source HS-code combinations by value.

    This is useful for answering:
        Which codes block deterministic conversion to target H5?
    """

    missing = coverage[
        coverage["conversion_need"].isin(
            ["missing", "unsupported_source_hs"]
        )
    ].copy()

    if missing.empty:
        return missing

    group_cols = [
        "source_hs",
        "source_cmd",
        "mapping_category",
        "conversion_need",
    ]

    out = (
        missing.groupby(group_cols, dropna=False)
        .agg(
            first_year=("first_year", "min"),
            last_year=("last_year", "max"),
            n_rows=("n_rows", "sum"),
            value_total=("value_total", "sum"),
        )
        .reset_index()
        .sort_values("value_total", ascending=False)
        .reset_index(drop=True)
    )

    if top_n is not None:
        out = out.head(int(top_n))

    return out

def build_hs_concordance_coverage_from_source_summary(
    source_summary: pd.DataFrame,
    concordance_df: pd.DataFrame,
    target_hs: str = "H5",
    weight_col: str = "weight",
    weight_sum_tolerance: float = 0.001,
) -> pd.DataFrame:
    return build_hs_concordance_coverage(
        source_summary=source_summary,
        concordance_df=concordance_df,
        target_hs=target_hs,
        weight_col=weight_col,
        weight_sum_tolerance=weight_sum_tolerance,
    )


def get_one_to_many_hs_mappings(
    coverage: pd.DataFrame,
    *,
    only_value_splits: bool = False,
    top_n: int | None = None,
) -> pd.DataFrame:
    """
    Return source HS/code pairs that map to more than one target code.

    Parameters
    ----------
    coverage:
        Output of build_hs_concordance_coverage().

    only_value_splits:
        If True, keep only mappings where the source value would be split
        across multiple target codes.

    top_n:
        Optional number of rows to return, sorted by trade value.

    Returns
    -------
    pd.DataFrame
        Filtered one-to-many mapping diagnostics.
    """

    required_cols = {
        "mapping_type",
        "produces_value_split",
        "trade_value_total",
        "n_target_codes",
    }

    missing = required_cols - set(coverage.columns)
    if missing:
        raise ValueError(
            f"coverage is missing required columns: {sorted(missing)}"
        )

    out = coverage.loc[
        coverage["mapping_type"].eq("one_to_many")
    ].copy()

    if only_value_splits:
        out = out.loc[out["produces_value_split"]].copy()

    out = out.sort_values(
        ["trade_value_total", "n_target_codes"],
        ascending=[False, False],
    ).reset_index(drop=True)

    if top_n is not None:
        out = out.head(top_n)

    return out

def build_hs_concordance_coverage_from_source_summary_deprecated(
    source_summary: pd.DataFrame,
    concordance_df: pd.DataFrame,
    target_hs: str = "H5",
    weight_col: str = "weight",
    weight_sum_tolerance: float = 0.001,
) -> pd.DataFrame:
    """
    Build HS concordance coverage diagnostics from a compact source-pair summary.

    source_summary must contain:
        source_hs
        source_cmd
        n_trade_rows
        first_year
        last_year
        trade_value_total
    """

    required_source_cols = {
        "source_hs",
        "source_cmd",
        "n_trade_rows",
        "first_year",
        "last_year",
        "trade_value_total",
    }

    missing_source = required_source_cols - set(source_summary.columns)
    if missing_source:
        raise ValueError(
            f"source_summary missing required columns: {sorted(missing_source)}"
        )

    required_concordance_cols = {
        "source_hs",
        "source_cmd",
        "target_hs",
        "target_cmd",
        weight_col,
    }

    missing_concordance = required_concordance_cols - set(concordance_df.columns)
    if missing_concordance:
        raise ValueError(
            "concordance_df missing required columns: "
            f"{sorted(missing_concordance)}"
        )

    src = source_summary.copy()
    src["source_hs"] = src["source_hs"].astype("string")
    src["source_cmd"] = src["source_cmd"].astype("string")

    conc = concordance_df.copy()
    conc["source_hs"] = conc["source_hs"].astype("string")
    conc["source_cmd"] = conc["source_cmd"].astype("string")
    conc["target_hs"] = conc["target_hs"].astype("string")
    conc["target_cmd"] = conc["target_cmd"].astype("string")
    conc[weight_col] = pd.to_numeric(conc[weight_col], errors="coerce")

    conc_target = conc.loc[conc["target_hs"] == target_hs].copy()

    mapping_summary = (
        conc_target.groupby(["source_hs", "source_cmd"], dropna=False, as_index=False)
        .agg(
            n_mapping_rows=("target_cmd", "size"),
            n_target_codes=("target_cmd", "nunique"),
            target_codes=("target_cmd", lambda x: sorted(set(x.dropna()))),
            weight_sum=(weight_col, "sum"),
            min_weight=(weight_col, "min"),
            max_weight=(weight_col, "max"),
        )
    )

    out = src.merge(
        mapping_summary,
        on=["source_hs", "source_cmd"],
        how="left",
    )

    out["has_mapping"] = out["n_mapping_rows"].notna()

    out["n_mapping_rows"] = out["n_mapping_rows"].fillna(0).astype("int64")
    out["n_target_codes"] = out["n_target_codes"].fillna(0).astype("int64")

    out["is_already_target_hs"] = out["source_hs"].eq(target_hs)

    out["mapping_type"] = "missing"

    out.loc[
        out["is_already_target_hs"],
        "mapping_type",
    ] = "already_target"

    out.loc[
        out["has_mapping"] & (out["n_target_codes"] == 1),
        "mapping_type",
    ] = "one_to_one"

    out.loc[
        out["has_mapping"] & (out["n_target_codes"] > 1),
        "mapping_type",
    ] = "one_to_many"

    out["produces_value_split"] = (
        out["has_mapping"]
        & (out["n_target_codes"] > 1)
        & out["min_weight"].notna()
        & out["max_weight"].notna()
        & (out["min_weight"] < 1.0)
    )

    lower = 1.0 - weight_sum_tolerance
    upper = 1.0 + weight_sum_tolerance

    out["weight_sum_status"] = "not_applicable"

    out.loc[
        out["has_mapping"],
        "weight_sum_status",
    ] = "check"

    out.loc[
        out["has_mapping"] & out["weight_sum"].between(lower, upper),
        "weight_sum_status",
    ] = "ok_approx_1"

    out.loc[
        out["has_mapping"] & ~out["weight_sum"].between(lower, upper),
        "weight_sum_status",
    ] = "not_1"

    out = out.sort_values(
        ["has_mapping", "source_hs", "trade_value_total", "source_cmd"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)

    return out

def get_missing_hs_mappings(
    coverage: pd.DataFrame,
    top_n: int | None = 50,
) -> pd.DataFrame:
    """
    Return source HS/code pairs that appear in the trade data
    but have no concordance into the target HS system.
    """

    missing = coverage.loc[~coverage["has_mapping"]].copy()

    missing = missing.sort_values(
        ["trade_value_total", "n_trade_rows"],
        ascending=[False, False],
    )

    if top_n is not None:
        missing = missing.head(top_n)

    return missing

def build_target_h5_mapping_interpretation_report(
    source_code_coverage: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a concise interpretation report from source-code-level coverage.

    This report is based on one row per source HS/code pair, so rows/value are
    not double-counted when one source code maps to multiple targets.
    """

    required = {
        "conversion_need",
        "source_hs",
        "source_cmd",
        "n_rows",
        "value_total",
    }

    missing = required - set(source_code_coverage.columns)
    if missing:
        raise KeyError(
            f"Source-code coverage table is missing required columns: "
            f"{sorted(missing)}. Available columns: "
            f"{list(source_code_coverage.columns)}"
        )

    total_rows = source_code_coverage["n_rows"].sum()
    total_value = source_code_coverage["value_total"].sum()

    total_source_hs_code_pairs = (
        source_code_coverage[["source_hs", "source_cmd"]]
        .drop_duplicates()
        .shape[0]
    )

    class_map = {
        "easy_without_weights": {
            "result_class": "Easy without weights",
            "interpretation": (
                "Already H5, 1:1, or deterministic aggregation"
            ),
            "sort_order": 1,
        },
        "needs_weights": {
            "result_class": "Needs weights",
            "interpretation": (
                "Source code may split across multiple H5 targets"
            ),
            "sort_order": 2,
        },
        "unsupported_source_hs": {
            "result_class": "Unsupported source HS",
            "interpretation": (
                "H2/H3 not covered by direct H4/H6 → H5 mappings"
            ),
            "sort_order": 3,
        },
        "missing": {
            "result_class": "Missing mapping",
            "interpretation": (
                "Supported source HS, but no H5 mapping was found"
            ),
            "sort_order": 4,
        },
    }

    rows: list[dict[str, object]] = []

    for conversion_need, group in source_code_coverage.groupby(
        "conversion_need",
        dropna=False,
    ):
        key = str(conversion_need)

        meta = class_map.get(
            key,
            {
                "result_class": key,
                "interpretation": "Other mapping category",
                "sort_order": 99,
            },
        )

        n_rows = int(group["n_rows"].sum())
        value_total = float(group["value_total"].sum())

        n_source_hs_code_pairs = (
            group[["source_hs", "source_cmd"]]
            .drop_duplicates()
            .shape[0]
        )

        share_source_hs_code_pairs = (
            n_source_hs_code_pairs / total_source_hs_code_pairs
            if total_source_hs_code_pairs
            else 0.0
        )

        n_source_codes = group["source_cmd"].nunique()
        n_source_hs = group["source_hs"].nunique()

        share_rows = n_rows / total_rows if total_rows else 0.0
        share_value = value_total / total_value if total_value else 0.0

        rows.append(
            {
                "result_class": meta["result_class"],
                "conversion_need": key,
                "n_source_hs_code_pairs": int(n_source_hs_code_pairs),
                "share_source_hs_code_pairs": share_source_hs_code_pairs,
                "share_source_hs_code_pairs_percent": (
                    share_source_hs_code_pairs * 100
                ),
                "n_source_codes": int(n_source_codes),
                "n_source_hs": int(n_source_hs),
                "n_rows": n_rows,
                "value_total": value_total,
                "share_rows": share_rows,
                "share_value": share_value,
                "share_rows_percent": share_rows * 100,
                "share_value_percent": share_value * 100,
                "interpretation": meta["interpretation"],
                "sort_order": meta["sort_order"],
            }
        )

    report = pd.DataFrame(rows)

    return (
        report.sort_values("sort_order")
        .drop(columns=["sort_order"])
        .reset_index(drop=True)
    )
    
def build_hs_concordance_coverage(
    source_summary: pd.DataFrame,
    concordance_df: pd.DataFrame,
    target_hs: str = "H5",
    weight_col: str = "weight",
    weight_sum_tolerance: float = 0.001) -> pd.DataFrame:
    """
    Build HS concordance coverage diagnostics from a compact source-pair summary.

    Expected source_summary columns:
        source_hs
        source_cmd
        n_trade_rows
        first_year
        last_year
        trade_value_total

    Expected concordance_df columns:
        source_hs
        source_cmd
        target_hs
        target_cmd
        weight
    """

    required_source_cols = {
        "source_hs",
        "source_cmd",
        "n_trade_rows",
        "first_year",
        "last_year",
        "trade_value_total",
    }

    missing_source = required_source_cols - set(source_summary.columns)
    if missing_source:
        raise ValueError(
            f"source_summary missing required columns: {sorted(missing_source)}"
        )

    required_concordance_cols = {
        "source_hs",
        "source_cmd",
        "target_hs",
        "target_cmd",
        weight_col,
    }

    missing_concordance = required_concordance_cols - set(concordance_df.columns)
    if missing_concordance:
        raise ValueError(
            "concordance_df missing required columns: "
            f"{sorted(missing_concordance)}"
        )

    src = source_summary.copy()
    src["source_hs"] = src["source_hs"].astype("string")
    src["source_cmd"] = src["source_cmd"].astype("string")

    conc = concordance_df.copy()
    conc["source_hs"] = conc["source_hs"].astype("string")
    conc["source_cmd"] = conc["source_cmd"].astype("string")
    conc["target_hs"] = conc["target_hs"].astype("string")
    conc["target_cmd"] = conc["target_cmd"].astype("string")
    conc[weight_col] = pd.to_numeric(conc[weight_col], errors="coerce")

    conc_target = conc.loc[conc["target_hs"] == target_hs].copy()

    mapping_summary = (
        conc_target.groupby(["source_hs", "source_cmd"], dropna=False, as_index=False)
        .agg(
            n_mapping_rows=("target_cmd", "size"),
            n_target_codes=("target_cmd", "nunique"),
            target_codes=("target_cmd", lambda x: sorted(set(x.dropna()))),
            weight_sum=(weight_col, "sum"),
            min_weight=(weight_col, "min"),
            max_weight=(weight_col, "max"),
        )
    )

    out = src.merge(
        mapping_summary,
        on=["source_hs", "source_cmd"],
        how="left",
    )

    out["has_mapping"] = out["n_mapping_rows"].notna()

    out["n_mapping_rows"] = out["n_mapping_rows"].fillna(0).astype("int64")
    out["n_target_codes"] = out["n_target_codes"].fillna(0).astype("int64")

    out["is_already_target_hs"] = out["source_hs"].eq(target_hs)

    out["mapping_type"] = "missing"

    out.loc[
        out["is_already_target_hs"],
        "mapping_type",
    ] = "already_target"

    out.loc[
        out["has_mapping"] & (out["n_target_codes"] == 1),
        "mapping_type",
    ] = "one_to_one"

    out.loc[
        out["has_mapping"] & (out["n_target_codes"] > 1),
        "mapping_type",
    ] = "one_to_many"

    out["produces_value_split"] = (
        out["has_mapping"]
        & (out["n_target_codes"] > 1)
        & out["min_weight"].notna()
        & out["max_weight"].notna()
        & (out["min_weight"] < 1.0)
    )

    lower = 1.0 - weight_sum_tolerance
    upper = 1.0 + weight_sum_tolerance

    out["weight_sum_status"] = "not_applicable"

    out.loc[
        out["has_mapping"],
        "weight_sum_status",
    ] = "check"

    out.loc[
        out["has_mapping"] & out["weight_sum"].between(lower, upper),
        "weight_sum_status",
    ] = "ok_approx_1"

    out.loc[
        out["has_mapping"] & ~out["weight_sum"].between(lower, upper),
        "weight_sum_status",
    ] = "not_1"

    out = out.sort_values(
        ["has_mapping", "source_hs", "trade_value_total", "source_cmd"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)

    return out