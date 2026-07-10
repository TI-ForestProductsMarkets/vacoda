"""
vacoda.hs_sync.allocation

Build pluggable HS-sync allocation weights for non-deterministic
source-to-target HS mappings.

The default strategy is equal_split:

    weight = 1 / number of distinct target codes for each source HS-code pair

This module only builds allocation weights. It does not apply them to raw trade
rows. That keeps the allocation strategy replaceable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import pandas as pd

from vacoda.utils.logger import get_logger
from vacoda.config.hs_sync import HsSyncConfig


logger = get_logger("HS_SYNC.ALLOCATION")


REQUIRED_MAPPING_COLUMNS = {
    "source_hs",
    "target_hs",
    "code.source",
    "code.target",
    "Relationship",
}


ALLOCATION_RELATIONSHIPS = {"1:n", "n:n"}


@dataclass(frozen=True)
class AllocationResult:
    """
    Container for allocation outputs.
    """

    weights: pd.DataFrame
    audit: pd.DataFrame


class AllocationStrategy(Protocol):
    """
    Protocol for replaceable HS-sync allocation strategies.
    """

    name: str

    def build_weights(self, mapping: pd.DataFrame) -> AllocationResult:
        """
        Build source -> target allocation weights.

        Parameters
        ----------
        mapping:
            Normalized mapping table with at least:
            source_hs, target_hs, code.source, code.target, Relationship.

        Returns
        -------
        AllocationResult
            weights:
                One row per source_hs/source_cmd/target_hs/target_cmd.
            audit:
                Source-code-level audit of weight sums and target counts.
        """


@dataclass(frozen=True)
class EqualSplitAllocation:
    """
    Equal-split allocation strategy.

    For every source HS-code pair that maps to multiple target codes, allocate
    uniformly across distinct target codes.
    """

    name: str = "equal_split"

    def build_weights(self, mapping: pd.DataFrame) -> AllocationResult:
        _validate_mapping_columns(mapping)

        candidate = mapping[
            mapping["Relationship"].isin(ALLOCATION_RELATIONSHIPS)
        ].copy()

        if candidate.empty:
            logger.warning("No non-deterministic mapping rows found.")
            return AllocationResult(
                weights=_empty_weights_frame(),
                audit=_empty_audit_frame(),
            )

        candidate["source_cmd"] = candidate["code.source"].astype(str).str.zfill(6)
        candidate["target_cmd"] = candidate["code.target"].astype(str).str.zfill(6)

        candidate = candidate[
            [
                "source_hs",
                "source_cmd",
                "target_hs",
                "target_cmd",
                "Relationship",
            ]
        ].drop_duplicates()

        # Count distinct target codes for each observed source HS-code pair.
        target_counts = (
            candidate.groupby(["source_hs", "source_cmd", "target_hs"], as_index=False)
            .agg(n_target_codes=("target_cmd", "nunique"))
        )

        weights = candidate.merge(
            target_counts,
            on=["source_hs", "source_cmd", "target_hs"],
            how="left",
            validate="many_to_one",
        )

        weights["allocation_weight"] = 1.0 / weights["n_target_codes"]
        weights["allocation_strategy"] = self.name
        weights["allocation_is_assumption"] = True

        weights = weights[
            [
                "source_hs",
                "source_cmd",
                "target_hs",
                "target_cmd",
                "Relationship",
                "n_target_codes",
                "allocation_weight",
                "allocation_strategy",
                "allocation_is_assumption",
            ]
        ].sort_values(
            [
                "source_hs",
                "source_cmd",
                "target_hs",
                "target_cmd",
            ]
        )

        audit = _build_weight_audit(weights)

        return AllocationResult(
            weights=weights.reset_index(drop=True),
            audit=audit,
        )


def _validate_mapping_columns(mapping: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_MAPPING_COLUMNS - set(mapping.columns))
    if missing:
        raise KeyError(
            "Mapping table is missing required allocation columns: "
            f"{missing}. Available columns: {list(mapping.columns)}"
        )


def _empty_weights_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "source_hs",
            "source_cmd",
            "target_hs",
            "target_cmd",
            "Relationship",
            "n_target_codes",
            "allocation_weight",
            "allocation_strategy",
            "allocation_is_assumption",
        ]
    )


def _empty_audit_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "source_hs",
            "source_cmd",
            "target_hs",
            "n_target_codes",
            "n_weight_rows",
            "weight_sum",
            "status",
        ]
    )


def _build_weight_audit(weights: pd.DataFrame) -> pd.DataFrame:
    audit = (
        weights.groupby(["source_hs", "source_cmd", "target_hs"], as_index=False)
        .agg(
            n_target_codes=("target_cmd", "nunique"),
            n_weight_rows=("target_cmd", "size"),
            weight_sum=("allocation_weight", "sum"),
        )
    )

    audit["status"] = audit["weight_sum"].sub(1.0).abs().le(1e-12).map(
        {
            True: "pass",
            False: "fail",
        }
    )

    return audit


def build_allocation_weights(
    *,
    mapping: pd.DataFrame,
    strategy: AllocationStrategy | None = None,
) -> AllocationResult:
    """
    Build allocation weights using a pluggable strategy.

    If no strategy is supplied, equal_split is used.
    """

    if strategy is None:
        strategy = EqualSplitAllocation()

    logger.info("Building HS-sync allocation weights.")
    logger.info("Allocation strategy: %s", strategy.name)

    result = strategy.build_weights(mapping)

    failed = result.audit[result.audit["status"] == "fail"]
    if not failed.empty:
        raise ValueError(
            "Allocation weight audit failed. Some source-code weights do not "
            "sum to 1.0:\n"
            + failed.to_string(index=False)
        )

    logger.info("Allocation weights built successfully.")
    logger.info("  Weight rows: %s", f"{len(result.weights):,}")
    logger.info("  Source-code pairs: %s", f"{len(result.audit):,}")

    return result


def write_allocation_outputs(
    *,
    weights: pd.DataFrame,
    audit: pd.DataFrame,
    output_dir: Path | None = None,
    filename_prefix: str | None = None,
    weights_path: Path | None = None,
    audit_path: Path | None = None,
    overwrite: bool = True,
) -> dict[str, Path]:
    """
    Write allocation weights and audit tables.
    """

    if weights_path is None or audit_path is None:
        if output_dir is None or filename_prefix is None:
            raise ValueError(
                "Either explicit output paths or output_dir/filename_prefix must be provided."
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        if weights_path is None:
            weights_path = output_dir / f"{filename_prefix}_allocation_weights.csv"
        if audit_path is None:
            audit_path = output_dir / f"{filename_prefix}_allocation_audit.csv"
    else:
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.parent.mkdir(parents=True, exist_ok=True)

    if not overwrite:
        for path in [weights_path, audit_path]:
            if path.exists():
                raise FileExistsError(f"Output already exists: {path}")

    weights.to_csv(weights_path, index=False)
    audit.to_csv(audit_path, index=False)

    logger.info("Wrote allocation weights: %s", weights_path)
    logger.info("Wrote allocation audit: %s", audit_path)

    return {
        "weights": weights_path,
        "audit": audit_path,
    }

def build_observed_needs_mapping(
    *,
    mapping_tables: dict[str, pd.DataFrame],
    source_code_coverage: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build mapping rows only for observed source HS-code pairs that need weights.

    Parameters
    ----------
    mapping_tables:
        Dict from mapping key, e.g. H2_to_H5, to normalized mapping DataFrame.

    source_code_coverage:
        Coverage table with one row per observed source_hs/source_cmd.

    Returns
    -------
    pd.DataFrame
        Mapping rows for observed non-deterministic source codes only.
    """

    required_source_cols = {
        "source_hs",
        "source_cmd",
        "conversion_need",
    }

    missing = sorted(required_source_cols - set(source_code_coverage.columns))
    if missing:
        raise KeyError(
            "source_code_coverage missing required columns: "
            f"{missing}. Available columns: {list(source_code_coverage.columns)}"
        )

    needs = source_code_coverage[
        source_code_coverage["conversion_need"] == "needs_weights"
    ][["source_hs", "source_cmd"]].copy()

    needs["source_cmd"] = needs["source_cmd"].astype(str).str.zfill(6)

    parts: list[pd.DataFrame] = []

    for mapping_key, mapping in mapping_tables.items():
        _validate_mapping_columns(mapping)

        tmp = mapping.copy()
        tmp["source_cmd"] = tmp["code.source"].astype(str).str.zfill(6)

        tmp = tmp.merge(
            needs,
            on=["source_hs", "source_cmd"],
            how="inner",
            validate="many_to_one",
        )

        tmp["mapping_key"] = mapping_key
        parts.append(tmp)

    if not parts:
        return pd.DataFrame()

    observed_mapping = pd.concat(parts, ignore_index=True)

    return observed_mapping

def build_deterministic_weight_rows(
    *,
    source_code_coverage: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build weight rows for deterministic/easy source HS-code pairs.

    These are source-code pairs whose H5 target is known without allocation:
    - already target HS
    - 1:1 mapping
    - n:1 deterministic aggregation

    Output grain:
        source_hs, source_cmd, target_hs, target_cmd
    """

    required = {
        "source_hs",
        "source_cmd",
        "target_hs",
        "target_codes",
        "conversion_need",
        "n_target_codes",
    }

    missing = sorted(required - set(source_code_coverage.columns))
    if missing:
        raise KeyError(
            "source_code_coverage is missing required columns for deterministic "
            f"weights: {missing}. Available columns: "
            f"{list(source_code_coverage.columns)}"
        )

    easy = source_code_coverage[
        source_code_coverage["conversion_need"] == "easy_without_weights"
    ].copy()

    if easy.empty:
        logger.warning("No easy_without_weights rows found.")
        return _empty_complete_weights_frame()

    easy["source_cmd"] = easy["source_cmd"].astype(str).str.zfill(6)

    # For already-target rows, target_codes should normally be the source code.
    # Still, use target_codes as the canonical coverage output.
    easy["target_codes"] = easy["target_codes"].fillna("").astype(str).str.strip()

    missing_target_codes = easy["target_codes"].eq("")
    if missing_target_codes.any():
        # Conservative fallback only for already-target rows.
        fallback_mask = missing_target_codes & easy["source_hs"].eq(easy["target_hs"])
        easy.loc[fallback_mask, "target_codes"] = easy.loc[
            fallback_mask,
            "source_cmd",
        ]

    # Deterministic rows should have exactly one target code.
    # If not, fail loudly: that means coverage classification or target_codes
    # aggregation is inconsistent.
    easy["target_code_list"] = easy["target_codes"].str.split(";")
    easy["n_target_codes_from_list"] = easy["target_code_list"].map(
        lambda values: len([v for v in values if str(v).strip()])
    )

    bad = easy[easy["n_target_codes_from_list"] != 1]
    if not bad.empty:
        raise ValueError(
            "Deterministic/easy source-code pairs should have exactly one "
            "target code, but some do not:\n"
            + bad[
                [
                    "source_hs",
                    "source_cmd",
                    "target_hs",
                    "target_codes",
                    "conversion_need",
                    "n_target_codes",
                ]
            ]
            .head(20)
            .to_string(index=False)
        )

    easy["target_cmd"] = easy["target_code_list"].map(lambda values: values[0])
    easy["target_cmd"] = easy["target_cmd"].astype(str).str.zfill(6)

    out = pd.DataFrame(
        {
            "source_hs": easy["source_hs"],
            "source_cmd": easy["source_cmd"],
            "target_hs": easy["target_hs"],
            "target_cmd": easy["target_cmd"],
            "conversion_need": easy["conversion_need"],
            "hs_sync_method": easy.apply(_deterministic_method_label, axis=1),
            "relationship_original": easy.get("relationship_original", pd.NA),
            "actual_mapping_category": easy.get("actual_mapping_category", pd.NA),
            "n_target_codes": 1,
            "allocation_weight": 1.0,
            "allocation_strategy": "deterministic",
            "allocation_is_assumption": False,
        }
    )

    return out.reset_index(drop=True)


def _deterministic_method_label(row: pd.Series) -> str:
    """
    Assign a readable method label for deterministic rows.
    """
    source_hs = str(row.get("source_hs", ""))
    target_hs = str(row.get("target_hs", ""))
    relationship = str(row.get("relationship_original", ""))

    if source_hs == target_hs:
        return "already_target"

    if relationship == "1:1":
        return "direct_1_to_1"

    if relationship == "n:1":
        return "deterministic_n_to_1"

    return "deterministic_without_weights"


def normalize_allocation_weight_rows(
    *,
    allocation_weights: pd.DataFrame,
    source_code_coverage: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize non-deterministic allocation weights into the complete
    mapping-weight-table schema.

    The allocation_weights input is produced by EqualSplitAllocation.
    """

    required = {
        "source_hs",
        "source_cmd",
        "target_hs",
        "target_cmd",
        "Relationship",
        "n_target_codes",
        "allocation_weight",
        "allocation_strategy",
        "allocation_is_assumption",
    }

    missing = sorted(required - set(allocation_weights.columns))
    if missing:
        raise KeyError(
            "allocation_weights missing required columns: "
            f"{missing}. Available columns: {list(allocation_weights.columns)}"
        )

    weights = allocation_weights.copy()

    weights["source_cmd"] = weights["source_cmd"].astype(str).str.zfill(6)
    weights["target_cmd"] = weights["target_cmd"].astype(str).str.zfill(6)

    # Attach source-code-level coverage metadata.
    coverage_cols = [
        col
        for col in [
            "source_hs",
            "source_cmd",
            "conversion_need",
            "relationship_original",
            "actual_mapping_category",
        ]
        if col in source_code_coverage.columns
    ]

    coverage_meta = source_code_coverage[coverage_cols].copy()
    coverage_meta["source_cmd"] = coverage_meta["source_cmd"].astype(str).str.zfill(6)

    weights = weights.merge(
        coverage_meta,
        on=["source_hs", "source_cmd"],
        how="left",
        validate="many_to_one",
    )

    missing_conversion = weights["conversion_need"].isna().sum()
    if missing_conversion:
        raise ValueError(
            "Some allocation-weight rows could not be matched back to "
            f"source_code_coverage. Missing rows: {missing_conversion}"
        )

    weights["hs_sync_method"] = weights["Relationship"].map(
        {
            "1:n": "equal_split_1_to_n",
            "n:n": "equal_split_n_to_n",
            "1:1": "equal_split_unexpected_1_to_1",
            "n:1": "equal_split_unexpected_n_to_1",
        }
    )

    out = pd.DataFrame(
        {
            "source_hs": weights["source_hs"],
            "source_cmd": weights["source_cmd"],
            "target_hs": weights["target_hs"],
            "target_cmd": weights["target_cmd"],
            "conversion_need": weights["conversion_need"],
            "hs_sync_method": weights["hs_sync_method"],
            "relationship_original": weights.get(
                "relationship_original",
                weights["Relationship"],
            ),
            "actual_mapping_category": weights.get(
                "actual_mapping_category",
                pd.NA,
            ),
            "n_target_codes": pd.to_numeric(
                weights["n_target_codes"],
                errors="coerce",
            ).astype("Int64"),
            "allocation_weight": pd.to_numeric(
                weights["allocation_weight"],
                errors="coerce",
            ),
            "allocation_strategy": weights["allocation_strategy"],
            "allocation_is_assumption": weights["allocation_is_assumption"],
        }
    )

    return out.reset_index(drop=True)


def build_complete_mapping_weight_table(
    *,
    source_code_coverage: pd.DataFrame,
    allocation_weights: pd.DataFrame,
) -> AllocationResult:
    """
    Build the complete target-HS mapping-weight table.

    Combines:
    - deterministic/easy mappings with weight = 1.0
    - non-deterministic mappings with supplied allocation weights

    Returns
    -------
    AllocationResult
        weights:
            Complete observed source→target weight table.
        audit:
            Source-code-level audit. Every observed source pair should have
            weight_sum = 1.0.
    """

    deterministic = build_deterministic_weight_rows(
        source_code_coverage=source_code_coverage,
    )

    allocated = normalize_allocation_weight_rows(
        allocation_weights=allocation_weights,
        source_code_coverage=source_code_coverage,
    )

    complete = pd.concat(
        [deterministic, allocated],
        ignore_index=True,
        sort=False,
    )

    complete["source_cmd"] = complete["source_cmd"].astype(str).str.zfill(6)
    complete["target_cmd"] = complete["target_cmd"].astype(str).str.zfill(6)

    complete["allocation_weight"] = pd.to_numeric(
        complete["allocation_weight"],
        errors="coerce",
    )

    complete = complete.sort_values(
        [
            "source_hs",
            "source_cmd",
            "target_hs",
            "target_cmd",
        ]
    ).reset_index(drop=True)

    audit = build_complete_mapping_weight_audit(
        complete_weights=complete,
        source_code_coverage=source_code_coverage,
    )

    failed = audit[audit["status"] == "fail"]
    if not failed.empty:
        raise ValueError(
            "Complete mapping-weight-table audit failed:\n"
            + failed.to_string(index=False)
        )

    logger.info("Complete mapping-weight table built successfully.")
    logger.info("  Weight rows: %s", f"{len(complete):,}")
    logger.info(
        "  Source-code pairs: %s",
        f"{complete[['source_hs', 'source_cmd']].drop_duplicates().shape[0]:,}",
    )

    return AllocationResult(
        weights=complete,
        audit=audit,
    )


def build_complete_mapping_weight_audit(
    *,
    complete_weights: pd.DataFrame,
    source_code_coverage: pd.DataFrame,
) -> pd.DataFrame:
    """
    Audit the complete mapping-weight table.

    Core invariant:
        for every observed source_hs/source_cmd pair,
        sum(allocation_weight) == 1.0
    """

    required_weights = {
        "source_hs",
        "source_cmd",
        "target_hs",
        "target_cmd",
        "allocation_weight",
    }

    missing_weights = sorted(required_weights - set(complete_weights.columns))
    if missing_weights:
        raise KeyError(
            "complete_weights missing required columns: "
            f"{missing_weights}. Available columns: {list(complete_weights.columns)}"
        )

    required_source = {
        "source_hs",
        "source_cmd",
        "conversion_need",
    }

    missing_source = sorted(required_source - set(source_code_coverage.columns))
    if missing_source:
        raise KeyError(
            "source_code_coverage missing required columns: "
            f"{missing_source}. Available columns: {list(source_code_coverage.columns)}"
        )

    weights = complete_weights.copy()
    source = source_code_coverage.copy()

    weights["source_cmd"] = weights["source_cmd"].astype(str).str.zfill(6)
    source["source_cmd"] = source["source_cmd"].astype(str).str.zfill(6)

    audit = (
        weights.groupby(["source_hs", "source_cmd", "target_hs"], as_index=False)
        .agg(
            n_weight_rows=("target_cmd", "size"),
            n_target_codes=("target_cmd", "nunique"),
            weight_sum=("allocation_weight", "sum"),
            min_weight=("allocation_weight", "min"),
            max_weight=("allocation_weight", "max"),
            allocation_strategies=(
                "allocation_strategy",
                lambda s: ";".join(sorted(set(s.astype(str)))),
            ),
        )
    )

    source_pairs = source[
        [
            "source_hs",
            "source_cmd",
            "conversion_need",
        ]
    ].drop_duplicates()

    audit = source_pairs.merge(
        audit,
        on=["source_hs", "source_cmd"],
        how="left",
        validate="one_to_one",
    )

    audit["has_weight_rows"] = audit["n_weight_rows"].notna()
    audit["weight_sum"] = pd.to_numeric(audit["weight_sum"], errors="coerce")

    audit["status"] = "pass"

    audit.loc[~audit["has_weight_rows"], "status"] = "fail"
    audit.loc[
        audit["has_weight_rows"] & audit["weight_sum"].sub(1.0).abs().gt(1e-12),
        "status",
    ] = "fail"

    audit["detail"] = ""

    audit.loc[~audit["has_weight_rows"], "detail"] = "missing weight rows"
    audit.loc[
        audit["has_weight_rows"] & audit["weight_sum"].sub(1.0).abs().gt(1e-12),
        "detail",
    ] = "weights do not sum to 1"

    audit = audit[
        [
            "source_hs",
            "source_cmd",
            "conversion_need",
            "target_hs",
            "n_target_codes",
            "n_weight_rows",
            "weight_sum",
            "min_weight",
            "max_weight",
            "allocation_strategies",
            "status",
            "detail",
        ]
    ].sort_values(["status", "source_hs", "source_cmd"])

    return audit.reset_index(drop=True)


def _empty_complete_weights_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "source_hs",
            "source_cmd",
            "target_hs",
            "target_cmd",
            "conversion_need",
            "hs_sync_method",
            "relationship_original",
            "actual_mapping_category",
            "n_target_codes",
            "allocation_weight",
            "allocation_strategy",
            "allocation_is_assumption",
        ]
    )


def write_complete_mapping_weight_outputs(
    *,
    weights: pd.DataFrame,
    audit: pd.DataFrame,
    output_dir: Path | None = None,
    filename_prefix: str | None = None,
    weights_path: Path | None = None,
    audit_path: Path | None = None,
    overwrite: bool = True,
) -> dict[str, Path]:
    """
    Write complete mapping-weight table and audit.
    """

    if weights_path is None or audit_path is None:
        if output_dir is None or filename_prefix is None:
            raise ValueError(
                "Either explicit output paths or output_dir/filename_prefix must be provided."
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        if weights_path is None:
            weights_path = output_dir / f"{filename_prefix}_complete_mapping_weights.csv"
        if audit_path is None:
            audit_path = output_dir / f"{filename_prefix}_complete_mapping_weights_audit.csv"
    else:
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.parent.mkdir(parents=True, exist_ok=True)

    if not overwrite:
        for path in [weights_path, audit_path]:
            if path.exists():
                raise FileExistsError(f"Output already exists: {path}")

    weights.to_csv(weights_path, index=False)
    audit.to_csv(audit_path, index=False)

    logger.info("Wrote complete mapping weights: %s", weights_path)
    logger.info("Wrote complete mapping weights audit: %s", audit_path)

    return {
        "complete_weights": weights_path,
        "complete_weights_audit": audit_path,
    }


def run_equal_split_allocation(
    *,
    cfg: HsSyncConfig,
    run_id: str,
    logger: None | object = None,
) -> dict[str, Path]:
    """
    Build equal-split allocation weights for observed needs-weights cases.

    This is a workflow backend function. It assumes coverage products have
    already been written.
    """

    if not cfg.weights.enabled:
        logger.info("Allocation step is disabled in config; skipping.")
        return {
            "status": "disabled",
            "run_id": run_id,
        }

    logger.info("Starting HS-sync allocation step.")
    logger.info("  Run ID: %s", run_id)
    logger.info("  Allocation target HS: %s", cfg.weights.target_hs)
    logger.info("  Allocation strategy: %s", cfg.weights.strategy.name)
    logger.info("  Allocation mappings source: %s", cfg.weights.input.mappings)

    if cfg.weights.strategy.name != "equal_split":
        raise ValueError(
            "Unsupported allocation strategy: "
            f"{cfg.weights.strategy.name!r}. "
            "This backend currently supports only 'equal_split'."
        )

    if cfg.weights.input.mappings != "coverage.mappings":
        raise ValueError(
            "Unsupported allocation.input.mappings value: "
            f"{cfg.weights.input.mappings!r}. "
            "Expected 'coverage.mappings'."
        )

    source_code_coverage_path = cfg.weights_input_coverage_path()
    logger.info("Loading source-code coverage from: %s", source_code_coverage_path)

    source_code_coverage = pd.read_csv(
        source_code_coverage_path,
        dtype={"source_cmd": str},
    )

    logger.info(
        "Loaded source-code coverage rows: %s",
        f"{len(source_code_coverage):,}",
    )

    mapping_paths = cfg.coverage_mapping_files()

    logger.info(
        "Loading %s mapping table(s) from: %s",
        len(mapping_paths),
        cfg.coverage_mapping_dir(),
    )

    mapping_tables = {
        key: pd.read_csv(
            path,
            dtype={
                "code.source": str,
                "code.target": str,
            },
        )
        for key, path in mapping_paths.items()
    }

    for key, df in mapping_tables.items():
        logger.info("  Mapping %s rows: %s", key, f"{len(df):,}")

    observed_mapping = build_observed_needs_mapping(
        mapping_tables=mapping_tables,
        source_code_coverage=source_code_coverage,
    )

    logger.info(
        "Observed mapping rows requiring allocation: %s",
        f"{len(observed_mapping):,}",
    )

    result = build_allocation_weights(
        mapping=observed_mapping,
    )

    logger.info("Allocation result weight rows: %s", f"{len(result.weights):,}")
    logger.info("Allocation audit rows: %s", f"{len(result.audit):,}")

    weights_path = cfg.weights_equal_split_weights_path()
    audit_path = cfg.weights_equal_split_audit_path()

    logger.info("Allocation weights output path: %s", weights_path)
    logger.info("Allocation audit output path: %s", audit_path)
    logger.info("Overwrite enabled: %s", cfg.weights.write.overwrite)

    outputs = write_allocation_outputs(
        weights=result.weights,
        audit=result.audit,
        weights_path=weights_path,
        audit_path=audit_path,
        overwrite=cfg.weights.write.overwrite,
    )

    n_failed = int((result.audit["status"] != "pass").sum())

    logger.info("HS-sync equal-split allocation step complete.")
    logger.info("  Weights file: %s", outputs["weights"])
    logger.info("  Audit file: %s", outputs["audit"])
    logger.info("  Allocation audit failed rows: %s", n_failed)

    return {
        "status": "completed",
        "run_id": run_id,
        "weights": outputs["weights"],
        "audit": outputs["audit"],
        "n_weight_rows": len(result.weights),
        "n_audit_rows": len(result.audit),
        "n_failed_audit_rows": n_failed,
    }