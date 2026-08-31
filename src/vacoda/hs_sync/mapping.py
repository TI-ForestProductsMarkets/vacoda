"""
Prepare UN Comtrade / UNSD HS correspondence tables for VACODA HS-sync.

Raw Excel files are stored under:

    references/comtrade/hs_correlation_groups/raw/

Normalized CSV files are written under:

    references/comtrade/hs_correlation_groups/normalized/

The normalized files always use VACODA source -> target direction, e.g.:

    from_HS2002_to_HS2017.csv
    from_HS2007_to_HS2017.csv
    from_HS2012_to_HS2017.csv
    from_HS2022_to_HS2017.csv

For files whose raw direction is HS2017 -> older HS, the relationship labels
are reversed:

    1:1 -> 1:1
    n:1 -> 1:n
    1:n -> n:1
    n:n -> n:n
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from logging import Logger

import pandas as pd
from vacoda.config.hs_sync import HsSyncConfig

RELATIONSHIP_REVERSE = {
    "1:1": "1:1",
    "n:1": "1:n",
    "1:n": "n:1",
    "n:n": "n:n",
}


@dataclass(frozen=True)
class MappingSpec:
    raw_filename: str
    raw_source_col: str
    raw_target_col: str
    normalized_source_hs: str
    normalized_target_hs: str
    normalized_source_system: str
    normalized_target_system: str
    reverse_direction: bool


MAPPING_SPECS: list[MappingSpec] = [
    MappingSpec(
        raw_filename="HS2017toHS2002ConversionAndCorrelationTables.xlsx",
        raw_source_col="HS2017",
        raw_target_col="HS2002",
        normalized_source_hs="H2",
        normalized_target_hs="H5",
        normalized_source_system="HS2002",
        normalized_target_system="HS2017",
        reverse_direction=True,
    ),
    MappingSpec(
        raw_filename="HS2017toHS2007ConversionAndCorrelationTables.xlsx",
        raw_source_col="HS2017",
        raw_target_col="HS2007",
        normalized_source_hs="H3",
        normalized_target_hs="H5",
        normalized_source_system="HS2007",
        normalized_target_system="HS2017",
        reverse_direction=True,
    ),
    MappingSpec(
        raw_filename="HS2017toHS2012ConversionAndCorrelationTables.xlsx",
        raw_source_col="HS2017",
        raw_target_col="HS2012",
        normalized_source_hs="H4",
        normalized_target_hs="H5",
        normalized_source_system="HS2012",
        normalized_target_system="HS2017",
        reverse_direction=True,
    ),
    MappingSpec(
        raw_filename="HS2022toHS2017ConversionAndCorrelationTables.xlsx",
        raw_source_col="HS2022",
        raw_target_col="HS2017",
        normalized_source_hs="H6",
        normalized_target_hs="H5",
        normalized_source_system="HS2022",
        normalized_target_system="HS2017",
        reverse_direction=False,
    ),
]

def _normalize_colname(value: object) -> str:
    """
    Normalize Excel header names.

    Examples:
        'HS 2022' -> 'HS2022'
        'HS2022'  -> 'HS2022'
        ' Relationship ' -> 'Relationship'
    """
    if pd.isna(value):
        return ""

    text = str(value).strip()
    text = " ".join(text.split())

    if text.upper().startswith("HS "):
        text = text.replace(" ", "")

    return text

def _clean_code(value: object) -> str | None:
    """
    Normalize HS code values to six-character strings.

    Handles Excel cases where codes may be read as numbers.
    """
    if pd.isna(value):
        return None

    text = str(value).strip()

    if text == "":
        return None

    # Excel sometimes reads HS codes as floats, e.g. 10121.0
    if text.endswith(".0"):
        text = text[:-2]

    # Keep only plain six-digit-like codes.
    if text.isdigit():
        text = text.zfill(6)

    return text

def _find_header_row(raw: pd.DataFrame, required_columns: set[str]) -> int:
    """
    Find the row containing the real header.

    UNSD sheets sometimes have a title row such as 'Between' before the
    actual header row. Column labels also vary between 'HS 2022' and 'HS2022'.
    """
    normalized_required = {
        _normalize_colname(col)
        for col in required_columns
    }

    for row_idx in range(len(raw)):
        values = {
            _normalize_colname(value)
            for value in raw.iloc[row_idx].tolist()
            if not pd.isna(value)
        }

        if normalized_required <= values:
            return row_idx

    raise ValueError(
        "Could not find header row containing columns: "
        + ", ".join(sorted(normalized_required))
    )

def read_relationship_sheet(path: Path, spec: MappingSpec) -> pd.DataFrame:
    """
    Read sheet 2 from a UNSD conversion/correlation workbook.

    Sheet 2 contains the relationship-aware table.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing raw mapping file: {path}")

    # sheet_name=1 means the second sheet.
    raw = pd.read_excel(path, sheet_name=1, header=None, dtype=str)

    raw_source_col = _normalize_colname(spec.raw_source_col)
    raw_target_col = _normalize_colname(spec.raw_target_col)


    required_columns = {
        raw_source_col,
        raw_target_col,
        "Relationship",
    }

    header_row = _find_header_row(raw, required_columns)

    df = pd.read_excel(
        path,
        sheet_name=1,
        header=header_row,
        dtype=str,
    )

    df = df.rename(columns={col: _normalize_colname(col) for col in df.columns})

    required_columns_normalized = {
        _normalize_colname(col)
        for col in required_columns
    }

    missing = required_columns_normalized - set(df.columns)

    if missing:
        raise ValueError(
            f"{path.name}: missing required columns after reading sheet 2: "
            + ", ".join(sorted(missing))
        )

    return df[[
        raw_source_col,
        raw_target_col,
        "Relationship",]].copy()


def normalize_mapping(df: pd.DataFrame, spec: MappingSpec) -> pd.DataFrame:
    """
    Convert one raw UNSD mapping table to VACODA source -> target direction.
    """
    raw_source = _normalize_colname(spec.raw_source_col)
    raw_target = _normalize_colname(spec.raw_target_col)
    
    df = df.copy()

    df[raw_source] = df[raw_source].map(_clean_code)
    df[raw_target] = df[raw_target].map(_clean_code)
    df["Relationship"] = df["Relationship"].astype(str).str.strip()

    df = df.dropna(subset=[raw_source, raw_target])
    df = df[df["Relationship"].isin(RELATIONSHIP_REVERSE)]

    if spec.reverse_direction:
        source_cmd = df[raw_target]
        target_cmd = df[raw_source]
        relationship = df["Relationship"].map(RELATIONSHIP_REVERSE)
    else:
        source_cmd = df[raw_source]
        target_cmd = df[raw_target]
        relationship = df["Relationship"]

    out = pd.DataFrame(
        {
            # Compatibility with your current coverage code.
            "code.source": source_cmd,
            "code.target": target_cmd,
            "Relationship": relationship,
            "adjustment": (f"{spec.normalized_source_system} to {spec.normalized_target_system}"),
            "group.id": pd.NA,

            # More explicit VACODA metadata.
            "source_hs": spec.normalized_source_hs,
            "target_hs": spec.normalized_target_hs,
            "source_system": spec.normalized_source_system,
            "target_system": spec.normalized_target_system,
            "relationship_original": df["Relationship"],
            "mapping_source": "UNSD_COMTRADE",
            "mapping_file": spec.raw_filename,
            "direction_reversed": spec.reverse_direction,
        }
    )

    out = out.drop_duplicates()
    out = out.sort_values(["code.source", "code.target"]).reset_index(drop=True)

    return out


def output_filename(spec: MappingSpec) -> str:
    return (
        f"from_{spec.normalized_source_system}_"
        f"to_{spec.normalized_target_system}.csv"
    )


def run_comtrade_mapping_preparation(
    *,
    cfg: HsSyncConfig,
    run_id: str,
    force: bool = False,
    logger: Logger,
) -> dict[str, object]:

    raw_dir = cfg.comtrade_hs_correlation_raw_dir()
    normalized_dir = cfg.comtrade_hs_correlation_normalized_dir()

    normalized_dir.mkdir(parents=True, exist_ok=True)


    input_paths = {
        spec.raw_filename: raw_dir / spec.raw_filename
        for spec in MAPPING_SPECS
    }

    output_paths = {
        spec.raw_filename: (
            normalized_dir / output_filename(spec)
        )
        for spec in MAPPING_SPECS
    }

    for name, path in input_paths.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing raw mapping file ({name}): {path}"
            )
    
    logger.info("Preparing UN Comtrade / UNSD HS mappings")
    logger.info("Run ID: %s", run_id)
    logger.info("Raw directory:        %s", raw_dir)
    logger.info("Normalized directory: %s", normalized_dir)

    outputs: dict[str, object] = {
        "status": "started",
        "run_id": run_id,
        "raw_dir": raw_dir,
        "normalized_dir": normalized_dir,
        "files": {},
    }

    file_outputs: dict[str, dict[str, object]] = {}

    for spec in MAPPING_SPECS:
        raw_path = raw_dir / spec.raw_filename
        out_path = normalized_dir / output_filename(spec)

        logger.info("=" * 72)
        logger.info("Input: %s", raw_path.name)
        logger.info(
            "Output direction: %s -> %s",
            spec.normalized_source_system,
            spec.normalized_target_system,
        )
        logger.info("Reverse direction: %s", spec.reverse_direction)

        raw_df = read_relationship_sheet(raw_path, spec)
        normalized = normalize_mapping(raw_df, spec)

        normalized.to_csv(out_path, index=False)

        relationship_counts = (
            normalized["Relationship"]
            .value_counts()
            .rename_axis("Relationship")
            .reset_index(name="n_rows")
            .sort_values("Relationship")
        )

        logger.info("Rows written: %s", f"{len(normalized):,}")
        logger.info("Output: %s", out_path)
        logger.info(
            "Relationship counts:\n%s",
            relationship_counts.to_string(index=False),
        )

        file_outputs[spec.raw_filename] = {
            "status": "written",
            "input": raw_path,
            "output": out_path,
            "rows": len(normalized),
        }

    outputs["status"] = "completed"
    outputs["files"] = file_outputs

    logger.info("Comtrade HS mapping preparation complete.")

    return {
        "status": "completed",
        "run_id": run_id,
        "raw_dir": raw_dir,
        "normalized_dir": normalized_dir,
        "files": file_outputs,
    }
