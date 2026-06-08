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

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = REPO_ROOT / "references" / "comtrade" / "hs_correlation_groups" / "raw"
NORMALIZED_DIR = (
    REPO_ROOT / "references" / "comtrade" / "hs_correlation_groups" / "normalized"
)


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

    required_columns = {
        spec.raw_source_col,
        spec.raw_target_col,
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

    return df[
        [
            spec.raw_source_col,
            spec.raw_target_col,
            "Relationship",
        ]
    ].copy()


def normalize_mapping(df: pd.DataFrame, spec: MappingSpec) -> pd.DataFrame:
    """
    Convert one raw UNSD mapping table to VACODA source -> target direction.
    """
    raw_source = spec.raw_source_col
    raw_target = spec.raw_target_col

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


def main() -> None:
    NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)

    print("Preparing UN Comtrade / UNSD HS mappings")
    print(f"Raw directory:        {RAW_DIR}")
    print(f"Normalized directory: {NORMALIZED_DIR}")
    print()

    for spec in MAPPING_SPECS:
        raw_path = RAW_DIR / spec.raw_filename

        print("=" * 72)
        print(f"Input:  {raw_path.name}")
        print(
            "Output direction: "
            f"{spec.normalized_source_system} -> {spec.normalized_target_system}"
        )
        print(f"Reverse direction: {spec.reverse_direction}")

        raw_df = read_relationship_sheet(raw_path, spec)
        normalized = normalize_mapping(raw_df, spec)

        out_path = NORMALIZED_DIR / output_filename(spec)
        normalized.to_csv(out_path, index=False)

        relationship_counts = (
            normalized["Relationship"]
            .value_counts()
            .rename_axis("Relationship")
            .reset_index(name="n_rows")
            .sort_values("Relationship")
        )

        print(f"Rows written: {len(normalized):,}")
        print(f"Output: {out_path}")
        print("Relationship counts:")
        print(relationship_counts.to_string(index=False))
        print()

    print("=" * 72)
    print("Done.")


if __name__ == "__main__":
    main()