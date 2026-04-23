"""
Script to convert metadata from excel sheets to parquet files for easier use in the pipeline.
This is a one-off utility script, not part of the main pipeline, for handling specific metadata
"""

import pandas as pd
from pathlib import Path

# --- Excel-related paths/settings ---
EXCEL_FILE = "data/raw/rawdata_44101210.xlsx"  # from raw_data.file

# Metadata sheet names from paths.yaml > metadata_sheets
METADATA_SHEETS = {
    "hs_version": "CD_HS",
    "reporter": "CD_Reporter",
    "commodity": "CD_Commodity",
    "flow": "CD_Flow",
    "unit": "CD_Unit",
    "estimation": "CD_EstimationCode",
}

# Example output paths for each metadata sheet (edit as needed)
METADATA_OUTPUT_PATHS = {
    key: f"data/metadata/{key}.parquet" for key in METADATA_SHEETS
}

def excel_sheet_to_parquet(
    excel_path: str,
    sheet_name: str,
    output_path: str
):
    # Load sheet
    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    # Optional: basic cleanup
    df.columns = df.columns.str.strip()

    # Convert all columns to string to avoid mixed-type errors
    df = df.astype(str)

    # Ensure output folder exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save as parquet
    df.to_parquet(output_path, index=False)

    print(f" Converted '{sheet_name}' → {output_path}")



if __name__ == "__main__":
    # Convert all metadata sheets to individual parquet files
    for key, sheet_name in METADATA_SHEETS.items():
        output_path = METADATA_OUTPUT_PATHS[key]
        excel_sheet_to_parquet(
            EXCEL_FILE,
            sheet_name=sheet_name,
            output_path=output_path
        )