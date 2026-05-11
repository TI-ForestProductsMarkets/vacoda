"""
This script reads commodity codes from an Excel file and generates a Python file containing a dictionary that maps each commodity code to a tuple of valid years. The
valid years are set to a default range (1988 to 2025) for all codes, but this can be adjusted as needed.

The generated Python file can then be used in the application to reference the valid years for each commodity code.
"""

from pathlib import Path
import pandas as pd

## Constants
OUTPUT_FILE_NAME= "cmdCode_dict.py"

INPUT_EXCEL = Path("data/reference/Datenauszüge_441012.xlsx")
OUTPUT_FILE = Path(f"src/vacoda/reference/{OUTPUT_FILE_NAME}")

SHEET_NAME = 'Commodity Codes TI-WF'

DEFAULT_VALID_FROM = 1988
DEFAULT_VALID_TO = 2025


def read_commodity_codes_from_excel(
    excel_path: Path,
    sheet_name: str | int = SHEET_NAME,
) -> list[str]:
    """
    Read commodity codes from an Excel sheet where the actual code list starts
    below a row containing 'Commodity Codes'.
    """

    raw = pd.read_excel(
        excel_path,
        sheet_name=sheet_name,
        header=None,
        dtype=str,
    )

    # Find the row that contains "Commodity Codes"
    marker_mask = raw.apply(
        lambda row: row.astype(str).str.contains(
            "Commodity Codes",
            case=False,
            na=False,
        ).any(),
        axis=1,
    )

    if not marker_mask.any():
        raise ValueError("Could not find a row containing 'Commodity Codes'.")

    marker_row = marker_mask.idxmax()

    # Assume codes are below the marker row, usually in the first column
    codes_raw = raw.iloc[marker_row + 1 :, 0]

    codes = (
        codes_raw
        .dropna()
        .astype(str)
        .str.strip()
    )

    # Keep only numeric commodity codes
    codes = codes[codes.str.fullmatch(r"\d+")]

    # Remove duplicates while preserving order
    return list(dict.fromkeys(codes.tolist()))


def format_hs_valid_years(
    codes: list[str],
    valid_from: int = DEFAULT_VALID_FROM,
    valid_to: int = DEFAULT_VALID_TO,
) -> str:
    """
    Format the commodity codes and their valid years into a Python dictionary string.
    """
    ## Lets write on the top of the file a comment about the generated nature of the file and the source of the data.
    
    lines = [
        "# This file is generated from an script build_hs_sets_from_excel.py that reads commodity codes from an Excel file.",
        "# Do not edit manually.",
        
        "HS_VALID_YEARS: dict[str, tuple[int, int]] = {",
    ]

    for code in codes:
        lines.append(f'    "{code}": ({valid_from}, {valid_to}),')

    lines.append("}")
    lines.append("")

    return "\n".join(lines)

def audit_cmdcodes(codes: list[str]) -> None:
    """
    Perform a simple audit of the commodity codes to check for duplicates and non-numeric entries.
    """
    unique_codes = set(codes)
    if len(unique_codes) != len(codes):
        print("Warning: Duplicate commodity codes found.")
    
    non_numeric_codes = [code for code in codes if not code.isdigit()]
    if non_numeric_codes:
        print(f"Warning: Non-numeric commodity codes found: {non_numeric_codes}")


def audit_cmdcodes_count(codes: list[str], generated_codes: list[str]) -> None:
    """
    Perform a single audit to make sure that number of commodity codes are equal in both excel and generated file.
    """
    #print(f"Number of commodity codes in Excel: {len(codes)}")
    #print(f"Number of commodity codes in generated file: {len(generated_codes)}")
    if len(codes) != len(generated_codes):
        print(f"Warning: Mismatch in number of commodity codes between Excel {len(codes)} and generated file {len(generated_codes)}.")
    else:
        print(f"Success: Number of commodity codes matches between Excel {len(codes)} and generated file {len(generated_codes)}.")


def main() -> None:
    codes = read_commodity_codes_from_excel(INPUT_EXCEL)

    audit_cmdcodes(codes)

    content = format_hs_valid_years(codes)

    audit_cmdcodes_count(codes, list(dict.fromkeys(codes)))

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(content, encoding="utf-8")

    print(f"Read {len(codes)} commodity codes.")
    print(f"Wrote: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()