"""
HS commodity reference sets for Comtrade downloads.

Use these as tokens in configs/comtrade.yaml, e.g.

    cmdCode: "WOOD_PANEL"
"""

from __future__ import annotations

# Single wood-sector code.
WOOD_PANEL = ["441012", "441011"]  # OSB and plywood, respectively.

# Multi wood-sector codes.
WOOD_ALL: list[str] = []


# Validity windows for 6-digit HS codes.
# Values are inclusive: start_year, end_year.
#
# For 441012:
# - OSB appears from HS2007 onward.
# - Use 2025 as current download horizon for now.
HS_VALID_YEARS: dict[str, tuple[int, int]] = {
    "441012": (2024, 2025),
    "441011": (2024, 2025),
}


def get_hs_valid_year_range(
    cmd_code: str,
    fallback_start: int = 1990,
    fallback_end: int = 2025,
) -> tuple[int, int]:
    """
    Return inclusive valid year range for a commodity code.

    If the code is not yet registered, return fallback years.
    """
    code = str(cmd_code).strip()

    if code in HS_VALID_YEARS:
        return HS_VALID_YEARS[code]

    return fallback_start, fallback_end