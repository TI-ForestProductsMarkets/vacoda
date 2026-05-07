"""
HS commodity reference sets for Comtrade downloads.

Use these as tokens in configs/comtrade.yaml, e.g.

    cmdCode: "WOOD_PANEL"
"""

from __future__ import annotations
from vacoda.reference.cmdCode_dict import HS_VALID_YEARS as DEFAULT_HS_VALID_YEARS


# Optional local override.
# If you define LOCAL_HS_VALID_YEARS below, it will be used instead of the imported default.

# LOCAL_HS_VALID_YEARS: dict[str, tuple[int, int]] = {
#     "441012": (2023, 2025),
#     "441011": (2024, 2025),
# }

try:
    HS_VALID_YEARS = LOCAL_HS_VALID_YEARS
except NameError:
    HS_VALID_YEARS = DEFAULT_HS_VALID_YEARS

# Single wood-sector code.
WOOD_PANEL = [key for key in HS_VALID_YEARS]  

# Multi wood-sector codes.
WOOD_ALL: list[str] = []


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