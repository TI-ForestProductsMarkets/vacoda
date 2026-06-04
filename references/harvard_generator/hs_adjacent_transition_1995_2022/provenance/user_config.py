"""
Weight Generator Configuration

This configuration file controls how the weight generator runs.
Edit the settings below to match your environment and requirements.
"""

from pathlib import Path
import sys
from datetime import date, timedelta
from src.config.source_target_pairs import SOURCE_TARGET_ENABLED_PAIRS


# =============================================================================
# PATHS CONFIGURATION
# =============================================================================

# top file directory path data downloaded from comtrade
RAW_DOWNLOADED_COMTRADE_DATA_PATH = (
    "/storage/vacoda/data/external/harvard_comtrade_bulk"
)

# =============================================================================
# BULK ENABLE/DISABLE SETTINGS
# =============================================================================
# Quick toggles for common scenarios
ENABLE_ALL_FORWARD = False
ENABLE_ALL_BACKWARD = False
ENABLE_ALL_CONVERSIONS = False

# =============================================================================
# CONVERSION PAIR SETTINGS
# =============================================================================
# Enable/disable specific source-target class pairs for conversion
# Set to True for the conversions you want to enable

# optimized weights are provided in the comtrade-downloader as a static data input

# BACKWARD HS CONVERSIONS (newer to older)
CONVERT_HS96_TO_HS92 = True
CONVERT_HS02_TO_HS96 = True
CONVERT_HS07_TO_HS02 = True
CONVERT_HS12_TO_HS07 = True
CONVERT_HS17_TO_HS12 = True
CONVERT_HS22_TO_HS17 = True

CONVERT_HS92_TO_SITC3 = False
CONVERT_SITC2_TO_SITC1 = False
CONVERT_SITC3_TO_SITC2 = False


# FORWARD HS CONVERSIONS (older to newer)
CONVERT_HS92_TO_HS96 = True
CONVERT_HS96_TO_HS02 = True
CONVERT_HS02_TO_HS07 = True
CONVERT_HS07_TO_HS12 = True
CONVERT_HS12_TO_HS17 = True
CONVERT_HS17_TO_HS22 = True

CONVERT_SITC3_TO_HS92 = False
CONVERT_SITC1_TO_SITC2 = False
CONVERT_SITC2_TO_SITC3 = False


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOG_LEVEL = "DEBUG"

# =============================================================================
# PROCESSING STEPS TO RUN (ADVANCED)
# =============================================================================

COMBINE_CONCORDANCES = True
CREATE_PRODUCT_GROUPS = True
BUILD_INPUT_MATRICES = True
GENERATE_WEIGHTS = False
GROUP_WEIGHTS = False

# =============================================================================
# SETUP
# =============================================================================


def get_enabled_conversions():
    """
    Returns a list of enabled conversion pairs based on the settings above.
    """
    enabled_pairs = []

    conversions = [
        (CONVERT_SITC3_TO_HS92, "SITC3", "HS1992", "forward", "1988", "1992"),
        (CONVERT_SITC2_TO_SITC3, "SITC2", "SITC3", "forward", "1987", "1988"),
        (CONVERT_SITC1_TO_SITC2, "SITC1", "SITC2", "forward", "1975", "1976"),
        (CONVERT_HS96_TO_HS92, "HS1996", "HS1992", "backward", "1996", "1995"),
        (CONVERT_HS02_TO_HS96, "HS2002", "HS1996", "backward", "2002", "2001"),
        (CONVERT_HS07_TO_HS02, "HS2007", "HS2002", "backward", "2007", "2006"),
        (CONVERT_HS12_TO_HS07, "HS2012", "HS2007", "backward", "2012", "2011"),
        (CONVERT_HS17_TO_HS12, "HS2017", "HS2012", "backward", "2017", "2016"),
        (CONVERT_HS22_TO_HS17, "HS2022", "HS2017", "backward", "2022", "2021"),
        (CONVERT_HS92_TO_SITC3, "HS1992", "SITC3", "backward", "1992", "1988"),
        (CONVERT_SITC3_TO_SITC2, "SITC3", "SITC2", "backward", "1988", "1987"),
        (CONVERT_SITC2_TO_SITC1, "SITC2", "SITC1", "backward", "1987", "1976"),
        (CONVERT_HS92_TO_HS96, "HS1992", "HS1996", "forward", "1995", "1996"),
        (CONVERT_HS96_TO_HS02, "HS1996", "HS2002", "forward", "2001", "2002"),
        (CONVERT_HS02_TO_HS07, "HS2002", "HS2007", "forward", "2006", "2007"),
        (CONVERT_HS07_TO_HS12, "HS2007", "HS2012", "forward", "2011", "2012"),
        (CONVERT_HS12_TO_HS17, "HS2012", "HS2017", "forward", "2016", "2017"),
        (CONVERT_HS17_TO_HS22, "HS2017", "HS2022", "forward", "2021", "2022"),
    ]

    if ENABLE_ALL_CONVERSIONS:
        for enabled, source, target, direction, source_year, target_year in conversions:
            enabled_pairs.append(
                {
                    "direction": direction,
                    "source_class": source,
                    "target_class": target,
                    "source_year": source_year,
                    "target_year": target_year,
                    "enabled": True,
                }
            )
    else:
        # individually set
        for enabled, source, target, direction, source_year, target_year in conversions:
            # enable bulk settings
            if ENABLE_ALL_FORWARD and direction == "forward":
                enabled = True
            elif ENABLE_ALL_BACKWARD and direction == "backward":
                enabled = True

            if enabled:
                enabled_pairs.append(
                    {
                        "direction": direction,
                        "source_class": source,
                        "target_class": target,
                        "source_year": source_year,
                        "target_year": target_year,
                        "enabled": True,
                    }
                )

    return enabled_pairs
