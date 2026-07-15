"""
vacoda.cif_fob.audit_bands
------------------------
This module defines the value and distance bands used in the training-evidence audit.
The bands are used to group the audit results for visualization and reporting.
"""
from __future__ import annotations

import numpy as np


VALUE_BAND_BINS: tuple[float, ...] = (
    0,
    10_000,
    100_000,
    1_000_000,
    10_000_000,
    np.inf,
)

VALUE_BAND_ORDER: tuple[str, ...] = (
    "<10k",
    "10k–100k",
    "100k–1M",
    "1M–10M",
    ">10M",
)


DISTANCE_BAND_BINS: tuple[float, ...] = (
    0,
    1_000,
    3_000,
    7_000,
    12_000,
    np.inf,
)

DISTANCE_BAND_ORDER: tuple[str, ...] = (
    "<1k km",
    "1k–3k km",
    "3k–7k km",
    "7k–12k km",
    ">12k km",
)

CIFFOB_FACTOR_BINS: tuple[float, ...] = (
    1.0,
    1.05,
    1.10,
    1.20,
    1.50,
    2.00,
    3.00,
)

CIFFOB_FACTOR_LABELS: tuple[str, ...] = (
    "1.00-1.05",
    "1.05-1.10",
    "1.10-1.20",
    "1.20-1.50",
    "1.50-2.00",
    "2.00-3.00",
)