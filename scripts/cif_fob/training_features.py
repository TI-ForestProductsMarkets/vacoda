"""
vacoda.cif_fob.training_features

"""
from __future__ import annotations
from pathlib import Path
from typing import Any
from vacoda.config.cif_fob import CifFobConfig


def build_training_features(
    *,
    cfg: CifFobConfig,
    input_path: Path,
    force: bool = False,
    logger,
) -> dict[str, Any]:
    ...