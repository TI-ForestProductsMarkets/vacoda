#!/usr/bin/env python3
"""
Run the VACODA master pipeline.
"""

from vacoda.pipeline.pipeline import run_pipeline


if __name__ == "__main__":
    config_path = "configs/pipeline.yaml"

    run_pipeline(config_path=config_path)