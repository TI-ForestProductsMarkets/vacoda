"""
vacoda.utils.timing

Small timing helpers for pipelines and workflows.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from collections.abc import Iterator


def format_seconds(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    if seconds < 60:
        return f"{seconds:.2f}s"

    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {sec:.1f}s"

    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {sec:.1f}s"


@contextmanager
def timed_step(
    logger: logging.Logger,
    step_num: int,
    step_name: str,
) -> Iterator[None]:
    """
    Log elapsed time for a workflow step.

    Example
    -------
    with timed_step(logger, 1, "Build year-HS counts"):
        ...
    """
    t0 = time.perf_counter()
    logger.info(f"STEP {step_num} — {step_name}")
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        logger.info(
            f"STEP {step_num} DONE — {step_name} "
            f"({format_seconds(elapsed)})"
        )


@contextmanager
def timed_workflow(
    logger: logging.Logger,
    workflow_name: str,
) -> Iterator[None]:
    """
    Log total elapsed time for a workflow.
    """
    t0 = time.perf_counter()
    logger.info("=" * 72)
    logger.info(f"START — {workflow_name}")
    logger.info("=" * 72)

    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        logger.info("=" * 72)
        logger.info(f"DONE — {workflow_name} ({format_seconds(elapsed)})")
        logger.info("=" * 72)