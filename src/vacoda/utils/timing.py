"""
vacoda.utils.timing

------------------------------
This will be removed in the future. Use vacoda.utils.runtime instead.
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


def _step_label(step_num: int, total_steps: int | None = None) -> str:
    if total_steps is None:
        return f"STEP {step_num}"
    return f"STEP {step_num}/{total_steps}"


@contextmanager
def timed_step(
    logger: logging.Logger,
    step_num: int,
    step_name: str,
    *,
    total_steps: int | None = None,
    separator: str = "=",
    width: int = 72,
) -> Iterator[None]:
    """
    Log a visually separated timed workflow step.
    """
    label = _step_label(step_num, total_steps)
    line = separator * width

    t0 = time.perf_counter()

    logger.info(line)
    logger.info("%s — %s", label, step_name)
    logger.info(line)

    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        logger.info("DONE — %s (%s)", step_name, format_seconds(elapsed))

@contextmanager
def timed_workflow(
    logger: logging.Logger,
    workflow_name: str,
) -> Iterator[None]:
    """
    Log total elapsed time for a workflow.
    """
    t0 = time.perf_counter()

    #logger.info("=" * 72)
    logger.info("START — %s", workflow_name)
    #logger.info("=" * 72)

    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        #logger.info("=" * 72)
        logger.info("DONE — %s (%s)", workflow_name, format_seconds(elapsed))
        #logger.info("=" * 72)