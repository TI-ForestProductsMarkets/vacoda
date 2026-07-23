
"""
vacoda.utils.runtime
--------------------------
Shared helpers for executing workflow steps and validating step results.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, Callable


def separator(logger) -> None:
    logger.info("=" * 72)


def log_step_header(
    *,
    logger,
    step_number: int,
    total_steps: int,
    title: str,
) -> None:
    separator(logger)
    logger.info(
        "STEP %s/%s — %s",
        step_number,
        total_steps,
        title,
    )
    separator(logger)


def run_step(
    *,
    logger,
    step_number: int,
    total_steps: int,
    title: str,
    runner: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    """Run one workflow step with consistent timing and logging."""
    log_step_header(
        logger=logger,
        step_number=step_number,
        total_steps=total_steps,
        title=title,
    )

    start = perf_counter()
    result = runner()
    elapsed = perf_counter() - start

    logger.info(
        "DONE — %s (%.2fs)",
        title,
        elapsed,
    )

    return result

def require_result_path(
    result: dict[str, Any],
    error_message: str,
    *keys: str,
) -> Path:
    """Return the first available path from a step result."""
    for key in keys:
        value = result.get(key)
        if value:
            return Path(value)

    raise RuntimeError(
        error_message + ": "
        + ", ".join(keys)
    )