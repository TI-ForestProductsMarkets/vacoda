
"""
vacoda.utils.runtime
--------------------------
Shared helpers for executing workflow steps and validating step results.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Any, TypeVar


ResultT = TypeVar("ResultT")


def format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"

    minutes, seconds = divmod(seconds, 60)

    if minutes < 60:
        return f"{int(minutes)}m {seconds:.1f}s"

    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {seconds:.1f}s"


def separator(
    logger: logging.Logger,
    *,
    character: str = "=",
    width: int = 72,
) -> None:
    logger.info(character * width)


def log_step_header(
    *,
    logger: logging.Logger,
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
    logger: logging.Logger,
    step_number: int,
    total_steps: int,
    title: str,
    runner: Callable[[], ResultT],
) -> ResultT:
    """Run one workflow step with consistent timing and logging."""
    log_step_header(
        logger=logger,
        step_number=step_number,
        total_steps=total_steps,
        title=title,
    )

    started = perf_counter()

    try:
        result = runner()
    except Exception:
        elapsed = perf_counter() - started
        logger.error(
            "FAILED — %s (%s)",
            title,
            format_seconds(elapsed),
        )
        raise

    elapsed = perf_counter() - started
    logger.info(
        "DONE — %s (%s)",
        title,
        format_seconds(elapsed),
    )

    return result

def _workflow_banner(
    logger: logging.Logger,
    text: str,
    *,
    width: int = 72,
) -> None:
    """Log a boxed workflow banner."""
    inner_width = width - 2
    logger.info("╔%s╗", "═" * inner_width)
    logger.info("║%-*s║", inner_width, f"  {text}")
    logger.info("╚%s╝", "═" * inner_width)


def log_pipeline_step_banner(
    logger: logging.Logger,
    step_name: str,
) -> None:
    """Log a prominent boundary between master-pipeline steps."""
    _workflow_banner(
        logger,
        f"▶▶▶  PIPELINE STEP  —  {step_name.upper()}",
    )
    
@contextmanager
def timed_pipeline_workflow(
    logger: logging.Logger,
    workflow_name: str,
) -> Iterator[None]:
    """Log total workflow execution time."""
    display_name = workflow_name.upper()
    started = perf_counter()

    logger.info("")
    _workflow_banner(
        logger,
        f"▶▶▶  START WORKFLOW  —  {display_name}",
    )

    try:
        yield
    except Exception:
        elapsed = perf_counter() - started

        _workflow_banner(
            logger,
            f"✖✖✖  FAILED WORKFLOW  —  {display_name} "
            f"({format_seconds(elapsed)})",
        )
        logger.info("")
        raise
    else:
        elapsed = perf_counter() - started

        _workflow_banner(
            logger,
            f"✔✔✔  END WORKFLOW  —  {display_name} "
            f"({format_seconds(elapsed)})",
        )
        logger.info("")


@contextmanager
def timed_workflow(
    logger: logging.Logger,
    workflow_name: str,
) -> Iterator[None]:
    started = perf_counter()

    try:
        yield
    except Exception:
        elapsed = perf_counter() - started
        logger.error(
            "FAILED — %s (%s)",
            workflow_name,
            format_seconds(elapsed),
        )
        raise
    else:
        elapsed = perf_counter() - started
        logger.info(
            "WORKFLOW COMPLETE — %s (%s)",
            workflow_name,
            format_seconds(elapsed),
        )

@contextmanager
def timed_workflow_deprecated(
    logger: logging.Logger,
    workflow_name: str,
) -> Iterator[None]:
    """Log total workflow execution time."""
    started = perf_counter()
    logger.info("Starting %s.", workflow_name)

    try:
        yield
    except Exception:
        elapsed = perf_counter() - started
        logger.error(
            "Failed %s after %s.",
            workflow_name,
            format_seconds(elapsed),
        )
        raise
    else:
        elapsed = perf_counter() - started
        logger.info(
            "Completed %s in %s.",
            workflow_name,
            format_seconds(elapsed),
        )


def require_result_path(
    result: dict[str, Any],
    error_message: str,
    *keys: str,) -> Path:
    for key in keys:
        value = result.get(key)

        if value not in (None, ""):
            return Path(value)

    raise RuntimeError(
        f"{error_message}: expected one of {', '.join(keys)}"
    )