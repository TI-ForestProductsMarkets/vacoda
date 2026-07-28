
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
from vacoda.utils.display import get_symbols, color_text
import re

from wcwidth import wcswidth

ResultT = TypeVar("ResultT")

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def _visible_width(text: str) -> int:
    """Return the number of terminal columns occupied by text."""
    plain_text = _ANSI_ESCAPE.sub("", text)
    width = wcswidth(plain_text)

    return max(width, 0)

def print_workflow_banner(
    text: str,
    *,
    symbol: str | None = None,
    width: int = 72,
) -> None:
    """Print a boxed banner for interactive terminal output."""
    inner_width = width - 2

    prefix = f"{symbol * 3}  " if symbol else ""
    content = f"  {prefix}{text}"

    padding = max(
        inner_width - _visible_width(content),
        0,
    )

    print(f"╔{'═' * inner_width}╗")
    print(f"║{content}{' ' * padding}║")
    print(f"╚{'═' * inner_width}╝")

def _workflow_banner(
    logger: logging.Logger,
    text: str,
    *,
    width: int = 72,
) -> None:
    """Log a boxed workflow banner with ANSI-aware alignment."""
    inner_width = width - 2
    content = f"  {text}"

    padding = max(
        inner_width - _visible_width(content),
        0,
    )

    logger.info("╔%s╗", "═" * inner_width)
    logger.info(
        "║%s%s║",
        content,
        " " * padding,
    )
    logger.info("╚%s╝", "═" * inner_width)

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
    run_symbol = color_text(get_symbols("play"), "cyan")
    separator(logger)
    logger.info(
        "%s  STEP %s/%s — %s",
        run_symbol,
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

def print_preflight_step(
    decision: Any,
) -> None:
    step_cfg = decision.step_config
    plan = decision.plan

    action_label = (
            color_text("[RUN]", "green")
            if decision.action == "run"
            else color_text("[SKIP]", "yellow")
        )

    pipeline_symbol = color_text(get_symbols("pipeline_step"), "cyan")
    checkpoint_symbol = get_symbols("checkpoint")
    output_symbol = get_symbols("output")
    audit_symbol = get_symbols("audit")
    flag_symbol = get_symbols("flag")

    print()
    print(
        f"{pipeline_symbol * 3}  "
        f"PIPELINE STEP  —  {decision.step_name.upper()}"
    )
    print(f"  Action:            {action_label}")
    print(f"  Reason:            {decision.reason}")
    print(f"  Config:            {step_cfg.config}")
    print(
        "  Expected status:   "
        + ", ".join(step_cfg.expected_status)
    )
    print(f"  {flag_symbol} Required flag:     {step_cfg.required_ready_flag}")
    print(f"  {checkpoint_symbol} Checkpoint:     {decision.checkpoint_path}")
    print(f"  {output_symbol} Output:         {plan.output_path}")
    print(f"  {audit_symbol} Audit:          {plan.audit_path}")
    print("----------------------------------------------------------------------")

def log_pipeline_step_banner(
    logger: logging.Logger,
    step_name: str,
) -> None:
    """Log a prominent boundary between master-pipeline steps."""

    symbol = color_text(
        get_symbols("pipeline_step"),
        "cyan",
    )

    _workflow_banner(
        logger,
        f"{symbol}{symbol}{symbol}  PIPELINE STEP  —  {step_name.upper()}",
    )
    
@contextmanager
def timed_pipeline_workflow(
    logger: logging.Logger,
    workflow_name: str,
) -> Iterator[None]:
    """Log total workflow execution time."""
    display_name = workflow_name.upper()
    started = perf_counter()
    start = color_text(get_symbols("play"), "red")
    checkmark = color_text(get_symbols("check_mark"), "green")
    cross = color_text(get_symbols("cross"), "red")

    logger.info("")
    _workflow_banner(
        logger,
        f"{start}{start}{start}  START WORKFLOW  —  {display_name}",
    )

    try:
        yield
    except Exception:
        elapsed = perf_counter() - started

        _workflow_banner(
            logger,
            f"{cross}{cross}{cross}  FAILED WORKFLOW  —  {display_name} "
            f"({format_seconds(elapsed)})",
        )
        logger.info("")
        raise
    else:
        elapsed = perf_counter() - started

        _workflow_banner(
            logger,
            f"{checkmark}{checkmark}{checkmark}  END WORKFLOW  —  {display_name} "
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