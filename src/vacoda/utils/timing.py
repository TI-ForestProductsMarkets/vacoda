import logging
from typing import Optional

def log_step_timing(logger: logging.Logger, step_num: int, step_name: str, t_step: float, t_now: Optional[float] = None):
    """
    Logs the timing for a pipeline step.
    Args:
        logger: Logger instance to use.
        step_num: Step number (1-based).
        step_name: Name/description of the step.
        t_step: Start time (perf_counter) for the step.
        t_now: Optional end time (perf_counter). If not provided, will call time.perf_counter().
    Returns:
        elapsed: Elapsed time in seconds (float)
    """
    import time
    if t_now is None:
        t_now = time.perf_counter()
    elapsed = t_now - t_step
    logger.info(f"[Timing] Step {step_num} ({step_name}): {elapsed:.2f} seconds")
    return elapsed
