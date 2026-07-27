"""
vacoda.utils.display
-------------------
Utility functions for displaying pandas Series and various symbols.

"""
from __future__ import annotations

import pandas as pd
from collections.abc import Iterable, Iterator
from typing import TypeVar

from tqdm import tqdm

T = TypeVar("T")

_SYMBOLS = {
    "warning": "\u26A0",              # ⚠
    "check": "\u2705",                # ✅
    "cross": "\u274C",                # ❌
    "info": "\u2139",                 # ℹ
    "star": "\u2B50",                 # ⭐
    "fire": "\U0001F525",             # 🔥
    "important": "\U0001F4A5",        # 💥
    "right_arrow": "\u27A1",          # ➡
    "right_finger": "\U0001F449",     # 👉
    "danger": "\U0001F6AB",           # 🚫
    "success": "\U0001F44D",          # 👍
    "failure": "\U0001F44E",          # 👎
    "sparkles": "\U0001F496",         # 💖
    "time": "\U0001F551",             # 🕑
    "duckdb": "\U0001F986",           # 🦆
    "pandas": "\U0001F43C",           # 🐼
    "python": "\U0001F40D",           # 🐍
    "warning_light": "\U0001F6A8",    # 🚨
    "construction": "\U0001F6A7",     # 🚧
    "recycle": "\U0001F504",          # 🔄
    "workflow": "\U0001F3C1",         # 🏁
    "check_mark": "\u2714",           # ✔
    "pipeline_step": "\u25C6",        # ◆
    "play": "\u25B6",                 # ▶
}

_ANSI_COLORS = {
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
    "cyan": "\033[36m",
    "reset": "\033[0m",
}


def color_text(text: str, color: str) -> str:
    try:
        prefix = _ANSI_COLORS[color]
    except KeyError as exc:
        raise ValueError(f"Unknown color: {color}") from exc

    return f"{prefix}{text}{_ANSI_COLORS['reset']}"

def get_symbols(name: str) -> str:
    try:
        return _SYMBOLS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown symbol name: {name}"
        ) from exc


def pretty_print_series(series: pd.Series, float_fmt: str = ",.2f", int_fmt: str = ","):
    """
    Nicely print a pandas Series with thousands separators and aligned output.
    Floats are formatted to 2 decimal places, ints with commas.
    If the key contains 'percentage', append a % sign.
    """
    for k, v in series.items():
        if isinstance(v, float):
            if 'percentage' in k.lower():
                print(f"{k:45} {format(v, float_fmt)}%")
            else:
                print(f"{k:45} {format(v, float_fmt)}")
        elif isinstance(v, int):
            print(f"{k:45} {format(v, int_fmt)}")
        else:
            print(f"{k:45} {v}")


def get_recovery_symbol(pct, threshold):
    if pct >= threshold:
        return "\u2705"   # ✅ strong recovery
    elif pct > 0:
        return "\u26A0"   # ⚠ partial recovery
    else:
        return "\u274C"   # ❌ no recovery

def get_symbols_deprecated(name):
    if name == "warning":
        return "\u26A0"   # ⚠
    elif name == "check":
        return "\u2705"   # ✅
    elif name == "cross":
        return "\u274C"   # ❌
    elif name == "info":
        return "\u2139"   # ℹ
    elif name == "star":
        return "\u2B50"   # ⭐
    elif name == "fire":
        return "\U0001F525"  # 🔥
    elif name == "important":
        return "\U0001F4A5"  # 💥
    elif name == "right_arrow":
        return "\u27A1"   # ➡
    elif name == "right_finger":
        return "\U0001F449"  # 👉
    elif name == "danger":
        return "\U0001F6AB"  #  🚫
    elif name == "success":
        return "\U0001F44D"  #  👍
    elif name == "failure":
        return "\U0001F44E"  #  👎
    elif name == "sparkles":
        return "\U0001F496"  # 💖
    elif name =="time":
        return "\U0001F551"  # 🕑
    elif name == "duckdb":
        return "\U0001F986"  # 🦆
    elif name == "pandas":
        return "\U0001F43C"  #  🐼
    elif name == "python":
        return "\U0001F40D"  #  🐍
    elif name == "warning_light":
        return "\U0001F6A8"  # 🚨
    elif name == "construction":
        return "\U0001F6A7"  # 🚧
    elif name == "recycle":
        return "\U0001F504"  # 🔄
    elif name == "workflow":
        return "\U0001F3C1"  # 🏁
    elif name == "pipeline_step":
        return "\U0001F4A5"  # 💥
    else:
        raise ValueError(f"Unknown symbol name: {name}")

def progress_bar(
    iterable: Iterable[T],
    *,
    total: int | None = None,
    desc: str = "Progress",
    unit: str = "it",
    enabled: bool = True,
    leave: bool = True,
) -> Iterator[T]:
    if not enabled:
        yield from iterable
        return

    yield from tqdm(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        dynamic_ncols=True,
        leave=leave,
    )