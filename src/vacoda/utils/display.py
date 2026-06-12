import pandas as pd

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

def get_symbols(name):
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

    else:
        raise ValueError(f"Unknown symbol name: {name}")