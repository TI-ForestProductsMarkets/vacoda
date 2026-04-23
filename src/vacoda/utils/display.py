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