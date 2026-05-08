from pathlib import Path
import pandas as pd

DATA_DIR = Path("/data/comtrade/data/raw/api/single")
files = sorted(DATA_DIR.rglob("*.csv"))

parts = []

for i, p in enumerate(files, 1):
    df = pd.read_csv(
        p,
        usecols=["RefYear", "ClassificationCode"],
        dtype={
            "RefYear": "int16",
            "ClassificationCode": "string",
        },
    )

    if df.empty:
        continue

    g = (
        df.groupby(["RefYear", "ClassificationCode"], dropna=False)
        .size()
        .reset_index(name="n_rows")
    )

    parts.append(g)

    if i % 1000 == 0:
        print(f"processed {i}/{len(files)}")

year_hs = pd.concat(parts, ignore_index=True)

year_hs = (
    year_hs.groupby(["RefYear", "ClassificationCode"], dropna=False)["n_rows"]
    .sum()
    .reset_index()
)

pivot = year_hs.pivot_table(
    index="RefYear",
    columns="ClassificationCode",
    values="n_rows",
    aggfunc="sum",
    fill_value=0,
)

print(pivot)