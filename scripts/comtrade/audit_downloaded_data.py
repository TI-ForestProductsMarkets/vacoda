"""
audit_downloaded_data.py
Script to audit the downloaded trade data files for size, column structure, and sample content.
"""

from pathlib import Path
from statistics import mean, median
import pandas as pd

DATA_DIR = Path("/data/comtrade/data/raw/api/single")
files = sorted(DATA_DIR.rglob("*.csv"))
sizes = [p.stat().st_size for p in files]

print("data_dir:", DATA_DIR)
print("csv_files:", len(files))
print("total_gb:", round(sum(sizes) / 1024**3, 2))
print("mean_mb:", round(mean(sizes) / 1024**2, 2))
print("median_mb:", round(median(sizes) / 1024**2, 2))
print("max_mb:", round(max(sizes) / 1024**2, 2))
print("min_bytes:", min(sizes))

print("\nsmallest files:")
for p in sorted(files, key=lambda x: x.stat().st_size)[:10]:
    print(f"{p.stat().st_size:>10}  {p.name}")

print("\nlargest files:")
for p in sorted(files, key=lambda x: x.stat().st_size, reverse=True)[:10]:
    print(f"{p.stat().st_size:>10}  {p.name}")

print("\nsample columns:")
sample = files[0]
df = pd.read_csv(sample, nrows=3)
print("sample_file:", sample.name)
print("sample_shape:", df.shape)
for c in df.columns:
    print(" ", c)
