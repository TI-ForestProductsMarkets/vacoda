"""
scripts.hs_sync.02_prepare_harvard_raw_parquet_layout
Convert partitioned staged output into the raw_parquet/HS/YEAR layout expected by the Harvard generator.
"""
from __future__ import annotations

import shutil
from pathlib import Path


def main() -> None:
    root = Path("/storage/vacoda/data/external/harvard_comtrade/as_reported")
    source_root = root / "raw_parquet_partitioned"
    target_root = root / "raw_parquet"

    target_root.mkdir(parents=True, exist_ok=True)

    for hs_dir in source_root.glob("hs_partition=*"):
        hs = hs_dir.name.split("=", 1)[1]

        for year_dir in hs_dir.glob("year_partition=*"):
            year = year_dir.name.split("=", 1)[1]
            out_dir = target_root / hs / year
            out_dir.mkdir(parents=True, exist_ok=True)

            for parquet_path in year_dir.glob("*.parquet"):
                shutil.copy2(parquet_path, out_dir / parquet_path.name)

    print(f"Prepared Harvard raw_parquet layout under: {target_root}")


if __name__ == "__main__":
    main()