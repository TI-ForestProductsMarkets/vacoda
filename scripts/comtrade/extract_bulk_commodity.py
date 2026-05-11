import gzip
import pandas as pd
from pathlib import Path

bulk_dir = Path("../data/raw/api/bulk")
out_file = Path("../data/interim/comtrade_441012_2023_from_bulk.parquet")
out_file.parent.mkdir(parents=True, exist_ok=True)

target = "441012"
target_flows = {"M", "X"} 

chunks_out = []
summary_rows = []

for path in sorted(bulk_dir.glob("Comtrade_bulk_441012_*.bin")):
    print(f"Reading {path.name}")

    total_rows = 0
    selected_rows = 0
    flow_counts = {}

    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        reader = pd.read_csv(f, sep="\t", dtype=str, chunksize=200_000)

        for chunk in reader:
            total_rows += len(chunk)

            mask = chunk["cmdCode"].eq(target)

            # Keep this if you want only import/export
            mask &= chunk["flowCode"].isin(target_flows)

            selected = chunk.loc[mask].copy()

            if not selected.empty:
                selected["source_file"] = path.name
                chunks_out.append(selected)

                selected_rows += len(selected)

                counts = selected["flowCode"].value_counts(dropna=False)
                for k, v in counts.items():
                    flow_counts[k] = flow_counts.get(k, 0) + int(v)

    summary_rows.append({
        "source_file": path.name,
        "total_rows_scanned": total_rows,
        "selected_rows": selected_rows,
        "flow_counts": flow_counts,
    })

summary = pd.DataFrame(summary_rows)

if chunks_out:
    result = pd.concat(chunks_out, ignore_index=True)
    result.to_parquet(out_file, index=False)

    print("Saved:", out_file)
    print("Shape:", result.shape)
    print("Flows:")
    print(result["flowCode"].value_counts(dropna=False))
    print("Reporters:", result["reporterCode"].nunique())
    print("Partners:", result["partnerCode"].nunique())
else:
    print("No rows found.")

print(summary)