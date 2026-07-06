from pathlib import Path

from vacoda.cif_fob.references.geography.cepii_gravity import (
    prepare_gravity_pair_geography,
)


project_root = Path(__file__).resolve().parents[4]
print(f"Project root: {project_root}")

INPUT_PATH = (
    project_root
    / "references"
    / "geography"
    / "cepii_gravity_2021"
    / "source"
    / "Gravity_V202211.csv"
)
OUTPUT_PATH = (
    project_root
    / "references"
    / "geography"
    / "cepii_gravity_2021"
    / "prepared"
    / "country_pair_geography.parquet"
)

print(f"Input path: {INPUT_PATH}")
print(f"Output path: {OUTPUT_PATH}")

try:
    prepare_gravity_pair_geography(
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
        reference_year=2021,
        threads=58,
    )
except Exception as e:
    print(f"An error occurred: {e}")


