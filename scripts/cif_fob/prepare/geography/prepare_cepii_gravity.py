"""
scripts.cif_fob.prepare.geography.prepare_cepii_gravity

------------
Script to prepare the CEPII gravity dataset for use in the CIF-FOB project.
This script reads the raw CEPII gravity data, processes it to extract relevant geographic information for
country pairs, and writes the processed data to a parquet file for downstream use.
"""
from vacoda.steps.cif_fob.references.geography.cepii_gravity import prepare_gravity_pair_geography
from vacoda.config.paths import get_geography_cepii_gravity_dir

INPUT_PATH =  get_geography_cepii_gravity_dir() / "source" / "Gravity_V202211.csv"
OUTPUT_PATH = get_geography_cepii_gravity_dir() / "prepared" / "country_pair_geography.parquet"

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


