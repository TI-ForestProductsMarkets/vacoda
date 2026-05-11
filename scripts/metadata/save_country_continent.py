import pandas as pd
import pycountry
import pycountry_convert as pc
from vacoda.io.paths import get_external_continent_mapping_file_path
rows = []

for country in pycountry.countries:
    try:
        alpha3 = country.alpha_3
        alpha2 = country.alpha_2

        continent_code = pc.country_alpha2_to_continent_code(alpha2)

        rows.append(
            {
                "iso3": alpha3,
                "continent_code": continent_code,
            }
        )
    except Exception:
        continue

df = pd.DataFrame(rows)

# Map to readable names
CONTINENT_MAP = {
    "AF": "Africa",
    "AS": "Asia",
    "EU": "Europe",
    "NA": "North America",
    "SA": "South America",
    "OC": "Oceania",
}

df["continent"] = df["continent_code"].map(CONTINENT_MAP)

# Save file
print("Saving continent mapping to:", get_external_continent_mapping_file_path())
df[["iso3", "continent"]].to_csv(get_external_continent_mapping_file_path(), index=False)

print("Saved mapping with shape:", df.shape)
