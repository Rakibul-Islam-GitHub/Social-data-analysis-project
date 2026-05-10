# Create compact 5-location-per-city S3 manifest

from pathlib import Path

import numpy as np
import pandas as pd

metadata_root = Path("data/metadata")
combined_dir = metadata_root / "combined"

BALANCED_MANIFEST_PATH = combined_dir / "s3_fetch_manifest_balanced_pm25_locations.csv"

COMPACT_MANIFEST_PATH = (
    combined_dir / "s3_fetch_manifest_compact_5_locations_per_city.csv"
)
COMPACT_SUMMARY_PATH = combined_dir / "s3_fetch_manifest_compact_5_city_summary.csv"

MAX_LOCATIONS_PER_CITY = 5

manifest = pd.read_csv(BALANCED_MANIFEST_PATH)

# Boolean cleanup for Colab CSV loading
bool_columns = [
    "has_pm25",
    "has_pm10",
    "eligible_particle_profile",
    "include_for_s3_main",
    "include_for_particle_profile",
    "include_pm10_only_optional",
]

for column in bool_columns:
    if column in manifest.columns:
        if manifest[column].dtype == "object":
            manifest[column] = (
                manifest[column]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({"true": True, "false": False})
                .fillna(False)
            )

# Coordinate cleanup
manifest["latitude"] = pd.to_numeric(manifest["latitude"], errors="coerce")
manifest["longitude"] = pd.to_numeric(manifest["longitude"], errors="coerce")

# Main compact manifest still uses PM2.5 locations only
main_manifest = manifest[manifest["has_pm25"] == True].copy()

# Priority:
# 1 = locations with both PM2.5 and PM10
# 2 = PM2.5-only locations
main_manifest["compact_priority"] = 2

main_manifest.loc[
    main_manifest["eligible_particle_profile"] == True, "compact_priority"
] = 1


def haversine_distance_km(lat1, lon1, lat2, lon2):
    # Approximate distance between two coordinates
    radius_km = 6371.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2

    c = 2 * np.arcsin(np.sqrt(a))

    return radius_km * c


def spatially_spread_selection(city_df, max_locations=5):
    # Select locations using pollutant priority and spatial spread
    city_df = city_df.copy()

    city_df = city_df.sort_values(
        ["compact_priority", "download_priority", "location_name", "location_id"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)

    if len(city_df) <= max_locations:
        city_df["compact_selection_reason"] = "kept_all_locations"
        return city_df

    selected_indices = []

    # First location: highest pollutant priority
    selected_indices.append(city_df.index[0])

    # Remaining locations: maximize distance from already selected locations
    while len(selected_indices) < max_locations:
        remaining_indices = [
            index for index in city_df.index if index not in selected_indices
        ]

        best_index = None
        best_score = -1

        for index in remaining_indices:
            row = city_df.loc[index]

            # Missing coordinates are allowed but receive low spatial score
            if pd.isna(row["latitude"]) or pd.isna(row["longitude"]):
                spatial_score = 0
            else:
                distances = []

                for selected_index in selected_indices:
                    selected_row = city_df.loc[selected_index]

                    if pd.isna(selected_row["latitude"]) or pd.isna(
                        selected_row["longitude"]
                    ):
                        continue

                    distance = haversine_distance_km(
                        row["latitude"],
                        row["longitude"],
                        selected_row["latitude"],
                        selected_row["longitude"],
                    )

                    distances.append(distance)

                spatial_score = min(distances) if distances else 0

            # PM2.5+PM10 locations still get a priority boost
            priority_bonus = 1000 if row["compact_priority"] == 1 else 0

            score = priority_bonus + spatial_score

            if score > best_score:
                best_score = score
                best_index = index

        selected_indices.append(best_index)

    selected = city_df.loc[selected_indices].copy()

    selected["compact_selection_reason"] = "priority_plus_spatial_spread"

    selected = selected.sort_values(
        ["compact_priority", "location_name", "location_id"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    return selected


compact_manifest = (
    main_manifest.groupby("city", group_keys=False)
    .apply(lambda city_df: spatially_spread_selection(city_df, MAX_LOCATIONS_PER_CITY))
    .reset_index(drop=True)
)

compact_manifest["compact_manifest_note"] = (
    "Compact first-pass S3 manifest. "
    "Maximum 5 PM2.5 locations per city. "
    "PM2.5+PM10 locations are prioritized, then spatial spread is used."
)

compact_city_summary = (
    compact_manifest.groupby("city")
    .agg(
        selected_locations=("location_id", "nunique"),
        selected_particle_profile_locations=("eligible_particle_profile", "sum"),
        selected_pm25_locations=("has_pm25", "sum"),
        selected_pm10_locations=("has_pm10", "sum"),
        min_latitude=("latitude", "min"),
        max_latitude=("latitude", "max"),
        min_longitude=("longitude", "min"),
        max_longitude=("longitude", "max"),
    )
    .reset_index()
    .sort_values(
        ["selected_locations", "selected_particle_profile_locations"],
        ascending=[False, False],
    )
)

compact_manifest.to_csv(COMPACT_MANIFEST_PATH, index=False)

compact_city_summary.to_csv(COMPACT_SUMMARY_PATH, index=False)

print("Compact manifest saved to:")
print(COMPACT_MANIFEST_PATH)

print("\nCompact city summary saved to:")
print(COMPACT_SUMMARY_PATH)

print("\nTotal selected locations:", len(compact_manifest))
print("Cities included:", compact_manifest["city"].nunique())

compact_city_summary
