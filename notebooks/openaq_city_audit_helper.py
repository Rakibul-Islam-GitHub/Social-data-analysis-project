"""
OpenAQ city metadata audit helper.

This module collects OpenAQ location IDs for candidate city groups and saves
pollutant-specific metadata files for later S3 downloads.

The core project rule is:
- PM2.5 locations are kept for the main comparison.
- PM10 locations are kept for context where available.
- Locations with both PM2.5 and PM10 are kept for particle-profile analysis.

Expected outputs per city group:
- <slug>_all_openaq_locations.csv
- <slug>_all_openaq_locations_with_tiers.csv
- <slug>_city_pollutant_summary.csv
- <slug>_pm25_core_location_ids_for_s3_download.csv
- <slug>_pm10_context_location_ids_for_s3_download.csv
- <slug>_particle_profile_location_ids_for_s3_download.csv
"""

import os
import time
from pathlib import Path

import pandas as pd
import requests


BASE_URL = "https://api.openaq.org/v3"


CITY_GROUPS = {
    "dense_policy_asia": {
        "label": "Dense but policy-relevant Asian comparison",
        "output_slug": "dense-policy-asia",
        "cities": [
            {
                "city": "Singapore",
                "country": "Singapore",
                "iso": "SG",
                "bbox": [103.55, 1.15, 104.10, 1.50],
                "group": "Dense but policy-relevant Asian comparison"
            },
            {
                "city": "Seoul",
                "country": "South Korea",
                "iso": "KR",
                "bbox": [126.70, 37.40, 127.25, 37.75],
                "group": "Dense but policy-relevant Asian comparison"
            },
            {
                "city": "Tokyo",
                "country": "Japan",
                "iso": "JP",
                "bbox": [139.45, 35.45, 140.05, 35.95],
                "group": "Dense but policy-relevant Asian comparison"
            },
            {
                "city": "Beijing",
                "country": "China",
                "iso": "CN",
                "bbox": [115.80, 39.55, 117.20, 40.30],
                "group": "Dense but policy-relevant Asian comparison"
            }
        ]
    },
    "developed_city": {
        "label": "Policy / developed-city comparison",
        "output_slug": "developed-city",
        "cities": [
            {
                "city": "London",
                "country": "United Kingdom",
                "iso": "GB",
                "bbox": [-0.55, 51.25, 0.35, 51.75],
                "group": "Policy / developed-city comparison"
            },
            {
                "city": "Los Angeles",
                "country": "United States",
                "iso": "US",
                "bbox": [-118.70, 33.70, -117.90, 34.35],
                "group": "Policy / developed-city comparison"
            },
            {
                "city": "New York",
                "country": "United States",
                "iso": "US",
                "bbox": [-74.30, 40.45, -73.65, 40.95],
                "group": "Policy / developed-city comparison"
            }
        ]
    },
    "optional_contrast": {
        "label": "Optional contrast",
        "output_slug": "optional-contrast",
        "cities": [
            {
                "city": "Mexico City",
                "country": "Mexico",
                "iso": "MX",
                "bbox": [-99.40, 19.10, -98.85, 19.65],
                "group": "Optional contrast"
            },
            {
                "city": "Cape Town",
                "country": "South Africa",
                "iso": "ZA",
                "bbox": [18.25, -34.10, 18.75, -33.70],
                "group": "Optional contrast"
            }
        ]
    }
}


def make_headers(api_key=None):
    # OpenAQ API key header configuration
    key = api_key or os.getenv("OPENAQ_API_KEY")

    if key is None:
        raise RuntimeError(
            "OPENAQ_API_KEY is missing. The key can be stored as an environment variable or passed to the function."
        )

    return {
        "X-API-Key": key
    }


def openaq_get(endpoint, params=None, headers=None, max_retries=3, sleep_seconds=2):
    # General OpenAQ GET request helper
    url = f"{BASE_URL}/{endpoint.lstrip('/')}"
    params = params or {}

    for attempt in range(max_retries):
        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=60
        )

        if response.status_code == 429 and attempt < max_retries - 1:
            time.sleep(sleep_seconds * (attempt + 1))
            continue

        response.raise_for_status()
        return response.json()

    response.raise_for_status()
    return None


def fetch_paginated(endpoint, params=None, headers=None, limit=1000, pause_seconds=0.2):
    # Pagination helper for OpenAQ list endpoints
    params = params.copy() if params else {}
    params["limit"] = limit

    page = 1
    results = []

    while True:
        params["page"] = page
        payload = openaq_get(endpoint, params=params, headers=headers)

        batch = payload.get("results", [])
        meta = payload.get("meta", {})
        found = meta.get("found")

        results.extend(batch)

        if len(batch) == 0:
            break

        if found is not None and len(results) >= found:
            break

        page += 1
        time.sleep(pause_seconds)

    return results


def get_parameter_ids(headers):
    # Parameter IDs resolved from OpenAQ metadata
    parameters = fetch_paginated("parameters", params={}, headers=headers, limit=1000)
    parameters_df = pd.json_normalize(parameters)

    parameter_lookup = (
        parameters_df
        .set_index("name")["id"]
        .to_dict()
    )

    pm25_id = parameter_lookup["pm25"]
    pm10_id = parameter_lookup["pm10"]

    return pm25_id, pm10_id, parameters_df


def parse_openaq_datetime(value):
    # Datetime parser for OpenAQ nested datetime fields
    if value is None:
        return pd.NaT

    if isinstance(value, dict):
        value = value.get("utc") or value.get("local")

    return pd.to_datetime(value, utc=True, errors="coerce")


def get_nested_name(value):
    # Name extractor for nested metadata fields
    if isinstance(value, dict):
        return value.get("name")
    return None


def summarize_sensor_dates(sensors, parameter_name):
    # Sensor date summary for one parameter at one location
    matching_sensors = []

    for sensor in sensors:
        parameter = sensor.get("parameter", {})
        if parameter.get("name") == parameter_name:
            matching_sensors.append(sensor)

    if len(matching_sensors) == 0:
        return {
            "sensor_ids": [],
            "first": pd.NaT,
            "last": pd.NaT,
            "sensor_count": 0
        }

    first_dates = [
        parse_openaq_datetime(sensor.get("datetimeFirst"))
        for sensor in matching_sensors
    ]

    last_dates = [
        parse_openaq_datetime(sensor.get("datetimeLast"))
        for sensor in matching_sensors
    ]

    first_dates = [date for date in first_dates if pd.notna(date)]
    last_dates = [date for date in last_dates if pd.notna(date)]

    return {
        "sensor_ids": [sensor.get("id") for sensor in matching_sensors],
        "first": min(first_dates) if first_dates else pd.NaT,
        "last": max(last_dates) if last_dates else pd.NaT,
        "sensor_count": len(matching_sensors)
    }


def location_to_audit_row(location, city_spec):
    # Location metadata converted into one audit row
    sensors = location.get("sensors", [])

    pm25_summary = summarize_sensor_dates(sensors, "pm25")
    pm10_summary = summarize_sensor_dates(sensors, "pm10")

    has_pm25 = pm25_summary["sensor_count"] > 0
    has_pm10 = pm10_summary["sensor_count"] > 0

    overlap_start = pd.NaT
    overlap_end = pd.NaT
    overlap_months = 0

    if has_pm25 and has_pm10:
        overlap_start = max(pm25_summary["first"], pm10_summary["first"])
        overlap_end = min(pm25_summary["last"], pm10_summary["last"])

        if pd.notna(overlap_start) and pd.notna(overlap_end) and overlap_end > overlap_start:
            overlap_months = (overlap_end - overlap_start).days / 30.44

    coordinates = location.get("coordinates") or {}

    return {
        "city": city_spec["city"],
        "country": city_spec["country"],
        "iso": city_spec["iso"],
        "city_group": city_spec["group"],

        "location_id": location.get("id"),
        "location_name": location.get("name"),
        "latitude": coordinates.get("latitude"),
        "longitude": coordinates.get("longitude"),
        "timezone": location.get("timezone"),

        "provider": get_nested_name(location.get("provider")),
        "owner": get_nested_name(location.get("owner")),

        "is_mobile": location.get("isMobile"),
        "is_monitor": location.get("isMonitor"),

        "has_pm25": has_pm25,
        "has_pm10": has_pm10,

        "pm25_sensor_count": pm25_summary["sensor_count"],
        "pm10_sensor_count": pm10_summary["sensor_count"],

        "pm25_sensor_ids": pm25_summary["sensor_ids"],
        "pm10_sensor_ids": pm10_summary["sensor_ids"],

        "pm25_first": pm25_summary["first"],
        "pm25_last": pm25_summary["last"],
        "pm10_first": pm10_summary["first"],
        "pm10_last": pm10_summary["last"],

        "overlap_start": overlap_start,
        "overlap_end": overlap_end,
        "overlap_months": overlap_months,

        "s3_prefix": f"s3://openaq-data-archive/records/csv.gz/locationid={location.get('id')}/"
    }


def fetch_locations_for_city(city_spec, pm25_id, pm10_id, headers):
    # Location query for one city bounding box
    bbox_string = ",".join(str(value) for value in city_spec["bbox"])

    params = {
        "bbox": bbox_string,
        "iso": city_spec["iso"],
        "parameters_id": f"{pm25_id},{pm10_id}",
        "mobile": "false",
        "limit": 1000
    }

    locations = fetch_paginated("locations", params=params, headers=headers, limit=1000)

    rows = [
        location_to_audit_row(location, city_spec)
        for location in locations
    ]

    return rows


def add_pollutant_tiers(audit_df):
    # City-location eligibility categories
    audit_with_tiers = audit_df.copy()

    audit_with_tiers["eligible_core_pm25"] = audit_with_tiers["has_pm25"]

    audit_with_tiers["eligible_pm10_context"] = audit_with_tiers["has_pm10"]

    audit_with_tiers["eligible_particle_profile"] = (
        audit_with_tiers["has_pm25"] &
        audit_with_tiers["has_pm10"]
    )

    audit_with_tiers["coverage_tier"] = "Exclude"

    audit_with_tiers.loc[
        audit_with_tiers["eligible_core_pm25"] &
        audit_with_tiers["eligible_particle_profile"],
        "coverage_tier"
    ] = "Tier A - PM2.5 and PM10"

    audit_with_tiers.loc[
        audit_with_tiers["eligible_core_pm25"] &
        ~audit_with_tiers["eligible_particle_profile"],
        "coverage_tier"
    ] = "Tier B - PM2.5 only"

    audit_with_tiers.loc[
        ~audit_with_tiers["eligible_core_pm25"] &
        audit_with_tiers["eligible_pm10_context"],
        "coverage_tier"
    ] = "Tier C - PM10 only"

    return audit_with_tiers


def create_city_pollutant_summary(audit_with_tiers):
    # City-level pollutant availability summary
    city_pollutant_summary = (
        audit_with_tiers
        .groupby("city")
        .agg(
            total_locations=("location_id", "nunique"),
            pm25_locations=("has_pm25", "sum"),
            pm10_locations=("has_pm10", "sum"),
            paired_locations=("eligible_particle_profile", "sum"),
            tier_a_locations=("coverage_tier", lambda x: (x == "Tier A - PM2.5 and PM10").sum()),
            tier_b_locations=("coverage_tier", lambda x: (x == "Tier B - PM2.5 only").sum()),
            tier_c_locations=("coverage_tier", lambda x: (x == "Tier C - PM10 only").sum())
        )
        .reset_index()
    )

    city_pollutant_summary["city_has_pm25"] = city_pollutant_summary["pm25_locations"] > 0

    city_pollutant_summary["city_has_pm10"] = city_pollutant_summary["pm10_locations"] > 0

    city_pollutant_summary["city_has_both"] = (
        city_pollutant_summary["city_has_pm25"] &
        city_pollutant_summary["city_has_pm10"]
    )

    city_pollutant_summary["recommended_action"] = "Exclude for now"

    city_pollutant_summary.loc[
        city_pollutant_summary["city_has_both"],
        "recommended_action"
    ] = "Keep for full audit"

    city_pollutant_summary.loc[
        city_pollutant_summary["city_has_pm25"] &
        ~city_pollutant_summary["city_has_pm10"],
        "recommended_action"
    ] = "Keep for PM2.5 core, exclude from ratio analysis"

    city_pollutant_summary.loc[
        ~city_pollutant_summary["city_has_pm25"] &
        city_pollutant_summary["city_has_pm10"],
        "recommended_action"
    ] = "Optional PM10 context only"

    city_pollutant_summary = city_pollutant_summary.sort_values(
        ["city_has_both", "pm25_locations", "paired_locations"],
        ascending=[False, False, False]
    )

    return city_pollutant_summary


def create_location_id_tables(audit_with_tiers):
    # Pollutant-specific location ID tables for S3 downloads
    id_columns = [
        "city",
        "country",
        "iso",
        "city_group",
        "location_id",
        "location_name",
        "latitude",
        "longitude",
        "timezone",
        "provider",
        "owner",
        "coverage_tier",
        "pm25_sensor_count",
        "pm10_sensor_count",
        "pm25_sensor_ids",
        "pm10_sensor_ids",
        "s3_prefix"
    ]

    pm25_core_ids = (
        audit_with_tiers[
            audit_with_tiers["eligible_core_pm25"]
        ][id_columns]
        .sort_values(["city", "coverage_tier", "location_name"])
        .copy()
    )

    pm10_context_ids = (
        audit_with_tiers[
            audit_with_tiers["eligible_pm10_context"]
        ][id_columns]
        .sort_values(["city", "coverage_tier", "location_name"])
        .copy()
    )

    particle_profile_ids = (
        audit_with_tiers[
            audit_with_tiers["eligible_particle_profile"]
        ][id_columns]
        .sort_values(["city", "coverage_tier", "location_name"])
        .copy()
    )

    return pm25_core_ids, pm10_context_ids, particle_profile_ids


def save_group_outputs(
    group_key,
    audit_df,
    audit_with_tiers,
    city_pollutant_summary,
    pm25_core_ids,
    pm10_context_ids,
    particle_profile_ids,
    output_root="data/metadata"
):
    # Group-specific metadata files
    group_info = CITY_GROUPS[group_key]
    output_slug = group_info["output_slug"]

    output_dir = Path(output_root) / output_slug
    output_dir.mkdir(parents=True, exist_ok=True)

    audit_df.to_csv(
        output_dir / f"{output_slug}_all_openaq_locations.csv",
        index=False
    )

    audit_with_tiers.to_csv(
        output_dir / f"{output_slug}_all_openaq_locations_with_tiers.csv",
        index=False
    )

    city_pollutant_summary.to_csv(
        output_dir / f"{output_slug}_city_pollutant_summary.csv",
        index=False
    )

    pm25_core_ids.to_csv(
        output_dir / f"{output_slug}_pm25_core_location_ids_for_s3_download.csv",
        index=False
    )

    pm10_context_ids.to_csv(
        output_dir / f"{output_slug}_pm10_context_location_ids_for_s3_download.csv",
        index=False
    )

    particle_profile_ids.to_csv(
        output_dir / f"{output_slug}_particle_profile_location_ids_for_s3_download.csv",
        index=False
    )

    return output_dir


def audit_city_group(group_key, api_key=None, output_root="data/metadata"):
    # Complete metadata audit for one city group
    headers = make_headers(api_key=api_key)
    pm25_id, pm10_id, parameters_df = get_parameter_ids(headers=headers)

    group_info = CITY_GROUPS[group_key]
    group_rows = []

    for city_spec in group_info["cities"]:
        city_rows = fetch_locations_for_city(
            city_spec=city_spec,
            pm25_id=pm25_id,
            pm10_id=pm10_id,
            headers=headers
        )
        group_rows.extend(city_rows)

    audit_df = pd.DataFrame(group_rows)

    if audit_df.empty:
        audit_with_tiers = audit_df.copy()
        city_pollutant_summary = pd.DataFrame()
        pm25_core_ids = pd.DataFrame()
        pm10_context_ids = pd.DataFrame()
        particle_profile_ids = pd.DataFrame()
    else:
        audit_with_tiers = add_pollutant_tiers(audit_df)
        city_pollutant_summary = create_city_pollutant_summary(audit_with_tiers)
        pm25_core_ids, pm10_context_ids, particle_profile_ids = create_location_id_tables(audit_with_tiers)

    output_dir = save_group_outputs(
        group_key=group_key,
        audit_df=audit_df,
        audit_with_tiers=audit_with_tiers,
        city_pollutant_summary=city_pollutant_summary,
        pm25_core_ids=pm25_core_ids,
        pm10_context_ids=pm10_context_ids,
        particle_profile_ids=particle_profile_ids,
        output_root=output_root
    )

    return {
        "group_key": group_key,
        "label": group_info["label"],
        "output_dir": output_dir,
        "audit": audit_df,
        "audit_with_tiers": audit_with_tiers,
        "city_pollutant_summary": city_pollutant_summary,
        "pm25_core_ids": pm25_core_ids,
        "pm10_context_ids": pm10_context_ids,
        "particle_profile_ids": particle_profile_ids,
        "parameters": parameters_df
    }


def audit_multiple_city_groups(group_keys=None, api_key=None, output_root="data/metadata"):
    # Complete metadata audit for multiple city groups
    if group_keys is None:
        group_keys = list(CITY_GROUPS.keys())

    results = {}

    for group_key in group_keys:
        result = audit_city_group(
            group_key=group_key,
            api_key=api_key,
            output_root=output_root
        )
        results[group_key] = result

    return results


def combine_group_outputs(results, output_root="data/metadata"):
    # Combined metadata files across audited groups
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summary_tables = [
        result["city_pollutant_summary"].assign(group_key=group_key)
        for group_key, result in results.items()
        if not result["city_pollutant_summary"].empty
    ]

    pm25_tables = [
        result["pm25_core_ids"].assign(group_key=group_key)
        for group_key, result in results.items()
        if not result["pm25_core_ids"].empty
    ]

    pm10_tables = [
        result["pm10_context_ids"].assign(group_key=group_key)
        for group_key, result in results.items()
        if not result["pm10_context_ids"].empty
    ]

    particle_tables = [
        result["particle_profile_ids"].assign(group_key=group_key)
        for group_key, result in results.items()
        if not result["particle_profile_ids"].empty
    ]

    combined_city_summary = pd.concat(summary_tables, ignore_index=True) if summary_tables else pd.DataFrame()
    combined_pm25_core_ids = pd.concat(pm25_tables, ignore_index=True) if pm25_tables else pd.DataFrame()
    combined_pm10_context_ids = pd.concat(pm10_tables, ignore_index=True) if pm10_tables else pd.DataFrame()
    combined_particle_profile_ids = pd.concat(particle_tables, ignore_index=True) if particle_tables else pd.DataFrame()

    combined_city_summary.to_csv(
        output_root / "combined_city_pollutant_summary_remaining_groups.csv",
        index=False
    )

    combined_pm25_core_ids.to_csv(
        output_root / "combined_pm25_core_location_ids_remaining_groups.csv",
        index=False
    )

    combined_pm10_context_ids.to_csv(
        output_root / "combined_pm10_context_location_ids_remaining_groups.csv",
        index=False
    )

    combined_particle_profile_ids.to_csv(
        output_root / "combined_particle_profile_location_ids_remaining_groups.csv",
        index=False
    )

    return {
        "combined_city_summary": combined_city_summary,
        "combined_pm25_core_ids": combined_pm25_core_ids,
        "combined_pm10_context_ids": combined_pm10_context_ids,
        "combined_particle_profile_ids": combined_particle_profile_ids
    }


if __name__ == "__main__":
    results = audit_multiple_city_groups(
        group_keys=[
            "dense_policy_asia",
            "developed_city",
            "optional_contrast"
        ],
        output_root="data/metadata"
    )

    combine_group_outputs(
        results=results,
        output_root="data/metadata"
    )

    for group_key, result in results.items():
        print(f"{group_key}: {result['output_dir']}")
        print(result["city_pollutant_summary"])
        print()
