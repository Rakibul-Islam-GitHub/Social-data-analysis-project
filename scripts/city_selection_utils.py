"""
City selection utilities for the air-quality project.

This module turns the coverage EDA output into final city lists for:
- main PM2.5 comparison
- PM2.5 / PM10 particle-profile analysis
- excluded or context cities

It is designed to be imported inside notebooks/EDA.ipynb after
city_coverage_selection_summary.csv has been created.
"""

from pathlib import Path
import re

import pandas as pd


DEFAULT_EXCLUDED_MAIN_CITIES = [
    "Kuwait City",
    "Abu Dhabi",
    "Cape Town",
]


DEFAULT_EXCLUSION_REASONS = {
    "kuwait_city": (
        "Excluded because negative PM2.5 values appeared in the coverage summary; "
        "requires additional validation before use."
    ),
    "abu_dhabi": (
        "Excluded because PM2.5 coverage does not align well with the "
        "Dubai-centered 2024–2025 comparison period."
    ),
    "cape_town": (
        "Excluded because available coverage ends too early for the main "
        "2025 web comparison."
    ),
}


def normalize_city_name(value):
    # Stable city name key for matching names like Kuwait City and kuwait_city
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def load_city_coverage_summary(path):
    # City coverage summary loaded from the EDA output
    df = pd.read_csv(path)

    for column in ["first_month", "last_month"]:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")

    return df


def create_final_city_selection_tables(
    city_coverage_summary,
    excluded_main_cities=None,
    particle_profile_min_pm10_months=6,
    keep_context_cities_in_main=True,
):
    # Final city selection tables after the coverage audit
    if excluded_main_cities is None:
        excluded_main_cities = DEFAULT_EXCLUDED_MAIN_CITIES

    df = city_coverage_summary.copy()

    if "city" not in df.columns:
        raise ValueError("city column is required.")

    if "recommended_use" not in df.columns:
        raise ValueError("recommended_use column is required.")

    if "usable_pm10_months" not in df.columns:
        raise ValueError("usable_pm10_months column is required.")

    df["city_key"] = df["city"].apply(normalize_city_name)

    excluded_city_keys = [
        normalize_city_name(city)
        for city in excluded_main_cities
    ]

    df["excluded_from_main"] = df["city_key"].isin(excluded_city_keys)

    if keep_context_cities_in_main:
        allowed_recommended_use = df["recommended_use"] != "Exclude for main analysis"
    else:
        allowed_recommended_use = df["recommended_use"].isin(
            [
                "Core PM2.5 comparison",
                "Core + particle-profile analysis",
            ]
        )

    df["final_pm25_main_city"] = (
        allowed_recommended_use
        & (~df["excluded_from_main"])
    )

    df["final_particle_profile_city"] = (
        df["final_pm25_main_city"]
        & (df["usable_pm10_months"] >= particle_profile_min_pm10_months)
    )

    df["final_city_role"] = "Excluded / context"

    df.loc[
        df["final_pm25_main_city"],
        "final_city_role"
    ] = "Main PM2.5 comparison"

    df.loc[
        df["final_particle_profile_city"],
        "final_city_role"
    ] = "Main PM2.5 + particle-profile analysis"

    df.loc[
        df["excluded_from_main"],
        "final_city_role"
    ] = "Excluded from main web story"

    df["selection_note"] = ""

    for city_key, reason in DEFAULT_EXCLUSION_REASONS.items():
        df.loc[
            df["city_key"] == city_key,
            "selection_note"
        ] = reason

    df.loc[
        df["final_pm25_main_city"]
        & (df["selection_note"] == ""),
        "selection_note"
    ] = "Selected for the main PM2.5 global comparison."

    df.loc[
        df["final_particle_profile_city"],
        "selection_note"
    ] = (
        "Selected for both the main PM2.5 comparison and "
        "PM2.5/PM10 particle-profile analysis."
    )

    final_city_selection_all = (
        df
        .sort_values(
            [
                "final_pm25_main_city",
                "final_particle_profile_city",
                "usable_pm25_months",
                "city",
            ],
            ascending=[False, False, False, True],
        )
        .reset_index(drop=True)
    )

    selected_cities_pm25_main = (
        final_city_selection_all[
            final_city_selection_all["final_pm25_main_city"]
        ]
        .copy()
        .reset_index(drop=True)
    )

    selected_cities_particle_profile = (
        final_city_selection_all[
            final_city_selection_all["final_particle_profile_city"]
        ]
        .copy()
        .reset_index(drop=True)
    )

    excluded_or_context_cities = (
        final_city_selection_all[
            ~final_city_selection_all["final_pm25_main_city"]
        ]
        .copy()
        .reset_index(drop=True)
    )

    return {
        "final_city_selection_all": final_city_selection_all,
        "selected_cities_pm25_main": selected_cities_pm25_main,
        "selected_cities_particle_profile": selected_cities_particle_profile,
        "excluded_or_context_cities": excluded_or_context_cities,
    }


def save_city_selection_tables(selection_tables, output_dir):
    # City selection tables saved for later processing and webpage documentation
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {}

    for table_name, table in selection_tables.items():
        output_path = output_dir / f"{table_name}.csv"
        table.to_csv(output_path, index=False)
        output_paths[table_name] = output_path

    return output_paths


def create_final_city_lists(
    city_coverage_summary_path,
    output_dir="data/processed/combined/city_selection",
    excluded_main_cities=None,
    particle_profile_min_pm10_months=6,
    keep_context_cities_in_main=True,
):
    # Full workflow from coverage summary to saved final city selection CSV files
    city_coverage_summary = load_city_coverage_summary(city_coverage_summary_path)

    selection_tables = create_final_city_selection_tables(
        city_coverage_summary=city_coverage_summary,
        excluded_main_cities=excluded_main_cities,
        particle_profile_min_pm10_months=particle_profile_min_pm10_months,
        keep_context_cities_in_main=keep_context_cities_in_main,
    )

    output_paths = save_city_selection_tables(
        selection_tables=selection_tables,
        output_dir=output_dir,
    )

    return selection_tables, output_paths


def print_city_selection_summary(selection_tables):
    # Compact text summary for notebook output
    final_city_selection_all = selection_tables["final_city_selection_all"]
    selected_cities_pm25_main = selection_tables["selected_cities_pm25_main"]
    selected_cities_particle_profile = selection_tables["selected_cities_particle_profile"]
    excluded_or_context_cities = selection_tables["excluded_or_context_cities"]

    print("Total cities:", len(final_city_selection_all))
    print("Selected PM2.5 main cities:", len(selected_cities_pm25_main))
    print("Selected particle-profile cities:", len(selected_cities_particle_profile))
    print("Excluded / context cities:", len(excluded_or_context_cities))
