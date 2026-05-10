"""
City-level Open-Meteo weather enrichment pipeline.

This script adds hourly Open-Meteo weather data to already processed city-level
OpenAQ files.

Expected input structure:
data/processed/openaq_by_city/
    bangkok/
        bangkok_city_hourly.csv or bangkok_city_hourly.parquet
        bangkok_city_daily.csv or bangkok_city_daily.parquet
        bangkok_city_monthly.csv or bangkok_city_monthly.parquet
    dubai/
        dubai_city_hourly.csv
        ...

Expected city selection file:
data/processed/combined/city_selection/selected_cities_pm25_main.csv

Main output structure:
data/processed/enriched/
    by_city/
        bangkok/
            bangkok_weather_hourly.csv
            bangkok_city_hourly_enriched.csv
            bangkok_city_daily_enriched.csv
            bangkok_city_monthly_enriched.csv
    combined/
        combined_city_hourly_enriched.csv
        combined_city_daily_enriched.csv
        combined_city_monthly_enriched.csv


"""

from __future__ import annotations

import math
import re
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import requests

PathLike = Union[str, Path]


OPENMETEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

DEFAULT_WEATHER_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
    "cloud_cover",
]

WEATHER_COLUMN_RENAME = {
    "temperature_2m": "temperature_2m_c",
    "relative_humidity_2m": "relative_humidity_2m_pct",
    "precipitation": "precipitation_mm",
    "wind_speed_10m": "wind_speed_10m_ms",
    "wind_direction_10m": "wind_direction_10m_deg",
    "surface_pressure": "surface_pressure_hpa",
    "cloud_cover": "cloud_cover_pct",
}

WHO_PM25_DAILY = 15.0
WHO_PM10_DAILY = 45.0


def clean_slug(value: str) -> str:
    # Stable folder and file name helper
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def load_table(path: PathLike) -> pd.DataFrame:
    # Tables can be loaded from CSV, CSV.GZ, or Parquet
    path = Path(path)
    suffixes = "".join(path.suffixes).lower()

    if suffixes.endswith(".parquet"):
        return pd.read_parquet(path)

    if suffixes.endswith(".csv") or suffixes.endswith(".csv.gz"):
        return pd.read_csv(path)

    raise ValueError(f"Unsupported table format: {path}")


def save_table(
    df: pd.DataFrame,
    path_without_suffix: PathLike,
    output_format: str = "both",
) -> list[Path]:
    # Table saver with CSV, Parquet, or both
    path_without_suffix = Path(path_without_suffix)
    path_without_suffix.parent.mkdir(parents=True, exist_ok=True)

    output_format = output_format.lower()
    saved_paths = []

    if output_format not in {"csv", "parquet", "both"}:
        raise ValueError("output_format must be 'csv', 'parquet', or 'both'.")

    if output_format in {"parquet", "both"}:
        parquet_path = path_without_suffix.with_suffix(".parquet")
        try:
            df.to_parquet(parquet_path, index=False)
            saved_paths.append(parquet_path)
        except Exception as exc:
            print(f"Parquet save failed for {path_without_suffix.name}. Error: {exc}")

    if output_format in {"csv", "both"}:
        csv_path = path_without_suffix.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        saved_paths.append(csv_path)

    return saved_paths


def find_city_file(city_folder: Path, city_slug: str, table_name: str) -> Path:
    # City file finder that prefers Parquet but can read CSV
    parquet_path = city_folder / f"{city_slug}_{table_name}.parquet"
    csv_path = city_folder / f"{city_slug}_{table_name}.csv"

    if parquet_path.exists():
        return parquet_path

    if csv_path.exists():
        return csv_path

    raise FileNotFoundError(
        f"Could not find {table_name} for {city_slug} in {city_folder}"
    )


def prepare_datetime_hour_utc(series: pd.Series) -> pd.Series:
    # UTC hourly datetime parser
    return pd.to_datetime(series, utc=True, errors="coerce").dt.floor("h")


def infer_date_range_from_city_hourly(city_hourly: pd.DataFrame) -> tuple[str, str]:
    # Date range inferred from available city-hourly data
    datetime_values = prepare_datetime_hour_utc(
        city_hourly["datetime_hour_utc"]
    ).dropna()

    if datetime_values.empty:
        raise ValueError("No valid datetime_hour_utc values were found.")

    start_date = datetime_values.min().date().isoformat()
    end_date = datetime_values.max().date().isoformat()

    return start_date, end_date


def filter_city_hourly_by_date(
    city_hourly: pd.DataFrame,
    start_date: Optional[str],
    end_date: Optional[str],
) -> pd.DataFrame:
    # City-hourly data filtered to the selected analysis window
    df = city_hourly.copy()

    df["datetime_hour_utc"] = prepare_datetime_hour_utc(df["datetime_hour_utc"])

    if start_date is not None:
        start_ts = pd.Timestamp(start_date, tz="UTC")
        df = df[df["datetime_hour_utc"] >= start_ts].copy()

    if end_date is not None:
        end_ts = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)
        df = df[df["datetime_hour_utc"] < end_ts].copy()

    return df.reset_index(drop=True)


def make_city_weather_point(city_hourly: pd.DataFrame, city_slug: str) -> dict:
    # One weather point per city using median city-hourly coordinates
    if "city_latitude" in city_hourly.columns:
        lat_column = "city_latitude"
    elif "lat" in city_hourly.columns:
        lat_column = "lat"
    else:
        raise ValueError("No latitude column was found.")

    if "city_longitude" in city_hourly.columns:
        lon_column = "city_longitude"
    elif "lon" in city_hourly.columns:
        lon_column = "lon"
    else:
        raise ValueError("No longitude column was found.")

    latitude = pd.to_numeric(city_hourly[lat_column], errors="coerce").median()
    longitude = pd.to_numeric(city_hourly[lon_column], errors="coerce").median()

    if pd.isna(latitude) or pd.isna(longitude):
        raise ValueError(f"Missing city weather coordinates for {city_slug}")

    city_name = (
        city_hourly["city"].dropna().iloc[0]
        if "city" in city_hourly.columns
        else city_slug
    )
    country = (
        city_hourly["country"].dropna().iloc[0]
        if "country" in city_hourly.columns and city_hourly["country"].notna().any()
        else pd.NA
    )
    city_group = (
        city_hourly["city_group"].dropna().iloc[0]
        if "city_group" in city_hourly.columns
        and city_hourly["city_group"].notna().any()
        else pd.NA
    )

    return {
        "city": city_name,
        "city_slug": city_slug,
        "country": country,
        "city_group": city_group,
        "weather_point_level": "city",
        "weather_point_id": f"city_{city_slug}",
        "weather_latitude": float(latitude),
        "weather_longitude": float(longitude),
    }


def fetch_openmeteo_hourly(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    variables: list[str] = DEFAULT_WEATHER_VARIABLES,
    timezone: str = "UTC",
    wind_speed_unit: str = "ms",
    temperature_unit: str = "celsius",
    precipitation_unit: str = "mm",
    timeout: int = 60,
    max_retries: int = 3,
    sleep_seconds: float = 2.0,
) -> pd.DataFrame:
    # Hourly historical weather request for one coordinate and date range
    params = {
        "latitude": float(latitude),
        "longitude": float(longitude),
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(variables),
        "timezone": timezone,
        "wind_speed_unit": wind_speed_unit,
        "temperature_unit": temperature_unit,
        "precipitation_unit": precipitation_unit,
    }

    last_error = None

    for attempt in range(max_retries):
        try:
            response = requests.get(
                OPENMETEO_ARCHIVE_URL,
                params=params,
                timeout=timeout,
            )

            if (
                response.status_code in {429, 500, 502, 503, 504}
                and attempt < max_retries - 1
            ):
                time.sleep(sleep_seconds * (attempt + 1))
                continue

            response.raise_for_status()
            payload = response.json()

            if "hourly" not in payload:
                raise ValueError(
                    f"Open-Meteo response did not include hourly data: {payload}"
                )

            weather = pd.DataFrame(payload["hourly"])

            if weather.empty:
                return weather

            weather = weather.rename(columns={"time": "datetime_hour_utc"})
            weather["datetime_hour_utc"] = prepare_datetime_hour_utc(
                weather["datetime_hour_utc"]
            )
            weather = weather.rename(columns=WEATHER_COLUMN_RENAME)

            weather["weather_latitude"] = float(latitude)
            weather["weather_longitude"] = float(longitude)
            weather["openmeteo_timezone"] = payload.get("timezone", timezone)
            weather["openmeteo_elevation"] = payload.get("elevation")

            return weather

        except Exception as exc:
            last_error = exc

            if attempt < max_retries - 1:
                time.sleep(sleep_seconds * (attempt + 1))
                continue

    raise RuntimeError(f"Open-Meteo request failed after retries: {last_error}")


def add_city_metadata_to_weather(
    weather: pd.DataFrame, weather_point: dict
) -> pd.DataFrame:
    # City metadata added to one weather table
    weather = weather.copy()

    for key, value in weather_point.items():
        if key not in {"weather_latitude", "weather_longitude"}:
            weather[key] = value

    front_columns = [
        "city",
        "city_slug",
        "country",
        "city_group",
        "weather_point_level",
        "weather_point_id",
        "datetime_hour_utc",
        "weather_latitude",
        "weather_longitude",
        "temperature_2m_c",
        "relative_humidity_2m_pct",
        "precipitation_mm",
        "wind_speed_10m_ms",
        "wind_direction_10m_deg",
        "surface_pressure_hpa",
        "cloud_cover_pct",
    ]

    front_columns = [column for column in front_columns if column in weather.columns]
    other_columns = [
        column for column in weather.columns if column not in front_columns
    ]

    return weather[front_columns + other_columns]


def join_weather_to_city_hourly(
    city_hourly: pd.DataFrame,
    weather_hourly: pd.DataFrame,
) -> pd.DataFrame:
    # Weather joined to city-hourly air quality
    city_df = city_hourly.copy()
    weather_df = weather_hourly.copy()

    city_df["datetime_hour_utc"] = prepare_datetime_hour_utc(
        city_df["datetime_hour_utc"]
    )
    weather_df["datetime_hour_utc"] = prepare_datetime_hour_utc(
        weather_df["datetime_hour_utc"]
    )

    weather_keep_columns = [
        "datetime_hour_utc",
        "weather_point_id",
        "weather_latitude",
        "weather_longitude",
        "temperature_2m_c",
        "relative_humidity_2m_pct",
        "precipitation_mm",
        "wind_speed_10m_ms",
        "wind_direction_10m_deg",
        "surface_pressure_hpa",
        "cloud_cover_pct",
        "openmeteo_timezone",
        "openmeteo_elevation",
    ]

    weather_keep_columns = [
        column for column in weather_keep_columns if column in weather_df.columns
    ]

    enriched = city_df.merge(
        weather_df[weather_keep_columns],
        on="datetime_hour_utc",
        how="left",
    )

    return enriched.sort_values("datetime_hour_utc").reset_index(drop=True)


def quantile_90(series: pd.Series) -> float:
    # Robust 90th percentile helper
    values = pd.to_numeric(series, errors="coerce").dropna()

    if values.empty:
        return np.nan

    return float(values.quantile(0.90))


def circular_mean_degrees(series: pd.Series) -> float:
    # Circular mean for wind direction in degrees
    values = pd.to_numeric(series, errors="coerce").dropna()

    if values.empty:
        return np.nan

    radians = np.deg2rad(values)
    sin_mean = np.sin(radians).mean()
    cos_mean = np.cos(radians).mean()

    if sin_mean == 0 and cos_mean == 0:
        return np.nan

    angle = math.degrees(math.atan2(sin_mean, cos_mean))

    return angle % 360


def make_nullable_exceedance(
    mean_values: pd.Series,
    valid_hours: pd.Series,
    threshold: float,
    min_valid_hours: int,
) -> pd.Series:
    # Nullable WHO exceedance flag with missing values for low-coverage days
    mean_values = pd.to_numeric(mean_values, errors="coerce")
    valid_hours = pd.to_numeric(valid_hours, errors="coerce")

    enough_hours = valid_hours >= min_valid_hours

    exceedance = pd.Series(pd.NA, index=mean_values.index, dtype="boolean")
    exceedance.loc[enough_hours] = mean_values.loc[enough_hours] > threshold

    return exceedance


def aggregate_city_daily_enriched(
    city_hourly_enriched: pd.DataFrame,
    min_valid_hours: int = 12,
) -> pd.DataFrame:
    # Daily city table containing air quality and weather features
    df = city_hourly_enriched.copy()
    df["datetime_hour_utc"] = prepare_datetime_hour_utc(df["datetime_hour_utc"])

    if "local_date" in df.columns:
        df["local_date"] = pd.to_datetime(df["local_date"], errors="coerce").dt.date
    else:
        df["local_date"] = df["datetime_hour_utc"].dt.date

    if "local_year" not in df.columns:
        df["local_year"] = pd.to_datetime(df["local_date"]).dt.year

    if "local_month" not in df.columns:
        df["local_month"] = pd.to_datetime(df["local_date"]).dt.month

    if "local_day" not in df.columns:
        df["local_day"] = pd.to_datetime(df["local_date"]).dt.day

    group_columns = [
        column
        for column in [
            "city",
            "country",
            "iso",
            "city_group",
            "local_date",
            "local_year",
            "local_month",
            "local_day",
        ]
        if column in df.columns
    ]

    aggregation_spec = {
        "active_hours": ("datetime_hour_utc", pd.Series.nunique),
    }

    pollutant_sources = {
        "pm25": "pm25_median",
        "pm10": "pm10_median",
    }

    for pollutant, source_column in pollutant_sources.items():
        if source_column in df.columns:
            aggregation_spec[f"{pollutant}_daily_mean"] = (source_column, "mean")
            aggregation_spec[f"{pollutant}_daily_median"] = (source_column, "median")
            aggregation_spec[f"{pollutant}_daily_p90"] = (source_column, quantile_90)
            aggregation_spec[f"{pollutant}_daily_max"] = (source_column, "max")
            aggregation_spec[f"valid_{pollutant}_hours"] = (
                source_column,
                lambda x: int(pd.to_numeric(x, errors="coerce").notna().sum()),
            )

    if "valid_pm25_locations" in df.columns:
        aggregation_spec["valid_pm25_locations_mean"] = ("valid_pm25_locations", "mean")
        aggregation_spec["valid_pm25_locations_max"] = ("valid_pm25_locations", "max")

    if "valid_pm10_locations" in df.columns:
        aggregation_spec["valid_pm10_locations_mean"] = ("valid_pm10_locations", "mean")
        aggregation_spec["valid_pm10_locations_max"] = ("valid_pm10_locations", "max")

    if "active_locations" in df.columns:
        aggregation_spec["active_locations_mean"] = ("active_locations", "mean")
        aggregation_spec["active_locations_max"] = ("active_locations", "max")

    if "pm25_pm10_ratio_from_city_medians" in df.columns:
        aggregation_spec["pm25_pm10_ratio_daily_mean"] = (
            "pm25_pm10_ratio_from_city_medians",
            "mean",
        )
        aggregation_spec["pm25_pm10_ratio_daily_median"] = (
            "pm25_pm10_ratio_from_city_medians",
            "median",
        )

    if "temperature_2m_c" in df.columns:
        aggregation_spec["temperature_2m_mean_c"] = ("temperature_2m_c", "mean")
        aggregation_spec["temperature_2m_min_c"] = ("temperature_2m_c", "min")
        aggregation_spec["temperature_2m_max_c"] = ("temperature_2m_c", "max")

    if "relative_humidity_2m_pct" in df.columns:
        aggregation_spec["relative_humidity_2m_mean_pct"] = (
            "relative_humidity_2m_pct",
            "mean",
        )

    if "precipitation_mm" in df.columns:
        aggregation_spec["precipitation_total_mm"] = ("precipitation_mm", "sum")

    if "wind_speed_10m_ms" in df.columns:
        aggregation_spec["wind_speed_10m_mean_ms"] = ("wind_speed_10m_ms", "mean")
        aggregation_spec["wind_speed_10m_p90_ms"] = ("wind_speed_10m_ms", quantile_90)
        aggregation_spec["wind_speed_10m_max_ms"] = ("wind_speed_10m_ms", "max")

    if "wind_direction_10m_deg" in df.columns:
        aggregation_spec["wind_direction_10m_circular_mean_deg"] = (
            "wind_direction_10m_deg",
            circular_mean_degrees,
        )

    if "surface_pressure_hpa" in df.columns:
        aggregation_spec["surface_pressure_mean_hpa"] = ("surface_pressure_hpa", "mean")

    if "cloud_cover_pct" in df.columns:
        aggregation_spec["cloud_cover_mean_pct"] = ("cloud_cover_pct", "mean")

    city_daily = (
        df.groupby(group_columns, dropna=False).agg(**aggregation_spec).reset_index()
    )

    if {"pm25_daily_mean", "valid_pm25_hours"}.issubset(city_daily.columns):
        city_daily["pm25_who_daily_exceedance"] = make_nullable_exceedance(
            mean_values=city_daily["pm25_daily_mean"],
            valid_hours=city_daily["valid_pm25_hours"],
            threshold=WHO_PM25_DAILY,
            min_valid_hours=min_valid_hours,
        )
        city_daily["pm25_daily_coverage_pct"] = (
            pd.to_numeric(city_daily["valid_pm25_hours"], errors="coerce") / 24 * 100
        )

    if {"pm10_daily_mean", "valid_pm10_hours"}.issubset(city_daily.columns):
        city_daily["pm10_who_daily_exceedance"] = make_nullable_exceedance(
            mean_values=city_daily["pm10_daily_mean"],
            valid_hours=city_daily["valid_pm10_hours"],
            threshold=WHO_PM10_DAILY,
            min_valid_hours=min_valid_hours,
        )
        city_daily["pm10_daily_coverage_pct"] = (
            pd.to_numeric(city_daily["valid_pm10_hours"], errors="coerce") / 24 * 100
        )

    return city_daily.sort_values(["city", "local_date"]).reset_index(drop=True)


def count_true_nullable(series: pd.Series) -> int:
    # Boolean counter that safely handles True, False, NaN, and pd.NA
    return int(pd.Series(series).astype("boolean").fillna(False).sum())


def aggregate_city_monthly_enriched(
    city_daily_enriched: pd.DataFrame,
    min_valid_days: int = 15,
) -> pd.DataFrame:
    # Monthly city table containing air quality and weather features
    df = city_daily_enriched.copy()

    if "local_year" not in df.columns or "local_month" not in df.columns:
        dates = pd.to_datetime(df["local_date"], errors="coerce")
        df["local_year"] = dates.dt.year
        df["local_month"] = dates.dt.month

    group_columns = [
        column
        for column in [
            "city",
            "country",
            "iso",
            "city_group",
            "local_year",
            "local_month",
        ]
        if column in df.columns
    ]

    aggregation_spec = {
        "valid_calendar_days_with_any_data": ("local_date", pd.Series.nunique),
    }

    for pollutant in ["pm25", "pm10"]:
        mean_column = f"{pollutant}_daily_mean"
        median_column = f"{pollutant}_daily_median"
        max_column = f"{pollutant}_daily_max"
        coverage_column = f"{pollutant}_daily_coverage_pct"
        exceedance_column = f"{pollutant}_who_daily_exceedance"

        if mean_column in df.columns:
            aggregation_spec[f"{pollutant}_monthly_mean"] = (mean_column, "mean")
            aggregation_spec[f"{pollutant}_monthly_median"] = (median_column, "median")
            aggregation_spec[f"{pollutant}_monthly_p90"] = (mean_column, quantile_90)
            aggregation_spec[f"{pollutant}_monthly_max"] = (max_column, "max")
            aggregation_spec[f"valid_{pollutant}_days"] = (
                mean_column,
                lambda x: int(pd.to_numeric(x, errors="coerce").notna().sum()),
            )

        if coverage_column in df.columns:
            aggregation_spec[f"{pollutant}_daily_coverage_pct_mean"] = (
                coverage_column,
                "mean",
            )

        if exceedance_column in df.columns:
            aggregation_spec[f"{pollutant}_exceedance_days"] = (
                exceedance_column,
                count_true_nullable,
            )

    if "pm25_pm10_ratio_daily_mean" in df.columns:
        aggregation_spec["pm25_pm10_ratio_monthly_mean"] = (
            "pm25_pm10_ratio_daily_mean",
            "mean",
        )
        aggregation_spec["pm25_pm10_ratio_monthly_median"] = (
            "pm25_pm10_ratio_daily_median",
            "median",
        )

    weather_mean_columns = [
        "temperature_2m_mean_c",
        "relative_humidity_2m_mean_pct",
        "wind_speed_10m_mean_ms",
        "surface_pressure_mean_hpa",
        "cloud_cover_mean_pct",
    ]

    for column in weather_mean_columns:
        if column in df.columns:
            monthly_column = column.replace("_mean_", "_monthly_mean_")
            aggregation_spec[monthly_column] = (column, "mean")

    if "precipitation_total_mm" in df.columns:
        aggregation_spec["precipitation_monthly_total_mm"] = (
            "precipitation_total_mm",
            "sum",
        )

    if "active_locations_mean" in df.columns:
        aggregation_spec["active_locations_mean"] = ("active_locations_mean", "mean")

    if "active_locations_max" in df.columns:
        aggregation_spec["active_locations_max"] = ("active_locations_max", "max")

    city_monthly = (
        df.groupby(group_columns, dropna=False).agg(**aggregation_spec).reset_index()
    )

    city_monthly["days_in_month"] = pd.to_datetime(
        city_monthly["local_year"].astype(int).astype(str)
        + "-"
        + city_monthly["local_month"].astype(int).astype(str).str.zfill(2)
        + "-01"
    ).dt.days_in_month

    for pollutant in ["pm25", "pm10"]:
        valid_days_column = f"valid_{pollutant}_days"
        exceedance_days_column = f"{pollutant}_exceedance_days"

        if valid_days_column in city_monthly.columns:
            city_monthly[f"{pollutant}_month_coverage_pct"] = (
                city_monthly[valid_days_column]
                / city_monthly["days_in_month"].replace(0, np.nan)
                * 100
            )

            city_monthly[f"{pollutant}_month_is_usable"] = (
                city_monthly[valid_days_column] >= min_valid_days
            )

        if (
            exceedance_days_column in city_monthly.columns
            and valid_days_column in city_monthly.columns
        ):
            city_monthly[f"{pollutant}_exceedance_pct"] = (
                city_monthly[exceedance_days_column]
                / city_monthly[valid_days_column].replace(0, np.nan)
                * 100
            )

    return city_monthly.sort_values(["city", "local_year", "local_month"]).reset_index(
        drop=True
    )


def process_one_city_weather(
    city_slug: str,
    input_root: PathLike = "data/processed/openaq_by_city",
    output_root: PathLike = "data/processed/enriched",
    start_date: Optional[str] = "2025-01-01",
    end_date: Optional[str] = "2025-12-31",
    output_format: str = "both",
    overwrite_weather: bool = False,
) -> dict:
    # Complete weather enrichment process for one city
    input_root = Path(input_root)
    output_root = Path(output_root)

    city_slug = clean_slug(city_slug)
    city_folder = input_root / city_slug

    city_output_folder = output_root / "by_city" / city_slug
    city_output_folder.mkdir(parents=True, exist_ok=True)

    city_hourly_path = find_city_file(city_folder, city_slug, "city_hourly")
    city_hourly = load_table(city_hourly_path)

    city_hourly = filter_city_hourly_by_date(
        city_hourly=city_hourly,
        start_date=start_date,
        end_date=end_date,
    )

    if city_hourly.empty:
        raise ValueError(
            f"No city-hourly data remains after date filtering for {city_slug}"
        )

    actual_start_date, actual_end_date = infer_date_range_from_city_hourly(city_hourly)

    weather_point = make_city_weather_point(
        city_hourly=city_hourly,
        city_slug=city_slug,
    )

    city_weather_points = pd.DataFrame([weather_point])

    weather_hourly_base = city_output_folder / f"{city_slug}_weather_hourly"
    weather_hourly_csv = weather_hourly_base.with_suffix(".csv")
    weather_hourly_parquet = weather_hourly_base.with_suffix(".parquet")

    if not overwrite_weather and weather_hourly_parquet.exists():
        weather_hourly = pd.read_parquet(weather_hourly_parquet)
    elif not overwrite_weather and weather_hourly_csv.exists():
        weather_hourly = pd.read_csv(weather_hourly_csv)
    else:
        weather_hourly = fetch_openmeteo_hourly(
            latitude=weather_point["weather_latitude"],
            longitude=weather_point["weather_longitude"],
            start_date=actual_start_date,
            end_date=actual_end_date,
        )

        weather_hourly = add_city_metadata_to_weather(
            weather=weather_hourly,
            weather_point=weather_point,
        )

        save_table(
            df=weather_hourly,
            path_without_suffix=weather_hourly_base,
            output_format=output_format,
        )

    city_hourly_enriched = join_weather_to_city_hourly(
        city_hourly=city_hourly,
        weather_hourly=weather_hourly,
    )

    city_daily_enriched = aggregate_city_daily_enriched(
        city_hourly_enriched=city_hourly_enriched,
    )

    city_monthly_enriched = aggregate_city_monthly_enriched(
        city_daily_enriched=city_daily_enriched,
    )

    save_table(
        df=city_weather_points,
        path_without_suffix=city_output_folder / f"{city_slug}_weather_point",
        output_format=output_format,
    )

    save_table(
        df=city_hourly_enriched,
        path_without_suffix=city_output_folder / f"{city_slug}_city_hourly_enriched",
        output_format=output_format,
    )

    save_table(
        df=city_daily_enriched,
        path_without_suffix=city_output_folder / f"{city_slug}_city_daily_enriched",
        output_format=output_format,
    )

    save_table(
        df=city_monthly_enriched,
        path_without_suffix=city_output_folder / f"{city_slug}_city_monthly_enriched",
        output_format=output_format,
    )

    log_row = {
        "city_slug": city_slug,
        "city": weather_point["city"],
        "country": weather_point["country"],
        "start_date": actual_start_date,
        "end_date": actual_end_date,
        "weather_latitude": weather_point["weather_latitude"],
        "weather_longitude": weather_point["weather_longitude"],
        "city_hourly_rows": len(city_hourly),
        "weather_hourly_rows": len(weather_hourly),
        "city_hourly_enriched_rows": len(city_hourly_enriched),
        "city_daily_enriched_rows": len(city_daily_enriched),
        "city_monthly_enriched_rows": len(city_monthly_enriched),
        "output_folder": str(city_output_folder),
        "status": "success",
        "error": None,
    }

    return {
        "log": log_row,
        "city_weather_points": city_weather_points,
        "weather_hourly": weather_hourly,
        "city_hourly_enriched": city_hourly_enriched,
        "city_daily_enriched": city_daily_enriched,
        "city_monthly_enriched": city_monthly_enriched,
    }


def combine_enriched_city_outputs(
    output_root: PathLike = "data/processed/enriched",
    output_format: str = "both",
    include_hourly_combined: bool = True,
) -> dict[str, pd.DataFrame]:
    # Combined enriched tables across all processed city folders
    output_root = Path(output_root)
    by_city_root = output_root / "by_city"
    combined_root = output_root / "combined"
    combined_root.mkdir(parents=True, exist_ok=True)

    daily_files = sorted(by_city_root.glob("*/*_city_daily_enriched.csv"))
    monthly_files = sorted(by_city_root.glob("*/*_city_monthly_enriched.csv"))
    weather_point_files = sorted(by_city_root.glob("*/*_weather_point.csv"))

    combined = {}

    if daily_files:
        combined_daily = pd.concat(
            [pd.read_csv(path) for path in daily_files], ignore_index=True
        )
        combined["combined_city_daily_enriched"] = combined_daily
        save_table(
            combined_daily,
            combined_root / "combined_city_daily_enriched",
            output_format,
        )

    if monthly_files:
        combined_monthly = pd.concat(
            [pd.read_csv(path) for path in monthly_files], ignore_index=True
        )
        combined["combined_city_monthly_enriched"] = combined_monthly
        save_table(
            combined_monthly,
            combined_root / "combined_city_monthly_enriched",
            output_format,
        )

    if weather_point_files:
        combined_weather_points = pd.concat(
            [pd.read_csv(path) for path in weather_point_files], ignore_index=True
        )
        combined["combined_city_weather_points"] = combined_weather_points
        save_table(
            combined_weather_points,
            combined_root / "combined_city_weather_points",
            output_format,
        )

    if include_hourly_combined:
        hourly_files = sorted(by_city_root.glob("*/*_city_hourly_enriched.csv"))

        if hourly_files:
            combined_hourly = pd.concat(
                [pd.read_csv(path) for path in hourly_files], ignore_index=True
            )
            combined["combined_city_hourly_enriched"] = combined_hourly
            save_table(
                combined_hourly,
                combined_root / "combined_city_hourly_enriched",
                output_format,
            )

    return combined


def process_selected_cities_weather(
    selected_cities_path: PathLike = "data/processed/combined/city_selection/selected_cities_pm25_main.csv",
    input_root: PathLike = "data/processed/openaq_by_city",
    output_root: PathLike = "data/processed/enriched",
    start_date: Optional[str] = "2025-01-01",
    end_date: Optional[str] = "2025-12-31",
    output_format: str = "both",
    overwrite_weather: bool = False,
    max_cities: Optional[int] = None,
    include_hourly_combined: bool = True,
) -> dict:
    # Complete weather enrichment process for all selected cities
    selected_cities = pd.read_csv(selected_cities_path)

    if "city_key" in selected_cities.columns:
        city_slugs = selected_cities["city_key"].astype(str).map(clean_slug).tolist()
    elif "city" in selected_cities.columns:
        city_slugs = selected_cities["city"].astype(str).map(clean_slug).tolist()
    else:
        raise ValueError("selected_cities_path must include city or city_key.")

    city_slugs = list(dict.fromkeys(city_slugs))

    if max_cities is not None:
        city_slugs = city_slugs[:max_cities]

    logs = []
    failed = []

    for index, city_slug in enumerate(city_slugs, start=1):
        print(f"[{index}/{len(city_slugs)}] Processing weather for {city_slug}")

        try:
            result = process_one_city_weather(
                city_slug=city_slug,
                input_root=input_root,
                output_root=output_root,
                start_date=start_date,
                end_date=end_date,
                output_format=output_format,
                overwrite_weather=overwrite_weather,
            )

            logs.append(result["log"])

        except Exception as exc:
            error_row = {
                "city_slug": city_slug,
                "status": "failed",
                "error": str(exc),
            }
            logs.append(error_row)
            failed.append(error_row)
            print(f"Failed for {city_slug}: {exc}")

    output_root = Path(output_root)
    log_df = pd.DataFrame(logs)

    log_path = output_root / "weather_enrichment_log.csv"
    output_root.mkdir(parents=True, exist_ok=True)
    log_df.to_csv(log_path, index=False)

    combined_tables = combine_enriched_city_outputs(
        output_root=output_root,
        output_format=output_format,
        include_hourly_combined=include_hourly_combined,
    )

    return {
        "log": log_df,
        "failed": failed,
        "combined_tables": combined_tables,
        "log_path": log_path,
    }


if __name__ == "__main__":
    result = process_selected_cities_weather(
        selected_cities_path="data/processed/combined/city_selection/selected_cities_pm25_main.csv",
        input_root="data/processed/openaq_by_city",
        output_root="data/processed/enriched",
        start_date="2025-01-01",
        end_date="2025-12-31",
        output_format="both",
        overwrite_weather=False,
        max_cities=None,
        include_hourly_combined=True,
    )

    print(result["log"])
