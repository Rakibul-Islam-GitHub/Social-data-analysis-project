"""
Weather-episode fingerprint visualization utilities.

This module creates an interactive small-multiple fingerprint plot:
- x-axis = daily mean wind speed
- y-axis = daily PM2.5 / PM10 ratio
- marker size = daily PM10 mean
- marker color = episode type

The plot connects particle profiles with weather context.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

WHO_PM25_DAILY = 15.0
WHO_PM10_DAILY = 45.0
MIN_VALID_HOURS = 12


CITY_DISPLAY_LOOKUP = {
    "dubai": "Dubai",
    "riyadh": "Riyadh",
    "delhi": "Delhi",
    "lahore": "Lahore",
    "dhaka": "Dhaka",
    "bangkok": "Bangkok",
    "jakarta": "Jakarta",
    "singapore": "Singapore",
    "seoul": "Seoul",
    "tokyo": "Tokyo",
    "beijing": "Beijing",
    "london": "London",
    "los_angeles": "Los Angeles",
    "new_york": "New York",
    "mexico_city": "Mexico City",
}


CITY_TO_COUNTRY = {
    "dubai": "United Arab Emirates",
    "riyadh": "Saudi Arabia",
    "delhi": "India",
    "lahore": "Pakistan",
    "dhaka": "Bangladesh",
    "bangkok": "Thailand",
    "jakarta": "Indonesia",
    "singapore": "Singapore",
    "seoul": "South Korea",
    "tokyo": "Japan",
    "beijing": "China",
    "london": "United Kingdom",
    "los_angeles": "United States",
    "new_york": "United States",
    "mexico_city": "Mexico",
}


CITY_TO_GROUP = {
    "dubai": "Gulf / desert urbanization",
    "riyadh": "Gulf / desert urbanization",
    "delhi": "High-pollution / fast-growing Asia",
    "lahore": "High-pollution / fast-growing Asia",
    "dhaka": "High-pollution / fast-growing Asia",
    "bangkok": "High-pollution / fast-growing Asia",
    "jakarta": "High-pollution / fast-growing Asia",
    "singapore": "Dense but policy-relevant Asian comparison",
    "seoul": "Dense but policy-relevant Asian comparison",
    "tokyo": "Dense but policy-relevant Asian comparison",
    "beijing": "Dense but policy-relevant Asian comparison",
    "london": "Policy / developed-city comparison",
    "los_angeles": "Policy / developed-city comparison",
    "new_york": "Policy / developed-city comparison",
    "mexico_city": "Optional contrast",
}


EPISODE_ORDER = [
    "Mixed high-pollution",
    "High-wind PM10-heavy",
    "Fine-particle high-PM2.5",
    "PM10-heavy low-ratio",
    "Lower-pollution",
]


EPISODE_COLORS = {
    "Mixed high-pollution": "#b2182b",
    "High-wind PM10-heavy": "#6b4c2a",
    "Fine-particle high-PM2.5": "#ef8a62",
    "PM10-heavy low-ratio": "#998ec3",
    "Lower-pollution": "#67a9cf",
}


EPISODE_SYMBOLS = {
    "Mixed high-pollution": "circle",
    "High-wind PM10-heavy": "diamond",
    "Fine-particle high-PM2.5": "triangle-up",
    "PM10-heavy low-ratio": "square",
    "Lower-pollution": "circle-open",
}


def clean_city_key(value):
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def fill_city_metadata(df):
    df = df.copy()
    df["city_key"] = df["city"].apply(clean_city_key)
    df["city_display"] = df["city_key"].map(CITY_DISPLAY_LOOKUP).fillna(df["city"])

    if "country" not in df.columns:
        df["country"] = pd.NA

    if "city_group" not in df.columns:
        df["city_group"] = pd.NA

    missing_country = df["country"].isna() | df["country"].astype(
        str
    ).str.lower().str.strip().isin(["", "nan", "none", "null"])

    missing_group = df["city_group"].isna() | df["city_group"].astype(
        str
    ).str.lower().str.strip().isin(["", "nan", "none", "null"])

    df.loc[missing_country, "country"] = df.loc[missing_country, "city_key"].map(
        CITY_TO_COUNTRY
    )
    df.loc[missing_group, "city_group"] = df.loc[missing_group, "city_key"].map(
        CITY_TO_GROUP
    )

    return df


def prepare_weather_episode_data(
    daily,
    selected_particle,
    min_valid_hours=MIN_VALID_HOURS,
    analysis_year=2025,
    ratio_min=0.0,
    ratio_max=2.5,
):
    daily = fill_city_metadata(daily)
    selected_particle = fill_city_metadata(selected_particle)

    selected_city_keys = selected_particle["city_key"].unique()
    daily = daily[daily["city_key"].isin(selected_city_keys)].copy()

    daily["local_date"] = pd.to_datetime(daily["local_date"], errors="coerce")
    daily = daily[daily["local_date"].dt.year == analysis_year].copy()

    numeric_columns = [
        "pm25_daily_mean",
        "pm25_daily_median",
        "pm10_daily_mean",
        "pm10_daily_median",
        "pm25_pm10_ratio_daily_mean",
        "pm25_pm10_ratio_daily_median",
        "valid_pm25_hours",
        "valid_pm10_hours",
        "wind_speed_10m_mean_ms",
        "wind_speed_10m_p90_ms",
        "wind_speed_10m_max_ms",
        "wind_direction_10m_circular_mean_deg",
        "temperature_2m_mean_c",
        "relative_humidity_2m_mean_pct",
        "cloud_cover_mean_pct",
        "precipitation_total_mm",
    ]

    for column in numeric_columns:
        if column in daily.columns:
            daily[column] = pd.to_numeric(daily[column], errors="coerce")

    if "pm25_pm10_ratio_daily_median" in daily.columns:
        daily["particle_ratio"] = daily["pm25_pm10_ratio_daily_median"]
    elif "pm25_pm10_ratio_daily_mean" in daily.columns:
        daily["particle_ratio"] = daily["pm25_pm10_ratio_daily_mean"]
    elif {"pm25_daily_median", "pm10_daily_median"}.issubset(daily.columns):
        daily["particle_ratio"] = np.where(
            (daily["pm10_daily_median"] > 0)
            & daily["pm25_daily_median"].notna()
            & daily["pm10_daily_median"].notna(),
            daily["pm25_daily_median"] / daily["pm10_daily_median"],
            np.nan,
        )
    else:
        raise ValueError(
            "Daily data needs PM2.5/PM10 ratio or PM2.5 and PM10 median columns."
        )

    daily["valid_particle_day"] = (
        (daily["valid_pm25_hours"].fillna(0) >= min_valid_hours)
        & (daily["valid_pm10_hours"].fillna(0) >= min_valid_hours)
        & daily["particle_ratio"].notna()
        & (daily["particle_ratio"] > ratio_min)
        & (daily["particle_ratio"] <= ratio_max)
        & daily["wind_speed_10m_mean_ms"].notna()
    )

    daily = daily[daily["valid_particle_day"]].copy()

    daily["city_ratio_median"] = daily.groupby("city_key")["particle_ratio"].transform(
        "median"
    )
    daily["city_wind_p75"] = daily.groupby("city_key")[
        "wind_speed_10m_mean_ms"
    ].transform(lambda x: x.quantile(0.75))

    daily["high_pm25_day"] = daily["pm25_daily_mean"] > WHO_PM25_DAILY
    daily["high_pm10_day"] = daily["pm10_daily_mean"] > WHO_PM10_DAILY
    daily["high_wind_day"] = daily["wind_speed_10m_mean_ms"] >= daily["city_wind_p75"]
    daily["low_ratio_day"] = daily["particle_ratio"] < daily["city_ratio_median"]
    daily["high_ratio_day"] = daily["particle_ratio"] >= daily["city_ratio_median"]

    daily["episode_type"] = "Lower-pollution"

    daily.loc[daily["high_pm25_day"] & daily["high_pm10_day"], "episode_type"] = (
        "Mixed high-pollution"
    )

    daily.loc[daily["high_pm10_day"] & daily["low_ratio_day"], "episode_type"] = (
        "PM10-heavy low-ratio"
    )

    daily.loc[daily["high_pm25_day"] & daily["high_ratio_day"], "episode_type"] = (
        "Fine-particle high-PM2.5"
    )

    daily.loc[
        daily["high_pm10_day"] & daily["high_wind_day"] & daily["low_ratio_day"],
        "episode_type",
    ] = "High-wind PM10-heavy"

    daily["episode_severity"] = (
        daily["pm25_daily_mean"].fillna(0) / WHO_PM25_DAILY
        + daily["pm10_daily_mean"].fillna(0) / WHO_PM10_DAILY
    )

    daily["month_label"] = daily["local_date"].dt.strftime("%b")
    daily["date_label"] = daily["local_date"].dt.strftime("%Y-%m-%d")

    summary = (
        daily.groupby(
            ["city_key", "city_display", "country", "city_group"], dropna=False
        )
        .agg(
            valid_episode_days=("local_date", "nunique"),
            median_ratio=("particle_ratio", "median"),
            median_wind=("wind_speed_10m_mean_ms", "median"),
            p75_wind=(
                "wind_speed_10m_mean_ms",
                lambda x: float(pd.to_numeric(x, errors="coerce").quantile(0.75)),
            ),
            mean_pm25=("pm25_daily_mean", "mean"),
            mean_pm10=("pm10_daily_mean", "mean"),
            mixed_days=(
                "episode_type",
                lambda x: int((x == "Mixed high-pollution").sum()),
            ),
            high_wind_pm10_days=(
                "episode_type",
                lambda x: int((x == "High-wind PM10-heavy").sum()),
            ),
            fine_particle_days=(
                "episode_type",
                lambda x: int((x == "Fine-particle high-PM2.5").sum()),
            ),
            pm10_heavy_days=(
                "episode_type",
                lambda x: int((x == "PM10-heavy low-ratio").sum()),
            ),
        )
        .reset_index()
    )

    summary["high_wind_pm10_share_pct"] = (
        summary["high_wind_pm10_days"]
        / summary["valid_episode_days"].replace(0, np.nan)
        * 100
    )

    summary["fine_particle_share_pct"] = (
        summary["fine_particle_days"]
        / summary["valid_episode_days"].replace(0, np.nan)
        * 100
    )

    summary = summary.sort_values(
        ["high_wind_pm10_share_pct", "mixed_days", "mean_pm10"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return daily, summary


def build_weather_episode_fingerprint(
    daily,
    selected_particle,
    min_valid_hours=MIN_VALID_HOURS,
    analysis_year=2025,
    ratio_min=0.0,
    ratio_max=2.5,
    columns=3,
):
    episode_daily, episode_summary = prepare_weather_episode_data(
        daily=daily,
        selected_particle=selected_particle,
        min_valid_hours=min_valid_hours,
        analysis_year=analysis_year,
        ratio_min=ratio_min,
        ratio_max=ratio_max,
    )

    city_order = episode_summary["city_display"].tolist()
    n_cities = len(city_order)
    rows = math.ceil(n_cities / columns)

    subplot_titles = []
    for city in city_order:
        row = episode_summary[episode_summary["city_display"] == city].iloc[0]
        subplot_titles.append(
            f"{city}<br><sup>median ratio {row['median_ratio']:.2f}, wind p75 {row['p75_wind']:.1f} m/s</sup>"
        )

    fig = make_subplots(
        rows=rows,
        cols=columns,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.06,
        vertical_spacing=0.12,
    )

    legend_shown = set()

    # Marker size scale
    pm10_values = pd.to_numeric(episode_daily["pm10_daily_mean"], errors="coerce")
    pm10_min = pm10_values.quantile(0.05)
    pm10_max = pm10_values.quantile(0.95)

    for index, city in enumerate(city_order):
        city_df = episode_daily[episode_daily["city_display"] == city].copy()
        city_summary = episode_summary[episode_summary["city_display"] == city].iloc[0]

        row = index // columns + 1
        col = index % columns + 1

        city_df["marker_size"] = 7 + 15 * (
            (city_df["pm10_daily_mean"].clip(pm10_min, pm10_max) - pm10_min)
            / max(pm10_max - pm10_min, 1)
        )

        city_df["hover_text"] = (
            "<b>"
            + city_df["city_display"]
            + "</b><br>"
            + "Date: "
            + city_df["date_label"]
            + "<br>"
            + "Episode: "
            + city_df["episode_type"]
            + "<br>"
            + "Wind speed: "
            + city_df["wind_speed_10m_mean_ms"].round(1).astype(str)
            + " m/s<br>"
            + "PM2.5 / PM10 ratio: "
            + city_df["particle_ratio"].round(2).astype(str)
            + "<br>"
            + "PM2.5 mean: "
            + city_df["pm25_daily_mean"].round(1).astype(str)
            + " ug/m3<br>"
            + "PM10 mean: "
            + city_df["pm10_daily_mean"].round(1).astype(str)
            + " ug/m3<br>"
            + "Temperature: "
            + city_df["temperature_2m_mean_c"].round(1).astype(str)
            + " deg C<br>"
            + "Cloud cover: "
            + city_df["cloud_cover_mean_pct"].round(0).astype(str)
            + "%"
        )

        for episode_type in EPISODE_ORDER:
            type_df = city_df[city_df["episode_type"] == episode_type].copy()

            if type_df.empty:
                continue

            fig.add_trace(
                go.Scatter(
                    x=type_df["wind_speed_10m_mean_ms"],
                    y=type_df["particle_ratio"],
                    mode="markers",
                    name=episode_type,
                    marker=dict(
                        size=type_df["marker_size"],
                        color=EPISODE_COLORS[episode_type],
                        symbol=EPISODE_SYMBOLS[episode_type],
                        opacity=0.72,
                        line=dict(width=0.6, color="white"),
                    ),
                    hovertext=type_df["hover_text"],
                    hoverinfo="text",
                    legendgroup=episode_type,
                    showlegend=episode_type not in legend_shown,
                ),
                row=row,
                col=col,
            )

            legend_shown.add(episode_type)

        fig.add_hline(
            y=city_summary["median_ratio"],
            line_dash="dot",
            line_width=1,
            line_color="rgba(50,50,50,0.45)",
            row=row,
            col=col,
        )

        fig.add_vline(
            x=city_summary["p75_wind"],
            line_dash="dot",
            line_width=1,
            line_color="rgba(50,50,50,0.45)",
            row=row,
            col=col,
        )

        fig.update_xaxes(title_text="Wind speed (m/s)", row=row, col=col)
        fig.update_yaxes(
            title_text="PM2.5 / PM10", range=[ratio_min, ratio_max], row=row, col=col
        )

    fig.update_layout(
        title=dict(
            text=(
                "Weather-Episode Fingerprints"
                "<br><sup>Each point is one city-day. X = wind speed, Y = PM2.5 / PM10 ratio, size = PM10, color = episode type.</sup>"
            ),
            x=0.02,
            xanchor="left",
        ),
        height=max(900, rows * 460),
        template="plotly_white",
        margin=dict(l=70, r=40, t=150, b=90),
        legend=dict(
            title="Episode type",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    return fig, episode_daily, episode_summary


def make_and_save_weather_episode_fingerprint(
    daily_path,
    selected_particle_path,
    figure_output_dir="outputs/figures",
    web_data_output_dir="web/data",
    min_valid_hours=MIN_VALID_HOURS,
    analysis_year=2025,
    ratio_min=0.0,
    ratio_max=2.5,
    columns=3,
):
    daily = pd.read_csv(daily_path)
    selected_particle = pd.read_csv(selected_particle_path)

    fig, episode_daily, episode_summary = build_weather_episode_fingerprint(
        daily=daily,
        selected_particle=selected_particle,
        min_valid_hours=min_valid_hours,
        analysis_year=analysis_year,
        ratio_min=ratio_min,
        ratio_max=ratio_max,
        columns=columns,
    )

    figure_output_dir = Path(figure_output_dir)
    web_data_output_dir = Path(web_data_output_dir)
    figure_output_dir.mkdir(parents=True, exist_ok=True)
    web_data_output_dir.mkdir(parents=True, exist_ok=True)

    fig.write_html(
        figure_output_dir / "weather_episode_fingerprint_2025.html",
        include_plotlyjs="cdn",
    )

    episode_daily.to_csv(
        web_data_output_dir / "weather_episode_fingerprint_daily_2025.csv",
        index=False,
    )

    episode_summary.to_csv(
        web_data_output_dir / "weather_episode_fingerprint_summary_2025.csv",
        index=False,
    )

    return fig, episode_daily, episode_summary
