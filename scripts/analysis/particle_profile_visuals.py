
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.colors as pc

WHO_PM25_DAILY = 15.0
WHO_PM10_DAILY = 45.0
MIN_VALID_HOURS = 12

CITY_DISPLAY_LOOKUP = {
    "dubai": "Dubai", "riyadh": "Riyadh", "delhi": "Delhi", "lahore": "Lahore",
    "dhaka": "Dhaka", "bangkok": "Bangkok", "jakarta": "Jakarta",
    "singapore": "Singapore", "seoul": "Seoul", "tokyo": "Tokyo",
    "beijing": "Beijing", "london": "London", "los_angeles": "Los Angeles",
    "new_york": "New York", "mexico_city": "Mexico City",
}

CITY_TO_COUNTRY = {
    "dubai": "United Arab Emirates", "riyadh": "Saudi Arabia", "delhi": "India",
    "lahore": "Pakistan", "dhaka": "Bangladesh", "bangkok": "Thailand",
    "jakarta": "Indonesia", "singapore": "Singapore", "seoul": "South Korea",
    "tokyo": "Japan", "beijing": "China", "london": "United Kingdom",
    "los_angeles": "United States", "new_york": "United States",
    "mexico_city": "Mexico",
}

CITY_TO_GROUP = {
    "dubai": "Gulf / desert urbanization", "riyadh": "Gulf / desert urbanization",
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

    missing_country = (
        df["country"].isna()
        | df["country"].astype(str).str.lower().str.strip().isin(["", "nan", "none", "null"])
    )
    missing_group = (
        df["city_group"].isna()
        | df["city_group"].astype(str).str.lower().str.strip().isin(["", "nan", "none", "null"])
    )

    df.loc[missing_country, "country"] = df.loc[missing_country, "city_key"].map(CITY_TO_COUNTRY)
    df.loc[missing_group, "city_group"] = df.loc[missing_group, "city_key"].map(CITY_TO_GROUP)
    return df


def prepare_particle_profile_data(
    daily,
    selected_particle,
    min_valid_hours=MIN_VALID_HOURS,
    ratio_min=0.0,
    ratio_max=2.5,
    analysis_year=2025,
):
    daily = fill_city_metadata(daily)
    selected_particle = fill_city_metadata(selected_particle)

    selected_city_keys = selected_particle["city_key"].unique()
    daily = daily[daily["city_key"].isin(selected_city_keys)].copy()

    daily["local_date"] = pd.to_datetime(daily["local_date"], errors="coerce")
    daily = daily[daily["local_date"].dt.year == analysis_year].copy()

    numeric_columns = [
        "pm25_daily_mean", "pm25_daily_median", "pm10_daily_mean", "pm10_daily_median",
        "pm25_pm10_ratio_daily_mean", "pm25_pm10_ratio_daily_median",
        "valid_pm25_hours", "valid_pm10_hours", "wind_speed_10m_mean_ms",
        "wind_speed_10m_p90_ms", "temperature_2m_mean_c",
        "relative_humidity_2m_mean_pct", "precipitation_total_mm", "cloud_cover_mean_pct",
    ]

    for column in numeric_columns:
        if column in daily.columns:
            daily[column] = pd.to_numeric(daily[column], errors="coerce")

    if "pm25_pm10_ratio_daily_median" in daily.columns:
        ratio_column = "pm25_pm10_ratio_daily_median"
    elif "pm25_pm10_ratio_daily_mean" in daily.columns:
        ratio_column = "pm25_pm10_ratio_daily_mean"
    elif {"pm25_daily_median", "pm10_daily_median"}.issubset(daily.columns):
        daily["pm25_pm10_ratio_daily_median"] = np.where(
            (daily["pm10_daily_median"] > 0)
            & daily["pm25_daily_median"].notna()
            & daily["pm10_daily_median"].notna(),
            daily["pm25_daily_median"] / daily["pm10_daily_median"],
            np.nan,
        )
        ratio_column = "pm25_pm10_ratio_daily_median"
    else:
        raise ValueError("Daily data needs PM2.5/PM10 ratio or PM2.5 and PM10 median columns.")

    daily["particle_ratio"] = pd.to_numeric(daily[ratio_column], errors="coerce")

    daily["particle_profile_coverage_ok"] = (
        (daily["valid_pm25_hours"].fillna(0) >= min_valid_hours)
        & (daily["valid_pm10_hours"].fillna(0) >= min_valid_hours)
        & daily["particle_ratio"].notna()
    )

    daily = daily[
        daily["particle_profile_coverage_ok"]
        & (daily["particle_ratio"] > ratio_min)
        & (daily["particle_ratio"] <= ratio_max)
    ].copy()

    daily["high_pm25_day"] = daily["pm25_daily_mean"] > WHO_PM25_DAILY
    daily["high_pm10_day"] = daily["pm10_daily_mean"] > WHO_PM10_DAILY

    if "wind_speed_10m_mean_ms" in daily.columns:
        wind_thresholds = (
            daily.groupby("city_key")["wind_speed_10m_mean_ms"]
            .quantile(0.75)
            .reset_index(name="city_wind_p75")
        )
        daily = daily.merge(wind_thresholds, on="city_key", how="left")
        daily["high_wind_day"] = daily["wind_speed_10m_mean_ms"] >= daily["city_wind_p75"]
    else:
        daily["city_wind_p75"] = np.nan
        daily["high_wind_day"] = False

    daily["city_ratio_median"] = daily.groupby("city_key")["particle_ratio"].transform("median")
    daily["weather_particle_flag"] = "Other valid day"

    daily.loc[
        daily["high_pm25_day"] & daily["high_pm10_day"],
        "weather_particle_flag"
    ] = "Mixed high-pollution day"

    daily.loc[
        daily["high_pm10_day"] & daily["high_wind_day"] & (daily["particle_ratio"] < daily["city_ratio_median"]),
        "weather_particle_flag"
    ] = "High-wind, PM10-heavy day"

    daily.loc[
        daily["high_pm25_day"] & (daily["particle_ratio"] >= daily["city_ratio_median"]),
        "weather_particle_flag"
    ] = "Fine-particle dominated high-PM2.5 day"

    summary = (
        daily.groupby(["city_key", "city_display", "country", "city_group"], dropna=False)
        .agg(
            valid_particle_days=("local_date", "nunique"),
            median_ratio=("particle_ratio", "median"),
            p25_ratio=("particle_ratio", lambda x: float(pd.to_numeric(x, errors="coerce").quantile(0.25))),
            p75_ratio=("particle_ratio", lambda x: float(pd.to_numeric(x, errors="coerce").quantile(0.75))),
            mean_pm25=("pm25_daily_mean", "mean"),
            mean_pm10=("pm10_daily_mean", "mean"),
            median_pm25=("pm25_daily_median", "median"),
            median_pm10=("pm10_daily_median", "median"),
            high_pm25_days=("high_pm25_day", "sum"),
            high_pm10_days=("high_pm10_day", "sum"),
            high_wind_days=("high_wind_day", "sum"),
            mean_wind_speed=("wind_speed_10m_mean_ms", "mean"),
            median_temperature=("temperature_2m_mean_c", "median"),
        )
        .reset_index()
    )

    summary["high_pm25_share_pct"] = (
        summary["high_pm25_days"] / summary["valid_particle_days"].replace(0, np.nan) * 100
    )
    summary["high_pm10_share_pct"] = (
        summary["high_pm10_days"] / summary["valid_particle_days"].replace(0, np.nan) * 100
    )
    summary = summary.sort_values("median_ratio", ascending=False).reset_index(drop=True)
    return daily, summary


def build_particle_profile_ridgeline(
    daily,
    selected_particle,
    min_valid_hours=MIN_VALID_HOURS,
    ratio_min=0.0,
    ratio_max=2.5,
    analysis_year=2025,
):
    particle_daily, particle_summary = prepare_particle_profile_data(
        daily=daily,
        selected_particle=selected_particle,
        min_valid_hours=min_valid_hours,
        ratio_min=ratio_min,
        ratio_max=ratio_max,
        analysis_year=analysis_year,
    )

    city_order = particle_summary["city_display"].tolist()
    palette = pc.qualitative.Set2 + pc.qualitative.Set3 + pc.qualitative.Pastel
    color_lookup = {city: palette[index % len(palette)] for index, city in enumerate(city_order)}

    fig = go.Figure()
    legend_shown = set()

    for city in city_order:
        city_df = particle_daily[particle_daily["city_display"] == city].copy()
        if city_df.empty:
            continue

        summary_row = particle_summary[particle_summary["city_display"] == city].iloc[0]
        city_color = color_lookup[city]

        city_df["hover_text"] = (
            "<b>" + city_df["city_display"] + "</b><br>"
            + "Date: " + city_df["local_date"].dt.strftime("%Y-%m-%d") + "<br>"
            + "PM2.5 / PM10 ratio: " + city_df["particle_ratio"].round(2).astype(str) + "<br>"
            + "PM2.5 mean: " + city_df["pm25_daily_mean"].round(1).astype(str) + " ug/m3<br>"
            + "PM10 mean: " + city_df["pm10_daily_mean"].round(1).astype(str) + " ug/m3<br>"
            + "Wind speed: " + city_df["wind_speed_10m_mean_ms"].round(1).astype(str) + " m/s<br>"
            + "Flag: " + city_df["weather_particle_flag"]
        )

        fig.add_trace(
            go.Violin(
                x=city_df["particle_ratio"],
                y=[city] * len(city_df),
                name=city,
                orientation="h",
                side="positive",
                width=1.85,
                spanmode="hard",
                points=False,
                line=dict(color=city_color, width=1.2),
                fillcolor=city_color,
                opacity=0.68,
                meanline=dict(visible=True, color="rgba(30,30,30,0.7)"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        for flag, marker_symbol, marker_size, marker_color in [
            ("High-wind, PM10-heavy day", "diamond", 8, "rgba(20,20,20,0.75)"),
            ("Mixed high-pollution day", "circle", 6, city_color),
            ("Fine-particle dominated high-PM2.5 day", "triangle-up", 7, city_color),
        ]:
            flag_df = city_df[city_df["weather_particle_flag"] == flag].copy()
            if flag_df.empty:
                continue

            fig.add_trace(
                go.Scatter(
                    x=flag_df["particle_ratio"],
                    y=[city] * len(flag_df),
                    mode="markers",
                    name=flag,
                    marker=dict(
                        symbol=marker_symbol,
                        size=marker_size,
                        color=marker_color,
                        line=dict(width=0.8, color="white"),
                        opacity=0.85,
                    ),
                    hovertext=flag_df["hover_text"],
                    hoverinfo="text",
                    legendgroup=flag,
                    showlegend=flag not in legend_shown,
                )
            )
            legend_shown.add(flag)

        fig.add_trace(
            go.Scatter(
                x=[summary_row["median_ratio"]],
                y=[city],
                mode="markers",
                marker=dict(symbol="line-ns-open", size=16, color="black", line=dict(width=2)),
                name="City median ratio",
                hovertext=(
                    f"<b>{city}</b><br>"
                    f"Median ratio: {summary_row['median_ratio']:.2f}<br>"
                    f"Valid particle-profile days: {int(summary_row['valid_particle_days'])}<br>"
                    f"High PM2.5 days: {int(summary_row['high_pm25_days'])}<br>"
                    f"High PM10 days: {int(summary_row['high_pm10_days'])}"
                ),
                hoverinfo="text",
                showlegend=False,
            )
        )

    fig.add_vline(
        x=0.5,
        line_width=1.2,
        line_dash="dash",
        line_color="rgba(80,80,80,0.6)",
        annotation_text="PM2.5 is half of PM10",
        annotation_position="top left",
    )

    fig.add_vline(
        x=1.0,
        line_width=1.2,
        line_dash="dot",
        line_color="rgba(80,80,80,0.6)",
        annotation_text="PM2.5 approx PM10",
        annotation_position="top right",
    )

    fig.update_layout(
        title=dict(
            text=(
                f"Particle Profiles Across Cities, {analysis_year}"
                "<br><sup>Daily PM2.5 / PM10 ratio distributions. Markers highlight weather-linked high-pollution days.</sup>"
            ),
            x=0.02,
            xanchor="left",
        ),
        xaxis=dict(title="Daily PM2.5 / PM10 ratio", range=[ratio_min, ratio_max], zeroline=False),
        yaxis=dict(title="City", categoryorder="array", categoryarray=city_order[::-1]),
        height=max(640, 72 * len(city_order)),
        template="plotly_white",
        margin=dict(l=150, r=50, t=110, b=80),
        legend=dict(
            title="Highlighted days",
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1,
        ),
    )

    return fig, particle_daily, particle_summary


def make_and_save_particle_profile_ridgeline(
    daily_path,
    selected_particle_path,
    figure_output_dir="outputs/figures",
    web_data_output_dir="web/data",
    min_valid_hours=MIN_VALID_HOURS,
    ratio_min=0.0,
    ratio_max=2.5,
    analysis_year=2025,
):
    daily = pd.read_csv(daily_path)
    selected_particle = pd.read_csv(selected_particle_path)

    fig, particle_daily, particle_summary = build_particle_profile_ridgeline(
        daily=daily,
        selected_particle=selected_particle,
        min_valid_hours=min_valid_hours,
        ratio_min=ratio_min,
        ratio_max=ratio_max,
        analysis_year=analysis_year,
    )

    figure_output_dir = Path(figure_output_dir)
    web_data_output_dir = Path(web_data_output_dir)
    figure_output_dir.mkdir(parents=True, exist_ok=True)
    web_data_output_dir.mkdir(parents=True, exist_ok=True)

    fig.write_html(
        figure_output_dir / "particle_profile_pm25_pm10_ridgeline_2025.html",
        include_plotlyjs="cdn",
    )

    particle_daily.to_csv(
        web_data_output_dir / "particle_profile_daily_2025.csv",
        index=False,
    )

    particle_summary.to_csv(
        web_data_output_dir / "particle_profile_summary_2025.csv",
        index=False,
    )

    return fig, particle_daily, particle_summary
