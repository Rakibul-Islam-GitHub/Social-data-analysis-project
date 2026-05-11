
"""
Dubai deep-dive visualization utilities.

Creates a compact Dubai section for analysis.ipynb:
1. Hour-of-day x month heatmap for Dubai's diurnal/seasonal PM2.5 fingerprint
2. Daily calendar-style PM2.5 and weather timeline
3. Weather-particle scatter for Dubai days

Inputs:
data/processed/enriched/by_city/dubai/dubai_city_hourly_enriched.csv
data/processed/enriched/combined/combined_city_daily_enriched_2025.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


WHO_PM25_DAILY = 15.0


def clean_pm_values(df):
    df = df.copy()

    pm25_columns = ["pm25_median", "pm25_mean", "pm25_p90", "pm25_min", "pm25_max"]
    pm10_columns = ["pm10_median", "pm10_mean", "pm10_p90", "pm10_min", "pm10_max"]

    for column in pm25_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
            df.loc[(df[column] < 0) | (df[column] > 1000), column] = np.nan

    for column in pm10_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
            df.loc[(df[column] < 0) | (df[column] > 2000), column] = np.nan

    return df


def load_dubai_hourly(hourly_path):
    hourly = pd.read_csv(hourly_path)
    hourly = clean_pm_values(hourly)

    hourly["datetime_hour_utc"] = pd.to_datetime(hourly["datetime_hour_utc"], utc=True, errors="coerce")

    if "datetime_hour_local" in hourly.columns:
        hourly["datetime_hour_local"] = pd.to_datetime(hourly["datetime_hour_local"], errors="coerce")
    else:
        hourly["datetime_hour_local"] = hourly["datetime_hour_utc"].dt.tz_convert(None)

    if "local_date" in hourly.columns:
        hourly["local_date"] = pd.to_datetime(hourly["local_date"], errors="coerce")
    else:
        hourly["local_date"] = hourly["datetime_hour_local"].dt.floor("D")

    if "local_hour" not in hourly.columns:
        hourly["local_hour"] = hourly["datetime_hour_local"].dt.hour

    hourly["local_year"] = pd.to_numeric(hourly["local_year"], errors="coerce").fillna(hourly["local_date"].dt.year).astype(int)
    hourly["local_month"] = pd.to_numeric(hourly["local_month"], errors="coerce").fillna(hourly["local_date"].dt.month).astype(int)

    hourly = hourly[hourly["local_year"] == 2025].copy()

    numeric_columns = [
        "pm25_median", "pm10_median", "pm25_pm10_ratio_from_city_medians",
        "wind_speed_10m_ms", "temperature_2m_c", "relative_humidity_2m_pct",
        "cloud_cover_pct", "precipitation_mm", "valid_pm25_locations", "active_locations"
    ]

    for column in numeric_columns:
        if column in hourly.columns:
            hourly[column] = pd.to_numeric(hourly[column], errors="coerce")

    return hourly


def load_dubai_daily(daily_path):
    daily = pd.read_csv(daily_path)
    daily["city_key"] = daily["city"].astype(str).str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_")
    daily = daily[daily["city_key"] == "dubai"].copy()

    daily["local_date"] = pd.to_datetime(daily["local_date"], errors="coerce")
    daily = daily[daily["local_date"].dt.year == 2025].copy()

    numeric_columns = [
        "pm25_daily_mean", "pm25_daily_median", "pm25_daily_p90", "pm25_daily_max",
        "pm10_daily_mean", "pm10_daily_median", "pm25_pm10_ratio_daily_median",
        "wind_speed_10m_mean_ms", "temperature_2m_mean_c",
        "relative_humidity_2m_mean_pct", "cloud_cover_mean_pct",
        "precipitation_total_mm", "valid_pm25_hours", "valid_pm10_hours"
    ]

    for column in numeric_columns:
        if column in daily.columns:
            daily[column] = pd.to_numeric(daily[column], errors="coerce")

    daily.loc[(daily["pm25_daily_mean"] < 0) | (daily["pm25_daily_mean"] > 1000), "pm25_daily_mean"] = np.nan

    daily["pm25_exceeds_who"] = (
        daily["pm25_daily_mean"].notna()
        & (daily["valid_pm25_hours"].fillna(0) >= 12)
        & (daily["pm25_daily_mean"] > WHO_PM25_DAILY)
    )

    daily["month_label"] = daily["local_date"].dt.strftime("%b")
    daily["date_label"] = daily["local_date"].dt.strftime("%Y-%m-%d")
    daily["day_of_year"] = daily["local_date"].dt.dayofyear

    return daily


def build_dubai_hour_month_heatmap(hourly):
    heatmap_data = (
        hourly
        .groupby(["local_hour", "local_month"], dropna=False)
        .agg(
            pm25_median=("pm25_median", "median"),
            pm10_median=("pm10_median", "median"),
            wind_speed_median=("wind_speed_10m_ms", "median"),
            valid_hours=("pm25_median", lambda x: int(pd.to_numeric(x, errors="coerce").notna().sum())),
        )
        .reset_index()
    )

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    pivot = (
        heatmap_data
        .pivot(index="local_hour", columns="local_month", values="pm25_median")
        .reindex(index=list(range(24)), columns=list(range(1, 13)))
    )

    wind_pivot = (
        heatmap_data
        .pivot(index="local_hour", columns="local_month", values="wind_speed_median")
        .reindex(index=list(range(24)), columns=list(range(1, 13)))
    )

    valid_pivot = (
        heatmap_data
        .pivot(index="local_hour", columns="local_month", values="valid_hours")
        .reindex(index=list(range(24)), columns=list(range(1, 13)))
    )

    hover_text = []

    for hour in pivot.index:
        row_text = []

        for month in pivot.columns:
            pm_value = pivot.loc[hour, month]
            wind_value = wind_pivot.loc[hour, month]
            valid_hours = valid_pivot.loc[hour, month]

            if pd.isna(pm_value):
                row_text.append(f"Hour {hour}:00<br>{month_labels[month-1]}<br>No PM2.5 data")
            else:
                row_text.append(
                    f"<b>Dubai</b><br>"
                    f"{month_labels[month-1]}, hour {hour}:00<br>"
                    f"Median PM2.5: {pm_value:.1f} ug/m3<br>"
                    f"Median wind: {wind_value:.1f} m/s<br>"
                    f"Valid hours: {valid_hours:.0f}"
                )

        hover_text.append(row_text)

    zmax = max(60, np.nanpercentile(pivot.values, 95))

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=month_labels,
            y=pivot.index,
            colorscale="Inferno_r",
            zmin=0,
            zmax=zmax,
            colorbar=dict(title="Median<br>PM2.5<br>ug/m3"),
            text=hover_text,
            hoverinfo="text",
        )
    )

    fig.update_layout(
        title=dict(
            text=(
                "Dubai's Diurnal Pollution Fingerprint, 2025"
                "<br><sup>Median PM2.5 by local hour and month. The chart shows when Dubai's high-PM2.5 periods occur within the day.</sup>"
            ),
            x=0.02,
            xanchor="left",
        ),
        xaxis_title="Month",
        yaxis_title="Local hour of day",
        height=650,
        template="plotly_white",
        margin=dict(l=80, r=40, t=95, b=70),
    )

    fig.update_yaxes(autorange="reversed", dtick=2)

    return fig, heatmap_data


def build_dubai_daily_weather_timeline(daily):
    daily = daily.sort_values("local_date").copy()

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        row_heights=[0.48, 0.26, 0.26],
        subplot_titles=[
            "Daily PM2.5 burden",
            "Wind speed and particle ratio",
            "Temperature and cloud cover",
        ],
        specs=[[{"secondary_y": False}], [{"secondary_y": True}], [{"secondary_y": True}]],
    )

    colors = np.where(daily["pm25_exceeds_who"], "#d73027", "#2c7fb8")

    fig.add_trace(
        go.Bar(
            x=daily["local_date"],
            y=daily["pm25_daily_mean"],
            marker=dict(color=colors),
            name="Daily mean PM2.5",
            hovertext=(
                "<b>Dubai</b><br>"
                + "Date: " + daily["date_label"] + "<br>"
                + "PM2.5 mean: " + daily["pm25_daily_mean"].round(1).astype(str) + " ug/m3<br>"
                + "PM2.5 median: " + daily["pm25_daily_median"].round(1).astype(str) + " ug/m3<br>"
                + "Valid hours: " + daily["valid_pm25_hours"].fillna(0).astype(int).astype(str)
            ),
            hoverinfo="text",
        ),
        row=1,
        col=1,
    )

    fig.add_hline(
        y=WHO_PM25_DAILY,
        line_dash="dash",
        line_color="rgba(150,0,0,0.65)",
        annotation_text="WHO daily PM2.5 guideline",
        annotation_position="top left",
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=daily["local_date"],
            y=daily["wind_speed_10m_mean_ms"],
            mode="lines",
            name="Wind speed",
            line=dict(color="#2166ac", width=2),
        ),
        row=2,
        col=1,
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=daily["local_date"],
            y=daily["pm25_pm10_ratio_daily_median"],
            mode="lines",
            name="PM2.5/PM10 ratio",
            line=dict(color="#b2182b", width=2),
        ),
        row=2,
        col=1,
        secondary_y=True,
    )

    fig.add_trace(
        go.Scatter(
            x=daily["local_date"],
            y=daily["temperature_2m_mean_c"],
            mode="lines",
            name="Temperature",
            line=dict(color="#f46d43", width=2),
        ),
        row=3,
        col=1,
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=daily["local_date"],
            y=daily["cloud_cover_mean_pct"],
            mode="lines",
            name="Cloud cover",
            line=dict(color="#636363", width=2),
        ),
        row=3,
        col=1,
        secondary_y=True,
    )

    fig.update_yaxes(title_text="PM2.5 ug/m3", row=1, col=1)
    fig.update_yaxes(title_text="Wind m/s", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="PM2.5/PM10", row=2, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Temp C", row=3, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Cloud %", row=3, col=1, secondary_y=True)

    fig.update_xaxes(tickformat="%b", dtick="M1", row=3, col=1)

    fig.update_layout(
        title=dict(
            text=(
                "Dubai Daily Pollution and Weather Timeline, 2025"
                "<br><sup>Daily PM2.5 exceedance is shown with red bars; weather context helps explain when conditions shift.</sup>"
            ),
            x=0.02,
            xanchor="left",
        ),
        height=850,
        template="plotly_white",
        margin=dict(l=80, r=70, t=110, b=70),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1,
        ),
    )

    return fig


def build_dubai_weather_particle_scatter(daily):
    plot_df = daily.copy()

    plot_df = plot_df[
        plot_df["pm25_pm10_ratio_daily_median"].notna()
        & plot_df["wind_speed_10m_mean_ms"].notna()
        & plot_df["pm10_daily_mean"].notna()
    ].copy()

    plot_df["episode_type"] = "Lower / moderate day"
    plot_df.loc[plot_df["pm25_daily_mean"] > WHO_PM25_DAILY, "episode_type"] = "High PM2.5 day"
    plot_df.loc[
        (plot_df["pm10_daily_mean"] > 45)
        & (plot_df["pm25_pm10_ratio_daily_median"] < plot_df["pm25_pm10_ratio_daily_median"].median()),
        "episode_type"
    ] = "PM10-heavy day"

    color_lookup = {
        "High PM2.5 day": "#d73027",
        "PM10-heavy day": "#6b4c2a",
        "Lower / moderate day": "#2c7fb8",
    }

    symbol_lookup = {
        "High PM2.5 day": "circle",
        "PM10-heavy day": "diamond",
        "Lower / moderate day": "circle-open",
    }

    fig = go.Figure()

    pm10_min = plot_df["pm10_daily_mean"].quantile(0.05)
    pm10_max = plot_df["pm10_daily_mean"].quantile(0.95)

    plot_df["marker_size"] = 8 + 20 * (
        (plot_df["pm10_daily_mean"].clip(pm10_min, pm10_max) - pm10_min)
        / max(pm10_max - pm10_min, 1)
    )

    for episode_type, type_df in plot_df.groupby("episode_type"):
        fig.add_trace(
            go.Scatter(
                x=type_df["wind_speed_10m_mean_ms"],
                y=type_df["pm25_pm10_ratio_daily_median"],
                mode="markers",
                name=episode_type,
                marker=dict(
                    size=type_df["marker_size"],
                    color=color_lookup[episode_type],
                    symbol=symbol_lookup[episode_type],
                    opacity=0.75,
                    line=dict(width=0.8, color="white"),
                ),
                hovertext=(
                    "<b>Dubai</b><br>"
                    + "Date: " + type_df["date_label"] + "<br>"
                    + "Episode: " + type_df["episode_type"] + "<br>"
                    + "Wind speed: " + type_df["wind_speed_10m_mean_ms"].round(1).astype(str) + " m/s<br>"
                    + "PM2.5/PM10 ratio: " + type_df["pm25_pm10_ratio_daily_median"].round(2).astype(str) + "<br>"
                    + "PM2.5 mean: " + type_df["pm25_daily_mean"].round(1).astype(str) + " ug/m3<br>"
                    + "PM10 mean: " + type_df["pm10_daily_mean"].round(1).astype(str) + " ug/m3"
                ),
                hoverinfo="text",
            )
        )

    fig.add_hline(
        y=plot_df["pm25_pm10_ratio_daily_median"].median(),
        line_dash="dot",
        line_color="rgba(50,50,50,0.5)",
        annotation_text="Dubai median ratio",
        annotation_position="top left",
    )

    fig.add_vline(
        x=plot_df["wind_speed_10m_mean_ms"].quantile(0.75),
        line_dash="dot",
        line_color="rgba(50,50,50,0.5)",
        annotation_text="wind p75",
        annotation_position="top right",
    )

    fig.update_layout(
        title=dict(
            text=(
                "Dubai Weather-Particle Fingerprint, 2025"
                "<br><sup>Each point is one day. X = wind speed, Y = PM2.5 / PM10 ratio, size = PM10.</sup>"
            ),
            x=0.02,
            xanchor="left",
        ),
        xaxis_title="Daily mean wind speed (m/s)",
        yaxis_title="Daily PM2.5 / PM10 ratio",
        height=650,
        template="plotly_white",
        margin=dict(l=80, r=40, t=100, b=70),
        legend=dict(
            title="Dubai day type",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    return fig, plot_df


def make_and_save_dubai_deep_dive(
    dubai_hourly_path,
    combined_daily_path,
    figure_output_dir="outputs/figures",
    web_data_output_dir="web/data",
):
    hourly = load_dubai_hourly(dubai_hourly_path)
    daily = load_dubai_daily(combined_daily_path)

    figure_output_dir = Path(figure_output_dir)
    web_data_output_dir = Path(web_data_output_dir)
    figure_output_dir.mkdir(parents=True, exist_ok=True)
    web_data_output_dir.mkdir(parents=True, exist_ok=True)

    fig_hour_month, dubai_hour_month = build_dubai_hour_month_heatmap(hourly)
    fig_timeline = build_dubai_daily_weather_timeline(daily)
    fig_scatter, dubai_scatter_data = build_dubai_weather_particle_scatter(daily)

    fig_hour_month.write_html(
        figure_output_dir / "dubai_hour_month_pm25_heatmap_2025.html",
        include_plotlyjs="cdn",
    )

    fig_timeline.write_html(
        figure_output_dir / "dubai_daily_pm25_weather_timeline_2025.html",
        include_plotlyjs="cdn",
    )

    fig_scatter.write_html(
        figure_output_dir / "dubai_weather_particle_fingerprint_2025.html",
        include_plotlyjs="cdn",
    )

    hourly.to_csv(web_data_output_dir / "dubai_hourly_enriched_2025.csv", index=False)
    daily.to_csv(web_data_output_dir / "dubai_daily_enriched_2025.csv", index=False)
    dubai_hour_month.to_csv(web_data_output_dir / "dubai_hour_month_pm25_2025.csv", index=False)
    dubai_scatter_data.to_csv(web_data_output_dir / "dubai_weather_particle_fingerprint_2025.csv", index=False)

    summary = {
        "valid_days": int(daily["pm25_daily_mean"].notna().sum()),
        "who_exceedance_days": int(daily["pm25_exceeds_who"].sum()),
        "who_exceedance_pct": float(daily["pm25_exceeds_who"].sum() / max(daily["pm25_daily_mean"].notna().sum(), 1) * 100),
        "median_pm25": float(daily["pm25_daily_median"].median()),
        "mean_pm25": float(daily["pm25_daily_mean"].mean()),
        "p90_pm25": float(daily["pm25_daily_mean"].quantile(0.90)),
        "median_wind": float(daily["wind_speed_10m_mean_ms"].median()),
        "median_particle_ratio": float(daily["pm25_pm10_ratio_daily_median"].median()) if "pm25_pm10_ratio_daily_median" in daily.columns else np.nan,
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(web_data_output_dir / "dubai_deep_dive_summary_2025.csv", index=False)

    return {
        "fig_hour_month": fig_hour_month,
        "fig_timeline": fig_timeline,
        "fig_scatter": fig_scatter,
        "hourly": hourly,
        "daily": daily,
        "summary": summary_df,
    }
