
"""
Exceedance visualization utilities.

Creates the WHO PM2.5 daily exceedance dot-matrix chart for analysis.ipynb.

Chart design:
- each square = one city-day
- red = daily mean PM2.5 above WHO 24-hour guideline
- blue = below guideline
- gray = low coverage / missing
- side bar = share of valid days above WHO guideline
"""

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


PathLike = Union[str, Path]

WHO_PM25_DAILY = 15.0
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


def clean_city_key(value) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def fill_city_metadata(df: pd.DataFrame) -> pd.DataFrame:
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


def longest_true_streak(values) -> int:
    longest = 0
    current = 0

    for value in values:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    return longest


def prepare_pm25_exceedance_data(
    daily: pd.DataFrame,
    selected_main: pd.DataFrame,
    who_threshold: float = WHO_PM25_DAILY,
    min_valid_hours: int = MIN_VALID_HOURS,
    analysis_year: int = 2025,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = fill_city_metadata(daily)
    selected_main = fill_city_metadata(selected_main)

    selected_city_keys = selected_main["city_key"].unique()
    daily = daily[daily["city_key"].isin(selected_city_keys)].copy()

    daily["local_date"] = pd.to_datetime(daily["local_date"], errors="coerce")
    daily = daily[daily["local_date"].dt.year == analysis_year].copy()

    numeric_columns = [
        "pm25_daily_mean",
        "pm25_daily_median",
        "pm25_daily_p90",
        "pm25_daily_max",
        "valid_pm25_hours",
        "pm25_daily_coverage_pct",
        "temperature_2m_mean_c",
        "relative_humidity_2m_mean_pct",
        "wind_speed_10m_mean_ms",
        "cloud_cover_mean_pct",
    ]

    for column in numeric_columns:
        if column in daily.columns:
            daily[column] = pd.to_numeric(daily[column], errors="coerce")

    if "pm25_daily_mean" not in daily.columns:
        raise ValueError("daily data must include pm25_daily_mean.")

    if "valid_pm25_hours" not in daily.columns:
        raise ValueError("daily data must include valid_pm25_hours.")

    daily.loc[daily["pm25_daily_mean"] < 0, "pm25_daily_mean"] = np.nan

    daily["coverage_ok"] = daily["valid_pm25_hours"].fillna(0) >= min_valid_hours

    daily["pm25_exceeds_who"] = (
        daily["coverage_ok"]
        & daily["pm25_daily_mean"].notna()
        & (daily["pm25_daily_mean"] > who_threshold)
    )

    daily["exceedance_status"] = "Below WHO daily guideline"

    daily.loc[daily["pm25_exceeds_who"], "exceedance_status"] = "Above WHO daily guideline"

    daily.loc[
        ~daily["coverage_ok"] | daily["pm25_daily_mean"].isna(),
        "exceedance_status"
    ] = "Low coverage / missing"

    daily["pm25_above_guideline"] = daily["pm25_daily_mean"] - who_threshold
    daily["day_of_year"] = daily["local_date"].dt.dayofyear
    daily["month_label"] = daily["local_date"].dt.strftime("%b")
    daily["date_label"] = daily["local_date"].dt.strftime("%Y-%m-%d")

    valid_daily = daily[daily["coverage_ok"] & daily["pm25_daily_mean"].notna()].copy()

    exceedance_summary = (
        valid_daily
        .groupby(["city_key", "city_display", "country", "city_group"], dropna=False)
        .agg(
            valid_days=("local_date", "nunique"),
            exceedance_days=("pm25_exceeds_who", "sum"),
            median_pm25=("pm25_daily_median", "median"),
            mean_pm25=("pm25_daily_mean", "mean"),
            p90_pm25=("pm25_daily_mean", lambda x: float(pd.to_numeric(x, errors="coerce").quantile(0.90))),
            max_pm25=("pm25_daily_mean", "max"),
            mean_coverage=("pm25_daily_coverage_pct", "mean"),
        )
        .reset_index()
    )

    exceedance_summary["exceedance_pct"] = (
        exceedance_summary["exceedance_days"]
        / exceedance_summary["valid_days"].replace(0, np.nan)
        * 100
    )

    streak_rows = []

    for city_key, city_df in daily.sort_values("local_date").groupby("city_key"):
        city_df = city_df.sort_values("local_date").copy()

        city_df["valid_exceedance_bool"] = (
            city_df["coverage_ok"]
            & city_df["pm25_exceeds_who"]
        )

        streak_rows.append(
            {
                "city_key": city_key,
                "longest_exceedance_streak_days": longest_true_streak(
                    city_df["valid_exceedance_bool"].tolist()
                ),
            }
        )

    streak_df = pd.DataFrame(streak_rows)

    exceedance_summary = exceedance_summary.merge(
        streak_df,
        on="city_key",
        how="left",
    )

    exceedance_summary = exceedance_summary.sort_values(
        ["exceedance_pct", "exceedance_days", "median_pm25"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return daily, exceedance_summary


def build_pm25_exceedance_dot_matrix(
    daily: pd.DataFrame,
    selected_main: pd.DataFrame,
    who_threshold: float = WHO_PM25_DAILY,
    min_valid_hours: int = MIN_VALID_HOURS,
    analysis_year: int = 2025,
) -> tuple[go.Figure, pd.DataFrame, pd.DataFrame]:
    daily, exceedance_summary = prepare_pm25_exceedance_data(
        daily=daily,
        selected_main=selected_main,
        who_threshold=who_threshold,
        min_valid_hours=min_valid_hours,
        analysis_year=analysis_year,
    )

    city_order = exceedance_summary["city_display"].tolist()
    plotly_y_order = city_order[::-1]

    status_order = [
        "Above WHO daily guideline",
        "Below WHO daily guideline",
        "Low coverage / missing",
    ]

    status_colors = {
        "Above WHO daily guideline": "#d73027",
        "Below WHO daily guideline": "#2c7fb8",
        "Low coverage / missing": "#d9d9d9",
    }

    daily["hover_text"] = (
        "<b>" + daily["city_display"] + "</b><br>"
        + "Date: " + daily["date_label"] + "<br>"
        + "Status: " + daily["exceedance_status"] + "<br>"
        + "Daily mean PM2.5: " + daily["pm25_daily_mean"].round(1).astype(str) + " ug/m3<br>"
        + "Daily median PM2.5: " + daily["pm25_daily_median"].round(1).astype(str) + " ug/m3<br>"
        + "Valid PM2.5 hours: " + daily["valid_pm25_hours"].fillna(0).astype(int).astype(str) + "<br>"
        + "Coverage: " + daily["pm25_daily_coverage_pct"].round(0).astype(str) + "%<br>"
    )

    if "wind_speed_10m_mean_ms" in daily.columns:
        daily["hover_text"] += "Wind speed: " + daily["wind_speed_10m_mean_ms"].round(1).astype(str) + " m/s<br>"

    if "temperature_2m_mean_c" in daily.columns:
        daily["hover_text"] += "Temperature: " + daily["temperature_2m_mean_c"].round(1).astype(str) + " deg C"

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.78, 0.22],
        shared_yaxes=True,
        horizontal_spacing=0.03,
        specs=[[{"type": "scatter"}, {"type": "bar"}]],
        subplot_titles=[
            "Daily PM2.5 exceedance matrix",
            "Share of valid days above WHO",
        ],
    )

    for status in status_order:
        status_df = daily[daily["exceedance_status"] == status].copy()

        fig.add_trace(
            go.Scatter(
                x=status_df["local_date"],
                y=status_df["city_display"],
                mode="markers",
                name=status,
                marker=dict(
                    symbol="square",
                    size=6.2,
                    color=status_colors[status],
                    line=dict(width=0),
                ),
                hovertext=status_df["hover_text"],
                hoverinfo="text",
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Bar(
            x=exceedance_summary["exceedance_pct"],
            y=exceedance_summary["city_display"],
            orientation="h",
            name="Exceedance burden",
            marker=dict(
                color=exceedance_summary["exceedance_pct"],
                colorscale="Reds",
                cmin=0,
                cmax=100,
                line=dict(width=0),
            ),
            text=exceedance_summary["exceedance_pct"].round(0).astype(int).astype(str) + "%",
            textposition="outside",
            hovertext=(
                "<b>" + exceedance_summary["city_display"] + "</b><br>"
                + "Valid days: " + exceedance_summary["valid_days"].astype(str) + "<br>"
                + "Exceedance days: " + exceedance_summary["exceedance_days"].astype(int).astype(str) + "<br>"
                + "Exceedance share: " + exceedance_summary["exceedance_pct"].round(1).astype(str) + "%<br>"
                + "Longest streak: " + exceedance_summary["longest_exceedance_streak_days"].astype(int).astype(str) + " days<br>"
                + "Median PM2.5: " + exceedance_summary["median_pm25"].round(1).astype(str) + " ug/m3"
            ),
            hoverinfo="text",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    month_starts = pd.date_range(f"{analysis_year}-01-01", f"{analysis_year}-12-01", freq="MS")

    for month_start in month_starts:
        fig.add_vline(
            x=month_start,
            line_width=0.6,
            line_dash="dot",
            line_color="rgba(80,80,80,0.35)",
            row=1,
            col=1,
        )

    fig.update_yaxes(
        categoryorder="array",
        categoryarray=plotly_y_order,
        title="City",
        row=1,
        col=1,
    )

    fig.update_yaxes(
        categoryorder="array",
        categoryarray=plotly_y_order,
        showticklabels=False,
        row=1,
        col=2,
    )

    fig.update_xaxes(
        title=f"Day in {analysis_year}",
        tickformat="%b",
        dtick="M1",
        row=1,
        col=1,
    )

    fig.update_xaxes(
        title="% of valid days",
        range=[0, max(105, exceedance_summary["exceedance_pct"].max() + 10)],
        row=1,
        col=2,
    )

    fig.update_layout(
        title=dict(
            text=(
                f"WHO PM2.5 Daily Exceedance Burden, {analysis_year}"
                f"<br><sup>Each square is one city-day. Red days exceed the WHO 24-hour PM2.5 guideline of {who_threshold:g} ug/m3; gray days have insufficient coverage.</sup>"
            ),
            x=0.02,
            xanchor="left",
        ),
        height=780,
        template="plotly_white",
        margin=dict(l=120, r=80, t=110, b=70),
        legend=dict(
            title="Daily status",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    return fig, daily, exceedance_summary


def make_and_save_pm25_exceedance_dot_matrix(
    daily_path: PathLike,
    selected_main_path: PathLike,
    figure_output_dir: PathLike = "outputs/figures",
    web_data_output_dir: PathLike = "web/data",
    who_threshold: float = WHO_PM25_DAILY,
    min_valid_hours: int = MIN_VALID_HOURS,
    analysis_year: int = 2025,
) -> tuple[go.Figure, pd.DataFrame, pd.DataFrame]:
    daily = pd.read_csv(daily_path)
    selected_main = pd.read_csv(selected_main_path)

    fig, daily_matrix, exceedance_summary = build_pm25_exceedance_dot_matrix(
        daily=daily,
        selected_main=selected_main,
        who_threshold=who_threshold,
        min_valid_hours=min_valid_hours,
        analysis_year=analysis_year,
    )

    figure_output_dir = Path(figure_output_dir)
    web_data_output_dir = Path(web_data_output_dir)

    figure_output_dir.mkdir(parents=True, exist_ok=True)
    web_data_output_dir.mkdir(parents=True, exist_ok=True)

    fig.write_html(
        figure_output_dir / "pm25_who_daily_exceedance_dot_matrix_2025.html",
        include_plotlyjs="cdn",
    )

    daily_matrix.to_csv(
        web_data_output_dir / "pm25_daily_exceedance_matrix_2025.csv",
        index=False,
    )

    exceedance_summary.to_csv(
        web_data_output_dir / "pm25_exceedance_burden_summary_2025.csv",
        index=False,
    )

    return fig, daily_matrix, exceedance_summary
