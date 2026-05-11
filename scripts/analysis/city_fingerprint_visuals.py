
"""
City fingerprint visualization utilities.

This module creates an interactive PCA-style city fingerprint map using:
- air quality burden
- PM2.5 / PM10 particle profile
- seasonality / volatility
- weather context
- WHO exceedance burden

The output is designed for analysis.ipynb and later webpage export.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


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


FEATURE_LABELS = {
    "annual_median_pm25": "Median PM2.5",
    "annual_p90_pm25": "P90 PM2.5",
    "pm25_exceedance_pct": "WHO exceedance %",
    "longest_pm25_streak": "Longest PM2.5 streak",
    "pm25_seasonal_amplitude": "PM2.5 seasonality",
    "pm25_monthly_volatility": "PM2.5 volatility",
    "annual_median_pm10": "Median PM10",
    "annual_median_ratio": "PM2.5/PM10 ratio",
    "annual_mean_wind": "Mean wind",
    "annual_mean_temperature": "Temperature",
    "annual_mean_humidity": "Humidity",
    "annual_mean_cloud_cover": "Cloud cover",
}


CLUSTER_COLORS = {
    0: "#1b9e77",
    1: "#d95f02",
    2: "#7570b3",
    3: "#e7298a",
    4: "#66a61e",
    5: "#e6ab02",
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


def longest_true_streak(values):
    longest = 0
    current = 0

    for value in values:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    return longest


def simple_kmeans(matrix, n_clusters=4, max_iter=100, random_state=42):
    # Lightweight deterministic k-means to avoid external dependencies
    rng = np.random.default_rng(random_state)

    n_rows = matrix.shape[0]
    n_clusters = int(min(max(n_clusters, 1), n_rows))

    # Initialize with spread-out rows based on first principal axis approximation
    if n_rows <= n_clusters:
        labels = np.arange(n_rows)
        centers = matrix.copy()
        return labels, centers

    order = np.argsort(matrix[:, 0])
    init_positions = np.linspace(0, n_rows - 1, n_clusters).round().astype(int)
    centers = matrix[order[init_positions]].copy()

    labels = np.zeros(n_rows, dtype=int)

    for _ in range(max_iter):
        distances = ((matrix[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = distances.argmin(axis=1)

        if np.array_equal(labels, new_labels):
            break

        labels = new_labels

        for cluster in range(n_clusters):
            if np.any(labels == cluster):
                centers[cluster] = matrix[labels == cluster].mean(axis=0)
            else:
                centers[cluster] = matrix[rng.integers(0, n_rows)]

    return labels, centers


def build_city_fingerprint_features(
    daily,
    monthly,
    selected_main,
    min_valid_hours=MIN_VALID_HOURS,
    analysis_year=2025,
):
    daily = fill_city_metadata(daily)
    monthly = fill_city_metadata(monthly)
    selected_main = fill_city_metadata(selected_main)

    selected_city_keys = selected_main["city_key"].unique()

    daily = daily[daily["city_key"].isin(selected_city_keys)].copy()
    monthly = monthly[monthly["city_key"].isin(selected_city_keys)].copy()

    daily["local_date"] = pd.to_datetime(daily["local_date"], errors="coerce")
    daily = daily[daily["local_date"].dt.year == analysis_year].copy()

    if "local_year" in monthly.columns:
        monthly = monthly[monthly["local_year"] == analysis_year].copy()

    numeric_daily = [
        "pm25_daily_mean",
        "pm25_daily_median",
        "pm25_daily_p90",
        "pm25_daily_max",
        "pm10_daily_mean",
        "pm10_daily_median",
        "pm25_pm10_ratio_daily_mean",
        "pm25_pm10_ratio_daily_median",
        "valid_pm25_hours",
        "valid_pm10_hours",
        "temperature_2m_mean_c",
        "relative_humidity_2m_mean_pct",
        "wind_speed_10m_mean_ms",
        "cloud_cover_mean_pct",
        "precipitation_total_mm",
    ]

    numeric_monthly = [
        "pm25_monthly_mean",
        "pm25_monthly_median",
        "pm25_monthly_p90",
        "pm10_monthly_mean",
        "pm10_monthly_median",
        "pm25_pm10_ratio_monthly_mean",
        "pm25_pm10_ratio_monthly_median",
        "pm25_month_coverage_pct",
        "pm10_month_coverage_pct",
        "valid_pm25_days",
        "valid_pm10_days",
    ]

    for column in numeric_daily:
        if column in daily.columns:
            daily[column] = pd.to_numeric(daily[column], errors="coerce")

    for column in numeric_monthly:
        if column in monthly.columns:
            monthly[column] = pd.to_numeric(monthly[column], errors="coerce")

    daily["coverage_ok_pm25"] = daily["valid_pm25_hours"].fillna(0) >= min_valid_hours
    daily["pm25_exceeds_who"] = (
        daily["coverage_ok_pm25"]
        & daily["pm25_daily_mean"].notna()
        & (daily["pm25_daily_mean"] > WHO_PM25_DAILY)
    )

    daily["coverage_ok_pm10"] = daily["valid_pm10_hours"].fillna(0) >= min_valid_hours

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
        daily["particle_ratio"] = np.nan

    daily_valid_pm25 = daily[daily["coverage_ok_pm25"] & daily["pm25_daily_mean"].notna()].copy()

    daily_features = (
        daily_valid_pm25
        .groupby(["city_key", "city_display", "country", "city_group"], dropna=False)
        .agg(
            valid_pm25_days=("local_date", "nunique"),
            annual_median_pm25=("pm25_daily_median", "median"),
            annual_mean_pm25=("pm25_daily_mean", "mean"),
            annual_p90_pm25=("pm25_daily_mean", lambda x: float(pd.to_numeric(x, errors="coerce").quantile(0.90))),
            pm25_exceedance_days=("pm25_exceeds_who", "sum"),
            annual_mean_temperature=("temperature_2m_mean_c", "mean"),
            annual_mean_humidity=("relative_humidity_2m_mean_pct", "mean"),
            annual_mean_wind=("wind_speed_10m_mean_ms", "mean"),
            annual_mean_cloud_cover=("cloud_cover_mean_pct", "mean"),
            annual_precipitation_total=("precipitation_total_mm", "sum"),
        )
        .reset_index()
    )

    daily_features["pm25_exceedance_pct"] = (
        daily_features["pm25_exceedance_days"]
        / daily_features["valid_pm25_days"].replace(0, np.nan)
        * 100
    )

    streak_rows = []

    for city_key, city_df in daily.sort_values("local_date").groupby("city_key"):
        city_df = city_df.sort_values("local_date").copy()
        streak_rows.append(
            {
                "city_key": city_key,
                "longest_pm25_streak": longest_true_streak(
                    (city_df["coverage_ok_pm25"] & city_df["pm25_exceeds_who"]).tolist()
                ),
            }
        )

    streak_df = pd.DataFrame(streak_rows)

    daily_features = daily_features.merge(
        streak_df,
        on="city_key",
        how="left",
    )

    particle_valid = daily[
        daily["coverage_ok_pm25"]
        & daily["coverage_ok_pm10"]
        & daily["particle_ratio"].notna()
        & (daily["particle_ratio"] > 0)
        & (daily["particle_ratio"] <= 2.5)
    ].copy()

    if not particle_valid.empty:
        particle_features = (
            particle_valid
            .groupby("city_key")
            .agg(
                valid_particle_days=("local_date", "nunique"),
                annual_median_pm10=("pm10_daily_median", "median"),
                annual_mean_pm10=("pm10_daily_mean", "mean"),
                annual_median_ratio=("particle_ratio", "median"),
            )
            .reset_index()
        )
    else:
        particle_features = pd.DataFrame(
            columns=["city_key", "valid_particle_days", "annual_median_pm10", "annual_mean_pm10", "annual_median_ratio"]
        )

    monthly_pm25 = monthly[monthly["pm25_monthly_median"].notna()].copy()

    monthly_features = (
        monthly_pm25
        .groupby("city_key")
        .agg(
            months_with_pm25=("local_month", "nunique"),
            pm25_monthly_min=("pm25_monthly_median", "min"),
            pm25_monthly_max=("pm25_monthly_median", "max"),
            pm25_monthly_volatility=("pm25_monthly_median", "std"),
            mean_pm25_month_coverage=("pm25_month_coverage_pct", "mean"),
        )
        .reset_index()
    )

    monthly_features["pm25_seasonal_amplitude"] = (
        monthly_features["pm25_monthly_max"] - monthly_features["pm25_monthly_min"]
    )

    features = (
        daily_features
        .merge(particle_features, on="city_key", how="left")
        .merge(monthly_features, on="city_key", how="left")
    )

    # Keep PM10 and ratio missingness explicit but allow PCA by imputing later
    features["annual_median_pm10"] = features["annual_median_pm10"].fillna(features["annual_median_pm10"].median())
    features["annual_median_ratio"] = features["annual_median_ratio"].fillna(features["annual_median_ratio"].median())

    return features


def run_pca_and_clustering(features, n_clusters=4):
    feature_columns = [
        "annual_median_pm25",
        "annual_p90_pm25",
        "pm25_exceedance_pct",
        "longest_pm25_streak",
        "pm25_seasonal_amplitude",
        "pm25_monthly_volatility",
        "annual_median_pm10",
        "annual_median_ratio",
        "annual_mean_wind",
        "annual_mean_temperature",
        "annual_mean_humidity",
        "annual_mean_cloud_cover",
    ]

    available_features = [column for column in feature_columns if column in features.columns]
    matrix = features[available_features].copy()

    for column in available_features:
        matrix[column] = pd.to_numeric(matrix[column], errors="coerce")
        matrix[column] = matrix[column].fillna(matrix[column].median())

    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0).replace(0, 1)

    z = (matrix - means) / stds
    z_matrix = z.to_numpy(dtype=float)

    # PCA with SVD
    u, singular_values, vt = np.linalg.svd(z_matrix, full_matrices=False)
    scores = u[:, :2] * singular_values[:2]

    explained_variance = (singular_values ** 2) / (len(z_matrix) - 1)
    explained_ratio = explained_variance / explained_variance.sum()

    loadings = pd.DataFrame(
        vt[:2, :].T,
        columns=["pc1_loading", "pc2_loading"],
        index=available_features,
    ).reset_index().rename(columns={"index": "feature"})

    loadings["feature_label"] = loadings["feature"].map(FEATURE_LABELS).fillna(loadings["feature"])
    loadings["loading_strength"] = np.sqrt(loadings["pc1_loading"] ** 2 + loadings["pc2_loading"] ** 2)

    labels, centers = simple_kmeans(scores, n_clusters=n_clusters, random_state=42)

    pca_features = features.copy()
    pca_features["pc1"] = scores[:, 0]
    pca_features["pc2"] = scores[:, 1]
    pca_features["cluster"] = labels
    pca_features["cluster_label"] = "Cluster " + (pca_features["cluster"] + 1).astype(str)

    return pca_features, loadings, explained_ratio[:2], available_features


def build_city_fingerprint_pca_plot(
    daily,
    monthly,
    selected_main,
    analysis_year=2025,
    n_clusters=4,
):
    features = build_city_fingerprint_features(
        daily=daily,
        monthly=monthly,
        selected_main=selected_main,
        analysis_year=analysis_year,
    )

    pca_features, loadings, explained_ratio, feature_columns = run_pca_and_clustering(
        features=features,
        n_clusters=n_clusters,
    )

    fig = go.Figure()

    max_exceedance = max(pca_features["pm25_exceedance_pct"].max(), 1)

    for cluster, cluster_df in pca_features.groupby("cluster"):
        color = CLUSTER_COLORS.get(int(cluster), "#666666")

        size = 16 + 24 * (
            cluster_df["pm25_exceedance_pct"].fillna(0)
            / max_exceedance
        )

        fig.add_trace(
            go.Scatter(
                x=cluster_df["pc1"],
                y=cluster_df["pc2"],
                mode="markers+text",
                name=f"Cluster {int(cluster) + 1}",
                text=cluster_df["city_display"],
                textposition="top center",
                marker=dict(
                    size=size,
                    color=color,
                    opacity=0.82,
                    line=dict(width=1.2, color="white"),
                ),
                hovertext=(
                    "<b>" + cluster_df["city_display"] + "</b><br>"
                    + "Cluster: " + cluster_df["cluster_label"] + "<br>"
                    + "Median PM2.5: " + cluster_df["annual_median_pm25"].round(1).astype(str) + " ug/m3<br>"
                    + "P90 PM2.5: " + cluster_df["annual_p90_pm25"].round(1).astype(str) + " ug/m3<br>"
                    + "WHO exceedance: " + cluster_df["pm25_exceedance_pct"].round(0).astype(str) + "%<br>"
                    + "Longest streak: " + cluster_df["longest_pm25_streak"].fillna(0).astype(int).astype(str) + " days<br>"
                    + "PM2.5/PM10 ratio: " + cluster_df["annual_median_ratio"].round(2).astype(str) + "<br>"
                    + "Mean wind: " + cluster_df["annual_mean_wind"].round(1).astype(str) + " m/s<br>"
                    + "Country: " + cluster_df["country"].astype(str)
                ),
                hoverinfo="text",
            )
        )

    # Loading arrows
    top_loadings = loadings.sort_values("loading_strength", ascending=False).head(7).copy()

    x_range = pca_features["pc1"].max() - pca_features["pc1"].min()
    y_range = pca_features["pc2"].max() - pca_features["pc2"].min()
    arrow_scale = 0.42 * max(x_range, y_range)

    for _, row in top_loadings.iterrows():
        x_end = row["pc1_loading"] * arrow_scale
        y_end = row["pc2_loading"] * arrow_scale

        fig.add_annotation(
            x=x_end,
            y=y_end,
            ax=0,
            ay=0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.2,
            arrowwidth=1.4,
            arrowcolor="rgba(40,40,40,0.65)",
        )

        fig.add_annotation(
            x=x_end * 1.08,
            y=y_end * 1.08,
            text=row["feature_label"],
            showarrow=False,
            font=dict(size=11, color="rgba(40,40,40,0.85)"),
            bgcolor="rgba(255,255,255,0.65)",
        )

    fig.add_hline(y=0, line_dash="dot", line_color="rgba(80,80,80,0.35)", line_width=1)
    fig.add_vline(x=0, line_dash="dot", line_color="rgba(80,80,80,0.35)", line_width=1)

    fig.update_layout(
        title=dict(
            text=(
                f"City Fingerprint Map, {analysis_year}"
                "<br><sup>PCA projection from pollution burden, seasonality, particle profile, weather context, and exceedance features. Marker size shows WHO exceedance burden.</sup>"
            ),
            x=0.02,
            xanchor="left",
        ),
        xaxis_title=f"Fingerprint axis 1 ({explained_ratio[0] * 100:.0f}% variance)",
        yaxis_title=f"Fingerprint axis 2 ({explained_ratio[1] * 100:.0f}% variance)",
        height=760,
        template="plotly_white",
        margin=dict(l=80, r=50, t=110, b=80),
        legend=dict(
            title="Cluster",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    return fig, pca_features, loadings


def make_and_save_city_fingerprint_pca(
    daily_path,
    monthly_path,
    selected_main_path,
    figure_output_dir="outputs/figures",
    web_data_output_dir="web/data",
    analysis_year=2025,
    n_clusters=4,
):
    daily = pd.read_csv(daily_path)
    monthly = pd.read_csv(monthly_path)
    selected_main = pd.read_csv(selected_main_path)

    fig, pca_features, loadings = build_city_fingerprint_pca_plot(
        daily=daily,
        monthly=monthly,
        selected_main=selected_main,
        analysis_year=analysis_year,
        n_clusters=n_clusters,
    )

    figure_output_dir = Path(figure_output_dir)
    web_data_output_dir = Path(web_data_output_dir)
    figure_output_dir.mkdir(parents=True, exist_ok=True)
    web_data_output_dir.mkdir(parents=True, exist_ok=True)

    fig.write_html(
        figure_output_dir / "city_fingerprint_pca_2025.html",
        include_plotlyjs="cdn",
    )

    pca_features.to_csv(
        web_data_output_dir / "city_fingerprint_features_2025.csv",
        index=False,
    )

    loadings.to_csv(
        web_data_output_dir / "city_fingerprint_pca_loadings_2025.csv",
        index=False,
    )

    return fig, pca_features, loadings
