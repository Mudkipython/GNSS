from __future__ import annotations

import io
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
PUBLIC_BUNDLE_PATH = APP_DIR / "data" / "precipice_dashboard_public_bundle.pkl"
BUNDLE_PATH = Path(os.environ.get("PRECIPICE_DASHBOARD_BUNDLE", PUBLIC_BUNDLE_PATH))

NAVY = "#12263A"
BLUE = "#1F6F8B"
MID_BLUE = "#4C72B0"
ICE = "#EAF3F8"
ORANGE = "#DD8452"
GREEN = "#2F855A"
RED = "#C44E52"
PURPLE = "#8172B3"
LABEL_NAMES = {0: "reliable proxy", 1: "suspect proxy"}
LABEL_COLORS = {"reliable_proxy": MID_BLUE, "suspect_proxy": ORANGE}
PLOT_CONFIG = {"displaylogo": False, "scrollZoom": False, "responsive": True}

METRIC_LABELS = {
    "oof_accuracy": "Accuracy",
    "oof_precision_suspect": "Precision: suspect",
    "oof_recall_suspect": "Recall: suspect",
    "oof_recall_reliable": "Recall: reliable",
    "oof_balanced_accuracy": "Balanced accuracy",
    "oof_f1_suspect": "F1: suspect",
    "oof_f2_suspect": "F2: suspect",
    "oof_r2_probability": "R²: probability",
    "mean_monthly_overfit_gap": "Train-validation gap",
}

st.set_page_config(
    page_title="PRECIPICE GNSS-IR review studio",
    page_icon=":material/water:",
    layout="wide",
)


def inject_style() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.2rem; max-width: 1460px;}
        .review-hero {
            padding: 1.35rem 1.5rem;
            border-radius: 16px;
            background: linear-gradient(115deg, #12263A 0%, #1F6F8B 72%, #4D8FA8 100%);
            color: white;
            box-shadow: 0 14px 32px rgba(18,38,58,.16);
            margin-bottom: .8rem;
        }
        .review-hero h1 {margin: 0; font-size: 2.05rem; letter-spacing: -.02em;}
        .review-hero p {margin: .45rem 0 0; color: #EAF3F8; max-width: 980px;}
        .eyebrow {font-size: .78rem; text-transform: uppercase; letter-spacing: .12em; opacity: .82;}
        .evidence-note {
            border-left: 4px solid #1F6F8B;
            background: #F1F7FA;
            border-radius: 8px;
            padding: .75rem .9rem;
            color: #29485A;
        }
        section[data-testid="stSidebar"] {border-right: 1px solid #BFD3DF;}
        div[data-testid="stMetric"] {background: #FFFFFF; border: 1px solid #D6E4EB;}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_bundle(path: str) -> dict:
    bundle_path = Path(path)
    if not bundle_path.exists():
        return {"tables_csv": {}, "json": {}, "images": {}, "metadata": {}}
    with bundle_path.open("rb") as handle:
        return pickle.load(handle)


DATA = load_bundle(str(BUNDLE_PATH))


def table(name: str, parse_dates: tuple[str, ...] = ()) -> pd.DataFrame:
    text = DATA.get("tables_csv", {}).get(name, "")
    if not text:
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(text), parse_dates=list(parse_dates))


def style_figure(fig: go.Figure, height: int | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, Arial, sans-serif", color=NAVY),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        hoverlabel=dict(bgcolor="white", font_color=NAVY),
        margin=dict(l=35, r=25, t=65, b=35),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    if height:
        fig.update_layout(height=height)
    fig.update_xaxes(showgrid=True, gridcolor="#E8EEF2")
    fig.update_yaxes(showgrid=True, gridcolor="#E8EEF2")
    return fig


def show_chart(fig: go.Figure, key: str) -> None:
    st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG, key=key)


def filter_dates(df: pd.DataFrame, column: str, date_range) -> pd.DataFrame:
    if df.empty or column not in df or not date_range:
        return df.copy()
    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
    return df[df[column].between(start, end, inclusive="left")].copy()


def as_percent(value) -> str:
    return "n/a" if value is None or pd.isna(value) else f"{float(value):.1%}"


def destination_point(lon: float, lat: float, bearing_deg: float, distance_km: float) -> tuple[float, float]:
    radius_km = 6371.0
    bearing = np.deg2rad(bearing_deg)
    lat1, lon1 = np.deg2rad(lat), np.deg2rad(lon)
    angular = distance_km / radius_km
    lat2 = np.arcsin(np.sin(lat1) * np.cos(angular) + np.cos(lat1) * np.sin(angular) * np.cos(bearing))
    lon2 = lon1 + np.arctan2(
        np.sin(bearing) * np.sin(angular) * np.cos(lat1),
        np.cos(angular) - np.sin(lat1) * np.sin(lat2),
    )
    return float(np.rad2deg(lon2)), float(np.rad2deg(lat2))


def look_sector_map(site_lon: float, site_lat: float, risk: float | None) -> go.Figure:
    risk = 0.0 if risk is None or pd.isna(risk) else float(risk)
    sector_lons, sector_lats = [site_lon], [site_lat]
    for azimuth in range(180, 251, 5):
        lon, lat = destination_point(site_lon, site_lat, azimuth, 12)
        sector_lons.append(lon)
        sector_lats.append(lat)
    sector_lons.append(site_lon)
    sector_lats.append(site_lat)

    fig = go.Figure()
    fig.add_trace(go.Scattergeo(
        lon=sector_lons,
        lat=sector_lats,
        mode="lines",
        fill="toself",
        fillcolor=f"rgba(221,132,82,{0.16 + 0.48 * np.clip(risk, 0, 1):.2f})",
        line=dict(color=ORANGE, width=2),
        name="approximate look sector",
        hovertemplate=f"Selected-day suspect risk={risk:.2f}<extra></extra>",
    ))
    for azimuth in (180, 215, 250):
        lon, lat = destination_point(site_lon, site_lat, azimuth, 12)
        fig.add_trace(go.Scattergeo(
            lon=[site_lon, lon], lat=[site_lat, lat], mode="lines+text",
            text=["", f"{azimuth}°"], textposition="top center",
            line=dict(color=BLUE, width=1.6), showlegend=False,
            hovertemplate=f"Representative azimuth {azimuth}°<extra></extra>",
        ))
    fig.add_trace(go.Scattergeo(
        lon=[site_lon], lat=[site_lat], mode="markers+text",
        marker=dict(size=14, color=RED, line=dict(color="white", width=2)),
        text=["PRECIPICE"], textposition="top right", name="station",
        hovertemplate="Approximate public display location<extra></extra>",
    ))
    fig.update_geos(
        projection_type="azimuthal equal area", center=dict(lat=site_lat, lon=site_lon),
        lataxis_range=[site_lat - 0.48, site_lat + 0.48],
        lonaxis_range=[site_lon - 1.25, site_lon + 1.25],
        showland=True, landcolor="#E8ECEF", showocean=True, oceancolor="#DDEDF5",
        coastlinecolor="#52606D", resolution=50,
    )
    fig.update_layout(title="Where the station looks", margin=dict(l=0, r=0, t=55, b=0))
    return style_figure(fig, 500)


def risk_calendar(predictions: pd.DataFrame, pipeline: str) -> go.Figure:
    df = predictions[predictions["pipeline"].eq(pipeline)].copy()
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["day"] = df["date"].dt.day
    calendar = df.pivot_table(index="month", columns="day", values="suspect_probability", aggfunc="mean")
    fig = go.Figure(go.Heatmap(
        z=calendar.values, x=calendar.columns, y=calendar.index,
        colorscale=[[0, "#F7FBFF"], [.35, "#F8DDAA"], [.7, "#F06B3D"], [1, "#8B1E2D"]],
        zmin=0, zmax=1, colorbar=dict(title="P(suspect)"),
        hovertemplate="%{y}-%{x}<br>P(suspect)=%{z:.2f}<extra></extra>",
    ))
    fig.update_layout(title="When the model raises review risk", xaxis_title="Day of month", yaxis_title="Month")
    return style_figure(fig, 430)


def water_context_plot(qc: pd.DataFrame, date_range) -> go.Figure:
    df = filter_dates(qc, "date", date_range)
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=.07,
        subplot_titles=("Daily GNSS-IR V5 mean", "Spline uncertainty", "Daily tidal range / damping context"),
        row_heights=[.48, .22, .30],
    )
    fig.add_trace(go.Scatter(x=df["date"], y=df["water_level_mean"], mode="lines", name="daily mean", line=dict(color=MID_BLUE, width=1.7)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["error_median"] * 100, mode="lines", name="median error", line=dict(color=RED, width=1.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["water_level_range"], mode="markers", name="daily range", marker=dict(color="#7A8790", size=5, opacity=.55)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["water_level_range_14d"], mode="lines", name="14-day range", line=dict(color=GREEN, width=2.1)), row=3, col=1)
    fig.add_hline(y=1.232, line_dash="dash", line_color=ORANGE, annotation_text="proxy threshold", row=3, col=1)
    fig.add_vrect(x0="2024-12-01", x1="2025-06-30", fillcolor="#9ECAE1", opacity=.12, line_width=0, row="all", col=1)
    fig.update_yaxes(title_text="m", row=1, col=1)
    fig.update_yaxes(title_text="cm", row=2, col=1)
    fig.update_yaxes(title_text="m/day", row=3, col=1)
    fig.update_layout(title="Interactive observing-product review", hovermode="x unified")
    return style_figure(fig, 760)


def tide_context_plot(six_hour: pd.DataFrame, daily: pd.DataFrame, date_range) -> go.Figure:
    hourly = filter_dates(six_hour, "datetime_utc", date_range)
    daily_view = filter_dates(daily, "date", date_range)
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=.07,
        subplot_titles=("Six-hour anomalies", "Tide-rate turning-point proxy", "Daily range comparison"),
        row_heights=[.45, .25, .30],
    )
    fig.add_trace(go.Scatter(x=hourly["datetime_utc"], y=hourly["tide_anomaly_m"], mode="lines", name="predicted tide", line=dict(color=ORANGE, width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=hourly["datetime_utc"], y=hourly["gnss_anomaly_m"], mode="lines", name="GNSS-IR V5", line=dict(color=MID_BLUE, width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=hourly["datetime_utc"], y=hourly["tide_rate_m_per_h"], mode="lines", name="tide rate", line=dict(color=PURPLE, width=1.3)), row=2, col=1)
    fig.add_hline(y=0, line_color="#7A8790", line_width=1, row=2, col=1)
    fig.add_trace(go.Scatter(x=daily_view["date"], y=daily_view["tide_range_24h"], mode="lines", name="tide range", line=dict(color=ORANGE, width=1.6)), row=3, col=1)
    fig.add_trace(go.Scatter(x=daily_view["date"], y=daily_view["gnss_range_24h"], mode="lines+markers", name="GNSS range", line=dict(color=MID_BLUE, width=1.2), marker=dict(size=4)), row=3, col=1)
    fig.update_yaxes(title_text="centered m", row=1, col=1)
    fig.update_yaxes(title_text="m/hour", row=2, col=1)
    fig.update_yaxes(title_text="m/day", row=3, col=1)
    fig.update_layout(title="Tide prediction as physical context", hovermode="x unified")
    return style_figure(fig, 760)


def pressure_coverage_plot(qc: pd.DataFrame, coverage: pd.DataFrame, date_range) -> go.Figure:
    df = filter_dates(qc, "date", date_range)
    anomaly = df["water_level_mean"] - df["water_level_mean"].median()
    fig = go.Figure(go.Scatter(
        x=df["date"], y=anomaly, mode="lines", name="GNSS-IR daily anomaly",
        line=dict(color=MID_BLUE, width=1.5),
        hovertemplate="%{x|%Y-%m-%d}<br>GNSS anomaly=%{y:.2f} m<extra></extra>",
    ))
    colors = ["rgba(221,132,82,.18)", "rgba(47,133,90,.18)"]
    for idx, row in coverage.iterrows():
        fig.add_vrect(
            x0=row["start"], x1=row["end"], fillcolor=colors[idx % len(colors)], line_width=0,
            annotation_text=f"{row['window']}", annotation_position="top left",
        )
    fig.update_layout(title="Pressure availability relative to the GNSS-IR record", yaxis_title="median-centered GNSS-IR (m)")
    return style_figure(fig, 480)


def diagnostic_plot(qc: pd.DataFrame, metric: str, date_range) -> go.Figure:
    df = filter_dates(qc, "date", date_range)
    reason_cols = [
        "seasonal_damped_period_proxy", "low_arc_availability_suspect",
        "low_v5_coverage_suspect", "low_tide_range_suspect",
    ]
    df["reasons"] = df[reason_cols].apply(
        lambda row: ", ".join(col.replace("_suspect", "").replace("_proxy", "") for col in reason_cols if bool(row[col])) or "reliable proxy",
        axis=1,
    )
    metric_labels = {
        "arc_count": "Daily arc count",
        "error_median": "Median spline error (m)",
        "valid_fraction": "Valid V5 fraction",
        "water_level_range": "Daily water-level range (m)",
        "water_level_range_14d": "14-day water-level range (m)",
    }
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=.08,
        subplot_titles=("Water level colored by proxy label", metric_labels[metric], "Proxy-label band"),
        row_heights=[.48, .32, .20],
    )
    for label, color in LABEL_COLORS.items():
        group = df[df["weak_label"].eq(label)]
        fig.add_trace(go.Scatter(
            x=group["date"], y=group["water_level_mean"], mode="markers", name=label.replace("_", " "),
            marker=dict(color=color, size=6), customdata=group[["reasons"]],
            hovertemplate="%{x|%Y-%m-%d}<br>water=%{y:.2f} m<br>%{customdata[0]}<extra></extra>",
        ), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df[metric], mode="lines+markers", name=metric_labels[metric], line=dict(color=GREEN, width=1.5), marker=dict(size=4)), row=2, col=1)
    thresholds = {"arc_count": 214, "valid_fraction": .5, "water_level_range": 1.232}
    if metric in thresholds:
        fig.add_hline(y=thresholds[metric], line_dash="dash", line_color=RED, annotation_text="proxy threshold", row=2, col=1)
    label_y = df["weak_label"].map({"reliable_proxy": 0, "suspect_proxy": 1})
    fig.add_trace(go.Scatter(
        x=df["date"], y=label_y, mode="markers", name="proxy label",
        marker=dict(color=df["weak_label"].map(LABEL_COLORS), size=7), text=df["reasons"],
        hovertemplate="%{x|%Y-%m-%d}<br>%{text}<extra></extra>",
    ), row=3, col=1)
    fig.update_yaxes(title_text="m", row=1, col=1)
    fig.update_yaxes(tickvals=[0, 1], ticktext=["reliable", "suspect"], row=3, col=1)
    fig.update_layout(title="Why was a day marked for review?", hovermode="x unified")
    return style_figure(fig, 720)


def reason_summary_plot(qc: pd.DataFrame, date_range) -> go.Figure:
    df = filter_dates(qc, "date", date_range)
    df["month"] = df["date"].dt.to_period("M").astype(str)
    reason_cols = [
        "seasonal_damped_period_proxy", "low_arc_availability_suspect",
        "low_v5_coverage_suspect", "low_tide_range_suspect",
    ]
    monthly = df.groupby("month")[reason_cols].sum().reset_index()
    monthly["reliable_proxy"] = df["weak_label"].eq("reliable_proxy").groupby(df["month"]).sum().values
    fig = go.Figure()
    palette = {"reliable_proxy": MID_BLUE, "seasonal_damped_period_proxy": ORANGE, "low_arc_availability_suspect": PURPLE, "low_v5_coverage_suspect": RED, "low_tide_range_suspect": GREEN}
    for column in ["reliable_proxy", *reason_cols]:
        fig.add_trace(go.Bar(x=monthly["month"], y=monthly[column], name=column.replace("_", " "), marker_color=palette[column]))
    fig.update_layout(title="Monthly reason composition", barmode="stack", xaxis_title="Month", yaxis_title="Days / triggered rules")
    return style_figure(fig, 430)


def correlation_plot(correlation: pd.DataFrame, selected: list[str]) -> go.Figure:
    frame = correlation[correlation["feature_x"].isin(selected) & correlation["feature_y"].isin(selected)]
    matrix = frame.pivot(index="feature_y", columns="feature_x", values="spearman").reindex(index=selected, columns=selected)
    fig = go.Figure(go.Heatmap(
        z=matrix.values, x=matrix.columns, y=matrix.index, zmin=-1, zmax=1, zmid=0,
        colorscale="RdBu_r", colorbar=dict(title="Spearman"),
        hovertemplate="%{y} vs %{x}<br>ρ=%{z:.2f}<extra></extra>",
    ))
    fig.update_layout(title="Daily signal-feature correlation structure")
    return style_figure(fig, 610)


def leaderboard_plot(df: pd.DataFrame, label_col: str, title: str) -> go.Figure:
    ordered = df.sort_values("oof_balanced_accuracy")
    fig = px.bar(
        ordered, x="oof_balanced_accuracy", y=label_col, orientation="h",
        color="mean_monthly_overfit_gap", color_continuous_scale="OrRd",
        hover_data=[c for c in ("model", "features", "feature_strategy", "oof_recall_suspect") if c in ordered],
        labels={"oof_balanced_accuracy": "OOF balanced accuracy", label_col: "", "mean_monthly_overfit_gap": "Gap"},
        title=title,
    )
    fig.update_xaxes(range=[0, 1])
    return style_figure(fig, max(410, 42 * len(ordered)))


def overfit_plot(df: pd.DataFrame, label_col: str) -> go.Figure:
    fig = px.scatter(
        df, x="mean_monthly_validation_balanced_accuracy", y="mean_train_balanced_accuracy",
        color="mean_monthly_overfit_gap", text=label_col, color_continuous_scale="OrRd",
        hover_data=[c for c in ("model", "features", "feature_strategy") if c in df],
        labels={
            "mean_monthly_validation_balanced_accuracy": "Validation balanced accuracy",
            "mean_train_balanced_accuracy": "Train balanced accuracy",
            "mean_monthly_overfit_gap": "Gap",
        },
        title="Generalization check",
    )
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#7A8790", dash="dash"), showlegend=False))
    fig.update_traces(textposition="top center")
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    return style_figure(fig, 520)


def metrics_heatmap(df: pd.DataFrame, label_col: str) -> go.Figure:
    columns = [column for column in METRIC_LABELS if column in df]
    matrix = df.set_index(label_col)[columns]
    fig = go.Figure(go.Heatmap(
        z=matrix.values, x=[METRIC_LABELS[c] for c in columns], y=matrix.index,
        colorscale="YlGnBu", zmin=-.2, zmax=1,
        hovertemplate="%{y}<br>%{x}: %{z:.3f}<extra></extra>", colorbar=dict(title="Score"),
    ))
    fig.update_layout(title="Metric matrix")
    return style_figure(fig, max(440, 42 * len(matrix)))


def prediction_timeline(predictions: pd.DataFrame, pipeline: str, date_range) -> go.Figure:
    df = predictions[predictions["pipeline"].eq(pipeline)].copy()
    df = filter_dates(df, "date", date_range)
    df["label_name"] = df["weak_label"].map(LABEL_NAMES)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["suspect_probability"], mode="lines", name="P(suspect)", line=dict(color=ORANGE, width=2)))
    fig.add_trace(go.Scatter(x=df["date"], y=df["weak_label"], mode="markers", name="proxy label", marker=dict(color=df["label_name"].map({"reliable proxy": MID_BLUE, "suspect proxy": ORANGE}), size=6)))
    fig.add_hline(y=.5, line_dash="dash", line_color="#7A8790", annotation_text="0.5 threshold")
    fig.update_layout(title=f"Out-of-fold timeline: {pipeline}", yaxis_title="Probability / label", hovermode="x unified")
    return style_figure(fig, 470)


summary = DATA.get("json", {}).get("week7_summary", {})
metadata = DATA.get("metadata", {})
primary = table("primary_v0v5_expanding_leaderboard")
signal = table("signal_only_expanding_leaderboard")
folds = table("expanding_window_fold_summary", ("train_start", "train_end", "validation_start", "validation_end"))
predictions = table("signal_only_expanding_oof_predictions", ("date",))
feature_counts = table("signal_only_feature_selection_counts")
target_definition = table("classification_target_definition")
teacher_sets = table("teacher_requested_feature_sets")
qc = table("daily_qc_context", ("date",))
correlation = table("signal_feature_correlation")
pressure_coverage = table("pressure_coverage", ("start", "end"))
tide_6h = table("tide_gnss_6h", ("datetime_utc",))
tide_daily = table("tide_gnss_daily", ("date",))
week5_leaderboard = table("week5_feature_version_leaderboard")
week5_families = table("week5_model_family_best")
week5_delta = table("week5_feature_delta")
week5_importance = table("week5_feature_importance")
week5_split = table("week5_split_summary", ("start_date", "end_date"))
week5_rationale = table("week5_split_rationale")

if qc.empty:
    st.error("The dashboard bundle is missing `daily_qc_context`. Rebuild and audit the public bundle.")
    st.stop()

min_date, max_date = qc["date"].min().date(), qc["date"].max().date()
model_options = sorted(predictions["pipeline"].dropna().unique()) if not predictions.empty else []
if not signal.empty:
    default_pipeline = signal.sort_values("oof_balanced_accuracy", ascending=False).iloc[0]["pipeline"]
else:
    default_pipeline = model_options[0] if model_options else None

inject_style()

with st.sidebar:
    st.title("PRECIPICE")
    st.caption("GNSS-IR evidence review")
    profile = metadata.get("disclosure_profile", "unknown")
    if profile == "public":
        st.badge("Public-safe bundle", icon=":material/verified_user:", color="green")
    else:
        st.badge("Private research bundle", icon=":material/lock:", color="orange")
    st.markdown("**Data resolution**")
    st.caption("Daily GNSS/QC · six-hour tide context · aggregate model evidence")
    st.markdown("**Notebook lineage**")
    st.caption("Week 5 classification diagnostics\n\nWeek 7 geography-style review")
    with st.expander("Disclosure boundary", icon=":material/security:"):
        st.write("No Level-1 arc rows, pressure measurements, five-minute V5 values, local paths, or downloadable raw tables are loaded by the public app.")
        st.caption("Values and charts displayed here are still public research outputs and require laboratory approval.")
    with st.expander("Key terms", icon=":material/menu_book:"):
        st.markdown(
            "**Proxy label** — QC-derived reliable/suspect review target.  \n"
            "**OOF** — prediction for a month excluded from model fitting.  \n"
            "**error_m** — spline uncertainty, not pressure error.  \n"
            "**Look sector** — approximate reflection geometry, not mapped sea ice."
        )

st.markdown(
    """
    <div class="review-hero">
      <div class="eyebrow">Grise Fjord · GNSS-IR coastal monitoring</div>
      <h1>PRECIPICE GNSS-IR review studio</h1>
      <p>Move from the observing product to physical context, proxy-label diagnostics, and time-aware model evidence without exposing source-resolution laboratory data.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

view = st.segmented_control(
    "Workspace",
    ["Overview", "Observe", "Diagnose", "Models"],
    default="Overview",
    label_visibility="collapsed",
    key="workspace_view",
)

site_lat = float(summary.get("site_lat", 76.4))
site_lon = float(summary.get("site_lon", -82.9))

if view == "Overview":
    st.header("Review the station in space and time")
    best_primary = primary.sort_values("oof_balanced_accuracy", ascending=False).iloc[0] if not primary.empty else None
    best_signal = signal.sort_values("oof_balanced_accuracy", ascending=False).iloc[0] if not signal.empty else None
    with st.container(horizontal=True):
        st.metric("Public record", f"{min_date:%b %Y} – {max_date:%b %Y}", border=True)
        st.metric("Daily review points", f"{len(qc):,}", border=True)
        st.metric("Best compact OOF BA", as_percent(best_primary["oof_balanced_accuracy"] if best_primary is not None else None), border=True)
        st.metric("Best signal-only OOF BA", as_percent(best_signal["oof_balanced_accuracy"] if best_signal is not None else None), border=True)

    if predictions.empty or not default_pipeline:
        st.warning("OOF prediction data are unavailable.")
    else:
        control_left, control_right = st.columns([1.2, 2.2], vertical_alignment="bottom")
        with control_left:
            pipeline = st.selectbox("Risk pipeline", model_options, index=model_options.index(default_pipeline), key="overview_pipeline")
        pipeline_rows = predictions[predictions["pipeline"].eq(pipeline)]
        with control_right:
            review_date = st.slider(
                "Review date", pipeline_rows["date"].min().date(), pipeline_rows["date"].max().date(),
                min(pd.Timestamp("2025-02-15").date(), pipeline_rows["date"].max().date()), format="YYYY-MM-DD",
            )
        nearest = pipeline_rows.loc[(pipeline_rows["date"] - pd.Timestamp(review_date)).abs().idxmin()]
        risk = float(nearest["suspect_probability"])
        left, right = st.columns([1.05, 1.25], gap="large")
        with left:
            show_chart(look_sector_map(site_lon, site_lat, risk), "overview_map")
        with right:
            st.metric("Selected-day risk", as_percent(risk), border=True)
            st.metric("Proxy label", LABEL_NAMES.get(int(nearest["weak_label"]), "n/a"), border=True)
            st.metric("OOF prediction", LABEL_NAMES.get(int(nearest["prediction"]), "n/a"), border=True)
            st.markdown(
                "<div class='evidence-note'><b>Interpretation:</b> the sector shows approximate viewing geometry. Its color is a date-specific review signal, not a spatial ice classification.</div>",
                unsafe_allow_html=True,
            )
        show_chart(risk_calendar(predictions, pipeline), "overview_calendar")

elif view == "Observe":
    st.header("Observe the product and its physical context")
    st.caption("Start from GNSS-IR behavior before interpreting classifications or models.")
    controls = st.columns([1.25, 2.2], vertical_alignment="bottom")
    with controls[0]:
        context_view = st.segmented_control("Context", ["Water level", "Tide", "Pressure coverage"], default="Water level")
    with controls[1]:
        observation_range = st.slider("Date window", min_date, max_date, (min_date, max_date), format="YYYY-MM-DD", key="observation_range")

    if context_view == "Water level":
        show_chart(water_context_plot(qc, observation_range), "water_context")
        st.caption("The shaded interval is seasonal proxy context. The 1.232 m line is a proxy-label threshold, not a physical ice boundary.")
    elif context_view == "Tide":
        if tide_6h.empty:
            st.warning("Tide context is unavailable in this bundle.")
        else:
            show_chart(tide_context_plot(tide_6h, tide_daily, observation_range), "tide_context")
            st.caption("Tide prediction is an external physical covariate. The public view uses six-hour means and has 2025 support only.")
    else:
        show_chart(pressure_coverage_plot(qc, pressure_coverage, observation_range), "pressure_context")
        st.caption("Only coverage intervals are published. Pressure measurements and vertical-datum values are not included in the public bundle.")
        if not pressure_coverage.empty:
            st.dataframe(
                pressure_coverage,
                hide_index=True,
                width="stretch",
                column_config={
                    "window": "Coverage window",
                    "start": st.column_config.DatetimeColumn("Start", format="YYYY-MM-DD HH:mm"),
                    "end": st.column_config.DatetimeColumn("End", format="YYYY-MM-DD HH:mm"),
                },
            )

elif view == "Diagnose":
    st.header("Diagnose why observations were flagged")
    diagnostic_view = st.segmented_control(
        "Diagnostic view", ["Target & QC", "Feature structure", "Definitions"], default="Target & QC"
    )
    if diagnostic_view == "Target & QC":
        left, right = st.columns([1.15, 2.1], vertical_alignment="bottom")
        with left:
            metric = st.selectbox(
                "Diagnostic variable",
                ["arc_count", "error_median", "valid_fraction", "water_level_range", "water_level_range_14d"],
                format_func=lambda value: {
                    "arc_count": "Arc availability",
                    "error_median": "Spline uncertainty",
                    "valid_fraction": "V5 coverage",
                    "water_level_range": "Daily range",
                    "water_level_range_14d": "14-day range",
                }[value],
            )
        with right:
            diagnostic_range = st.slider("Date window", min_date, max_date, (min_date, max_date), format="YYYY-MM-DD", key="diagnostic_range")
        show_chart(diagnostic_plot(qc, metric, diagnostic_range), "diagnostic_timeline")
        show_chart(reason_summary_plot(qc, diagnostic_range), "reason_summary")
        st.warning("These labels organize QC review. They are not independently observed ice/open-water/wind classes.", icon=":material/warning:")

    elif diagnostic_view == "Feature structure":
        available = sorted(correlation["feature_x"].unique()) if not correlation.empty else []
        preferred = [name for name in ["sp_median", "ptn_median", "clr_median", "ms_median", "df_median", "rh_median", "vs_median", "af_median"] if name in available]
        selected = st.multiselect("Features shown in the correlation matrix", available, default=preferred)
        if selected:
            show_chart(correlation_plot(correlation, selected), "feature_correlation")
        col_a, col_b = st.columns([1.2, .8], gap="large")
        with col_a:
            if not feature_counts.empty:
                count_fig = px.bar(
                    feature_counts.sort_values("selected_fold_count"), x="selected_fold_count", y="feature",
                    orientation="h", title="Selection frequency across expanding-window folds",
                    labels={"selected_fold_count": "Selected folds", "feature": ""},
                )
                count_fig.update_traces(marker_color=GREEN)
                show_chart(style_figure(count_fig, max(410, 28 * len(feature_counts))), "feature_counts")
        with col_b:
            st.markdown(
                "<div class='evidence-note'><b>Why this matters</b><br><br>Strongly correlated LSP-derived variables contain duplicated information. The compact pipelines reduce redundancy and are easier to audit.</div>",
                unsafe_allow_html=True,
            )
            st.caption("The public bundle contains correlation coefficients, not daily arc-feature values.")

    else:
        left, right = st.columns(2, gap="large")
        with left:
            st.subheader("Proxy-label rules")
            st.dataframe(target_definition, hide_index=True, width="stretch")
        with right:
            st.subheader("Teacher-requested feature versions")
            st.dataframe(teacher_sets, hide_index=True, width="stretch")
        st.markdown(
            "<div class='evidence-note'><b>Evidence boundary:</b> a model trained on QC-derived labels can reproduce the review logic, but it cannot establish independent physical truth.</div>",
            unsafe_allow_html=True,
        )

else:
    st.header("Compare models without mixing validation designs")
    design = st.segmented_control(
        "Validation design", ["Expanding-window OOF", "Week 5 blocked diagnostic"], default="Expanding-window OOF"
    )

    if design == "Expanding-window OOF":
        st.caption("Primary generalization evidence: earlier months train the model and the next unseen month validates it.")
        family = st.segmented_control("Experiment family", ["Compact V-series", "Signal-only pipelines"], default="Compact V-series")
        if family == "Compact V-series":
            active, label_col = primary.copy(), "version"
        else:
            active, label_col = signal.copy(), "pipeline"
        best = active.sort_values("oof_balanced_accuracy", ascending=False).iloc[0]
        with st.container(horizontal=True):
            st.metric("Best experiment", str(best[label_col]), border=True)
            st.metric("OOF balanced accuracy", as_percent(best["oof_balanced_accuracy"]), border=True)
            st.metric("Suspect recall", as_percent(best["oof_recall_suspect"]), border=True)
            st.metric("Train-validation gap", f"{best['mean_monthly_overfit_gap']:.3f}", border=True)

        model_view = st.segmented_control("Evidence", ["Ranking", "Generalization", "Metric matrix", "OOF timeline"], default="Ranking")
        if model_view == "Ranking":
            show_chart(leaderboard_plot(active, label_col, "Time-aware model ranking"), "model_ranking")
        elif model_view == "Generalization":
            show_chart(overfit_plot(active, label_col), "model_overfit")
        elif model_view == "Metric matrix":
            show_chart(metrics_heatmap(active, label_col), "model_heatmap")
        elif family == "Signal-only pipelines" and model_options:
            pipeline = st.selectbox("Pipeline", model_options, index=model_options.index(default_pipeline), key="timeline_pipeline")
            timeline_range = st.slider(
                "Timeline window", predictions["date"].min().date(), predictions["date"].max().date(),
                (predictions["date"].min().date(), predictions["date"].max().date()), format="YYYY-MM-DD",
            )
            show_chart(prediction_timeline(predictions, pipeline, timeline_range), "model_timeline")
        else:
            st.info("OOF probability timelines are available for the signal-only pipelines.")

        with st.expander("Validation folds and sortable metrics", icon=":material/table_chart:"):
            st.dataframe(folds, hide_index=True, width="stretch")
            st.dataframe(active.rename(columns=METRIC_LABELS), hide_index=True, width="stretch")

    else:
        st.warning(
            "Week 5 uses season-aware blocked holdout months. It is useful for feature diagnostics, but its scores must not be ranked against expanding-window OOF results.",
            icon=":material/experiment:",
        )
        week5_view = st.segmented_control(
            "Week 5 evidence", ["Feature versions", "Model families", "Added-feature gain", "Feature importance", "Split design"],
            default="Feature versions",
        )
        if week5_view == "Feature versions":
            ordered = week5_leaderboard.sort_values("balanced_accuracy")
            fig = px.bar(
                ordered, x="balanced_accuracy", y="version", orientation="h", color="model_family",
                hover_data=["model", "features", "recall_suspect", "f2_suspect"],
                labels={"balanced_accuracy": "Blocked-holdout balanced accuracy", "version": ""},
                title="Best model within each feature version",
            )
            fig.update_xaxes(range=[0, 1])
            show_chart(style_figure(fig, 470), "week5_versions")
        elif week5_view == "Model families":
            fig = px.bar(
                week5_families, x="version", y="balanced_accuracy", color="model_family", barmode="group",
                hover_data=["model", "features"], title="Model-family sensitivity within each feature version",
                labels={"balanced_accuracy": "Balanced accuracy", "version": "Feature version"},
            )
            show_chart(style_figure(fig, 520), "week5_families")
        elif week5_view == "Added-feature gain":
            best_delta = week5_delta.sort_values("delta_balanced_accuracy_vs_ms_only", ascending=False).groupby("version", as_index=False).head(1)
            fig = px.bar(
                best_delta.sort_values("delta_balanced_accuracy_vs_ms_only"),
                x="delta_balanced_accuracy_vs_ms_only", y="version", orientation="h", color="delta_balanced_accuracy_vs_ms_only",
                color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
                labels={"delta_balanced_accuracy_vs_ms_only": "Δ balanced accuracy vs ms-only", "version": ""},
                title="Best added-feature improvement over the ms-only benchmark",
            )
            fig.add_vline(x=0, line_color=NAVY, line_width=1)
            show_chart(style_figure(fig, 470), "week5_delta")
        elif week5_view == "Feature importance":
            fig = px.bar(
                week5_importance.sort_values("importance_mean"), x="importance_mean", y="feature", orientation="h",
                color="version", error_x="importance_std", hover_data=["best_model"],
                labels={"importance_mean": "Balanced-accuracy drop after shuffling", "feature": ""},
                title="Permutation importance in each version's best model",
            )
            show_chart(style_figure(fig, 520), "week5_importance")
        else:
            split_long = week5_split.melt(
                id_vars=["split", "start_date", "end_date", "total_days"],
                value_vars=["reliable_proxy", "suspect_proxy"], var_name="class", value_name="days",
            )
            fig = px.bar(split_long, x="split", y="days", color="class", barmode="stack", title="Blocked diagnostic class balance")
            show_chart(style_figure(fig, 430), "week5_split")
            if not week5_rationale.empty:
                st.dataframe(week5_rationale, hide_index=True, width="stretch")

st.caption(
    "First-pass review interface. Proxy-label performance is not independent environmental validation; pressure values remain private and the public bundle is intentionally downsampled."
)
