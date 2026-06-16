from __future__ import annotations

import io
import json
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


APP_DIR = Path(__file__).resolve().parent


def find_project_root() -> Path:
    for candidate in [Path.cwd(), APP_DIR, *APP_DIR.parents]:
        if (candidate / "outputs").exists() or (candidate / "Data_Raw").exists():
            return candidate
    if APP_DIR.parent.name == "apps":
        return APP_DIR.parents[1]
    return APP_DIR.parent


PROJECT_ROOT = find_project_root()
W6_DIR = PROJECT_ROOT / "outputs" / "eda" / "week_6_progress"
W7_DIR = PROJECT_ROOT / "outputs" / "eda" / "week_7_geography_visual_progress"
V5_PATH = PROJECT_ROOT / "Data_Raw" / "PRECIPICE" / "processed" / "spline_v5" / "precipice_sealevel_v5.csv"
BUNDLE_PATH = APP_DIR / "data" / "precipice_dashboard_bundle.pkl"

LABEL_NAMES = {0: "reliable proxy", 1: "suspect proxy"}
LABEL_COLORS = {"reliable proxy": "#4C72B0", "suspect proxy": "#DD8452"}
ATLAS_BLUE = "#1F6F8B"
ATLAS_ICE = "#EAF3F8"
ATLAS_NAVY = "#12263A"
ATLAS_ORANGE = "#DD8452"
ATLAS_GREEN = "#2F855A"
METRIC_LABELS = {
    "oof_accuracy": "Accuracy",
    "oof_precision_suspect": "Precision: suspect",
    "oof_recall_suspect": "Recall: suspect",
    "oof_recall_reliable": "Recall: reliable",
    "oof_balanced_accuracy": "Balanced accuracy",
    "oof_f1_suspect": "F1: suspect",
    "oof_f2_suspect": "F2: suspect",
    "oof_r2_probability": "R2: probability",
    "mean_monthly_overfit_gap": "Train-validation BA gap",
}


st.set_page_config(
    page_title="PRECIPICE classification review",
    page_icon=":material/analytics:",
    layout="wide",
)


def inject_style() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.4rem;
            max-width: 1420px;
        }
        div[data-testid="stMetric"] {
            background: linear-gradient(180deg, #ffffff 0%, #f3f8fb 100%);
            border: 1px solid #c9dbe5;
            box-shadow: 0 8px 22px rgba(31, 111, 139, 0.08);
        }
        .geo-hero {
            border: 1px solid #bfd3df;
            border-radius: 14px;
            padding: 1.25rem 1.35rem;
            background:
                linear-gradient(110deg, rgba(18,38,58,0.95), rgba(31,111,139,0.84)),
                repeating-linear-gradient(45deg, rgba(255,255,255,0.08) 0 1px, transparent 1px 18px);
            color: white;
            box-shadow: 0 16px 35px rgba(18, 38, 58, 0.18);
            margin-bottom: 1rem;
        }
        .geo-hero h1 {
            margin: 0;
            font-size: 2.05rem;
            letter-spacing: 0;
        }
        .geo-hero p {
            max-width: 980px;
            margin: 0.45rem 0 0 0;
            color: #e7f4f8;
            font-size: 1rem;
        }
        .geo-badge {
            display: inline-block;
            padding: 0.18rem 0.55rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.14);
            border: 1px solid rgba(255,255,255,0.32);
            color: #f7fbff;
            font-size: 0.78rem;
            margin-right: 0.35rem;
            margin-bottom: 0.35rem;
        }
        .geo-card {
            border: 1px solid #c9dbe5;
            border-radius: 12px;
            padding: 1rem;
            background: #ffffff;
            box-shadow: 0 6px 16px rgba(31, 111, 139, 0.07);
            min-height: 118px;
        }
        .geo-card h3 {
            margin: 0 0 0.35rem 0;
            font-size: 1rem;
            color: #12263a;
        }
        .geo-card p {
            margin: 0;
            color: #52606d;
            font-size: 0.92rem;
        }
        .map-note {
            border-left: 4px solid #1F6F8B;
            background: #eef7fb;
            padding: 0.75rem 0.9rem;
            border-radius: 8px;
            color: #17364a;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero() -> None:
    st.markdown(
        """
        <div class="geo-hero">
            <span class="geo-badge">GNSS-IR coastal monitoring</span>
            <span class="geo-badge">Grise Fjord / PRECIPICE</span>
            <span class="geo-badge">QC-first classification review</span>
            <h1>PRECIPICE field atlas dashboard</h1>
            <p>
                A geography-style review interface for water-level products, pressure/tide context,
                proxy labels, model diagnostics, and single-station look-sector risk.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def atlas_card(title: str, body: str) -> None:
    st.markdown(f'<div class="geo-card"><h3>{title}</h3><p>{body}</p></div>', unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_bundle(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {"tables_csv": {}, "json": {}, "images": {}, "metadata": {}}
    with p.open("rb") as f:
        return pickle.load(f)


DATA_BUNDLE = load_bundle(str(BUNDLE_PATH))


@st.cache_data(show_spinner=False)
def read_csv(path: str, parse_dates: tuple[str, ...] = ()) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, parse_dates=list(parse_dates))


@st.cache_data(show_spinner=False)
def read_json(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def bundled_csv(name: str, path: Path, parse_dates: tuple[str, ...] = ()) -> pd.DataFrame:
    tables = DATA_BUNDLE.get("tables_csv", {})
    if name in tables:
        return pd.read_csv(io.StringIO(tables[name]), parse_dates=list(parse_dates))
    return read_csv(str(path), parse_dates=parse_dates)


def bundled_json(name: str, path: Path) -> dict:
    values = DATA_BUNDLE.get("json", {})
    if name in values:
        return values[name]
    return read_json(str(path))


@st.cache_data(show_spinner=False)
def load_v5(path: str) -> pd.DataFrame:
    df = read_csv(path, parse_dates=("datetime_utc",))
    if df.empty:
        return df
    keep = [c for c in ["datetime_utc", "water_level_m", "error_m"] if c in df.columns]
    df = df[keep].dropna(subset=["datetime_utc"]).sort_values("datetime_utc")
    return df


def existing(path: Path) -> Path | None:
    return path if path.exists() else None


def as_percent(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.1%}"


def date_filter(df: pd.DataFrame, date_col: str, start, end) -> pd.DataFrame:
    if df.empty or date_col not in df.columns or not start or not end:
        return df
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1)
    return df[(df[date_col] >= start_ts) & (df[date_col] < end_ts)].copy()


def display_figure(path: Path, caption: str = "") -> None:
    if path.exists():
        st.image(str(path), width="stretch")
        if caption:
            st.caption(caption)
    elif path.name in DATA_BUNDLE.get("images", {}):
        st.image(DATA_BUNDLE["images"][path.name], width="stretch")
        if caption:
            st.caption(caption)
    else:
        st.warning(f"Missing figure: `{path}`", icon=":material/warning:")


def nice_metrics_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    rename = {c: METRIC_LABELS.get(c, c) for c in df.columns}
    return df.rename(columns=rename)


def leaderboard_plot(df: pd.DataFrame, label_col: str, title: str) -> go.Figure:
    if df.empty or "oof_balanced_accuracy" not in df.columns:
        return go.Figure()
    plot_df = df.sort_values("oof_balanced_accuracy", ascending=True).copy()
    fig = px.bar(
        plot_df,
        x="oof_balanced_accuracy",
        y=label_col,
        orientation="h",
        color="mean_monthly_overfit_gap" if "mean_monthly_overfit_gap" in plot_df.columns else None,
        color_continuous_scale="OrRd",
        hover_data=[c for c in ["model", "features", "feature_strategy", "oof_recall_suspect", "oof_precision_suspect"] if c in plot_df.columns],
        title=title,
        labels={"oof_balanced_accuracy": "OOF balanced accuracy", label_col: ""},
    )
    fig.update_layout(height=max(360, 42 * len(plot_df)), margin=dict(l=10, r=10, t=55, b=20))
    fig.update_xaxes(range=[0, 1])
    return fig


def metrics_heatmap(df: pd.DataFrame, label_col: str, title: str) -> go.Figure:
    metric_cols = [c for c in METRIC_LABELS if c in df.columns]
    if df.empty or not metric_cols:
        return go.Figure()
    heat = df.set_index(label_col)[metric_cols]
    fig = go.Figure(
        data=go.Heatmap(
            z=heat.values,
            x=[METRIC_LABELS[c] for c in heat.columns],
            y=heat.index,
            colorscale="YlGnBu",
            zmin=-0.2,
            zmax=1,
            hovertemplate="%{y}<br>%{x}: %{z:.3f}<extra></extra>",
            colorbar=dict(title="score"),
        )
    )
    fig.update_layout(title=title, height=max(380, 42 * len(heat)), margin=dict(l=10, r=10, t=55, b=80))
    return fig


def overfit_plot(df: pd.DataFrame, label_col: str, title: str) -> go.Figure:
    required = {"mean_train_balanced_accuracy", "mean_monthly_validation_balanced_accuracy"}
    if df.empty or not required.issubset(df.columns):
        return go.Figure()
    fig = px.scatter(
        df,
        x="mean_monthly_validation_balanced_accuracy",
        y="mean_train_balanced_accuracy",
        color="mean_monthly_overfit_gap" if "mean_monthly_overfit_gap" in df.columns else None,
        text=label_col,
        color_continuous_scale="Reds",
        hover_data=[c for c in ["model", "features", "feature_strategy"] if c in df.columns],
        title=title,
        labels={
            "mean_monthly_validation_balanced_accuracy": "Mean validation balanced accuracy",
            "mean_train_balanced_accuracy": "Mean train balanced accuracy",
            "mean_monthly_overfit_gap": "Gap",
        },
    )
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="gray", dash="dash"), showlegend=False))
    fig.update_traces(textposition="top center")
    fig.update_layout(height=480, margin=dict(l=10, r=10, t=55, b=20))
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    return fig


def destination_point(lon: float, lat: float, bearing_deg: float, distance_km: float) -> tuple[float, float]:
    radius_km = 6371.0
    bearing = np.deg2rad(bearing_deg)
    lat1 = np.deg2rad(lat)
    lon1 = np.deg2rad(lon)
    angular = distance_km / radius_km
    lat2 = np.arcsin(np.sin(lat1) * np.cos(angular) + np.cos(lat1) * np.sin(angular) * np.cos(bearing))
    lon2 = lon1 + np.arctan2(
        np.sin(bearing) * np.sin(angular) * np.cos(lat1),
        np.cos(angular) - np.sin(lat1) * np.sin(lat2),
    )
    return float(np.rad2deg(lon2)), float(np.rad2deg(lat2))


def look_sector_map(site_lon: float, site_lat: float, risk: float | None = None) -> go.Figure:
    risk = 0.0 if risk is None or pd.isna(risk) else float(risk)
    sector_lons = [site_lon]
    sector_lats = [site_lat]
    for az in range(180, 251, 5):
        lon, lat = destination_point(site_lon, site_lat, az, 12)
        sector_lons.append(lon)
        sector_lats.append(lat)
    sector_lons.append(site_lon)
    sector_lats.append(site_lat)

    fig = go.Figure()
    fig.add_trace(
        go.Scattergeo(
            lon=sector_lons,
            lat=sector_lats,
            mode="lines",
            fill="toself",
            fillcolor=f"rgba(221, 132, 82, {0.18 + 0.45 * min(max(risk, 0), 1):.2f})",
            line=dict(color=ATLAS_ORANGE, width=2),
            name="approx. look sector",
            hovertemplate="Approximate GNSS-IR look sector<br>mean suspect risk=%{customdata:.2f}<extra></extra>",
            customdata=[risk] * len(sector_lons),
        )
    )
    for az in [180, 215, 250]:
        lon, lat = destination_point(site_lon, site_lat, az, 12)
        fig.add_trace(
            go.Scattergeo(
                lon=[site_lon, lon],
                lat=[site_lat, lat],
                mode="lines+text",
                text=["", f"{az}°"],
                textposition="top center",
                line=dict(color=ATLAS_BLUE, width=2),
                showlegend=False,
                hovertemplate=f"Representative azimuth {az}°<extra></extra>",
            )
        )
    fig.add_trace(
        go.Scattergeo(
            lon=[site_lon],
            lat=[site_lat],
            mode="markers+text",
            marker=dict(size=15, color="#B8322A", line=dict(color="white", width=2)),
            text=["PRECIPICE"],
            textposition="top right",
            name="station",
            hovertemplate="PRECIPICE GNSS-IR<br>lat=%{lat:.3f}<br>lon=%{lon:.3f}<extra></extra>",
        )
    )
    fig.update_geos(
        projection_type="azimuthal equal area",
        center=dict(lat=site_lat, lon=site_lon),
        lataxis_range=[site_lat - 0.48, site_lat + 0.48],
        lonaxis_range=[site_lon - 1.25, site_lon + 1.25],
        showland=True,
        landcolor="#E8ECEF",
        showocean=True,
        oceancolor="#DDEDF5",
        coastlinecolor="#52606D",
        showlakes=False,
        resolution=50,
    )
    fig.update_layout(
        title="Station-centered look-sector map",
        height=520,
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=0.02, xanchor="left", x=0.02),
    )
    return fig


def risk_calendar(preds: pd.DataFrame, pipeline: str) -> go.Figure:
    if preds.empty or "date" not in preds.columns:
        return go.Figure()
    df = preds[preds["pipeline"].eq(pipeline)].copy() if "pipeline" in preds.columns and pipeline else preds.copy()
    if df.empty:
        return go.Figure()
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["day"] = df["date"].dt.day
    cal = df.pivot_table(index="month", columns="day", values="suspect_probability", aggfunc="mean")
    fig = go.Figure(
        go.Heatmap(
            z=cal.values,
            x=cal.columns,
            y=cal.index,
            colorscale=[[0, "#F7FBFF"], [0.35, "#F8DDAA"], [0.7, "#F06B3D"], [1, "#8B1E2D"]],
            zmin=0,
            zmax=1,
            colorbar=dict(title="P(suspect)"),
            hovertemplate="month=%{y}<br>day=%{x}<br>P(suspect)=%{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Daily suspect-risk calendar",
        height=470,
        margin=dict(l=10, r=10, t=50, b=30),
        xaxis_title="day of month",
        yaxis_title="month",
    )
    return fig


def prediction_timeline(preds: pd.DataFrame, pipeline: str, start, end) -> go.Figure:
    if preds.empty:
        return go.Figure()
    df = preds[preds["pipeline"].eq(pipeline)].copy() if "pipeline" in preds.columns else preds.copy()
    df = date_filter(df, "date", start, end)
    if df.empty:
        return go.Figure()
    df["label_name"] = df["weak_label"].map(LABEL_NAMES).fillna(df["weak_label"].astype(str))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["suspect_probability"],
            mode="lines",
            name="P(suspect)",
            line=dict(color="#DD8452", width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>P(suspect)=%{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["weak_label"],
            mode="markers",
            name="proxy label",
            marker=dict(
                size=7,
                color=df["label_name"].map(LABEL_COLORS),
                symbol="circle",
            ),
            hovertext=df["label_name"],
            hovertemplate="%{x|%Y-%m-%d}<br>%{hovertext}<extra></extra>",
        )
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="0.5 threshold")
    fig.update_layout(
        title=f"Out-of-fold suspect probability: {pipeline}",
        yaxis_title="suspect probability / label",
        height=420,
        margin=dict(l=10, r=10, t=55, b=20),
    )
    return fig


def show_file_links(paths: Iterable[Path]) -> None:
    rows = []
    for p in paths:
        bundled = p.name in DATA_BUNDLE.get("images", {}) or p.name in DATA_BUNDLE.get("metadata", {}).get("bundled_artifacts", [])
        rows.append({"artifact": p.name, "exists locally": p.exists(), "bundled": bundled, "path": str(p)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


summary = bundled_json("week7_summary", W7_DIR / "week7_geography_progress_summary.json")
primary = bundled_csv("primary_v0v5_expanding_leaderboard", W6_DIR / "primary_v0v5_expanding_leaderboard.csv")
signal = bundled_csv("signal_only_expanding_leaderboard", W6_DIR / "signal_only_expanding_leaderboard.csv")
folds = bundled_csv("expanding_window_fold_summary", W6_DIR / "expanding_window_fold_summary.csv", parse_dates=("train_start", "train_end", "validation_start", "validation_end"))
preds = bundled_csv("signal_only_expanding_oof_predictions", W6_DIR / "signal_only_expanding_oof_predictions.csv", parse_dates=("date",))
feature_counts = bundled_csv("signal_only_feature_selection_counts", W6_DIR / "signal_only_feature_selection_counts.csv")
target_definition = bundled_csv("classification_target_definition", W6_DIR / "classification_target_definition.csv")
teacher_sets = bundled_csv("teacher_requested_feature_sets", W6_DIR / "teacher_requested_feature_sets.csv")
v5 = load_v5(str(V5_PATH))
if v5.empty:
    v5 = bundled_csv("v5_spline_light", V5_PATH, parse_dates=("datetime_utc",))

if not v5.empty:
    min_date = v5["datetime_utc"].min().date()
    max_date = v5["datetime_utc"].max().date()
else:
    min_date = pd.Timestamp("2024-08-20").date()
    max_date = pd.Timestamp("2025-10-26").date()

inject_style()

with st.sidebar:
    st.title("Field atlas")
    st.caption("PRECIPICE GNSS-IR review dashboard.")
    explanation_level = st.segmented_control("Explanation level", ["Beginner", "Detailed"], default="Beginner")
    date_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date
    model_options = signal["pipeline"].tolist() if not signal.empty and "pipeline" in signal.columns else []
    default_model = model_options[0] if model_options else ""
    selected_pipeline = st.selectbox("Signal-only pipeline", model_options, index=0 if model_options else None)
    st.caption("No model is retrained in this app.")

hero()

if DATA_BUNDLE.get("metadata"):
    st.caption(f"Dashboard bundle loaded: {DATA_BUNDLE['metadata'].get('created_from', 'local artifact bundle')}")

st.warning(
    "Current labels are QC-derived proxy labels. They support review and discussion, but they are not independent physical truth for ice/open water/wind.",
    icon=":material/warning:",
)

tabs = st.tabs([
    "Atlas overview",
    "Water-level product",
    "Validation & tide",
    "Label logic",
    "Signal features",
    "Model evidence",
    "Map review",
    "Glossary",
])

with tabs[0]:
    st.header("Atlas overview")
    st.write(
        "GNSS-IR uses reflected GNSS signals to estimate water level and diagnose surface/sensor conditions. "
        "The current classification task is framed as **reliable vs suspect review support**, not confirmed physical surface-state classification."
    )

    best_primary = primary.iloc[0] if not primary.empty else {}
    best_signal = signal.iloc[0] if not signal.empty else {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("V5 valid range", f"{summary.get('v5_valid_start', 'n/a')[:10]} to {summary.get('v5_valid_end', 'n/a')[:10]}", border=True)
    c2.metric("Arc files", summary.get("n_arc_files", "n/a"), border=True)
    c3.metric("Best V0-V5 BA", as_percent(best_primary.get("oof_balanced_accuracy") if hasattr(best_primary, "get") else None), border=True)
    c4.metric("Best signal-only BA", as_percent(best_signal.get("oof_balanced_accuracy") if hasattr(best_signal, "get") else None), border=True)

    st.subheader("Field interpretation guide")
    guide_cols = st.columns(4)
    with guide_cols[0]:
        atlas_card("1. Observe", "Start from the V5 water-level product and spline uncertainty before discussing models.")
    with guide_cols[1]:
        atlas_card("2. Validate", "Use pressure data only in the two overlap windows; do not treat it as full-year truth.")
    with guide_cols[2]:
        atlas_card("3. Diagnose", "Inspect arc availability and signal features to understand why a day is marked suspect.")
    with guide_cols[3]:
        atlas_card("4. Review spatially", "Map the station and look sector, not a continuous sea-ice surface.")

    st.subheader("Site-atlas preview")
    site_lat = float(summary.get("site_lat", 76.42))
    site_lon = float(summary.get("site_lon", -82.9))
    if not preds.empty and selected_pipeline:
        mean_risk_preview = preds[preds["pipeline"].eq(selected_pipeline)]["suspect_probability"].mean()
    else:
        mean_risk_preview = None
    col_a, col_b = st.columns([1.1, 0.9])
    with col_a:
        st.plotly_chart(look_sector_map(site_lon, site_lat, mean_risk_preview), width="stretch", key="overview_sector_map")
    with col_b:
        st.markdown(
            """
            <div class="map-note">
            <b>Cartographic interpretation:</b> the colored sector is a review footprint for the GNSS-IR reflection geometry.
            It is not a mapped sea-ice polygon, and it should not be interpreted as a continuous spatial prediction.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.subheader("Workflow")
        flow = pd.DataFrame(
            [
                ["1", "GNSS raw/SNR", "Reflected signal contains water/surface information."],
                ["2", "LSP arcs", "Arc-level retrievals summarize reflector-height candidates."],
                ["3", "QC/spline", "Bad retrievals and uncertain spline nodes are checked first."],
                ["4", "Validation/context", "Pressure validates overlap windows; tide gives phase context."],
                ["5", "Proxy labels", "Reliable/suspect labels organize review, not final physical truth."],
                ["6", "Models", "Expanding-window validation tests whether features help review support."],
            ],
            columns=["step", "stage", "plain-language meaning"],
        )
        st.dataframe(flow, hide_index=True, width="stretch")

with tabs[1]:
    st.header("Water-level product")
    st.write("Start with the data product before modeling. The key question is whether the water-level product and uncertainty look physically plausible.")
    display_figure(W7_DIR / "fig2_v5_product_overview.png", "Full-year V5 water level, coverage, and uncertainty context.")

    with st.expander("Interactive V5 zoom", expanded=True):
        vf = date_filter(v5, "datetime_utc", start_date, end_date)
        if vf.empty:
            st.warning("V5 data not available for this date range.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=vf["datetime_utc"], y=vf["water_level_m"], mode="lines", name="water_level_m", line=dict(color="#2B6CB0", width=1)))
            if "error_m" in vf.columns:
                fig.add_trace(go.Scatter(x=vf["datetime_utc"], y=vf["error_m"], mode="lines", name="error_m", yaxis="y2", line=dict(color="#C44E52", width=1)))
                fig.update_layout(yaxis2=dict(title="error_m", overlaying="y", side="right"))
            fig.update_layout(title="Zoomable V5 water level and uncertainty", yaxis_title="water level (m)", height=480)
            st.plotly_chart(fig, width="stretch", key="v5_zoom_chart")

    with st.expander("Beginner note: what is error_m?", expanded=explanation_level == "Beginner"):
        st.write(
            "`error_m` is an uncertainty/error estimate from the GNSS-IR spline product. "
            "It is not calculated by comparing with the pressure sensor at every time. "
            "In the benchmark-paper logic, large spline error is a direct QC warning."
        )

with tabs[2]:
    st.header("Validation and tide context")
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Pressure overlap")
        display_figure(W7_DIR / "fig3_pressure_overlap_context.png", "Pressure data are validation evidence only where the overlap exists.")
    with col_b:
        st.subheader("Tide context")
        display_figure(W7_DIR / "fig6_tide_context.png", "Tide prediction helps interpret phase and turning points; it is not pressure truth.")

    if not folds.empty:
        st.subheader("Expanding-window validation folds")
        st.dataframe(folds, hide_index=True, width="stretch")

with tabs[3]:
    st.header("Proxy label logic")
    st.write("This section explains the current target. It is useful, but it is not a final physical surface-state label.")
    display_figure(W7_DIR / "fig4_daily_proxy_label_diagnostics.png", "Proxy label timeline and monthly reason composition.")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.subheader("How labels were made")
        if not target_definition.empty:
            st.dataframe(target_definition, hide_index=True, width="stretch")
        else:
            st.write("Target-definition table is missing.")
    with col_b:
        st.subheader("Teacher-requested feature versions")
        if not teacher_sets.empty:
            st.dataframe(teacher_sets, hide_index=True, width="stretch")

    with st.expander("Important caveat", expanded=True):
        st.write(
            "A model trained on proxy labels can only learn to reproduce the proxy-label logic. "
            "It cannot prove true ice/open-water/wind classification until independent evidence is added."
        )

with tabs[4]:
    st.header("Signal features and feature selection")
    display_figure(W7_DIR / "fig4b_representative_signal_diagnostics_timeline.png", "Representative GNSS-IR signal features shown on separate axes.")
    display_figure(W7_DIR / "fig7_signal_feature_correlation_heatmap.png", "Correlated features explain why selected-feature pipelines are easier to interpret than all-input models.")

    col_a, col_b = st.columns([0.55, 0.45])
    with col_a:
        st.subheader("Feature selection counts")
        if not feature_counts.empty:
            fig = px.bar(
                feature_counts.sort_values("selected_fold_count", ascending=True),
                x="selected_fold_count",
                y="feature",
                orientation="h",
                title="How often each signal feature was selected across folds",
                labels={"selected_fold_count": "selected fold count", "feature": ""},
            )
            fig.update_layout(height=max(360, 24 * len(feature_counts)))
            st.plotly_chart(fig, width="stretch", key="feature_selection_counts_chart")
        else:
            st.info("Feature selection count output not found.")
    with col_b:
        st.subheader("Why not use everything blindly?")
        st.write(
            "Many GNSS-IR variables are related because they come from the same SNR/LSP retrieval process. "
            "Feature selection helps reduce duplicated information and makes the model easier to explain."
        )
        st.write("The full signal-only model is still useful as a stress test, but not necessarily the best interpretation model.")

with tabs[5]:
    st.header("Model evidence")
    st.write("The primary result uses monthly expanding-window validation: earlier months train, the next unseen month validates.")

    model_view = st.segmented_control("Model result group", ["V0-V5 compact features", "Signal-only optimized"], default="V0-V5 compact features")
    if model_view == "V0-V5 compact features":
        active = primary.copy()
        label_col = "version"
        title = "V0-V5 expanding-window comparison"
    else:
        active = signal.copy()
        label_col = "pipeline"
        title = "Signal-only expanding-window comparison"

    model_key = "v0v5" if model_view == "V0-V5 compact features" else "signal_only"
    st.plotly_chart(leaderboard_plot(active, label_col, title), width="stretch", key=f"{model_key}_leaderboard")
    st.plotly_chart(metrics_heatmap(active, label_col, "All key metrics"), width="stretch", key=f"{model_key}_metrics_heatmap")
    st.plotly_chart(overfit_plot(active, label_col, "Overfitting check: train vs validation"), width="stretch", key=f"{model_key}_overfit")

    st.subheader("Sortable metrics table")
    st.dataframe(nice_metrics_table(active), hide_index=True, width="stretch")

    if selected_pipeline:
        st.subheader("Prediction timeline")
        safe_pipeline = selected_pipeline.replace(" ", "_").replace("/", "_")
        st.plotly_chart(prediction_timeline(preds, selected_pipeline, start_date, end_date), width="stretch", key=f"prediction_timeline_{safe_pipeline}")

    with st.expander("How to read the scores", expanded=explanation_level == "Beginner"):
        st.write(
            "**Balanced accuracy** averages reliable-day recall and suspect-day recall, so it is better than raw accuracy when classes are imbalanced. "
            "**Recall suspect** asks how many suspect days were caught. "
            "**Overfit gap** compares train and validation performance; a large gap means the model may not generalize."
        )

with tabs[6]:
    st.header("Map review")
    st.write("These views make the model output geographically understandable without pretending that one station can map the whole bay.")
    site_lat = float(summary.get("site_lat", 76.42))
    site_lon = float(summary.get("site_lon", -82.9))
    risk_for_map = preds[preds["pipeline"].eq(selected_pipeline)]["suspect_probability"].mean() if selected_pipeline and not preds.empty else None
    col_a, col_b = st.columns([0.52, 0.48])
    with col_a:
        st.plotly_chart(look_sector_map(site_lon, site_lat, risk_for_map), width="stretch", key="map_review_sector_map")
        st.plotly_chart(risk_calendar(preds, selected_pipeline), width="stretch", key="map_review_risk_calendar")
    with col_b:
        display_figure(W7_DIR / "fig1b_cartopy_site_context.png", "Arctic projection and approximate look-sector context.")
        display_figure(W7_DIR / "fig1_site_map.png", "Basic site location.")
        display_figure(W7_DIR / "fig8_spatiotemporal_risk_map_calendar.png", "Station/look-sector risk plus daily risk calendar.")

    st.markdown(
        """
        <div class="map-note">
        <b>Map-reading rule:</b> treat the sector and calendar as a review guide for where/when to inspect Sentinel-2,
        SAR, ERA5 weather, or field notes. Do not report it as a gridded sea-ice classification map.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Saved interactive HTML dashboards")
    show_file_links(
        [
            W7_DIR / "interactive_v5_observing_dashboard.html",
            W7_DIR / "interactive_label_review_dashboard.html",
            W7_DIR / "interactive_model_dashboard.html",
            W7_DIR / "interactive_spatiotemporal_risk_dashboard.html",
            W7_DIR / "interactive_prediction_sector_timeslider.html",
            W7_DIR / "interactive_tide_context_dashboard.html",
        ]
    )
    st.caption("These are saved notebook artifacts. The Streamlit app recreates the most important interactions directly where possible.")

with tabs[7]:
    st.header("Glossary")
    glossary = pd.DataFrame(
        [
            ["water_level_m", "Spline-fitted GNSS-IR water-level estimate.", "Core observing product."],
            ["error_m", "Spline uncertainty/error estimate.", "Direct QC context; high values warn that retrieval is less trustworthy."],
            ["arc_count", "Number of daily arc-level retrievals.", "Low availability can indicate weak coverage or retrieval issues."],
            ["ms", "Mean SNR signal strength summary.", "Signal context; useful benchmark feature, not the label definition."],
            ["sp", "Lomb-Scargle spectral peak strength.", "Stronger/cleaner peak can indicate a clearer reflector-height retrieval."],
            ["ptn", "Peak-to-noise style summary.", "Related to spectral quality; often correlated with sp/clr/pr."],
            ["clr", "Confidence level of retrieval.", "Low confidence is a quality warning in paper-style QC."],
            ["rh", "Reflector height estimate.", "Intermediate retrieval related to water/surface height."],
            ["df", "Damping-related signal feature.", "May respond to surface roughness, ice, or geometry; interpret cautiously."],
            ["af", "Area-factor signal feature.", "Shape/area summary from the retrieval stage."],
            ["vs", "Variance of detrended SNR.", "Signal variability; can rise with rough/noisy conditions."],
            ["proxy label", "Rule/QC-derived reliable-vs-suspect label.", "Useful for review, not final physical truth."],
            ["OOF", "Out-of-fold prediction.", "Prediction for a validation month not used to train that fold."],
        ],
        columns=["term", "plain-language meaning", "why it matters"],
    )
    st.dataframe(glossary, hide_index=True, width="stretch")

    st.subheader("Source artifacts")
    show_file_links(
        [
            V5_PATH,
            W6_DIR / "primary_v0v5_expanding_leaderboard.csv",
            W6_DIR / "signal_only_expanding_leaderboard.csv",
            W6_DIR / "signal_only_expanding_oof_predictions.csv",
            W7_DIR / "week7_geography_progress_summary.json",
        ]
    )
