from __future__ import annotations

import io
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
BUNDLE_PATH = PUBLIC_BUNDLE_PATH

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
            padding: 2.3rem 2.5rem;
            border-radius: 22px;
            background:
                radial-gradient(circle at 82% 18%, rgba(234,243,248,.24), transparent 26%),
                linear-gradient(118deg, #0A1E2E 0%, #123F57 52%, #1F6F8B 100%);
            color: white;
            box-shadow: 0 20px 45px rgba(18,38,58,.20);
            margin-bottom: 1rem;
        }
        .review-hero h1 {margin: .18rem 0 0; font-size: 2.55rem; line-height: 1.08; letter-spacing: -.035em; max-width: 900px;}
        .review-hero p {margin: .85rem 0 0; color: #EAF3F8; max-width: 850px; font-size: 1.05rem; line-height: 1.55;}
        .eyebrow {font-size: .78rem; text-transform: uppercase; letter-spacing: .12em; opacity: .82;}
        .chapter-label {color: #1F6F8B; font-size: .76rem; font-weight: 700; letter-spacing: .13em; text-transform: uppercase; margin: 2rem 0 .25rem;}
        .chapter-title {color: #12263A; font-size: 2rem; font-weight: 700; line-height: 1.15; letter-spacing: -.025em; margin: 0 0 .5rem;}
        .chapter-copy {color: #52606D; max-width: 860px; font-size: 1rem; line-height: 1.65; margin-bottom: 1.1rem;}
        .story-card {background: linear-gradient(180deg,#FFFFFF,#F7FAFC); border: 1px solid #D6E4EB; border-radius: 16px; padding: 1.05rem 1.1rem; min-height: 170px; box-shadow: 0 8px 22px rgba(31,111,139,.06);}
        .story-card .step {color: #DD8452; font-size: .76rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase;}
        .story-card h3 {color: #12263A; margin: .45rem 0 .35rem; font-size: 1.08rem;}
        .story-card p {color: #52606D; line-height: 1.5; font-size: .92rem;}
        .safety-strip {background: #0F3042; color: white; border-radius: 18px; padding: 1.3rem 1.45rem; margin: .8rem 0;}
        .safety-strip h3 {margin: 0 0 .35rem; color: white;}
        .safety-strip p {margin: 0; color: #DDEDF5; line-height: 1.55;}
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


def arctic_context_map(site_lon: float, site_lat: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scattergeo(
        lon=[site_lon], lat=[site_lat], mode="markers+text",
        marker=dict(size=16, color=ORANGE, line=dict(color="white", width=2)),
        text=["PRECIPICE · Grise Fjord"], textposition="top right",
        hovertemplate="High Canadian Arctic<br>Coarsened public display location<extra></extra>",
        name="field station",
    ))
    fig.add_trace(go.Scattergeo(
        lon=[-73.57, site_lon], lat=[45.50, site_lat], mode="lines",
        line=dict(color="rgba(31,111,139,.45)", width=1.4, dash="dot"),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scattergeo(
        lon=[-73.57], lat=[45.50], mode="markers+text",
        marker=dict(size=7, color=BLUE), text=["Montréal"], textposition="bottom right",
        hoverinfo="skip", showlegend=False,
    ))
    fig.update_geos(
        projection_type="orthographic",
        projection_rotation=dict(lon=-75, lat=67, roll=0),
        showland=True, landcolor="#ECE9E1", showocean=True, oceancolor="#D9EBF4",
        showlakes=True, lakecolor="#D9EBF4", coastlinecolor="#6D7C84",
        showcountries=True, countrycolor="rgba(109,124,132,.35)", resolution=50,
    )
    fig.update_layout(title="A field station near the top of the world", margin=dict(l=0, r=0, t=55, b=0), showlegend=False)
    return style_figure(fig, 540)


def detective_case_plot(qc: pd.DataFrame, target_date: pd.Timestamp) -> go.Figure:
    window = qc[qc["date"].between(target_date - pd.Timedelta(days=10), target_date + pd.Timedelta(days=10))]
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=.12,
        subplot_titles=("Daily water-level mean", "14-day tidal-range context"),
        row_heights=[.58, .42],
    )
    fig.add_trace(go.Scatter(x=window["date"], y=window["water_level_mean"], mode="lines+markers", name="water level", line=dict(color=MID_BLUE, width=2), marker=dict(size=5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=window["date"], y=window["water_level_range_14d"], mode="lines+markers", name="14-day range", line=dict(color=GREEN, width=2), marker=dict(size=5)), row=2, col=1)
    fig.add_vline(x=target_date, line_color=ORANGE, line_width=2, annotation_text="case date", row="all", col=1)
    fig.add_hline(y=1.232, line_dash="dash", line_color=RED, annotation_text="review threshold", row=2, col=1)
    fig.update_yaxes(title_text="m", row=1, col=1)
    fig.update_yaxes(title_text="m", row=2, col=1)
    fig.update_layout(title="Signal detective: what would you flag?", hovermode="x unified", showlegend=False)
    return style_figure(fig, 500)


def next_detective_case() -> None:
    st.session_state.detective_round += 1
    st.session_state.detective_revealed = False
    st.session_state.pop("detective_guess", None)


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


def tide_reflection_animation(six_hour: pd.DataFrame, selected_date) -> go.Figure:
    day_start = pd.Timestamp(selected_date).normalize()
    hourly_index = pd.date_range(day_start, day_start + pd.Timedelta(hours=23), freq="1h")
    interpolation_index = pd.date_range(day_start, day_start + pd.Timedelta(days=1), freq="1h")
    source = six_hour[
        six_hour["datetime_utc"].between(day_start, day_start + pd.Timedelta(days=1), inclusive="both")
    ].set_index("datetime_utc")[["tide_anomaly_m", "gnss_anomaly_m"]]
    hourly = (
        source.reindex(source.index.union(interpolation_index))
        .sort_index()
        .interpolate(method="time", limit_direction="both")
        .reindex(hourly_index)
    )
    hourly["tide_rate"] = hourly["tide_anomaly_m"].diff().fillna(0)
    hours = np.arange(24)

    def phase_label(rate: float) -> str:
        if rate > 0.03:
            return "rising tide"
        if rate < -0.03:
            return "falling tide"
        return "near a turning point"

    def water_y(tide_value: float) -> float:
        return 2.15 + 0.60 * float(tide_value)

    initial_tide = float(hourly.iloc[0]["tide_anomaly_m"])
    initial_gnss = float(hourly.iloc[0]["gnss_anomaly_m"]) if pd.notna(hourly.iloc[0]["gnss_anomaly_m"]) else np.nan
    initial_water = water_y(initial_tide)

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[.50, .50],
        vertical_spacing=.12,
        subplot_titles=("Reflection cartoon", "Tide and GNSS-IR anomalies"),
    )
    fig.add_trace(go.Scatter(
        x=[0, 10, 10, 0, 0], y=[0, 0, initial_water, initial_water, 0],
        mode="lines", fill="toself", fillcolor="rgba(88,153,186,.42)",
        line=dict(color=BLUE, width=2), name="water surface", hoverinfo="skip",
        showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[0, 2.4, 2.4, 0, 0], y=[0, 0, 3.45, 4.05, 0],
        mode="lines", fill="toself", fillcolor="#E7E1D4",
        line=dict(color="#7A6F61", width=1.3), name="shore", hoverinfo="skip", showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[8.3, 4.5, 1.65], y=[8.4, initial_water, 3.55],
        mode="lines", line=dict(color=ORANGE, width=3), name="reflected signal",
        hoverinfo="skip", showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[8.3], y=[8.4], mode="markers+text", text=["GNSS satellite"], textposition="top center",
        marker=dict(size=18, color=ORANGE, symbol="diamond", line=dict(color="white", width=1.5)),
        name="satellite", hovertemplate="GNSS satellite broadcast<extra></extra>", showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[1.65], y=[3.55], mode="markers+text", text=["shore receiver"], textposition="top center",
        marker=dict(size=16, color=RED, symbol="triangle-up", line=dict(color="white", width=1.5)),
        name="receiver", hovertemplate="PRECIPICE receiver<extra></extra>", showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[4.5], y=[initial_water], mode="markers+text",
        text=[f"{initial_tide:+.2f} m<br>{phase_label(float(hourly.iloc[0]['tide_rate']))}"],
        textposition="bottom center", marker=dict(size=11, color="#FFFFFF", line=dict(color=ORANGE, width=3)),
        name="reflection point", hoverinfo="skip", showlegend=False,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=hours, y=hourly["tide_anomaly_m"], mode="lines", name="predicted tide anomaly",
        line=dict(color=ORANGE, width=2.5), hovertemplate="hour=%{x}:00<br>tide=%{y:+.2f} m<extra></extra>",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=hours, y=hourly["gnss_anomaly_m"], mode="lines", name="GNSS-IR anomaly",
        line=dict(color=MID_BLUE, width=2), hovertemplate="hour=%{x}:00<br>GNSS=%{y:+.2f} m<extra></extra>",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=[0], y=[initial_tide], mode="markers+text", text=[phase_label(float(hourly.iloc[0]["tide_rate"]))],
        textposition="top right", marker=dict(size=13, color=ORANGE, line=dict(color="white", width=1.5)),
        name="current tide", hoverinfo="skip", showlegend=False,
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=[0], y=[initial_gnss], mode="markers", marker=dict(size=11, color=MID_BLUE, line=dict(color="white", width=1.5)),
        name="current GNSS-IR", hoverinfo="skip", showlegend=False,
    ), row=2, col=1)

    frames = []
    for hour, (_, row) in enumerate(hourly.iterrows()):
        tide_value = float(row["tide_anomaly_m"])
        gnss_value = float(row["gnss_anomaly_m"]) if pd.notna(row["gnss_anomaly_m"]) else np.nan
        surface = water_y(tide_value)
        phase = phase_label(float(row["tide_rate"]))
        frames.append(go.Frame(
            name=f"{hour:02d}:00",
            data=[
                go.Scatter(x=[0, 10, 10, 0, 0], y=[0, 0, surface, surface, 0]),
                go.Scatter(x=[8.3, 4.5, 1.65], y=[8.4, surface, 3.55]),
                go.Scatter(x=[4.5], y=[surface], text=[f"{tide_value:+.2f} m<br>{phase}"]),
                go.Scatter(x=[hour], y=[tide_value], text=[phase]),
                go.Scatter(x=[hour], y=[gnss_value]),
            ],
            traces=[0, 2, 5, 8, 9],
        ))
    fig.frames = frames

    slider_steps = [
        dict(
            label=frame.name,
            method="animate",
            args=[[frame.name], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}],
        )
        for frame in frames
    ]
    fig.update_layout(
        height=880,
        updatemenus=[dict(
            type="buttons", direction="left", x=.98, y=-.13, xanchor="right", yanchor="top",
            buttons=[
                dict(label="Play 24 hours", method="animate", args=[None, {"fromcurrent": True, "frame": {"duration": 420, "redraw": True}, "transition": {"duration": 150}}]),
                dict(label="Pause", method="animate", args=[[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}}]),
            ],
        )],
        sliders=[dict(
            active=0, currentvalue={"prefix": "Hour: "}, pad={"t": 48},
            steps=slider_steps, x=.08, len=.84, y=-.07,
        )],
    )
    fig.update_xaxes(visible=False, range=[0, 10], row=1, col=1)
    fig.update_yaxes(visible=False, range=[0, 9.4], row=1, col=1)
    fig.update_xaxes(title_text="UTC hour", range=[0, 23], dtick=3, row=2, col=1)
    fig.update_yaxes(title_text="median-centered anomaly (m)", range=[-1.35, 1.35], row=2, col=1)
    fig = style_figure(fig, 880)
    fig.update_layout(
        margin=dict(l=35, r=25, t=105, b=95),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=.5),
    )
    return fig


def ice_damping_animation(damping_percent: int) -> go.Figure:
    """Conceptual exhibit: ice influence compresses the retrieved GNSS-IR range."""
    hours = np.arange(24)
    open_water_tide = np.sin(2 * np.pi * (hours - 2) / 12.42)
    retrieval_scale = 1 - float(damping_percent) / 100
    affected_retrieval = retrieval_scale * open_water_tide

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[.48, .52],
        vertical_spacing=.12,
        subplot_titles=("Reflection cartoon", "Expected vs retrieved amplitude"),
    )
    initial_surface = 2.35 + .42 * open_water_tide[0]
    fig.add_trace(go.Scatter(
        x=[0, 10, 10, 0, 0], y=[0, 0, initial_surface, initial_surface, 0],
        mode="lines", fill="toself", fillcolor="rgba(88,153,186,.48)",
        line=dict(color=BLUE, width=2), name="water", hoverinfo="skip",
        showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=np.linspace(.4, 9.6, 14),
        y=np.full(14, initial_surface + .08),
        mode="lines+markers", line=dict(color="#B8D8E8", width=10),
        marker=dict(size=8, color="#F4FAFD", line=dict(color="#8DBDD2", width=1)),
        name="ice-influenced surface", hoverinfo="skip", showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[8.3, 5.0, 1.6], y=[8.0, initial_surface + .08, 3.7],
        mode="lines", line=dict(color=ORANGE, width=3, dash="dot"),
        name="disturbed reflection path", hoverinfo="skip", showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[8.3, 1.6], y=[8.0, 3.7], mode="markers+text",
        text=["satellite", "receiver"], textposition=["top center", "top center"],
        marker=dict(size=[18, 16], color=[ORANGE, RED], symbol=["diamond", "triangle-up"], line=dict(color="white", width=1.5)),
        showlegend=False, hoverinfo="skip",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[5.0], y=[initial_surface + .08], mode="markers+text",
        text=[f"retrieved range: {retrieval_scale:.0%}"], textposition="bottom center",
        marker=dict(size=13, color="white", line=dict(color=PURPLE, width=3)),
        name="retrieval response", hoverinfo="skip", showlegend=False,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=hours, y=open_water_tide, mode="lines", name="open-water tide pattern",
        line=dict(color=ORANGE, width=2.6), hovertemplate="hour=%{x}:00<br>expected=%{y:+.2f}<extra></extra>",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=hours, y=affected_retrieval, mode="lines", name="ice-affected GNSS-IR retrieval",
        line=dict(color=MID_BLUE, width=2.6), hovertemplate="hour=%{x}:00<br>retrieved=%{y:+.2f}<extra></extra>",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=[0], y=[open_water_tide[0]], mode="markers", marker=dict(size=13, color=ORANGE, line=dict(color="white", width=1.5)),
        name="current expected", hoverinfo="skip", showlegend=False,
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=[0], y=[affected_retrieval[0]], mode="markers", marker=dict(size=13, color=MID_BLUE, line=dict(color="white", width=1.5)),
        name="current retrieval", hoverinfo="skip", showlegend=False,
    ), row=2, col=1)

    frames = []
    ice_x = np.linspace(.4, 9.6, 14)
    for hour in hours:
        surface = 2.35 + .42 * open_water_tide[hour]
        frames.append(go.Frame(
            name=f"{hour:02d}:00",
            data=[
                go.Scatter(x=[0, 10, 10, 0, 0], y=[0, 0, surface, surface, 0]),
                go.Scatter(x=ice_x, y=np.full(len(ice_x), surface + .08)),
                go.Scatter(x=[8.3, 5.0, 1.6], y=[8.0, surface + .08, 3.7]),
                go.Scatter(x=[5.0], y=[surface + .08], text=[f"retrieved range: {retrieval_scale:.0%}"]),
                go.Scatter(x=[hour], y=[open_water_tide[hour]]),
                go.Scatter(x=[hour], y=[affected_retrieval[hour]]),
            ],
            traces=[0, 1, 2, 4, 7, 8],
        ))
    fig.frames = frames
    fig.update_layout(
        height=850,
        updatemenus=[dict(
            type="buttons", direction="left", x=.98, y=-.15, xanchor="right", yanchor="top",
            buttons=[
                dict(label="Play", method="animate", args=[None, {"fromcurrent": True, "frame": {"duration": 380, "redraw": True}, "transition": {"duration": 120}}]),
                dict(label="Pause", method="animate", args=[[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}}]),
            ],
        )],
        sliders=[dict(
            active=0, currentvalue={"prefix": "Hour: "}, pad={"t": 48}, x=.08, len=.84, y=-.08,
            steps=[dict(label=f.name, method="animate", args=[[f.name], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}]) for f in frames],
        )],
    )
    fig.update_xaxes(visible=False, range=[0, 10], row=1, col=1)
    fig.update_yaxes(visible=False, range=[0, 9], row=1, col=1)
    fig.update_xaxes(title_text="UTC hour", range=[0, 23], dtick=3, row=2, col=1)
    fig.update_yaxes(title_text="normalized amplitude", range=[-1.25, 1.25], row=2, col=1)
    fig = style_figure(fig, 850)
    fig.update_layout(
        margin=dict(l=35, r=25, t=105, b=95),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=.5),
    )
    return fig


def qc_sorting_animation() -> go.Figure:
    """Illustrative tokens explain the two QC levels and required spline refit."""
    n_tokens = 36
    stages = [
        ("1 · Raw arcs", 36, "Many reflection candidates<br>arrive from Level 1."),
        ("2 · PTN / CLR screen", 24, "A paper-style sensitivity screen<br>retains qualifying arcs."),
        ("3 · Refit the spline", 12, "Filtered arcs must be used<br>to fit a new Level-2 spline."),
        ("4 · error_m screen", 10, "The Level-2 uncertainty gate<br>retains review-ready spline points."),
    ]
    stage_axis_labels = ["Raw<br>arcs", "PTN/CLR<br>gate", "Refit<br>spline", "error_m<br>gate"]

    def token_state(stage_index: int) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
        rng = np.random.default_rng(17 + stage_index)
        retained = stages[stage_index][1]
        x = stage_index + rng.normal(0, .10, n_tokens)
        y = 1.2 + (np.arange(n_tokens) % 6) * .42 + rng.normal(0, .025, n_tokens)
        colors = [GREEN if index < retained else "#CBD5E1" for index in range(n_tokens)]
        symbols = ["diamond" if stage_index == 2 and index < retained else "circle" for index in range(n_tokens)]
        return x, y, colors, symbols

    x0, y0, colors0, symbols0 = token_state(0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x0, y=y0, mode="markers", name="illustrative observations",
        marker=dict(size=13, color=colors0, symbol=symbols0, line=dict(color="white", width=1)),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[1.5], y=[4.35], mode="text", text=[stages[0][2]],
        textfont=dict(size=17, color=NAVY), showlegend=False, hoverinfo="skip",
    ))
    for stage_index, (_, _, _) in enumerate(stages):
        fig.add_annotation(x=stage_index, y=.55, text=stage_axis_labels[stage_index], showarrow=False, font=dict(size=11, color=NAVY))
        if stage_index < len(stages) - 1:
            fig.add_annotation(x=stage_index + .55, y=2.25, ax=stage_index + .25, ay=2.25, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=3, arrowwidth=2, arrowcolor="#94A3B8", text="")

    frames = []
    for stage_index, (label, retained, message) in enumerate(stages):
        x, y, colors, symbols = token_state(stage_index)
        frames.append(go.Frame(
            name=label,
            data=[
                go.Scatter(x=x, y=y, marker=dict(size=13, color=colors, symbol=symbols, line=dict(color="white", width=1))),
                go.Scatter(x=[1.5], y=[4.35], text=[f"{message}<br><b>{retained} of 36 teaching tokens remain visible</b>"]),
            ],
            traces=[0, 1],
        ))
    fig.frames = frames
    fig.update_layout(
        title="QC sorting line",
        height=540,
        xaxis=dict(range=[-.55, 3.55], visible=False),
        yaxis=dict(range=[.25, 4.75], visible=False),
        updatemenus=[dict(
            type="buttons", direction="left", x=.98, y=-.17, xanchor="right", yanchor="top",
            buttons=[
                dict(label="Run the sorting line", method="animate", args=[None, {"fromcurrent": True, "frame": {"duration": 1000, "redraw": True}, "transition": {"duration": 280}}]),
                dict(label="Pause", method="animate", args=[[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}}]),
            ],
        )],
        sliders=[dict(
            active=0, currentvalue={"visible": False}, pad={"t": 48}, x=.08, len=.84, y=-.08,
            steps=[dict(label=f.name, method="animate", args=[[f.name], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}]) for f in frames],
        )],
    )
    return style_figure(fig, 540)


def expanding_window_animation(folds: pd.DataFrame) -> go.Figure:
    """Animate the real public fold summary as a chronological training train."""
    ordered = folds.sort_values("validation_start").reset_index(drop=True)
    start_month = ordered["train_start"].min().to_period("M")
    end_month = ordered["validation_end"].max().to_period("M")
    months = pd.period_range(start_month, end_month, freq="M")
    month_labels = [period.strftime("%b<br>%y") if index % 2 == 0 or index == len(months) - 1 else "" for index, period in enumerate(months)]
    month_hover = [period.strftime("%B %Y") for period in months]

    def fold_state(row: pd.Series) -> tuple[list[str], list[str], str]:
        train_end = row["train_end"].to_period("M")
        validation_month = row["validation_start"].to_period("M")
        colors = [MID_BLUE if period <= train_end else ORANGE if period == validation_month else "#E2E8F0" for period in months]
        roles = ["past training month" if period <= train_end else "unseen validation month" if period == validation_month else "future month" for period in months]
        message = f"Train through {train_end.strftime('%b %Y')}<br>test {validation_month.strftime('%b %Y')}"
        return colors, roles, message

    colors0, roles0, message0 = fold_state(ordered.iloc[0])
    custom0 = [[month_hover[index], roles0[index]] for index in range(len(months))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.arange(len(months)), y=np.ones(len(months)), mode="markers+text",
        text=month_labels, textposition="bottom center", customdata=custom0,
        textfont=dict(size=10, color=NAVY),
        marker=dict(size=42, color=colors0, symbol="square", line=dict(color="white", width=2)),
        hovertemplate="%{customdata[0]}<br>%{customdata[1]}<extra></extra>", name="calendar months", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[(len(months) - 1) / 2], y=[2.25], mode="text", text=[message0],
        textfont=dict(size=20, color=NAVY), showlegend=False, hoverinfo="skip",
    ))
    frames = []
    for index, row in ordered.iterrows():
        colors, roles, message = fold_state(row)
        custom = [[month_hover[month_index], roles[month_index]] for month_index in range(len(months))]
        validation_label = row["validation_start"].strftime("%Y-%m")
        frames.append(go.Frame(
            name=f"Fold {index + 1} · {validation_label}",
            data=[
                go.Scatter(marker=dict(size=42, color=colors, symbol="square", line=dict(color="white", width=2)), customdata=custom),
                go.Scatter(text=[f"{message}<br><span style='font-size:13px'>n train = {int(row['n_train']):,} · n validation = {int(row['n_validation']):,}</span>"]),
            ],
            traces=[0, 1],
        ))
    fig.frames = frames
    fig.update_layout(
        title="Time-aware validation train",
        height=470,
        xaxis=dict(range=[-.8, len(months) - .2], visible=False),
        yaxis=dict(range=[.25, 2.75], visible=False),
        updatemenus=[dict(
            type="buttons", direction="left", x=.98, y=-.20, xanchor="right", yanchor="top",
            buttons=[
                dict(label="Run all folds", method="animate", args=[None, {"fromcurrent": True, "frame": {"duration": 850, "redraw": True}, "transition": {"duration": 250}}]),
                dict(label="Pause", method="animate", args=[[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}}]),
            ],
        )],
        sliders=[dict(
            active=0, currentvalue={"visible": False}, pad={"t": 48}, x=.08, len=.84, y=-.08,
            steps=[dict(label=f.name, method="animate", args=[[f.name], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}]) for f in frames],
        )],
    )
    return style_figure(fig, 470)


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

st.session_state.setdefault("detective_round", 0)
st.session_state.setdefault("detective_revealed", False)

inject_style()

with st.sidebar:
    st.title("PRECIPICE")
    st.caption("An interactive Arctic field exhibit")
    st.markdown("**Exhibit guide**")
    st.caption("Overview — story and map\n\nLearn — animated science gallery\n\nObserve — water and tide\n\nDiagnose — QC evidence\n\nModels — validation results")
    st.markdown("**Notebook lineage**")
    st.caption("Week 5 diagnostic comparison\n\nWeek 6 expanding-window evidence\n\nWeek 7 geographic interpretation")
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
      <div class="eyebrow">PRECIPICE · Interactive Arctic field exhibit</div>
      <h1>Listening to Arctic water with satellite reflections</h1>
      <p>A guided journey from a remote receiver near Grise Fjord to the reflected GNSS signals, quality-control clues, and time-aware models used to review the water-level record.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

view = st.segmented_control(
    "Workspace",
    ["Overview", "Learn", "Observe", "Diagnose", "Models"],
    default="Overview",
    label_visibility="collapsed",
    key="workspace_view",
)

site_lat = float(summary.get("site_lat", 76.4))
site_lon = float(summary.get("site_lon", -82.9))

if view == "Overview":
    st.markdown("<div class='chapter-label'>Chapter 01 · Place</div><div class='chapter-title'>First, find the field station</div><div class='chapter-copy'>PRECIPICE sits in the High Canadian Arctic. The left map gives continental context; the right map shows the approximate direction in which the shore receiver observes reflected satellite signals over water.</div>", unsafe_allow_html=True)
    map_left, map_right = st.columns([.92, 1.08], gap="large")
    with map_left:
        show_chart(arctic_context_map(site_lon, site_lat), "arctic_context_map")
    with map_right:
        show_chart(look_sector_map(site_lon, site_lat, None), "local_context_map")

    best_primary = primary.sort_values("oof_balanced_accuracy", ascending=False).iloc[0] if not primary.empty else None
    best_signal = signal.sort_values("oof_balanced_accuracy", ascending=False).iloc[0] if not signal.empty else None
    with st.container(horizontal=True):
        st.metric("Field record", f"{min_date:%b %Y} – {max_date:%b %Y}", border=True)
        st.metric("Daily review points", f"{len(qc):,}", border=True)
        st.metric("Level-1 files summarized", f"{summary.get('n_arc_files', 'n/a')}", border=True)
        st.metric("Tide context", "2025 only", border=True)

    st.markdown("<div class='chapter-label'>Chapter 02 · Principle</div><div class='chapter-title'>How can a satellite become a water-level instrument?</div><div class='chapter-copy'>A useful analogy is an echo: the direct and reflected signals arrive by different paths. GNSS-IR studies that path difference. Unlike an acoustic echo, the receiver is measuring radio-wave interference, not sound.</div>", unsafe_allow_html=True)
    story_cols = st.columns(3, gap="large")
    story = [
        ("01 · Broadcast", "Satellites provide the signal", "GNSS satellites continuously transmit radio signals. The receiver records both the direct path and reflections arriving from the water-facing sector."),
        ("02 · Reflection", "The surface acts like a moving mirror", "As water level and surface conditions change, the extra reflected path changes too. Ice, roughness, and geometry can disturb the pattern."),
        ("03 · Retrieval", "Interference becomes a height estimate", "Spectral analysis extracts reflector-height candidates. QC and spline fitting then turn many arcs into a continuous observing product."),
    ]
    for column, (step, title, body) in zip(story_cols, story):
        with column:
            st.markdown(f"<div class='story-card'><div class='step'>{step}</div><h3>{title}</h3><p>{body}</p></div>", unsafe_allow_html=True)

    st.markdown("<div class='chapter-label'>Chapter 03 · Interactive</div><div class='chapter-title'>Try the signal-detective challenge</div><div class='chapter-copy'>Look at a short section of the public, daily-resolution record. Decide whether the highlighted day looks ordinary or deserves closer QC review. Then reveal what the proxy rules and out-of-fold model reported.</div>", unsafe_allow_html=True)

    if predictions.empty or not default_pipeline:
        st.warning("The detective challenge needs OOF predictions, which are unavailable in this bundle.")
    else:
        challenge_predictions = predictions[predictions["pipeline"].eq(default_pipeline)].drop_duplicates("date", keep="last")
        challenge = qc.merge(challenge_predictions[["date", "suspect_probability", "prediction"]], on="date", how="inner")
        low_cases = challenge[challenge["suspect_probability"] < .35]
        high_cases = challenge[challenge["suspect_probability"] > .65]
        case_pool = high_cases if st.session_state.detective_round % 2 else low_cases
        case_position = (st.session_state.detective_round * 17 + 5) % max(len(case_pool), 1)
        case = case_pool.iloc[case_position] if not case_pool.empty else challenge.iloc[0]
        case_date = pd.Timestamp(case["date"])

        game_left, game_right = st.columns([1.45, .75], gap="large")
        with game_left:
            show_chart(detective_case_plot(qc, case_date), "detective_case_plot")
        with game_right:
            with st.container(border=True):
                st.subheader(f"Case {st.session_state.detective_round + 1}")
                st.caption(f"Highlighted date: {case_date:%Y-%m-%d}")
                guess = st.segmented_control(
                    "Your review decision",
                    ["Looks reliable", "Needs review"],
                    default=None,
                    key="detective_guess",
                )
                if st.button("Reveal evidence", icon=":material/visibility:", disabled=guess is None, width="stretch"):
                    st.session_state.detective_revealed = True

                if st.session_state.detective_revealed:
                    predicted_suspect = int(case["prediction"]) == 1
                    guessed_suspect = guess == "Needs review"
                    if guessed_suspect == predicted_suspect:
                        st.success("You matched the model's review decision.", icon=":material/check_circle:")
                    else:
                        st.info("You disagreed with the model — inspect the clues below.", icon=":material/search:")
                    reason_columns = [
                        "seasonal_damped_period_proxy", "low_arc_availability_suspect",
                        "low_v5_coverage_suspect", "low_tide_range_suspect",
                    ]
                    reasons = [name.replace("_suspect", "").replace("_proxy", "").replace("_", " ") for name in reason_columns if bool(case[name])]
                    st.metric("OOF suspect probability", as_percent(case["suspect_probability"]), border=True)
                    st.write("**Triggered review clues:** " + (", ".join(reasons) if reasons else "none"))
                    st.caption("This compares your judgment with a QC proxy/model, not with independently observed environmental truth.")
                    st.button("Next case", icon=":material/arrow_forward:", on_click=next_detective_case, width="stretch")

    st.markdown("<div class='chapter-label'>Chapter 04 · Trust</div><div class='chapter-title'>What is safe to show in this public exhibit?</div>", unsafe_allow_html=True)
    st.markdown("<div class='safety-strip'><h3>Public by design, not public by accident</h3><p>The application loads one audited, fixed bundle. It cannot scan the laboratory data folders or switch to a private bundle at runtime.</p></div>", unsafe_allow_html=True)
    safety_cols = st.columns(4, gap="medium")
    safety_items = [
        ("Daily", "GNSS/QC resolution", "No five-minute V5 series"),
        ("6-hour", "Tide/GNSS context", "No hourly source table"),
        ("Windows only", "Pressure availability", "No pressure measurements"),
        ("0 rows", "Level-1 arcs published", "Only counts and correlations"),
    ]
    for column, (value, label, note) in zip(safety_cols, safety_items):
        with column:
            st.metric(label, value, border=True)
            st.caption(note)
    st.caption("Still public: daily derived water-level values, model probabilities, aggregate metrics, correlation coefficients, and a coarsened site location. These outputs still require laboratory approval.")

    with st.expander("Open expert view: daily risk map and calendar", icon=":material/map:"):
        pipeline = st.selectbox("Risk pipeline", model_options, index=model_options.index(default_pipeline), key="overview_pipeline")
        pipeline_rows = predictions[predictions["pipeline"].eq(pipeline)]
        review_date = st.slider(
            "Review date", pipeline_rows["date"].min().date(), pipeline_rows["date"].max().date(),
            min(pd.Timestamp("2025-02-15").date(), pipeline_rows["date"].max().date()), format="YYYY-MM-DD",
        )
        nearest = pipeline_rows.loc[(pipeline_rows["date"] - pd.Timestamp(review_date)).abs().idxmin()]
        expert_left, expert_right = st.columns([1, 1.25], gap="large")
        with expert_left:
            show_chart(look_sector_map(site_lon, site_lat, float(nearest["suspect_probability"])), "overview_map")
        with expert_right:
            show_chart(risk_calendar(predictions, pipeline), "overview_calendar")

elif view == "Learn":
    st.markdown(
        "<div class='chapter-label'>Hands-on science gallery</div>"
        "<div class='chapter-title'>Move the controls before reading the research plots</div>"
        "<div class='chapter-copy'>These short exhibits isolate one idea at a time. They are visual explanations, not additional measurements or model results.</div>",
        unsafe_allow_html=True,
    )
    learning_exhibit = st.segmented_control(
        "Choose an exhibit",
        ["Ice and retrieval damping", "QC sorting line", "Time-aware validation"],
        default="Ice and retrieval damping",
        key="learning_exhibit",
    )

    if learning_exhibit == "Ice and retrieval damping":
        intro, control = st.columns([2.2, 1], vertical_alignment="bottom")
        with intro:
            st.subheader("Why can winter GNSS-IR look too flat?")
            st.write("Think of the tide as a full-size dance and the GNSS-IR retrieval as its shadow. Ice-influenced reflections can make the shadow look smaller even when this exhibit does not change the underlying tide.")
        with control:
            damping_percent = st.slider(
                "Conceptual retrieval damping",
                min_value=0,
                max_value=90,
                value=65,
                step=5,
                format="%d%%",
                key="learning_damping",
            )
        show_chart(ice_damping_animation(damping_percent), "ice_damping_animation")
        st.caption("Teaching boundary: the orange curve is a synthetic open-water tide pattern and the blue curve is a synthetic compressed retrieval. This does not demonstrate that sea ice physically reduces the true ocean tide by the selected percentage.")

    elif learning_exhibit == "QC sorting line":
        st.subheader("How do many reflection arcs become a review-ready spline?")
        st.write("Follow the colored tokens through two different QC levels. The important middle step is easy to miss: filtering Level-1 arcs does not automatically edit an existing Level-2 product; the spline must be fitted again.")
        show_chart(qc_sorting_animation(), "qc_sorting_animation")
        st.caption("Teaching boundary: the 36 tokens and survival counts are illustrative, not PRECIPICE retention statistics. The paper-style PTN/CLR thresholds are a sensitivity analysis; the current V5 configuration used relaxed PTN/CLR limits and an error_m limit of 1 m.")

    else:
        st.subheader("How does a model prove it can handle a future month?")
        st.write("Each blue carriage is a month the model has already seen. The orange carriage is the next unseen month. After validation, that month joins the training history and the train moves forward.")
        if folds.empty:
            st.warning("The public fold summary is unavailable in this bundle.")
        else:
            show_chart(expanding_window_animation(folds), "expanding_window_animation")
            st.caption("Evidence boundary: these frames use the published expanding-window fold dates and sample counts. Blue months train the model; the orange month is excluded until its validation prediction is made.")

elif view == "Observe":
    st.header("Observe the product and its physical context")
    st.caption("Start from GNSS-IR behavior before interpreting classifications or models.")
    controls = st.columns([1.6, 2.0], vertical_alignment="bottom")
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
            tide_view = st.segmented_control(
                "Tide exhibit",
                ["Daily animation", "Seasonal context"],
                default="Daily animation",
                key="tide_exhibit_view",
            )
            if tide_view == "Daily animation":
                available_start = tide_6h["datetime_utc"].min().date()
                available_end = tide_6h["datetime_utc"].max().date()
                default_day = min(max(pd.Timestamp("2025-07-15").date(), available_start), available_end)
                selected_tide_day = st.date_input(
                    "Choose a 2025 date",
                    value=default_day,
                    min_value=available_start,
                    max_value=available_end,
                    key="tide_animation_date",
                )
                show_chart(tide_reflection_animation(tide_6h, selected_tide_day), "tide_reflection_animation")
                day_summary = tide_daily[tide_daily["date"].dt.date.eq(selected_tide_day)]
                if not day_summary.empty:
                    row = day_summary.iloc[0]
                    metric_left, metric_right = st.columns(2)
                    metric_left.metric("Predicted tide range", f"{row['tide_range_24h']:.2f} m", border=True)
                    metric_right.metric("GNSS-IR daily range", f"{row['gnss_range_24h']:.2f} m" if pd.notna(row["gnss_range_24h"]) else "n/a", border=True)
                st.caption(
                    "Teaching schematic only: geometry and vertical motion are exaggerated. The 24 animation frames are interpolated in the browser from the audited six-hour public table; this is not an electromagnetic ray-tracing model."
                )
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

elif view == "Models":
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
