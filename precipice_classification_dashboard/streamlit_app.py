from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parents[1]
W6_DIR = PROJECT_ROOT / "outputs" / "eda" / "week_6_progress"
W7_DIR = PROJECT_ROOT / "outputs" / "eda" / "week_7_geography_visual_progress"
V5_PATH = PROJECT_ROOT / "Data_Raw" / "PRECIPICE" / "processed" / "spline_v5" / "precipice_sealevel_v5.csv"

LABEL_NAMES = {0: "reliable proxy", 1: "suspect proxy"}
LABEL_COLORS = {"reliable proxy": "#4C72B0", "suspect proxy": "#DD8452"}
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
        st.image(str(path), use_container_width=True)
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
        rows.append({"artifact": p.name, "exists": p.exists(), "path": str(p)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


summary = read_json(str(W7_DIR / "week7_geography_progress_summary.json"))
primary = read_csv(str(W6_DIR / "primary_v0v5_expanding_leaderboard.csv"))
signal = read_csv(str(W6_DIR / "signal_only_expanding_leaderboard.csv"))
folds = read_csv(str(W6_DIR / "expanding_window_fold_summary.csv"), parse_dates=("train_start", "train_end", "validation_start", "validation_end"))
preds = read_csv(str(W6_DIR / "signal_only_expanding_oof_predictions.csv"), parse_dates=("date",))
feature_counts = read_csv(str(W6_DIR / "signal_only_feature_selection_counts.csv"))
target_definition = read_csv(str(W6_DIR / "classification_target_definition.csv"))
teacher_sets = read_csv(str(W6_DIR / "teacher_requested_feature_sets.csv"))
v5 = load_v5(str(V5_PATH))

if not v5.empty:
    min_date = v5["datetime_utc"].min().date()
    max_date = v5["datetime_utc"].max().date()
else:
    min_date = pd.Timestamp("2024-08-20").date()
    max_date = pd.Timestamp("2025-10-26").date()

with st.sidebar:
    st.title("PRECIPICE review")
    st.caption("Local Streamlit layer for Week 5-7 outputs.")
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

st.title("PRECIPICE GNSS-IR classification review")
st.caption("A teaching dashboard for QC-first coastal monitoring, proxy labels, and Week 6 model results.")

st.warning(
    "Current labels are QC-derived proxy labels. They support review and discussion, but they are not independent physical truth for ice/open water/wind.",
    icon=":material/warning:",
)

tabs = st.tabs([
    "Overview",
    "Data product",
    "Validation & tide",
    "Labels",
    "Features",
    "Models",
    "Geography review",
    "Glossary",
])

with tabs[0]:
    st.header("Project overview")
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

    col_a, col_b = st.columns([1.15, 0.85])
    with col_a:
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
        st.dataframe(flow, hide_index=True, use_container_width=True)
    with col_b:
        st.subheader("What to remember")
        st.info("Pressure sensor data do not cover the whole GNSS-IR year.", icon=":material/info:")
        st.info("Tide prediction is context, not a training label.", icon=":material/timeline:")
        st.info("One station means map output is a look-sector review, not a sea-ice map.", icon=":material/map:")

with tabs[1]:
    st.header("GNSS-IR V5 observing product")
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
            st.plotly_chart(fig, use_container_width=True)

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
        st.dataframe(folds, hide_index=True, use_container_width=True)

with tabs[3]:
    st.header("Proxy labels")
    st.write("This section explains the current target. It is useful, but it is not a final physical surface-state label.")
    display_figure(W7_DIR / "fig4_daily_proxy_label_diagnostics.png", "Proxy label timeline and monthly reason composition.")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.subheader("How labels were made")
        if not target_definition.empty:
            st.dataframe(target_definition, hide_index=True, use_container_width=True)
        else:
            st.write("Target-definition table is missing.")
    with col_b:
        st.subheader("Teacher-requested feature versions")
        if not teacher_sets.empty:
            st.dataframe(teacher_sets, hide_index=True, use_container_width=True)

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
            st.plotly_chart(fig, use_container_width=True)
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
    st.header("Model comparison")
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

    st.plotly_chart(leaderboard_plot(active, label_col, title), use_container_width=True)
    st.plotly_chart(metrics_heatmap(active, label_col, "All key metrics"), use_container_width=True)
    st.plotly_chart(overfit_plot(active, label_col, "Overfitting check: train vs validation"), use_container_width=True)

    st.subheader("Sortable metrics table")
    st.dataframe(nice_metrics_table(active), hide_index=True, use_container_width=True)

    if selected_pipeline:
        st.subheader("Prediction timeline")
        st.plotly_chart(prediction_timeline(preds, selected_pipeline, start_date, end_date), use_container_width=True)

    with st.expander("How to read the scores", expanded=explanation_level == "Beginner"):
        st.write(
            "**Balanced accuracy** averages reliable-day recall and suspect-day recall, so it is better than raw accuracy when classes are imbalanced. "
            "**Recall suspect** asks how many suspect days were caught. "
            "**Overfit gap** compares train and validation performance; a large gap means the model may not generalize."
        )

with tabs[6]:
    st.header("Geography review")
    st.write("These figures make the model output geographically understandable without pretending that one station can map the whole bay.")
    col_a, col_b = st.columns(2)
    with col_a:
        display_figure(W7_DIR / "fig1_site_map.png", "Basic site location.")
        display_figure(W7_DIR / "fig1b_cartopy_site_context.png", "Arctic projection and approximate look-sector context.")
    with col_b:
        display_figure(W7_DIR / "fig8_spatiotemporal_risk_map_calendar.png", "Station/look-sector risk plus daily risk calendar.")

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
    st.dataframe(glossary, hide_index=True, use_container_width=True)

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
