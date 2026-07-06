from __future__ import annotations

import argparse
import json
import pickle
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parents[1]
W6_DIR = PROJECT_ROOT / "outputs" / "eda" / "week_6_progress"
W7_DIR = PROJECT_ROOT / "outputs" / "eda" / "week_7_geography_visual_progress"
W5_DIR = PROJECT_ROOT / "outputs" / "eda" / "week_5_classification_progress"
DATA_ROOT = PROJECT_ROOT / "Data_Raw" / "PRECIPICE"
V5_PATH = PROJECT_ROOT / "Data_Raw" / "PRECIPICE" / "processed" / "spline_v5" / "precipice_sealevel_v5.csv"
ARC_DIR = DATA_ROOT / "processed" / "arcs_v5"
TIDE_PATH = DATA_ROOT / "grisefjord_tidalpredictions_present_2050_2150_2300" / "GriseFjord_allConstituents_presentDay_2025.csv"
PRESSURE_COVERAGE_PATH = APP_DIR / "data" / "pressure_coverage_public.csv"
PUBLIC_OUT_PATH = APP_DIR / "data" / "precipice_dashboard_public_bundle.pkl"
PRIVATE_OUT_PATH = APP_DIR / "data" / "precipice_dashboard_bundle.pkl"


TABLES = {
    "primary_v0v5_expanding_leaderboard": W6_DIR / "primary_v0v5_expanding_leaderboard.csv",
    "signal_only_expanding_leaderboard": W6_DIR / "signal_only_expanding_leaderboard.csv",
    "expanding_window_fold_summary": W6_DIR / "expanding_window_fold_summary.csv",
    "signal_only_expanding_oof_predictions": W6_DIR / "signal_only_expanding_oof_predictions.csv",
    "signal_only_feature_selection_counts": W6_DIR / "signal_only_feature_selection_counts.csv",
    "classification_target_definition": W6_DIR / "classification_target_definition.csv",
    "teacher_requested_feature_sets": W6_DIR / "teacher_requested_feature_sets.csv",
}

WEEK5_TABLES = {
    "week5_feature_version_leaderboard": W5_DIR / "feature_version_leaderboard.csv",
    "week5_model_family_best": W5_DIR / "model_family_best_by_version.csv",
    "week5_feature_delta": W5_DIR / "feature_version_delta_vs_ms_only.csv",
    "week5_feature_importance": W5_DIR / "feature_importance_best_models.csv",
    "week5_split_summary": W5_DIR / "train_test_split_summary.csv",
    "week5_split_rationale": W5_DIR / "split_design_rationale.csv",
}

FIGURES = [
    "fig1_site_map.png",
    "fig1b_cartopy_site_context.png",
    "fig2_v5_product_overview.png",
    "fig3_pressure_overlap_context.png",
    "fig4_daily_proxy_label_diagnostics.png",
    "fig4b_representative_signal_diagnostics_timeline.png",
    "fig5_week6_model_progress_summary.png",
    "fig6_tide_context.png",
    "fig7_signal_feature_correlation_heatmap.png",
    "fig8_spatiotemporal_risk_map_calendar.png",
]

PUBLIC_LEADERBOARD_DROP = {
    "most_common_best_params",
    "primary_selection_rule",
    "oof_confusion_matrix_0_reliable_1_suspect",
}
PUBLIC_OOF_COLUMNS = [
    "date",
    "pipeline",
    "model",
    "weak_label",
    "prediction",
    "suspect_probability",
]
ARC_FEATURE_COLUMNS = ["rh", "sp", "ptn", "clr", "pr", "ms", "vs", "af", "df", "minelv", "maxelv", "azi"]


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def csv_text(df: pd.DataFrame) -> str:
    return df.to_csv(index=False)


def build_v5(profile: str) -> str:
    if not V5_PATH.exists():
        return ""
    df = pd.read_csv(V5_PATH, parse_dates=["datetime_utc"])
    keep = [c for c in ["datetime_utc", "water_level_m", "error_m"] if c in df.columns]
    df = df[keep].dropna(subset=["datetime_utc"]).sort_values("datetime_utc")
    if profile == "public":
        # The public application needs temporal context, not the laboratory's
        # full five-minute Level-2 series. Daily aggregation removes 99%+ of
        # the rows while preserving the presentation workflow.
        df = (
            df.set_index("datetime_utc")
            .resample("1D")
            .agg({"water_level_m": "mean", "error_m": "median"})
            .dropna(how="all")
            .round(3)
            .reset_index()
        )
    return csv_text(df)


def load_v5() -> pd.DataFrame:
    if not V5_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(V5_PATH, parse_dates=["datetime_utc"])
    keep = [c for c in ["datetime_utc", "water_level_m", "error_m"] if c in df.columns]
    return df[keep].dropna(subset=["datetime_utc"]).sort_values("datetime_utc")


def daily_v5_context(v5: pd.DataFrame) -> pd.DataFrame:
    valid = v5.dropna(subset=["water_level_m"]).copy()
    if valid.empty:
        return pd.DataFrame()
    expected_per_day = 24 * 12
    daily = valid.set_index("datetime_utc").resample("1D").agg(
        water_level_mean=("water_level_m", "mean"),
        water_level_range=("water_level_m", lambda x: x.max() - x.min()),
        error_median=("error_m", "median"),
        valid_points=("water_level_m", "count"),
    )
    daily["valid_fraction"] = (daily["valid_points"] / expected_per_day).clip(upper=1)
    daily["water_level_range_14d"] = daily["water_level_range"].rolling(14, center=True, min_periods=7).mean()
    return daily.round(3).reset_index().rename(columns={"datetime_utc": "date"})


def parse_arc_date(path: Path) -> pd.Timestamp:
    match = re.search(r"_l1_(\d{6})_24h", path.name)
    return pd.to_datetime(match.group(1), format="%y%m%d") if match else pd.NaT


def daily_arc_summary() -> pd.DataFrame:
    rows = []
    for path in sorted(ARC_DIR.glob("precipice_l1_*_24h.csv")):
        try:
            frame = pd.read_csv(path, usecols=lambda c: c in ARC_FEATURE_COLUMNS)
        except Exception:
            continue
        row = {"date": parse_arc_date(path), "arc_count": len(frame)}
        for col in ARC_FEATURE_COLUMNS:
            if col in frame.columns:
                values = pd.to_numeric(frame[col], errors="coerce")
                row[f"{col}_median"] = values.median()
                row[f"{col}_std"] = values.std()
        rows.append(row)
    return pd.DataFrame(rows).dropna(subset=["date"]).sort_values("date") if rows else pd.DataFrame()


def public_qc_context(v5_daily: pd.DataFrame, arcs_daily: pd.DataFrame) -> pd.DataFrame:
    if v5_daily.empty:
        return pd.DataFrame()
    out = v5_daily.merge(arcs_daily[["date", "arc_count"]], on="date", how="left") if not arcs_daily.empty else v5_daily.copy()
    out["seasonal_damped_period_proxy"] = out["date"].between("2024-12-01", "2025-06-30")
    out["low_arc_availability_suspect"] = out["arc_count"] <= 214
    out["low_v5_coverage_suspect"] = out["valid_fraction"] < 0.50
    out["low_tide_range_suspect"] = out["water_level_range"] <= 1.232
    reason_cols = [
        "seasonal_damped_period_proxy",
        "low_arc_availability_suspect",
        "low_v5_coverage_suspect",
        "low_tide_range_suspect",
    ]
    out["weak_label"] = np.where(out[reason_cols].any(axis=1), "suspect_proxy", "reliable_proxy")
    keep = [
        "date",
        "water_level_mean",
        "water_level_range",
        "water_level_range_14d",
        "error_median",
        "valid_fraction",
        "arc_count",
        "weak_label",
        *reason_cols,
    ]
    return out[keep].round(3)


def public_signal_correlation(arcs_daily: pd.DataFrame) -> pd.DataFrame:
    if arcs_daily.empty:
        return pd.DataFrame()
    feature_cols = [c for c in arcs_daily if c.endswith("_median") or c.endswith("_std")]
    corr = arcs_daily[feature_cols].corr(method="spearman")
    long = corr.rename_axis("feature_y").reset_index().melt(id_vars="feature_y", var_name="feature_x", value_name="spearman")
    return long.dropna(subset=["spearman"]).round(3)


def public_tide_context(v5: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not TIDE_PATH.exists() or v5.empty:
        return pd.DataFrame(), pd.DataFrame()
    tide = pd.read_csv(TIDE_PATH, parse_dates=["time"]).rename(columns={"tide_height": "predicted_tide_m"})
    tide["tide_anomaly_m"] = tide["predicted_tide_m"] - tide["predicted_tide_m"].mean()
    tide["tide_rate_m_per_h"] = tide["predicted_tide_m"].diff()
    valid = v5.dropna(subset=["water_level_m"]).copy()
    valid = valid[valid["datetime_utc"].between(tide["time"].min(), tide["time"].max())]
    valid["gnss_anomaly_m"] = valid["water_level_m"] - valid["water_level_m"].mean()

    tide_6h = tide.set_index("time")[["tide_anomaly_m", "tide_rate_m_per_h"]].resample("6h").mean()
    gnss_6h = valid.set_index("datetime_utc")[["gnss_anomaly_m"]].resample("6h").mean()
    six_hour = tide_6h.join(gnss_6h, how="outer").round(3).reset_index().rename(columns={"index": "datetime_utc", "time": "datetime_utc"})

    tide_daily = tide.set_index("time").resample("1D").agg(
        tide_range_24h=("predicted_tide_m", lambda x: x.max() - x.min())
    )
    gnss_daily = valid.set_index("datetime_utc").resample("1D").agg(
        gnss_range_24h=("water_level_m", lambda x: x.max() - x.min())
    )
    daily = tide_daily.join(gnss_daily, how="outer").round(3).reset_index().rename(columns={"index": "date", "time": "date"})
    return six_hour, daily


def public_pressure_coverage() -> pd.DataFrame:
    if not PRESSURE_COVERAGE_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(PRESSURE_COVERAGE_PATH, parse_dates=["start", "end"])


def public_table(name: str, df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if name == "signal_only_expanding_oof_predictions":
        keep = [c for c in PUBLIC_OOF_COLUMNS if c in df.columns]
        df = df[keep].copy()
        if "suspect_probability" in df.columns:
            df["suspect_probability"] = df["suspect_probability"].round(3)
        return df
    drop = [c for c in PUBLIC_LEADERBOARD_DROP if c in df.columns]
    return df.drop(columns=drop).round(4)


def public_summary(summary: dict) -> dict:
    allowed = {
        "v5_valid_start",
        "v5_valid_end",
        "n_arc_files",
        "site_lat",
        "site_lon",
        "week6_best_v0v5",
        "week6_best_signal_only",
    }
    clean = {key: summary[key] for key in allowed if key in summary}
    # Coarsen the display location to avoid publishing unnecessary precision.
    for key in ("site_lat", "site_lon"):
        if key in clean:
            clean[key] = round(float(clean[key]), 1)
    return clean


def build_bundle(profile: str) -> tuple[dict, Path]:
    out_path = PUBLIC_OUT_PATH if profile == "public" else PRIVATE_OUT_PATH
    bundle = {
        "version": 2,
        "tables_csv": {},
        "json": {},
        "images": {},
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "created_from": f"Week 6/7 PRECIPICE {profile} dashboard artifacts",
            "disclosure_profile": profile,
            "bundled_artifacts": [],
        },
    }

    for name, path in TABLES.items():
        df = read_table(path)
        if df.empty:
            continue
        if profile == "public":
            df = public_table(name, df)
        bundle["tables_csv"][name] = csv_text(df)
        bundle["metadata"]["bundled_artifacts"].append(name)

    for name, path in WEEK5_TABLES.items():
        df = read_table(path)
        if df.empty:
            continue
        if profile == "public":
            drop = [c for c in ["notes", "confusion_matrix_0_reliable_1_suspect"] if c in df.columns]
            df = df.drop(columns=drop).round(4)
        bundle["tables_csv"][name] = csv_text(df)
        bundle["metadata"]["bundled_artifacts"].append(name)

    v5_text = build_v5(profile)
    if v5_text:
        bundle["tables_csv"]["v5_spline_light"] = v5_text
        bundle["metadata"]["bundled_artifacts"].append("v5_spline_light_daily" if profile == "public" else "v5_spline_light")

    summary_path = W7_DIR / "week7_geography_progress_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        bundle["json"]["week7_summary"] = public_summary(summary) if profile == "public" else summary
        bundle["metadata"]["bundled_artifacts"].append("week7_summary")

    v5 = load_v5()
    v5_daily = daily_v5_context(v5)
    arcs_daily = daily_arc_summary()
    derived_tables = {
        "daily_qc_context": public_qc_context(v5_daily, arcs_daily),
        "signal_feature_correlation": public_signal_correlation(arcs_daily),
        "pressure_coverage": public_pressure_coverage(),
    }
    tide_6h, tide_daily = public_tide_context(v5)
    derived_tables["tide_gnss_6h"] = tide_6h
    derived_tables["tide_gnss_daily"] = tide_daily
    for name, df in derived_tables.items():
        if not df.empty:
            bundle["tables_csv"][name] = csv_text(df)
            bundle["metadata"]["bundled_artifacts"].append(name)

    if profile == "private":
        for filename in FIGURES:
            path = W7_DIR / filename
            if path.exists():
                bundle["images"][filename] = path.read_bytes()
                bundle["metadata"]["bundled_artifacts"].append(filename)

    return bundle, out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a public-sanitized or private dashboard bundle.")
    parser.add_argument("--profile", choices=("public", "private"), default="public")
    args = parser.parse_args()

    bundle, out_path = build_bundle(args.profile)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(bundle, f, protocol=4)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {out_path} ({size_mb:.2f} MB; profile={args.profile})")
    print(f"Tables: {len(bundle['tables_csv'])}, images: {len(bundle['images'])}")


if __name__ == "__main__":
    main()
