from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parents[1]
W6_DIR = PROJECT_ROOT / "outputs" / "eda" / "week_6_progress"
W7_DIR = PROJECT_ROOT / "outputs" / "eda" / "week_7_geography_visual_progress"
V5_PATH = PROJECT_ROOT / "Data_Raw" / "PRECIPICE" / "processed" / "spline_v5" / "precipice_sealevel_v5.csv"
OUT_PATH = APP_DIR / "data" / "precipice_dashboard_bundle.pkl"


TABLES = {
    "primary_v0v5_expanding_leaderboard": W6_DIR / "primary_v0v5_expanding_leaderboard.csv",
    "signal_only_expanding_leaderboard": W6_DIR / "signal_only_expanding_leaderboard.csv",
    "expanding_window_fold_summary": W6_DIR / "expanding_window_fold_summary.csv",
    "signal_only_expanding_oof_predictions": W6_DIR / "signal_only_expanding_oof_predictions.csv",
    "signal_only_feature_selection_counts": W6_DIR / "signal_only_feature_selection_counts.csv",
    "classification_target_definition": W6_DIR / "classification_target_definition.csv",
    "teacher_requested_feature_sets": W6_DIR / "teacher_requested_feature_sets.csv",
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


def csv_text(path: Path) -> str:
    return path.read_text() if path.exists() else ""


def build_v5_light() -> str:
    if not V5_PATH.exists():
        return ""
    df = pd.read_csv(V5_PATH, parse_dates=["datetime_utc"])
    keep = [c for c in ["datetime_utc", "water_level_m", "error_m"] if c in df.columns]
    df = df[keep].dropna(subset=["datetime_utc"]).sort_values("datetime_utc")
    return df.to_csv(index=False)


def main() -> None:
    bundle = {
        "version": 1,
        "tables_csv": {},
        "json": {},
        "images": {},
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "created_from": "Week 6/7 PRECIPICE dashboard artifacts",
            "bundled_artifacts": [],
        },
    }

    for name, path in TABLES.items():
        if path.exists():
            bundle["tables_csv"][name] = csv_text(path)
            bundle["metadata"]["bundled_artifacts"].append(path.name)

    v5_text = build_v5_light()
    if v5_text:
        bundle["tables_csv"]["v5_spline_light"] = v5_text
        bundle["metadata"]["bundled_artifacts"].append("v5_spline_light")

    summary_path = W7_DIR / "week7_geography_progress_summary.json"
    if summary_path.exists():
        bundle["json"]["week7_summary"] = json.loads(summary_path.read_text())
        bundle["metadata"]["bundled_artifacts"].append(summary_path.name)

    for filename in FIGURES:
        path = W7_DIR / filename
        if path.exists():
            bundle["images"][filename] = path.read_bytes()
            bundle["metadata"]["bundled_artifacts"].append(filename)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("wb") as f:
        pickle.dump(bundle, f, protocol=4)

    size_mb = OUT_PATH.stat().st_size / (1024 * 1024)
    print(f"Wrote {OUT_PATH} ({size_mb:.2f} MB)")
    print(f"Tables: {len(bundle['tables_csv'])}, images: {len(bundle['images'])}")


main()
