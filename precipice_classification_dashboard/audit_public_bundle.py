from __future__ import annotations

import csv
import io
import pickle
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
BUNDLE_PATH = APP_DIR / "data" / "precipice_dashboard_public_bundle.pkl"
PUBLIC_APP_PATH = APP_DIR / "streamlit_app.py"

FORBIDDEN_TEXT = (
    "Data_Raw",
    "outputs/",
    "/Users/",
    "\\Users\\",
    ".mat",
    "precipice_dashboard_bundle.pkl",
)
FORBIDDEN_TABLE_KEYS = ("arc", "pressure_raw", "snr_raw")
RAW_ARC_COLUMNS = {"rh", "sp", "ptn", "clr", "pr", "ms", "vs", "af", "df"}
PUBLIC_OOF_COLUMNS = {"date", "pipeline", "model", "weak_label", "prediction", "suspect_probability"}
PUBLIC_V5_COLUMNS = {"datetime_utc", "water_level_m", "error_m"}
TIDE_6H_COLUMNS = {"datetime_utc", "tide_anomaly_m", "tide_rate_m_per_h", "gnss_anomaly_m"}
TIDE_DAILY_COLUMNS = {"date", "tide_range_24h", "gnss_range_24h"}
PRESSURE_COVERAGE_COLUMNS = {"window", "start", "end"}
DAILY_QC_COLUMNS = {
    "date",
    "water_level_mean",
    "water_level_range",
    "water_level_range_14d",
    "error_median",
    "valid_fraction",
    "arc_count",
    "weak_label",
    "seasonal_damped_period_proxy",
    "low_arc_availability_suspect",
    "low_v5_coverage_suspect",
    "low_tide_range_suspect",
}


def fail(message: str) -> None:
    raise SystemExit(f"PUBLIC BUNDLE AUDIT FAILED: {message}")


def rows_and_columns(text: str) -> tuple[list[dict[str, str]], set[str]]:
    rows = list(csv.DictReader(io.StringIO(text)))
    return rows, set(rows[0]) if rows else set()


def main() -> None:
    if not BUNDLE_PATH.exists():
        fail(f"missing {BUNDLE_PATH}")
    if BUNDLE_PATH.stat().st_size > 10 * 1024 * 1024:
        fail("bundle is larger than 10 MB")
    if "PRECIPICE_DASHBOARD_BUNDLE" in PUBLIC_APP_PATH.read_text():
        fail("public app permits a runtime bundle-path override")

    with BUNDLE_PATH.open("rb") as f:
        bundle = pickle.load(f)

    if bundle.get("metadata", {}).get("disclosure_profile") != "public":
        fail("metadata does not declare disclosure_profile=public")

    serialized_text = repr({
        "metadata": bundle.get("metadata", {}),
        "json": bundle.get("json", {}),
        "table_keys": list(bundle.get("tables_csv", {})),
    })
    for token in FORBIDDEN_TEXT:
        if token in serialized_text:
            fail(f"found forbidden path/token: {token}")

    if bundle.get("images"):
        fail("public bundle must not embed static research figures")

    tables = bundle.get("tables_csv", {})
    for key, text in tables.items():
        if any(token in key.lower() for token in FORBIDDEN_TABLE_KEYS):
            fail(f"forbidden raw-style table key: {key}")
        for token in FORBIDDEN_TEXT:
            if token in text:
                fail(f"table {key} contains forbidden path/token: {token}")

    v5_rows, v5_columns = rows_and_columns(tables.get("v5_spline_light", ""))
    if v5_columns != PUBLIC_V5_COLUMNS:
        fail(f"unexpected public V5 columns: {sorted(v5_columns)}")
    if len(v5_rows) > 500:
        fail(f"public V5 has {len(v5_rows)} rows; expected daily aggregation")

    oof_rows, oof_columns = rows_and_columns(tables.get("signal_only_expanding_oof_predictions", ""))
    if oof_columns != PUBLIC_OOF_COLUMNS:
        fail(f"unexpected public OOF columns: {sorted(oof_columns)}")
    for row in oof_rows:
        probability = row.get("suspect_probability", "")
        if "." in probability and len(probability.rstrip("0").split(".")[-1]) > 3:
            fail("OOF probability contains more than three decimal places")

    daily_rows, daily_columns = rows_and_columns(tables.get("daily_qc_context", ""))
    if daily_columns != DAILY_QC_COLUMNS:
        fail(f"unexpected daily QC columns: {sorted(daily_columns)}")
    if len(daily_rows) > 500:
        fail(f"daily QC table has {len(daily_rows)} rows")

    pressure_rows, pressure_columns = rows_and_columns(tables.get("pressure_coverage", ""))
    if pressure_columns != PRESSURE_COVERAGE_COLUMNS:
        fail(f"unexpected pressure coverage columns: {sorted(pressure_columns)}")
    if len(pressure_rows) > 10:
        fail("pressure coverage table contains unexpected row-level data")

    tide_rows, tide_columns = rows_and_columns(tables.get("tide_gnss_6h", ""))
    if tide_columns != TIDE_6H_COLUMNS:
        fail(f"unexpected six-hour tide columns: {sorted(tide_columns)}")
    if len(tide_rows) > 1600:
        fail("tide/GNSS public table is finer than the approved six-hour profile")
    _, tide_daily_columns = rows_and_columns(tables.get("tide_gnss_daily", ""))
    if tide_daily_columns != TIDE_DAILY_COLUMNS:
        fail(f"unexpected daily tide columns: {sorted(tide_daily_columns)}")

    for key, text in tables.items():
        _, columns = rows_and_columns(text)
        if RAW_ARC_COLUMNS.issubset(columns):
            fail(f"table {key} contains the complete raw arc feature schema")

    print(f"PUBLIC BUNDLE AUDIT PASSED: {BUNDLE_PATH}")
    print(f"size_mb={BUNDLE_PATH.stat().st_size / (1024 * 1024):.2f}")
    print(
        f"daily_v5_rows={len(v5_rows)} daily_qc_rows={len(daily_rows)} "
        f"tide_6h_rows={len(tide_rows)} oof_rows={len(oof_rows)} images={len(bundle.get('images', {}))}"
    )


if __name__ == "__main__":
    main()
