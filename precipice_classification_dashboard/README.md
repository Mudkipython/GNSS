# PRECIPICE classification review dashboard

This Streamlit app is a local presentation and exploration layer for the Week 5-7 PRECIPICE GNSS-IR classification work.

It does not retrain models and does not regenerate notebooks. It reads existing outputs from:

- `outputs/eda/week_6_progress/`
- `outputs/eda/week_7_geography_visual_progress/`
- `Data_Raw/PRECIPICE/processed/spline_v5/precipice_sealevel_v5.csv`

Run from this folder:

```bash
uv sync
uv run streamlit run streamlit_app.py
```

Open the local URL that Streamlit prints, usually `http://localhost:8501`.

Important interpretation:

- The current classification target is a QC-derived proxy label.
- Pressure sensor data validate only two overlap windows.
- Tide predictions are geographic context, not pressure truth.
- The geography map is a single-station look-sector review view, not a spatial interpolation of sea ice.
