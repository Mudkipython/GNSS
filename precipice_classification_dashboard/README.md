# PRECIPICE classification review dashboard

This Streamlit app is a local presentation and exploration layer for the Week 5-7 PRECIPICE GNSS-IR classification work.
The interface is organized as a geography-style field atlas: site context, observing product, validation/tide context, proxy-label logic, model evidence, and map review.

It does not retrain models and does not regenerate notebooks. It reads existing outputs from:

- `outputs/eda/week_6_progress/`
- `outputs/eda/week_7_geography_visual_progress/`
- `Data_Raw/PRECIPICE/processed/spline_v5/precipice_sealevel_v5.csv`

For Streamlit Cloud or GitHub deployment, the app can also run from a compact bundled artifact:

- `data/precipice_dashboard_bundle.pkl`

Build or refresh it locally with:

```bash
uv run python build_data_bundle.py
```

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
- Runtime dependencies stay light for Streamlit Cloud; heavy GeoPandas/Cartopy products are pre-rendered into the bundle or represented with Plotly geometry.
