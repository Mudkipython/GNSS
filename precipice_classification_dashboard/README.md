# PRECIPICE GNSS-IR review studio

This Streamlit app is an interactive presentation and exploration layer for the
Week 5 and Week 7 PRECIPICE GNSS-IR classification work. Four task-oriented
views cover station context, the observing product, QC/proxy-label diagnostics,
and model evidence.

The deployed app does not retrain models, regenerate notebooks, scan project
folders, or read laboratory source files. It reads one audited bundle only.

The local bundle builder reads approved inputs from:

- `outputs/eda/week_6_progress/`
- `outputs/eda/week_7_geography_visual_progress/`
- `Data_Raw/PRECIPICE/processed/spline_v5/precipice_sealevel_v5.csv`

For Streamlit Cloud or GitHub deployment, the app runs from an explicitly
sanitized public artifact:

- `data/precipice_dashboard_public_bundle.pkl`

Build and audit it locally with:

```bash
uv run python build_data_bundle.py --profile public
uv run python audit_public_bundle.py
```

The public bundle contains daily aggregated V5/QC values, six-hour tide/GNSS
anomalies, pre-approved pressure coverage intervals without measurements,
reduced/rounded OOF predictions, correlation coefficients, aggregate model
tables, and coarse display coordinates. Interactive charts are rebuilt in the
app; static research figures are not embedded. The bundle does not contain
Level-1 arc rows, pressure measurements, the five-minute V5 table, or absolute
local paths.

The legacy `data/precipice_dashboard_bundle.pkl` is a private local artifact.
It contains the complete five-minute V5 series and must not be committed or
uploaded to Streamlit Cloud. The app-level `.gitignore` blocks private pickle
bundles while explicitly allowing the sanitized public bundle.

To build an internal full-resolution bundle deliberately:

```bash
uv run python build_data_bundle.py --profile private
PRECIPICE_DASHBOARD_BUNDLE=data/precipice_dashboard_bundle.pkl \
  uv run streamlit run streamlit_app.py
```

`.gitignore` is not retroactive. If a private bundle or raw file was already
committed, remove it from Git's index and audit repository history before
making the repository public. Rotating to the public bundle does not erase an
older committed copy.

Important disclosure boundary: any chart, coordinate, metric, or time series
shown by a public Streamlit app is itself public information. The sanitized
bundle prevents accidental publication of source-resolution tables; it does
not make displayed research results confidential. If the figures or daily
values are not approved for release, use a private repository plus restricted
app access instead of a public deployment.

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
- Static notebook figures are reconstructed as Plotly views from audited, downsampled tables.
