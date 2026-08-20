# Stage 1: PRECIPICE GNSS-IR Reliability Classification

This repository contains the Streamlit review app for Stage 1 of the
PRECIPICE GNSS-IR project at Ausuittuq (Grise Fiord), Nunavut. The shore-mounted
receiver looks across Jones Sound and records reflected satellite signals that
can be processed into a water-level time series. Sea ice, incomplete
observations, and weak reflection geometry can change the signal, so Stage 1
identifies days that need closer quality-control review.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://gnss-class.streamlit.app/)

**Live demo:** <https://gnss-class.streamlit.app/>  
The deployment may require Streamlit sign-in.

## What Stage 1 does

| Item | Definition |
|---|---|
| Task | Daily reliability classification for the GNSS-IR record |
| Output | `reliable_proxy` or `suspect_proxy` |
| Role | Prioritize days for human QC review |
| Primary evidence | 12 monthly expanding-window out-of-fold validation periods |
| Evidence period | August 2024 to October 2025 |
| Public interface | Precomputed results only; the app does not train or tune models |

The target is a **weak QC label** built from a known seasonal damping window,
low arc availability, low V5 coverage, and low daily water-level range. It is
not a direct observation of sea ice, sensor failure, or environmental state.
The classifier learns which days resemble the review rules; it does not
establish physical truth.

Stage 1 is the first part of a two-stage workflow. Stage 2 reconstructs missing
water levels with tide and weather information after the reliability review.
CHS and ECCC variables belong to the later reconstruction workflow and future
Stage 1 challenger experiments. They were not retroactively added to the
reported Stage 1 benchmark.

## Analysis workflow

| Step | What happens | Leakage control |
|---|---|---|
| 1. Daily aggregation | Level-1 arc diagnostics and V5 coverage are summarized by UTC day | Source-resolution rows stay in the private research workspace |
| 2. Weak-label construction | Transparent QC rules assign the daily proxy target | `ms_median`, the benchmark feature, is not used to construct the target |
| 3. Feature comparison | Small teacher-requested feature sets and signal-only pipelines are compared | Correlated features are screened and reported |
| 4. Temporal validation | Earlier months train the model and the next unseen month is evaluated | Feature selection and tuning are repeated inside each training fold |
| 5. Human review | Out-of-fold probabilities and diagnostics are shown in Streamlit | The app cannot retrain the model or change its evidence |

The earlier season-aware blocked split remains in the app as a diagnostic. The
monthly expanding-window results are the primary generalization evidence, and
the two validation designs are not ranked against each other.

## Main Stage 1 result

The focused daily dataset contains 452 days, including 434 days with arc files,
from 1 August 2024 to 26 October 2025. The weak target contains 186
`reliable_proxy` days and 266 `suspect_proxy` days.

| Model comparison | Out-of-fold balanced accuracy | Interpretation |
|---|---:|---|
| `ms_median + sp_std`, plain logistic regression | 0.895 | Best controlled V0-V5 feature comparison |
| Train-fold Lasso-selected signal features, balanced logistic regression | 0.888 | Best signal-only pipeline |
| `ms_median` benchmark | 0.515 | Single-feature reference |

These scores measure agreement with the QC-derived proxy target. They should
not be reported as sea-ice classification accuracy or independent sensor
validation.

## What the app shows

The Streamlit app turns the Stage 1 outputs into a guided review:

- Arctic and station-scale site context;
- an explanation of GNSS interferometric reflectometry;
- daily V5, uncertainty, coverage, and damping diagnostics;
- tide context and two pressure-sensor coverage windows;
- the proxy-label definition and a signal-review exercise; and
- feature comparisons, expanding-window folds, and out-of-fold predictions.

Teaching animations use synthetic curves or clearly identified aggregates.
They are explanatory graphics, not additional measurements.

## Repository contents

| Path | Purpose |
|---|---|
| `streamlit_app.py` | Public, read-only Streamlit interface |
| `build_data_bundle.py` | Local builder for public or private app bundles |
| `audit_public_bundle.py` | Disclosure audit for the public bundle |
| `data/precipice_dashboard_public_bundle.pkl` | Reduced artifact used by the deployed app |
| `.streamlit/config.toml` | App theme and server settings |
| `pyproject.toml` and `uv.lock` | Reproducible Python environment |

The research notebooks and full sensor data are maintained outside this
deployment directory. The app does not scan the project tree or regenerate
notebook results.

## Run the app

Requirements: Python 3.11+ and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python audit_public_bundle.py
uv run streamlit run streamlit_app.py
```

Open the local URL printed by Streamlit, normally
`http://localhost:8501`.

## Data access and disclosure boundary

The deployed app reads one fixed file:

```text
data/precipice_dashboard_public_bundle.pkl
```

The public artifact contains:

- daily aggregated V5 and QC values;
- six-hour tide and GNSS anomaly series;
- selected, rounded out-of-fold prediction fields;
- aggregate model and feature-selection tables;
- pressure coverage intervals without pressure measurements; and
- coarsened coordinates for display.

It excludes:

- raw NMEA and SNR files;
- Level-1 arc rows and the raw arc-feature schema;
- the source-resolution five-minute V5 table;
- pressure measurements;
- private notebook outputs and static research figures;
- secrets, absolute local paths, and private bundle overrides.

### Pickle is not encryption

A `.pkl` file does not hide its contents. Anyone who obtains it can normally
recover the serialized tables. Converting a private CSV to pickle would still
disclose the data.

This project can publish the bundled file only because the builder aggregates,
selects, rounds, and coarsens its contents before serialization. The audit then
checks the disclosure profile, allowed schemas, row limits, precision, file
size, forbidden source names, raw-style table keys, and embedded images.

The public artifact passed the local audit on **11 August 2026**:

```text
size                         0.28 MB
daily V5 rows                  433
daily QC rows                  433
six-hour tide/GNSS rows      1,460
reduced OOF rows             2,880
embedded images                  0
```

These controls prevent the app from exposing source-resolution laboratory
tables. Every value displayed by the app remains visible to its audience. If a
reduced value is not approved for that audience, it must stay out of the bundle
and the repository.

## Rebuild the public bundle

Run the builder only inside the authorized research workspace, where the
private inputs are available:

```bash
uv run python build_data_bundle.py --profile public
uv run python audit_public_bundle.py
```

The builder reads approved local Stage 1 outputs, the private V5 source table,
and contextual tide information. It writes aggregates to the deployment
directory; it does not copy the private inputs. A public build must not be
released unless the audit passes.

The application-level and project-level `.gitignore` files block raw data
directories, the source V5 filename, secrets, environments, and private pickle
files. Only the audited public bundle is allowlisted. Git ignore rules do not
remove files from history, so repository history must be checked before access
is widened.

## Data and software provenance

| Input or software | Role in this repository | Public status |
|---|---|---|
| PRECIPICE GNSS-IR observations | Private source for daily Stage 1 summaries | Raw record not distributed here |
| Tide predictions | Physical context and one weak-label diagnostic | Reduced anomalies only |
| Pressure-sensor record | Independent context for two overlap windows | Coverage intervals only |
| [`gnssir-rt-dev`](https://github.com/Precipice-Sensors/gnssir-rt-dev) | Processing software for the custom PRECIPICE sensors | Linked software repository |
| [`gnssrefl`](https://github.com/kristinemlarson/gnssrefl) | Reference open GNSS-IR toolkit | Public software and documentation |
| [Elena Savidge's processing and satellite-context repository](https://github.com/elenasavidge/mcgill-gnssir-satellite-work) | Companion site, processing, Sentinel-2, sea-ice, ICESat-2, and SWOT context | Private collaborator repository |

The companion repository documents the GNSS-IR processing chain and satellite
setting. This Stage 1 repository focuses on daily reliability classification
and the public review interface.

## Interpretation limits

- The proxy target supports QC triage; it is not an independently labelled
  environmental class.
- Pressure observations cover two periods and are not used as a universal
  truth record.
- Tide predictions provide physical context and do not share the GNSS-IR
  vertical reference.
- The station map shows an approximate look sector, not a spatial sea-ice
  interpolation.
- Model importance describes fitted prediction behavior and is not a causal
  estimate.

## Project collaborators

Elena Savidge¹, Natalya Gomez¹, David Didier², Jeremy Baudry², Terry Noah³,
Dave Purnell⁴, and Ruihe Zhang¹

¹ McGill University  
² Université du Québec à Rimouski  
³ Ausuittuq Adventures  
⁴ PRECIPICE

Stage 1 classification and Streamlit review app: Ruihe Zhang  
Project supervision: Prof. Natalya Gomez  
Scientific collaboration and GNSS-IR/satellite context: Elena Savidge

## Release check

Before publishing a new app version:

1. Build with `--profile public`.
2. Require a passing `audit_public_bundle.py` result.
3. Inspect the exact staged-file list for raw data, secrets, private bundles,
   source-resolution tables, and local paths.
4. Review the app as its least-privileged intended viewer.
5. Confirm that every displayed result is approved for that audience.
