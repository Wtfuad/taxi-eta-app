# Repository guidance

## Project and layout

This repository serves a Porto taxi trip-duration model through a Streamlit dashboard
and a FastAPI REST API. Training happens in an external v4 notebook; this repository
contains inference code, exported model artifacts, and a notebook export cell.

- `taxi_eta/features.py`: feature functions copied verbatim from the training notebook.
- `taxi_eta/predictor.py`: artifact loading, feature assembly, stacked LightGBM models,
  raw/log blending, and empirical prediction intervals.
- `taxi_eta/traces.py`: shared trace parsing, validation, time conversion, route
  resampling, and batch input handling; `TripRequest` is the internal input type.
- `taxi_eta/config.py`: environment settings, with relative model paths resolved
  against the repository root.
- `app/main.py`: Streamlit UI, map drawing, replay, batch uploads, and password gate.
- `api/main.py`, `api/routes.py`, `api/schemas.py`: FastAPI lifecycle, endpoints,
  and Pydantic request/response models.
- `models/`: exported serving artifacts, including tracked LightGBM text models.
- `scripts/export_artifacts_cell.py`: replacement cell for the training notebook;
  requires notebook globals and is not a standalone training script.
- `tests/`: pytest unit, prediction parity, FastAPI, and headless Streamlit tests.
  `tests/fixtures/model/` contains a separate synthetic model.

## Setup and commands

Run commands from the repository root. The Docker runtime uses Python 3.12.
Use a working local virtual environment; if an existing `.venv` references a missing
Python installation, recreate it with an available compatible interpreter.

PowerShell setup:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

`requirements-dev.txt` includes the runtime requirements and pytest/httpx.
Most runtime libraries are pinned in `requirements.txt`; keep dependency changes
intentional. There is no configured formatter, linter, or separate build step.

```powershell
# Full suite
.\.venv\Scripts\python.exe -m pytest

# Focused checks, selected for the affected area
.\.venv\Scripts\python.exe -m pytest tests/test_traces.py tests/test_predictor.py
.\.venv\Scripts\python.exe -m pytest tests/test_api.py
.\.venv\Scripts\python.exe -m pytest tests/test_app_smoke.py

# Local dashboard and API (run separately)
.\.venv\Scripts\python.exe -m streamlit run app/main.py
.\.venv\Scripts\python.exe -m uvicorn api.main:app --reload --port 8000

# Optional synthetic preview; set before either server command
$env:MODEL_DIR = 'tests/fixtures/model'
```

Dashboard: `http://localhost:8501`; API docs: `http://localhost:8000/docs`.
The Dockerfile starts Streamlit by default; API serving requires overriding the
container command, as documented in the README.

## Contracts to preserve

- Keep prediction logic in the shared `taxi_eta` package so dashboard and API
  results agree. Match the surrounding Python style and avoid unrelated rewrites.
- Do not independently refactor the copied functions in `features.py`. Feature
  changes require corresponding notebook changes, artifact re-export, and copying
  the functions again to preserve training/serving parity.
- Preserve configured feature order, training constants, lookup fallbacks, auxiliary
  model averaging, and the log-space blend when modifying inference.
- The existing parity guard is
  `tests/test_predictor.py::test_parity_with_notebook_predictions`, using fixture
  expectations with absolute tolerance `1e-6` seconds. Do not regenerate expected
  predictions merely to accommodate a regression.
- Coordinates are `[longitude, latitude]`, sampled every 15 seconds. Internal
  timestamps are Unix seconds in UTC; naive user datetimes use `Europe/Lisbon`.
  Preserve latitude/longitude swap detection and DST handling in shared parsing.
- Predictions represent **total** duration, with remaining duration computed by
  subtracting elapsed time. Total duration is floored at elapsed time plus one
  15-second step. Intervals are empirical validation-error ranges, not guarantees.
- Preserve trace errors versus warnings and per-row batch failure reporting: an
  invalid row must not discard otherwise valid rows.
- Load portable JSON, CSV, and LightGBM text artifacts; do not introduce pickle.
  Preserve support for both `aux/` and legacy `_aux/` model directories and clear
  errors for missing/incomplete models or mismatched constants.
- Treat model files and parity fixtures as versioned outputs. Avoid replacing or
  editing them unless the task explicitly requires an artifact update. Synthetic
  fixtures demonstrate parity and behavior, not real-world model quality; retain
  the UI's synthetic-model warning.
- Logs should contain operational counts/timings rather than raw GPS traces.

## Configuration and verification

Settings are `MODEL_DIR` (defaults to `models/`), `MAX_BATCH_ROWS` (5000),
`APP_PASSWORD` (empty), and `LOG_LEVEL` (`INFO`). `APP_PASSWORD` gates the Streamlit
UI; it does not provide FastAPI authentication. Keep secrets out of tracked files.

Run the relevant existing tests after code changes; use the full suite for shared
pipeline or cross-interface changes. Headless Streamlit tests cover Python-side
behavior; interactive map drawing and tile rendering require browser verification.
Report any checks that could not run. Keep README commands and API documentation
aligned when changing user-visible behavior, configuration, or deployment.
