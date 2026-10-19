# Porto Taxi Trip-Time Predictor: dashboard

A web dashboard around the trip-time model from the training notebook (taxi-time-pred v4). Given the GPS points a taxi
has produced so far, the start time and how the taxi was called, it predicts the **total** trip time and a range.

Ways to enter a trip: paste or upload a GPS trace, draw a route on a map, replay a trip from the dataset, or upload a CSV with many trips.
Plus a *Model & performance* page with the validation metrics and the limitations.

```
 training notebook (Kaggle)                          this repository (serving)
 ┌──────────────────────────┐   taxi_eta_artifacts   ┌──────────────────────────────────────────────┐
 │ final fit + export cell  │ ─────── .zip ────────▶ │ models/  (LightGBM text, CSV, JSON; no pickle)│
 └──────────────────────────┘                        │ taxi_eta/features.py  (copied verbatim)       │
                                                     │ taxi_eta/predictor.py (same steps as notebook)│
                                                     │ taxi_eta/traces.py    (parse, validate, draw) │
                                                     │ app/main.py           (Streamlit interface)   │
                                                     └──────────────────────────────────────────────┘
```

## 1. Export the model from the notebook (once per model version)

1. Run the v4 notebook up to and including the validation cells (so that `tot_va`, `best`, `tot_raw`, `tot_log`, `W_BLEND`, `tables` exist).
2. **Replace cell 28** ("9. Final fit and submission") with the content of `scripts/export_artifacts_cell.py` and run it.
   It does what the old cell did (same seeds and models: with synthetic data the submission was identical to the old cell's) and also writes
   `./artifacts/` and `taxi_eta_artifacts.zip`.
3. Download the zip (Kaggle: Output tab) and unzip it **into `models/`** of this repository.

## 2. Run locally

```bash
# Create and activate virtual environment
python -m venv .venv && source .venv/bin/activate    # On Windows: .venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt

# Run the Streamlit interactive dashboard (opens http://localhost:8501)
streamlit run app/main.py

# Run the FastAPI REST API service (opens http://localhost:8000, docs at http://localhost:8000/docs)
uvicorn api.main:app --reload --port 8000

# Preview with SYNTHETIC test model (banner shown; numbers are meaningless)
MODEL_DIR=tests/fixtures/model streamlit run app/main.py
MODEL_DIR=tests/fixtures/model uvicorn api.main:app --reload --port 8000

# Run full test suite (parity, unit, Streamlit, and FastAPI)
pytest
```

## 3. Deploy to a cloud server with a shared URL

```bash
docker build -t taxi-eta .
docker run -p 8501:8501 -e APP_PASSWORD='choose-a-password' taxi-eta
```

Any host that runs a container works (a virtual machine with Docker, or a managed container service). Notes:

- Set **`APP_PASSWORD`** so the shared URL is not open to everyone. This is a single shared password, not user accounts.
- Terminate **HTTPS** at the platform or a reverse proxy (the app itself serves plain HTTP). The proxy must allow **WebSockets**, which Streamlit needs.
- Health check: `GET /_stcore/health` returns `ok`. The image honours `$PORT` if the host sets it.
- Resources: the model files are small; start with 1 CPU and 1 to 2 GB RAM and watch memory with large batch files.
- Streamlit keeps session state in memory per instance. Run **one replica**, or enable sticky sessions if you scale out.
- The model folder is copied into the image at build time. To ship a new model, export again, replace `models/`, rebuild and redeploy.

| Environment variable | Default | Meaning |
|---|---|---|
| `MODEL_DIR` | `models` | folder with the exported artifacts |
| `APP_PASSWORD` | empty (no gate) | shared password |
| `MAX_BATCH_ROWS` | `5000` | rows allowed in one batch upload |
| `LOG_LEVEL` | `INFO` | Python log level (logs contain counts and timings, never traces) |

## 4. What each part guarantees

- **Same features as training.** `taxi_eta/features.py` holds the notebook's functions copied verbatim (extracted programmatically). `tests/test_predictor.py::test_parity_with_notebook_predictions`
  checks that the dashboard reproduces the notebook's predictions to 1e-6 s on 300 traces (fixtures in `tests/fixtures`).
- **Fail loudly.** Missing or incomplete model files, or training constants that differ from `features.py`, stop the app with a clear message instead of predicting silently.
- **Input validation.** Bad traces (empty, non-numeric, out-of-range, too long) are errors; suspicious ones (outside Porto, GPS jumps, no movement, over 3 hours) are warnings. A bad row in a batch file is reported and skipped, not fatal.
- **Honest uncertainty.** The range shown with each prediction comes from the model's validation errors for the same number of observed points. It is empirical, not a guarantee.
- **No pickle.** Artifacts are LightGBM text models, CSV and JSON, so loading a model cannot execute code and files are portable across pandas versions.

## 5. Tested and not tested

Tested here: the 34 unit, parity and headless-UI tests (also in a fresh virtualenv with the pinned requirements); a real `streamlit run` server answering its health check;
the export cell against the v4 notebook code on **synthetic** trips (identical submission to the original cell).

**Not tested:** the model on the real Kaggle data (all numbers in the tests come from synthetic trips); the interactive map in a real browser (drawing and map tiles;
the tests only run the Python side); the Docker build (no Docker in the build environment); behaviour under load or many simultaneous users.
Do these checks before relying on the deployment: draw a route in a browser, run a batch file of your own, and compare a few dashboard predictions with the notebook.

## 6. Known limits

- Porto only; GPS points every 15 seconds; trained on July 2013 to June 2014 trips of at most 3 hours.
- *Replay* trips were not used as training rows but did contribute to the historical averages, so they look better than new trips.
- A drawn route has no timing: it is converted to 15-second points at the speed you choose, which is an assumption, not data.
- The map tiles come from OpenStreetMap and need an internet connection in the user's browser.

## 7. FastAPI REST API

The application includes a production-ready FastAPI service in `api/`:

- **Interactive Swagger Docs**: `http://localhost:8000/docs`
- **ReDoc API Reference**: `http://localhost:8000/redoc`
- **Endpoints**:
  - `GET /health` — Service readiness & model status
  - `GET /model/info` — Loaded model version, feature names, parameters
  - `GET /model/metrics` — Validation RMSLE/RMSE benchmarks and slice evaluations
  - `POST /predict` — Single trip ETA prediction with empirical interval bounds
  - `POST /predict/curve` — Point-by-point prefix progression along a trace
  - `POST /predict/batch` — Bulk predictions via JSON array
  - `POST /predict/batch/csv` — Bulk predictions via CSV upload
  - `POST /tools/resample-route` — Convert drawn polyline coordinates to 15s GPS steps

## 8. Ideas for later

User accounts; prediction logging and drift monitoring (compare live feature means with `metrics.json`); CI that runs `pytest` on every change.

## Layout

```
api/main.py              FastAPI service entrypoint   taxi_eta/features.py   feature code copied from the notebook
api/routes.py            FastAPI route handlers       taxi_eta/traces.py     parsing, validation, resampling, batch reading
api/schemas.py           Pydantic schemas             taxi_eta/predictor.py  loads artifacts, predicts
app/main.py              Streamlit dashboard          taxi_eta/config.py     environment settings
scripts/                 notebook export cell         models/                put the real artifacts here
tests/                   pytest suite + fixtures
```
