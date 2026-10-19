"""Tests for the FastAPI REST API service."""
from __future__ import annotations

import io
import json
import pytest
from fastapi.testclient import TestClient

from conftest import FIX, ROOT
from taxi_eta.predictor import Predictor
from api.main import app
from api.routes import set_predictor

GOOD_TRACE = [
    [-8.6110, 41.1496],
    [-8.6105, 41.1498],
    [-8.6098, 41.1500],
    [-8.6090, 41.1503],
]


@pytest.fixture(scope="module")
def client():
    import os
    old_md = os.environ.get("MODEL_DIR")
    os.environ["MODEL_DIR"] = str(FIX / "model")
    pred = Predictor(str(FIX / "model"))
    set_predictor(pred, None)
    from api.routes import get_predictor
    app.dependency_overrides[get_predictor] = lambda: pred
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
    if old_md is not None:
        os.environ["MODEL_DIR"] = old_md
    else:
        os.environ.pop("MODEL_DIR", None)


def test_api_root(client):
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert "service" in data
    assert data["documentation"] == "/docs"
    assert "endpoints" in data


def test_health_check(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["synthetic"] is True


def test_model_info(client):
    res = client.get("/model/info")
    assert res.status_code == 200
    data = res.json()
    assert data["version"].startswith("v4")
    assert data["n_features"] > 0
    assert "elapsed_time_s" in data["features"]


def test_model_metrics(client):
    res = client.get("/model/metrics")
    assert res.status_code == 200
    data = res.json()
    assert "baseline" in data
    assert "models" in data


def test_predict_single_success(client):
    payload = {
        "polyline": GOOD_TRACE,
        "timestamp": 1407000000,
        "call_type": "B",
        "origin_stand": 15,
        "taxi_id": 20000589,
        "level": 0.8,
        "include_features": True,
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["observed_points"] == 4
    assert data["elapsed_s"] == 45.0
    assert data["total_s"] >= data["elapsed_s"] + 15.0
    assert data["remaining_s"] >= 15.0
    assert data["formatted_total"]
    assert data["features"] is not None
    assert "dist_travelled_km" in data["features"]


def test_predict_single_text_polyline(client):
    payload = {
        "polyline": json.dumps(GOOD_TRACE),
        "timestamp": "2014-08-14 18:00:00",
        "call_type": "C",
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["observed_points"] == 4
    assert data["total_s"] > 0


def test_predict_invalid_trace(client):
    payload = {
        "polyline": "invalid coordinates",
        "timestamp": 1407000000,
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 422


def test_predict_curve(client):
    payload = {
        "polyline": GOOD_TRACE,
        "timestamp": 1407000000,
        "step": 1,
        "level": 0.8,
    }
    res = client.post("/predict/curve", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert len(data["curve"]) == 4
    assert [p["k"] for p in data["curve"]] == [1, 2, 3, 4]


def test_predict_batch_json(client):
    payload = {
        "trips": [
            {
                "id": "trip_1",
                "polyline": GOOD_TRACE,
                "timestamp": 1407000000,
                "call_type": "C",
                "actual_total_s": 600.0,
            },
            {
                "id": "trip_2",
                "polyline": [[-8.6110, 41.1496]],
                "timestamp": 1407000000,
            },
            {
                "id": "bad_trip",
                "polyline": "garbage",
                "timestamp": 1407000000,
            },
        ],
        "level": 0.8,
    }
    res = client.post("/predict/batch", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert len(data["results"]) == 3
    assert data["results"][0]["status"] == "ok"
    assert data["results"][0]["prediction"] is not None
    assert data["results"][0]["error_s"] is not None
    assert data["results"][1]["status"] == "ok"
    assert data["results"][2]["status"] == "error"
    assert data["summary"]["successful_predictions"] == 2
    assert data["summary"]["failed_rows"] == 1


def test_predict_batch_csv(client):
    csv_content = (
        "trip_id,polyline,timestamp,call_type,origin_stand,taxi_id,actual_total_s\n"
        f't1,"{json.dumps(GOOD_TRACE)}",1407000000,B,15,20000589,700.0\n'
        f't2,"{json.dumps(GOOD_TRACE)}",1407000000,C,,,650.0\n'
    )
    files = {"file": ("trips.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    res = client.post("/predict/batch/csv", files=files)
    assert res.status_code == 200
    data = res.json()
    assert len(data["results"]) == 2
    assert data["summary"]["successful_predictions"] == 2
    assert "rmsle" in data["summary"]


def test_resample_drawn_route(client):
    payload = {
        "vertices": [
            [-8.6110, 41.1496],
            [-8.5990, 41.1520],
            [-8.5855, 41.1486],
        ],
        "speed_kmh": 30.0,
    }
    res = client.post("/tools/resample-route", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["n_points"] > 2
    assert data["driven_seconds"] > 0
