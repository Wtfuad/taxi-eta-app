import json
import shutil

import numpy as np
import pandas as pd
import pytest

from conftest import FIX
from taxi_eta.features import CALL_MAP, STEP_S
from taxi_eta.predictor import ArtifactError, Predictor
from taxi_eta.traces import TripRequest


@pytest.fixture(scope="module")
def predictor():
    return Predictor(FIX / "model")


@pytest.fixture(scope="module")
def requests_expected():
    df = pd.read_csv(FIX / "test_traces.csv")
    exp = pd.read_csv(FIX / "expected_total_s.csv").set_index("TRIP_ID")["total_s"]
    reqs = []
    for _, r in df.iterrows():
        pts = np.array(json.loads(r.POLYLINE), dtype=float).reshape(-1, 2)
        stand = -1 if pd.isna(r.ORIGIN_STAND) else int(r.ORIGIN_STAND)
        reqs.append(TripRequest(pts, int(r.TIMESTAMP), CALL_MAP[r.CALL_TYPE], stand, int(r.TAXI_ID)))
    return reqs, exp.loc[df.TRIP_ID].values


def test_parity_with_notebook_predictions(predictor, requests_expected):
    """The contract: the dashboard must reproduce the notebook's submission values (train/serve skew guard)."""
    reqs, expected = requests_expected
    got = predictor.predict(reqs).total_s.values
    assert np.allclose(got, expected, rtol=0, atol=1e-6), np.abs(got - expected).max()


def test_predictions_are_sane(predictor, requests_expected):
    reqs, _ = requests_expected
    out = predictor.predict(reqs)
    assert np.isfinite(out[["total_s", "remaining_s", "low_s", "high_s"]].values).all()
    assert (out.total_s >= out.elapsed_s + STEP_S - 1e-9).all()
    assert (out.remaining_s >= STEP_S - 1e-9).all()
    assert (out.low_s <= out.high_s).all() and (out.low_s >= out.elapsed_s + STEP_S - 1e-9).all()


def test_batch_equals_single(predictor, requests_expected):
    reqs, _ = requests_expected
    batch = predictor.predict(reqs[:5]).total_s.values
    single = np.array([predictor.predict([r]).total_s.iloc[0] for r in reqs[:5]])
    assert np.allclose(batch, single, atol=1e-9)


def test_unknown_taxi_and_stand_do_not_break(predictor, requests_expected):
    reqs, _ = requests_expected
    r = reqs[0]
    a = predictor.predict([TripRequest(r.pts, r.ts, r.call, r.stand, None)]).total_s.iloc[0]
    b = predictor.predict([TripRequest(r.pts, r.ts, r.call, r.stand, 99999999)]).total_s.iloc[0]   # taxi never seen in training
    assert np.isfinite(a) and np.isfinite(b) and abs(a - b) < 1e-9                                  # both fall back to the global prior


def test_single_point_trace_works(predictor):
    out = predictor.predict([TripRequest(np.array([[-8.611, 41.1496]]), 1407000000, 2, -1, None)])
    assert np.isfinite(out.total_s.iloc[0]) and out.observed_points.iloc[0] == 1


def test_curve_matches_direct_prediction(predictor, requests_expected):
    reqs, _ = requests_expected
    r = next(x for x in reqs if len(x.pts) >= 12)
    cur = predictor.predict_curve(r.pts, r)
    assert list(cur.k) == list(range(1, len(r.pts) + 1))
    assert np.isclose(cur.total_s.iloc[-1], predictor.predict([r]).total_s.iloc[0])


def test_interval_level_ordering(predictor, requests_expected):
    reqs, _ = requests_expected
    a, b = predictor.predict(reqs[:20], 0.8), predictor.predict(reqs[:20], 0.9)
    assert (b.low_s <= a.low_s + 1e-9).all() and (b.high_s >= a.high_s - 1e-9).all()


def test_empty_request_list(predictor):
    assert predictor.predict([]).empty


def test_missing_model_gives_clear_error(tmp_path):
    with pytest.raises(ArtifactError, match="No model found"):
        Predictor(tmp_path)


def test_incomplete_model_is_rejected(tmp_path):
    shutil.copytree(FIX / "model", tmp_path / "m")
    aux_p = (tmp_path / "m" / "aux") if (tmp_path / "m" / "aux").exists() else (tmp_path / "m" / "_aux")
    next(aux_p.glob("*.txt")).unlink()
    with pytest.raises(ArtifactError, match="Incomplete model files"):
        Predictor(tmp_path / "m")


def test_constant_mismatch_is_rejected(tmp_path):
    shutil.copytree(FIX / "model", tmp_path / "m")
    cfg = json.loads((tmp_path / "m" / "config.json").read_text())
    cfg["constants"]["MAX_SPEED_KMH"] = 120.0
    (tmp_path / "m" / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(ArtifactError, match="MAX_SPEED_KMH"):
        Predictor(tmp_path / "m")
