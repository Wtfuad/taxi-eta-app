"""Porto taxi trip-time predictor: Streamlit dashboard.

Run:  streamlit run app/main.py        (settings come from environment variables, see README)
"""
from __future__ import annotations

import hmac
import json
import logging
import sys
from datetime import date, datetime, time as dtime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from folium.plugins import Draw
from streamlit_folium import st_folium

from taxi_eta.config import Settings
from taxi_eta.features import CALL_MAP, STEP_S
from taxi_eta.predictor import ArtifactError, Predictor
from taxi_eta.traces import (MAX_POINTS, TEMPLATE_CSV, TripRequest, local_to_unix, parse_trace_text, read_batch,
                             resample_route, to_polyline_json, validate_points)

st.set_page_config(page_title="Porto Taxi Trip-Time Predictor", page_icon="🚖", layout="wide")
SET = Settings.from_env()
logging.basicConfig(level=SET.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("taxi_eta.app")
PORTO = (41.1579, -8.6291)
CALL_LABELS = ["A: dispatched by phone", "B: picked up at a taxi stand", "C: street hail"]


# ----------------------------------------------------------------------------- helpers
def fmt_dur(seconds: float) -> str:
    if seconds is None or not np.isfinite(seconds):
        return "n/a"
    s = int(round(seconds))
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h} h {m:02d} min" if h else f"{m} min {s:02d} s"


def check_password() -> bool:
    if not SET.app_password:
        return True
    if st.session_state.get("auth_ok"):
        return True
    st.title("🚖 Porto Taxi Trip-Time Predictor")
    entered = st.text_input("Password", type="password")
    if entered:
        if hmac.compare_digest(entered.encode(), SET.app_password.encode()):
            st.session_state["auth_ok"] = True
            st.rerun()
        else:
            st.error("Wrong password.")
    return False


@st.cache_resource(show_spinner="Loading the model ...")
def load_predictor(model_dir: str) -> Predictor:
    return Predictor(model_dir)


@st.cache_data(show_spinner=False)
def load_demo(model_dir: str):
    p = Path(model_dir) / "demo_trips.csv"
    return pd.read_csv(p) if p.exists() else None


def trace_map(pts: np.ndarray, rest: np.ndarray | None = None) -> folium.Map:
    ll = [[float(p[1]), float(p[0])] for p in pts]
    m = folium.Map(location=ll[0], zoom_start=14, tiles="OpenStreetMap", control_scale=True)
    if rest is not None and len(rest) > 1:
        folium.PolyLine([[float(p[1]), float(p[0])] for p in rest], color="#888888", weight=3, dash_array="6", tooltip="rest of the trip (hidden from the model)").add_to(m)
    if len(ll) > 1:
        folium.PolyLine(ll, color="#3b6ea5", weight=5, opacity=0.9, tooltip="observed trace").add_to(m)
    folium.CircleMarker(ll[0], radius=7, color="green", fill=True, fill_opacity=1, tooltip="start").add_to(m)
    folium.CircleMarker(ll[-1], radius=7, color="red", fill=True, fill_opacity=1, tooltip="current position").add_to(m)
    allp = ll + ([[float(p[1]), float(p[0])] for p in rest] if rest is not None else [])
    if len(allp) > 1:
        lat, lon = [p[0] for p in allp], [p[1] for p in allp]
        m.fit_bounds([[min(lat), min(lon)], [max(lat), max(lon)]], padding=(25, 25))
    return m


def example_trace() -> str:
    return to_polyline_json(resample_route([(-8.6110, 41.1496), (-8.5990, 41.1520), (-8.5855, 41.1486)], 25))


# ----------------------------------------------------------------------------- gate and model
if not check_password():
    st.stop()
try:
    predictor = load_predictor(str(SET.model_dir))
except ArtifactError as exc:
    st.error(f"The model could not be loaded: {exc}")
    st.stop()
info = predictor.info

if info.get("synthetic"):
    st.warning("⚠️ This is a PREVIEW model trained on synthetic data. Its predictions are meaningless; it only shows how the interface works.")

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🚖 Trip-time predictor")
    st.caption(f"Model {info['version']} · {info['n_features']} features · {info['n_final_models']} final models")
    st.subheader("Trip details")
    d = st.date_input("Start date", value=date(2014, 8, 14))
    t = st.time_input("Start time (Porto time)", value=dtime(18, 0))
    call_label = st.radio("How was the taxi called?", CALL_LABELS, index=2, key="call")
    call = CALL_LABELS.index(call_label)
    stand = int(st.number_input("Taxi stand", min_value=1, max_value=63, value=15, step=1)) if call == 1 else -1
    taxi_txt = st.text_input("Taxi ID (optional)", value="", key="taxi", help="Leave empty if unknown; the model then uses the average over all taxis.")
    taxi = int(taxi_txt) if taxi_txt.strip().isdigit() else None
    if taxi_txt.strip() and taxi is None:
        st.warning("The taxi ID must be a number; it is ignored.")
    level = st.select_slider("Prediction range", options=[0.8, 0.9], value=0.8, format_func=lambda v: f"{int(v * 100)}%", key="level")
    st.divider()
    st.caption("Estimates from a statistical model trained on Porto taxi trips of 2013-2014. Not for operational dispatch decisions.")

BASE = {"ts": local_to_unix(datetime.combine(d, t)), "call": call, "stand": stand, "taxi": taxi}


# ----------------------------------------------------------------------------- single-trip result
def show_prediction(pts: np.ndarray, base: dict, truth: float | None = None, rest: np.ndarray | None = None, key: str = "res"):
    errors, warnings = validate_points(pts)
    for w in warnings:
        st.warning(w)
    if errors:
        for e in errors:
            st.error(e)
        return
    req = TripRequest(pts, **base)
    pred, feats = predictor.predict_with_features([req], level)
    r = pred.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted total trip time", fmt_dur(r.total_s))
    c2.metric("Driven so far", fmt_dur(r.elapsed_s), help=f"{int(r.observed_points)} GPS points observed")
    c3.metric("Predicted time remaining", fmt_dur(r.remaining_s))
    c4.metric(f"{int(level * 100)}% range for the total", f"{fmt_dur(r.low_s)} to {fmt_dur(r.high_s)}" if np.isfinite(r.low_s) else "n/a")
    st.caption(f"The range comes from the model's errors on held-out validation trips with a similar number of observed points "
               f"({int(level * 100)}% of them fell inside such a range). It is empirical, not a guarantee.")
    if truth is not None:
        a1, a2 = st.columns(2)
        a1.metric("Actual total trip time", fmt_dur(truth))
        a2.metric("Prediction error", f"{(r.total_s - truth) / 60:+.1f} min", help="predicted minus actual")
    st_folium(trace_map(pts, rest), height=380, use_container_width=True, returned_objects=[], key=f"map_{key}")
    with st.expander("Inputs the model used"):
        st.dataframe(feats[predictor.features].T.rename(columns={0: "value"}), height=300, width="stretch")
    result = {"start_unix": base["ts"], "observed_points": int(r.observed_points), "predicted_total_s": round(float(r.total_s), 1),
              "predicted_remaining_s": round(float(r.remaining_s), 1), f"low_{int(level * 100)}_s": round(float(r.low_s), 1),
              f"high_{int(level * 100)}_s": round(float(r.high_s), 1), "model_version": info["version"]}
    st.download_button("Download this result (JSON)", json.dumps(result, indent=1), "prediction.json", "application/json", key=f"dl_{key}")
    log.info("single prediction: %d points, total %.0f s", len(pts), r.total_s)


def method_paste():
    st.session_state.setdefault("trace_text", "")
    st.text_area("GPS trace", key="trace_text", height=150,
                 placeholder='[[-8.6110,41.1496],[-8.6105,41.1498],...]   or one "longitude,latitude" pair per line',
                 help="One point every 15 seconds, oldest first. JSON (as in the competition data), GeoJSON LineString, or one pair per line. "
                      "Latitude/longitude order is detected automatically.")
    cols = st.columns([1, 3])
    cols[0].button("Load an example trace", on_click=lambda: st.session_state.update(trace_text=example_trace()))
    up = cols[1].file_uploader("...or upload a file (.txt, .csv, .json)", type=["txt", "csv", "json"], key="trace_file")
    text = st.session_state.trace_text
    if up is not None:
        if up.size > 2_000_000:
            st.error("The file is larger than 2 MB.")
            return
        text = up.getvalue().decode("utf-8", errors="replace")
    if not text.strip():
        st.info("Paste a trace, upload a file, or load the example to get a prediction.")
        return
    tr = parse_trace_text(text)
    for n in tr.notes:
        st.info(n)
    if tr.errors:
        for e in tr.errors:
            st.error(e)
        return
    show_prediction(tr.pts, BASE, key="paste")


def method_draw():
    st.write("Draw the route the taxi has driven so far (polyline tool, top left of the map; double-click to finish). "
             "Because a drawing has no timing, it is converted to GPS points 15 seconds apart at the speed you choose.")
    speed = st.slider("Assumed average speed (km/h)", 5, 80, 25, help="City traffic is typically 15-30 km/h.")
    m = folium.Map(location=PORTO, zoom_start=12, tiles="OpenStreetMap", control_scale=True)
    Draw(export=False, draw_options={"polyline": {"shapeOptions": {"color": "#3b6ea5"}}, "polygon": False, "rectangle": False,
                                     "circle": False, "marker": False, "circlemarker": False}, edit_options={"edit": False}).add_to(m)
    out = st_folium(m, height=460, use_container_width=True, key="draw_map", returned_objects=["all_drawings"])
    drawings = (out or {}).get("all_drawings") or []
    lines = [x for x in drawings if (x.get("geometry") or {}).get("type") == "LineString"]
    if not lines:
        st.info("Draw a route on the map to get a prediction.")
        return
    verts = lines[-1]["geometry"]["coordinates"]
    pts = resample_route(verts, speed)
    st.caption(f"{len(verts)} map points became {len(pts)} GPS points at {speed} km/h ({fmt_dur((len(pts) - 1) * STEP_S)} driven).")
    if len(pts) > MAX_POINTS:
        st.error(f"That is {len(pts):,} points; the limit is {MAX_POINTS:,}. Draw a shorter route or raise the speed.")
        return
    show_prediction(pts, BASE, key="draw")


def method_replay():
    demo = load_demo(str(SET.model_dir))
    if demo is None or demo.empty:
        st.info("No demo trips are included in this deployment (demo_trips.csv is missing from the model folder).")
        return
    st.info("These trips come from the dataset. They were not used as training rows, but they did contribute to the historical averages, "
            "so accuracy here can look better than on brand-new trips. See the *Model & performance* tab for validation numbers.")
    demo = demo.reset_index(drop=True)
    idx = st.selectbox("Trip", demo.index, format_func=lambda i: f"Trip {int(demo.trip_id[i])} · actual {fmt_dur(demo.true_total_s[i])}")
    row = demo.loc[idx]
    pts_all = np.array(json.loads(row["polyline"]), dtype=np.float64)
    n = len(pts_all)
    k = st.slider("GPS points observed so far", 1, n - 1, min(max(n // 3, 1), n - 1),
                  help="Move the slider to see how the prediction changes as more of the trip is observed.")
    stand_v = -1 if pd.isna(row["origin_stand"]) else int(row["origin_stand"])
    base = {"ts": int(row["timestamp"]), "call": CALL_MAP[str(row["call_type"])], "stand": stand_v, "taxi": int(row["taxi_id"])}
    st.caption(f"Trip details (start time, call type, stand, taxi) come from the dataset; the sidebar is ignored here. Call type {row['call_type']}.")
    show_prediction(pts_all[:k], base, truth=float(row["true_total_s"]), rest=pts_all[k - 1:], key="replay")
    if st.checkbox("Show how the estimate evolves as the trip progresses", value=True):
        cur = predictor.predict_curve(pts_all[:n - 1], TripRequest(pts_all, **base), step=1, level=level)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cur.k, y=cur.high_s / 60, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=cur.k, y=cur.low_s / 60, mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(59,110,165,0.2)",
                                 name=f"{int(level * 100)}% range"))
        fig.add_trace(go.Scatter(x=cur.k, y=cur.total_s / 60, mode="lines", name="predicted total", line=dict(color="#3b6ea5", width=3)))
        fig.add_hline(y=float(row["true_total_s"]) / 60, line_dash="dash", line_color="#c0504d", annotation_text="actual total")
        fig.add_vline(x=k, line_dash="dot", line_color="gray")
        fig.update_layout(xaxis_title="GPS points observed", yaxis_title="Total trip time (min)", height=360, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
        st.plotly_chart(fig, width="stretch")


# ----------------------------------------------------------------------------- batch
def tab_batch():
    st.write("Upload a CSV with one trip per row. Required columns: **polyline** (JSON list of [longitude, latitude]) and **timestamp** "
             "(Unix seconds, or a date-time in Porto time). Optional: trip_id, call_type (A/B/C), origin_stand, taxi_id, and "
             "actual_total_s to score the predictions. The competition's test file format works as is.")
    st.download_button("Download a CSV template", TEMPLATE_CSV, "trips_template.csv", "text/csv")
    up = st.file_uploader("CSV file", type=["csv"], key="batch_file")
    if up is None:
        return
    if up.size > 100_000_000:
        st.error("The file is larger than 100 MB.")
        return
    try:
        df = pd.read_csv(up)
    except Exception as exc:
        st.error(f"The file could not be read as CSV: {exc}")
        return
    if len(df) > SET.max_batch_rows:
        st.error(f"The file has {len(df):,} rows; the limit is {SET.max_batch_rows:,}. Split it into several files.")
        return
    try:
        requests, meta = read_batch(df)
    except ValueError as exc:
        st.error(str(exc))
        return
    ok = [i for i, r in enumerate(requests) if r is not None]
    st.write(f"**{len(df):,}** rows read, **{len(ok):,}** usable, **{len(df) - len(ok):,}** with problems.")
    if len(ok) < len(df):
        with st.expander("Rows with problems"):
            st.dataframe(pd.DataFrame(meta).query("status == 'error'")[["id", "message"]], width="stretch")
    if not ok:
        return
    if st.button("Run predictions", type="primary", key="run_batch"):
        bar = st.progress(0.0, text="Predicting ...")
        parts = []
        chunk = 500
        for s in range(0, len(ok), chunk):
            idxs = ok[s:s + chunk]
            parts.append(predictor.predict([requests[i] for i in idxs], level).assign(_row=idxs))
            bar.progress(min(1.0, (s + chunk) / len(ok)), text=f"Predicting ... {min(s + chunk, len(ok)):,} / {len(ok):,}")
        bar.empty()
        res = pd.concat(parts, ignore_index=True).set_index("_row")
        out = pd.DataFrame(meta)
        for c in ["observed_points", "elapsed_s", "total_s", "remaining_s", "low_s", "high_s"]:
            out[c] = np.nan
            out.loc[res.index, c] = res[c].values
        out["predicted_total_min"] = out.total_s / 60
        out["predicted_remaining_min"] = out.remaining_s / 60
        out["low_min"], out["high_min"], out["elapsed_min"] = out.low_s / 60, out.high_s / 60, out.elapsed_s / 60
        cols = ["id", "status", "message", "observed_points", "elapsed_min", "predicted_total_min", "predicted_remaining_min", "low_min", "high_min"]
        if out["actual_total_s"].notna().any():
            out["actual_total_min"] = out.actual_total_s / 60
            out["error_min"] = out.predicted_total_min - out.actual_total_min
            cols += ["actual_total_min", "error_min"]
        st.session_state["batch_out"] = (out, cols, int(level * 100))
        log.info("batch prediction: %d rows", len(ok))
    if "batch_out" in st.session_state:
        out, cols, lv = st.session_state["batch_out"]
        st.dataframe(out[cols].round(2), height=320, width="stretch")
        st.download_button("Download predictions (CSV)", out[cols].round(3).to_csv(index=False), "predictions.csv", "text/csv")
        good = out[out.status == "ok"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Trips predicted", f"{len(good):,}")
        c2.metric("Median predicted total", fmt_dur(good.total_s.median()))
        c3.metric("Longest predicted total", fmt_dur(good.total_s.max()))
        fig = go.Figure(go.Histogram(x=good.predicted_total_min.clip(upper=120), nbinsx=60, marker_color="#3b6ea5"))
        fig.update_layout(xaxis_title="Predicted total trip time (min, clipped at 120)", yaxis_title="Trips", height=300, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")
        sc = good.dropna(subset=["actual_total_s"])
        if len(sc):
            rmsle = float(np.sqrt(np.mean((np.log1p(sc.total_s) - np.log1p(sc.actual_total_s)) ** 2)))
            rmse = float(np.sqrt(np.mean((sc.total_s - sc.actual_total_s) ** 2)))
            st.success(f"Scored on {len(sc):,} trips with a known actual time: RMSLE {rmsle:.4f}, RMSE {rmse:.0f} s.")


# ----------------------------------------------------------------------------- model page
def tab_model():
    M = predictor.metrics
    st.subheader("Validation performance")
    if not M:
        st.info("No metrics file (metrics.json) in the model folder.")
    else:
        rows = [("Baseline: same time for every trip", M["baseline"]["rmsle"], M["baseline"]["rmse_s"]),
                ("Model, raw target", M["models"]["raw"]["rmsle"], M["models"]["raw"]["rmse_s"]),
                ("Model, log target", M["models"]["log"]["rmsle"], M["models"]["log"]["rmse_s"]),
                (f"Blend (used here, weight on raw = {M['blend_weight_raw']:.1f})", M["models"]["blend"]["rmsle"], M["models"]["blend"]["rmse_s"])]
        st.dataframe(pd.DataFrame(rows, columns=["model", "RMSLE (competition metric)", "RMSE (s)"]).round(4), hide_index=True, width="stretch")
        st.caption(f"Scored on {M['validation_rows']:,} cuts from the latest 15% of trips (by start time), which the validation models never saw. "
                   "The deployed models were refit on all data, so these numbers are a close but not exact guide.")
        if M.get("slices"):
            name = st.selectbox("Where is the model strong or weak?", list(M["slices"]))
            st.dataframe(pd.DataFrame(M["slices"][name]).set_index("slice"), width="stretch")
    st.subheader("Prediction ranges")
    q = predictor.err_q
    if q:
        st.write("The range around each prediction is taken from the model's own validation errors, separately for each number of observed GPS points. "
                 "Early in a trip the ranges are wide because little is known about where the passenger is going.")
        tab = pd.DataFrame({b: {k: float(np.expm1(-v)) for k, v in qq.items()} for b, qq in q["quantiles"].items()}).T
        tab.columns = [f"q{c}" for c in tab.columns]
        st.caption("Typical ratio true / predicted total time (minus 1) at each quantile of the error: negative = the trip took less than predicted.")
        st.dataframe((tab).round(3), width="stretch")
    st.subheader("Limitations")
    st.markdown(
        "- Trained on Porto taxi trips from July 2013 to June 2014 with GPS points every 15 seconds; other cities or sampling rates are not supported.\n"
        "- Trips longer than 3 hours were excluded from training, so such trips are not predicted reliably.\n"
        "- Short prefixes (a few GPS points) and very long trips have the largest errors; long trips tend to be under-predicted.\n"
        "- Traffic incidents, weather and events are unknown to the model.\n"
        "- This is a demonstration of a statistical model, not a validated dispatch system.")
    st.subheader("Model details")
    st.json({**info, "feature_names": predictor.features}, expanded=False)


def tab_about():
    st.markdown(
        "**What it does.** Given the GPS points a taxi has produced so far (one every 15 seconds), the start time and how the taxi was called, "
        "it predicts the **total** duration of the trip.\n\n"
        "**How it works.**\n"
        "1. The trace is turned into features: distance driven, speeds over several windows, stops, heading, position, time of day.\n"
        "2. Historical averages (by pickup area and hour, stand and taxi) and a table of what happened to past trips at the same place and heading are added.\n"
        "3. Small models estimate how far the taxi still has to go; their estimates are further inputs.\n"
        "4. Gradient-boosted tree models (LightGBM, several seeds) predict the remaining time in two ways and the results are blended.\n\n"
        "**Data handling.** Traces you enter or upload are processed in memory for the prediction and are not saved by this application; "
        "only counts and timings are logged. The hosting platform may keep its own access logs.")


# ----------------------------------------------------------------------------- layout
st.title("Porto Taxi Trip-Time Predictor")
st.caption("Predict how long a taxi trip will take from the part of the GPS trace seen so far.")
tab1, tab2, tab3, tab4 = st.tabs(["Single trip", "Batch upload", "Model & performance", "About"])
with tab1:
    method = st.radio("How do you want to enter the trip?", ["Paste or upload a trace", "Draw on a map", "Replay a dataset trip"], horizontal=True, key="method")
    if method.startswith("Paste"):
        method_paste()
    elif method.startswith("Draw"):
        method_draw()
    else:
        method_replay()
with tab2:
    tab_batch()
with tab3:
    tab_model()
with tab4:
    tab_about()
