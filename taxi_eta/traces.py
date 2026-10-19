"""Parsing, validation and resampling of GPS traces entered by users, plus batch-file reading.

A trace is an (n, 2) float array of [longitude, latitude] points, one every 15 seconds, exactly as in the
competition data. Nothing here depends on the model.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .features import CALL_MAP, MAX_SPEED_KMH, STEP_S, haversine

PORTO_BBOX = {"lon_min": -8.80, "lon_max": -8.40, "lat_min": 40.95, "lat_max": 41.35}
MAX_POINTS = 2000                      # hard limit per trace (about 8 hours)
LONG_TRIP_S = 3 * 3600                 # the model was trained on trips of at most 3 hours
_NUM = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
LOCAL_TZ = "Europe/Lisbon"


@dataclass
class TraceResult:
    pts: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    notes: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and len(self.pts) >= 1


@dataclass
class TripRequest:
    pts: np.ndarray
    ts: int                            # trip start, Unix seconds (UTC)
    call: int = 2                      # 0 = A (dispatch), 1 = B (stand), 2 = C (street)
    stand: int = -1                    # stand id, -1 if not a stand trip
    taxi: Optional[int] = None         # taxi id, None if unknown


# ----------------------------------------------------------------------------- time
def local_to_unix(dt) -> int:
    """Naive datetimes are read as Porto local time; aware datetimes are converted."""
    t = pd.Timestamp(dt)
    if t.tzinfo is None:
        t = t.tz_localize(LOCAL_TZ, ambiguous=True, nonexistent="shift_forward")
    return int(t.timestamp())


# ----------------------------------------------------------------------------- parsing
def _as_pairs(obj):
    """Accept [[lon, lat], ...], GeoJSON LineString / Feature, or {"coordinates": [...]}."""
    if isinstance(obj, dict):
        if "coordinates" in obj:
            return obj["coordinates"]
        if "geometry" in obj and isinstance(obj["geometry"], dict):
            return obj["geometry"].get("coordinates")
        return None
    return obj


def parse_trace_text(text: str) -> TraceResult:
    res = TraceResult()
    text = (text or "").strip()
    if not text:
        res.errors.append("The trace is empty.")
        return res
    arr = None
    if text[0] in "[{":
        try:
            pairs = _as_pairs(json.loads(text))
            arr = np.array([[float(p[0]), float(p[1])] for p in pairs], dtype=np.float64)
        except Exception:
            arr = None                                           # fall back to line parsing
    if arr is None:
        rows = []
        for ln, line in enumerate(re.split(r"[\n;]+", text), start=1):
            if ln == 1 and re.search(r"[A-Za-z]", line):
                continue                                         # header line such as "lon,lat"
            nums = _NUM.findall(line)
            if len(nums) < 2:
                if ln == 1 or not line.strip():
                    continue                                     # header or blank line
                res.errors.append(f"Line {ln} does not contain two numbers: {line.strip()[:40]!r}")
                continue
            rows.append([float(nums[0]), float(nums[1])])
        if res.errors:
            return res
        arr = np.array(rows, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) == 0:
        res.errors.append("Could not read any [longitude, latitude] points.")
        return res
    # latitude, longitude order? (Porto: latitude about 41, longitude about -8.6)
    if np.all(np.abs(arr[:, 0] - 41.15) < 1.5) and np.all(np.abs(arr[:, 1] + 8.6) < 1.5):
        arr = arr[:, ::-1].copy()
        res.notes.append("The columns looked like (latitude, longitude); they were swapped to (longitude, latitude).")
    res.pts = arr
    errors, warnings = validate_points(arr)
    res.errors += errors
    res.warnings += warnings
    return res


def validate_points(pts: np.ndarray):
    errors, warnings = [], []
    n = len(pts)
    if n < 1:
        return ["The trace has no points."], warnings
    if n > MAX_POINTS:
        errors.append(f"The trace has {n:,} points; the limit is {MAX_POINTS:,}.")
    if not np.all(np.isfinite(pts)):
        errors.append("The trace contains missing or non-numeric coordinates.")
        return errors, warnings
    if np.any(np.abs(pts[:, 0]) > 180) or np.any(np.abs(pts[:, 1]) > 90):
        errors.append("Some coordinates are outside the valid range (longitude +/-180, latitude +/-90).")
        return errors, warnings
    b = PORTO_BBOX
    outside = int(np.sum((pts[:, 0] < b["lon_min"]) | (pts[:, 0] > b["lon_max"]) | (pts[:, 1] < b["lat_min"]) | (pts[:, 1] > b["lat_max"])))
    if outside:
        warnings.append(f"{outside} point(s) lie outside the Porto area the model was trained on; the prediction may be unreliable.")
    if n >= 2:
        sp = haversine(pts[:-1, 0], pts[:-1, 1], pts[1:, 0], pts[1:, 1]) * 3600.0 / STEP_S
        jumps = int(np.sum(sp > MAX_SPEED_KMH))
        if jumps:
            warnings.append(f"{jumps} step(s) imply more than {MAX_SPEED_KMH:.0f} km/h (GPS jumps or missing points).")
        if np.all(sp < 0.5):
            warnings.append("The taxi has not moved at all in this trace.")
    else:
        warnings.append("Only the start point is known; the prediction is based on position and time only.")
    if (n - 1) * STEP_S > LONG_TRIP_S:
        warnings.append("The trip is already longer than 3 hours, the longest trip type the model was trained on.")
    return errors, warnings


def to_polyline_json(pts: np.ndarray) -> str:
    return json.dumps([[round(float(a), 6), round(float(b), 6)] for a, b in pts], separators=(",", ":"))


# ----------------------------------------------------------------------------- drawing
def resample_route(vertices, speed_kmh: float) -> np.ndarray:
    """Turn a drawn route (a few vertices) into points one 15-second step apart at a constant speed."""
    v = np.asarray(vertices, dtype=np.float64)
    if len(v) < 2:
        return v[:1].copy()
    seg = haversine(v[:-1, 0], v[:-1, 1], v[1:, 0], v[1:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] <= 0:
        return v[:1].copy()
    step = max(float(speed_kmh), 0.1) * STEP_S / 3600.0
    d = np.arange(0.0, cum[-1] + 1e-12, step)
    return np.column_stack([np.interp(d, cum, v[:, 0]), np.interp(d, cum, v[:, 1])])


# ----------------------------------------------------------------------------- batch files
_ALIASES = {
    "id": ["tripid", "id", "trip", "name"],
    "polyline": ["polyline", "trace", "gps", "points", "path", "route"],
    "timestamp": ["timestamp", "starttime", "start", "ts", "datetime", "time"],
    "call": ["calltype", "call"],
    "stand": ["originstand", "stand"],
    "taxi": ["taxiid", "taxi", "driver"],
    "actual": ["actualtotals", "actualtotal", "traveltime", "truetotals", "actual"],
}


def _norm(c: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(c).lower())


def map_columns(df: pd.DataFrame) -> dict:
    by_norm = {_norm(c): c for c in df.columns}
    out = {}
    for key, names in _ALIASES.items():
        for n in names:
            if n in by_norm:
                out[key] = by_norm[n]
                break
    return out


def _parse_timestamp(v):
    if pd.isna(v):
        raise ValueError("missing timestamp")
    try:
        f = float(v)
        return int(f / 1000) if f > 1e11 else int(f)           # Unix seconds (or milliseconds)
    except (TypeError, ValueError):
        t = pd.to_datetime(str(v), errors="coerce")
        if pd.isna(t):
            raise ValueError(f"cannot read timestamp {v!r}")
        return local_to_unix(t)


def read_batch(df: pd.DataFrame):
    """Return (requests, info). requests[i] is None for rows that cannot be used; info has one dict per row."""
    cols = map_columns(df)
    missing = [k for k in ("polyline", "timestamp") if k not in cols]
    if missing:
        raise ValueError(f"Required column(s) not found: {', '.join(missing)}. "
                         f"Expected at least a polyline column and a timestamp column; found: {list(df.columns)}")
    requests, info = [], []
    for i, row in df.iterrows():
        rid = row[cols["id"]] if "id" in cols else i + 1
        try:
            tr = parse_trace_text(str(row[cols["polyline"]]))
            if not tr.ok:
                raise ValueError("; ".join(tr.errors))
            ts = _parse_timestamp(row[cols["timestamp"]])
            call = 2
            if "call" in cols and not pd.isna(row[cols["call"]]):
                call = CALL_MAP.get(str(row[cols["call"]]).strip().upper(), None)
                if call is None:
                    raise ValueError(f"unknown call type {row[cols['call']]!r} (use A, B or C)")
            stand = -1
            if "stand" in cols and not pd.isna(row[cols["stand"]]):
                stand = int(float(row[cols["stand"]]))
            taxi = None
            if "taxi" in cols and not pd.isna(row[cols["taxi"]]):
                taxi = int(float(row[cols["taxi"]]))
            requests.append(TripRequest(tr.pts, ts, call, stand, taxi))
            msgs = tr.notes + tr.warnings
            if "call" not in cols:
                msgs.append("no call type column: assumed C (street)")
            info.append({"id": rid, "status": "ok", "message": "; ".join(msgs),
                         "actual_total_s": float(row[cols["actual"]]) if "actual" in cols and not pd.isna(row[cols["actual"]]) else np.nan})
        except Exception as exc:                                   # one bad row must not stop the batch
            requests.append(None)
            info.append({"id": rid, "status": "error", "message": str(exc)[:200], "actual_total_s": np.nan})
    return requests, info


TEMPLATE_CSV = (
    "trip_id,polyline,timestamp,call_type,origin_stand,taxi_id\n"
    'example_1,"[[-8.6110,41.1496],[-8.6105,41.1498],[-8.6098,41.1500],[-8.6090,41.1503]]",2014-08-14 18:00:00,C,,20000589\n'
    'example_2,"[[-8.6291,41.1579],[-8.6288,41.1583],[-8.6282,41.1588]]",2014-08-14 18:00:00,B,15,20000080\n'
)
