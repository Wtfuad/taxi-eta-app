"""Feature engineering for the Porto taxi trip-time model.

The function bodies below are copied VERBATIM from the training notebook (taxi-time-pred v4) so that
training and serving compute identical features. Do not edit them here: change the notebook, re-export
the artifacts, and copy the functions again. tests/test_features_parity.py guards the contract.
"""
import numpy as np
import pandas as pd

# constants: must equal the values used when the model was trained (checked against config.json on load)
STEP_S = 15.0
MAX_SPEED_KMH = 160.0
R_EARTH = 6371.0
CENTER = (-8.6110, 41.1496)
AIRPORT = (-8.6814, 41.2481)
CALL_MAP = {"A": 0, "B": 1, "C": 2}
CONSTANTS = {"STEP_S": STEP_S, "MAX_SPEED_KMH": MAX_SPEED_KMH, "R_EARTH": R_EARTH,
             "CENTER": list(CENTER), "AIRPORT": list(AIRPORT)}


def haversine(lon1, lat1, lon2, lat2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def grid_id(lon, lat, dlon=0.006, dlat=0.0045):
    gx = np.floor((np.asarray(lon, float) + 8.80) / dlon)
    gy = np.floor((np.asarray(lat, float) - 41.00) / dlat)
    gid = np.clip(gx, 0, 999) * 1000 + np.clip(gy, 0, 999)
    return np.where(np.isnan(gid), -1, gid).astype(np.int64)


def local_time(ts):
    return pd.to_datetime(ts, unit="s", utc=True).dt.tz_convert("Europe/Lisbon")


BASE_KEYS = ["pickup_lon", "pickup_lat", "curr_lon", "curr_lat", "dist_travelled_km", "displacement_km", "tortuosity",
             "avg_speed_kmh", "speed_last3_kmh", "speed_last5_kmh", "speed_last10_kmh", "speed_std_kmh", "max_speed_kmh",
             "stop_frac", "accel_last", "head_sin", "head_cos", "heading_change", "n_gaps"]


STALL_KEYS = ["stop_run_end_s", "longest_stop_s", "n_stop_runs", "stop_frac_last20", "stop_frac_last60",
              "speed_last20_kmh", "moving_speed_kmh", "moving_time_s", "dcenter_d5", "dcenter_d20",
              "dairport_d10", "max_from_pickup_km", "frac_of_max_from_pickup"]


def _bearing(a, b, coslat):
    dx, dy = (b[0] - a[0]) * coslat, b[1] - a[1]
    return np.arctan2(dx, dy) if dx * dx + dy * dy > (5e-5) ** 2 else np.nan


def trip_features(pts, k):
    """Features of the first k points of a trip (k = len(pts) for a test trip)."""
    f = dict.fromkeys(BASE_KEYS + STALL_KEYS, np.nan)
    f["elapsed_time_s"] = (k - 1) * STEP_S if k > 0 else 0.0
    if k == 0:
        return f
    obs = pts[:k]
    (lon0, lat0), (lon1, lat1) = obs[0], obs[-1]
    f.update(pickup_lon=lon0, pickup_lat=lat0, curr_lon=lon1, curr_lat=lat1)
    disp = float(haversine(lon0, lat0, lon1, lat1))
    f["displacement_km"] = disp
    if k == 1:
        f.update(dist_travelled_km=0.0, tortuosity=1.0, n_gaps=0)
        return f

    sd = haversine(obs[:-1, 0], obs[:-1, 1], obs[1:, 0], obs[1:, 1])     # km per step
    sp = sd * 3600.0 / STEP_S                                            # km/h
    spc = np.minimum(sp, MAX_SPEED_KMH)
    dist = float(sd.sum())
    f["dist_travelled_km"] = dist
    f["tortuosity"] = float(np.clip(dist / max(disp, 0.05), 0, 50))
    f["avg_speed_kmh"] = dist * 3600.0 / (len(sd) * STEP_S)
    for m in (3, 5, 10):
        f[f"speed_last{m}_kmh"] = float(sd[-m:].mean() * 3600.0 / STEP_S)
    f["speed_std_kmh"] = float(spc.std())
    f["max_speed_kmh"] = float(spc.max())
    f["stop_frac"] = float((sp < 1.0).mean())
    f["n_gaps"] = int((sp > MAX_SPEED_KMH).sum())
    if len(sd) >= 10:
        f["accel_last"] = float((sd[-5:].mean() - sd[-10:-5].mean()) * 3600.0 / STEP_S)

    coslat = np.cos(np.radians(lat1))
    b_last = _bearing(obs[max(0, k - 6)], obs[-1], coslat)
    b_first = _bearing(obs[0], obs[min(5, k - 1)], coslat)
    if not np.isnan(b_last):
        f["head_sin"], f["head_cos"] = float(np.sin(b_last)), float(np.cos(b_last))
        if not np.isnan(b_first):
            f["heading_change"] = float(abs((b_last - b_first + np.pi) % (2 * np.pi) - np.pi))

    # stalls and trends
    st = sp < 1.0
    runs_edge = np.diff(np.concatenate(([0], st.astype(np.int8), [0])))
    starts, ends = np.flatnonzero(runs_edge == 1), np.flatnonzero(runs_edge == -1)
    runs = ends - starts
    f["n_stop_runs"] = len(runs)
    f["longest_stop_s"] = (runs.max() if len(runs) else 0) * STEP_S
    f["stop_run_end_s"] = (runs[-1] if len(runs) and ends[-1] == len(st) else 0) * STEP_S      # currently stopped for
    f["stop_frac_last20"], f["stop_frac_last60"] = float(st[-20:].mean()), float(st[-60:].mean())
    f["speed_last20_kmh"] = float(sd[-20:].mean() * 3600.0 / STEP_S)
    mv = ~st
    f["moving_time_s"] = float(mv.sum() * STEP_S)
    f["moving_speed_kmh"] = float(np.minimum(sp[mv], MAX_SPEED_KMH).mean()) if mv.any() else 0.0
    dc, da = haversine(obs[:, 0], obs[:, 1], *CENTER), haversine(obs[:, 0], obs[:, 1], *AIRPORT)
    f["dcenter_d5"] = float(dc[-1] - dc[max(0, k - 6)])
    f["dcenter_d20"] = float(dc[-1] - dc[max(0, k - 21)])
    f["dairport_d10"] = float(da[-1] - da[max(0, k - 11)])
    dp = haversine(obs[:, 0], obs[:, 1], lon0, lat0)
    mx = float(dp.max())
    f["max_from_pickup_km"] = mx
    f["frac_of_max_from_pickup"] = float(dp[-1] / mx) if mx > 0.05 else 1.0
    return f


def add_derived(df):
    """Time-of-day and geometry features computed from the prefix features."""
    st, cur = local_time(df["ts"]), local_time(df["ts"] + df["elapsed_time_s"])
    df["start_hour"] = st.dt.hour + st.dt.minute / 60
    df["cur_hour"] = cur.dt.hour + cur.dt.minute / 60 + cur.dt.second / 3600
    df["sin_cur"], df["cos_cur"] = np.sin(2 * np.pi * df["cur_hour"] / 24), np.cos(2 * np.pi * df["cur_hour"] / 24)
    df["dow"] = st.dt.dayofweek

    df["delta_lon"], df["delta_lat"] = df["curr_lon"] - df["pickup_lon"], df["curr_lat"] - df["pickup_lat"]
    df["pickup_to_center_km"] = haversine(df["pickup_lon"], df["pickup_lat"], *CENTER)
    df["curr_to_center_km"] = haversine(df["curr_lon"], df["curr_lat"], *CENTER)
    df["curr_to_airport_km"] = haversine(df["curr_lon"], df["curr_lat"], *AIRPORT)

    # heading relative to the centre / airport / pickup (cosine of the angle) and a crude time to get there
    lon, lat = df["curr_lon"].values, df["curr_lat"].values
    coslat = np.cos(np.radians(lat))
    for name, tlon, tlat in [("center", CENTER[0], CENTER[1]), ("airport", AIRPORT[0], AIRPORT[1]),
                             ("pickup", df["pickup_lon"].values, df["pickup_lat"].values)]:
        dx, dy = (tlon - lon) * coslat, tlat - lat
        df[f"head_to_{name}_cos"] = (df["head_sin"].values * dx + df["head_cos"].values * dy) / (np.sqrt(dx * dx + dy * dy) + 1e-9)
    spd = np.maximum(df["moving_speed_kmh"].values, 5.0)
    df["center_eta_s"] = df["curr_to_center_km"].values * 3600.0 / spd
    df["airport_eta_s"] = df["curr_to_airport_km"].values * 3600.0 / spd
    return df


LOC_FEATS = ["loc_rem_log", "loc_rem_s", "loc_rem_km", "loc_n", "cell_rem_log", "cell_n"]


def heading_bucket(sin_, cos_):
    """8 compass buckets, 8 = unknown / stationary."""
    ang = np.nan_to_num(np.arctan2(sin_, cos_))
    return np.where(np.isnan(sin_), 8, np.floor((ang + np.pi) / (2 * np.pi) * 8).astype(np.int64) % 8)


def add_loc_features(df, tab, fold=None, m=10.0):
    """fold: trip fold of each TRAINING row (removed from the table); -1 or None = use the whole table."""
    cell = grid_id(df["curr_lon"].values, df["curr_lat"].values, 0.008, 0.006)
    key = cell * 9 + heading_bucket(df["head_sin"].values, df["head_cos"].values)
    fo = np.full(len(df), -1, dtype=np.int64) if fold is None else np.asarray(fold, dtype=np.int64)

    def stats(tot, tf, k):
        t = tot.reindex(k).fillna(0.0).values
        s = tf.reindex(pd.MultiIndex.from_arrays([k, fo])).fillna(0.0).values
        return t - s                                                   # columns: lr, rr, dd, n

    sc, sk = stats(tab["c"], tab["cf"], cell), stats(tab["k"], tab["kf"], key)
    cm = (sc[:, :3] + m * tab["glob"]) / (sc[:, 3:4] + m)              # cell mean, shrunk to the global mean
    km = (sk[:, :3] + m * cm) / (sk[:, 3:4] + m)                       # (cell, heading) mean, shrunk to the cell mean
    df["loc_rem_log"], df["loc_rem_s"], df["loc_rem_km"] = km[:, 0], km[:, 1], km[:, 2]
    df["loc_n"], df["cell_rem_log"], df["cell_n"] = sk[:, 3], cm[:, 0], sc[:, 3]
    return df


AUX_TARGETS = ["rem_path_km", "rem_km", "dd_lon", "dd_lat"]


AUX_COLS = ["aux_path_km", "aux_rem_km", "aux_dd_lon", "aux_dd_lat"]


AUX_DERIVED = ["aux_dest_to_center_km", "aux_dest_to_airport_km", "aux_eta_moving_s", "aux_eta_avg_s"]


def attach_aux(df, A):
    for j, c in enumerate(AUX_COLS):
        df[c] = A[:, j]
    dlon, dlat = df["curr_lon"].values + df["aux_dd_lon"].values, df["curr_lat"].values + df["aux_dd_lat"].values
    df["aux_dest_to_center_km"] = haversine(dlon, dlat, *CENTER)
    df["aux_dest_to_airport_km"] = haversine(dlon, dlat, *AIRPORT)
    df["aux_eta_moving_s"] = df["aux_path_km"].values * 3600.0 / np.maximum(df["moving_speed_kmh"].values, 5.0)
    df["aux_eta_avg_s"] = df["aux_path_km"].values * 3600.0 / np.maximum(df["avg_speed_kmh"].values, 3.0)
    return df
