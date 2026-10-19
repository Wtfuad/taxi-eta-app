"""Inference: loads the artifacts exported from the training notebook and predicts total trip time.

The sequence below mirrors `build_test_frame` + the final prediction cell of the notebook:
    trace -> prefix features -> derived features -> priors -> location table -> stacked destination models
          -> final raw / log models (several seeds) -> log-space blend -> total time.
Artifacts are plain files (JSON, CSV, LightGBM text models); no pickle is loaded.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import features as F
from .traces import TripRequest

log = logging.getLogger("taxi_eta.predictor")

K_EDGES = [0, 2, 5, 10, 20, 40, 80, 1e9]
K_LABELS = ["1-2", "3-5", "6-10", "11-20", "21-40", "41-80", "81+"]


class ArtifactError(RuntimeError):
    pass


def k_bucket(k: np.ndarray) -> np.ndarray:
    return pd.cut(np.asarray(k, dtype=float), K_EDGES, labels=K_LABELS).astype(str)


class Predictor:
    def __init__(self, model_dir):
        self.dir = Path(model_dir)
        cfg_path = self.dir / "config.json"
        if not cfg_path.exists():
            raise ArtifactError(f"No model found: {cfg_path} does not exist. Export the artifacts from the training notebook "
                                f"(see README) and put them in {self.dir}.")
        t0 = time.time()
        self.cfg = json.loads(cfg_path.read_text())
        self._check_constants()
        self.features = list(self.cfg["FEATURES"])
        self.aux_in = list(self.cfg["AUX_IN"])
        self.cats = list(self.cfg["CATS"])
        self.w_blend = float(self.cfg["W_BLEND"])
        self.aux_folds = int(self.cfg["AUX_FOLDS"])
        self.prior_keys = dict(self.cfg["PRIOR_KEYS"])
        self.g = float(self.cfg["g_full"])
        self.lookups = {n: pd.read_csv(self.dir / "priors" / f"{n}.csv").set_index("key")["value"] for n in self.prior_keys}
        self.tab = self._load_loc_table()
        self.aux_models = {}
        aux_dir = self.dir / "aux"
        if not aux_dir.exists() and (self.dir / "_aux").exists():
            aux_dir = self.dir / "_aux"
        for f in sorted(aux_dir.glob("aux_f*_t*.txt")):
            fi, tj = f.stem.replace("aux_f", "").split("_t")
            self.aux_models[(int(fi), int(tj))] = lgb.Booster(model_file=str(f))
        self.raw_models = [lgb.Booster(model_file=str(p)) for p in sorted((self.dir / "final").glob("raw_*.txt"))]
        self.log_models = [lgb.Booster(model_file=str(p)) for p in sorted((self.dir / "final").glob("log_*.txt"))]
        n_aux = self.aux_folds * len(F.AUX_TARGETS)
        if len(self.aux_models) != n_aux or not self.raw_models or len(self.raw_models) != len(self.log_models):
            raise ArtifactError(f"Incomplete model files: {len(self.aux_models)}/{n_aux} stacked models, "
                                f"{len(self.raw_models)} raw and {len(self.log_models)} log final models.")
        self.err_q = self._load_json("error_quantiles.json")
        self.metrics = self._load_json("metrics.json")
        log.info("model %s loaded in %.1f s (%d features, %d final models)", self.cfg.get("version"), time.time() - t0,
                 len(self.features), len(self.raw_models) + len(self.log_models))

    # ------------------------------------------------------------------ loading helpers
    def _load_json(self, name):
        p = self.dir / name
        return json.loads(p.read_text()) if p.exists() else None

    def _check_constants(self):
        """Guard against train/serve skew: constants used in training must equal those in features.py."""
        saved = self.cfg.get("constants", {})
        for k, v in F.CONSTANTS.items():
            if k in saved and not np.allclose(np.asarray(saved[k], dtype=float), np.asarray(v, dtype=float)):
                raise ArtifactError(f"Constant {k} differs between training ({saved[k]}) and features.py ({v}). "
                                    f"Re-export the model or update features.py.")

    def _load_loc_table(self):
        d = self.dir / "loc"
        t = {"k": pd.read_csv(d / "k.csv").set_index("k"), "kf": pd.read_csv(d / "kf.csv").set_index(["k", "f"]),
             "c": pd.read_csv(d / "c.csv").set_index("c"), "cf": pd.read_csv(d / "cf.csv").set_index(["c", "f"]),
             "glob": np.asarray(self.cfg["loc_glob"], dtype=float)}
        return t

    # ------------------------------------------------------------------ public info
    @property
    def info(self) -> dict:
        return {"version": self.cfg.get("version", "unknown"), "created_utc": self.cfg.get("created_utc"),
                "n_features": len(self.features), "n_final_models": len(self.raw_models) + len(self.log_models),
                "blend_weight_raw": self.w_blend, "training_cuts": self.cfg.get("n_training_cuts"),
                "synthetic": bool(self.cfg.get("synthetic", False)), "lightgbm_trained_with": self.cfg.get("lightgbm_version")}

    # ------------------------------------------------------------------ pipeline
    def feature_frame(self, requests: Sequence[TripRequest]) -> pd.DataFrame:
        """All model inputs for the given requests (same steps as build_test_frame in the notebook)."""
        if not requests:
            return pd.DataFrame(columns=self.features)
        df = pd.DataFrame([F.trip_features(np.asarray(r.pts, dtype=np.float64), len(r.pts)) for r in requests])
        df["ts"] = np.array([r.ts for r in requests], dtype=np.int64)
        df["call"] = np.array([r.call for r in requests], dtype=int)
        df["stand"] = np.array([r.stand for r in requests], dtype=int)
        df = F.add_derived(df)
        taxi = np.array([np.nan if r.taxi is None else float(r.taxi) for r in requests])
        keys = {"k_grid_hb": F.grid_id(df["pickup_lon"], df["pickup_lat"]) * 10 + (df["start_hour"] // 3).astype(int).values,
                "stand": df["stand"].values, "taxi": taxi}
        for name, col in self.prior_keys.items():
            df[name] = pd.Series(np.asarray(keys[col])).map(self.lookups[name]).fillna(self.g).values
        df = F.add_loc_features(df, self.tab, fold=None)
        A = np.zeros((len(df), len(F.AUX_TARGETS)))
        X = df[self.aux_in]
        for (fi, tj), m in sorted(self.aux_models.items()):
            A[:, tj] += m.predict(X) / self.aux_folds
        df = F.attach_aux(df, A)
        return df

    def predict(self, requests: Sequence[TripRequest], level: float = 0.8) -> pd.DataFrame:
        """One row per request: predicted total / remaining time in seconds and an empirical interval."""
        t0 = time.time()
        df = self.feature_frame(requests)
        if df.empty:
            return pd.DataFrame(columns=["observed_points", "elapsed_s", "total_s", "remaining_s", "low_s", "high_s"])
        X = df[self.features]
        p_raw = np.mean([m.predict(X) for m in self.raw_models], axis=0)
        p_log = np.mean([m.predict(X) for m in self.log_models], axis=0)
        el = df["elapsed_time_s"].values
        log_raw = np.log1p(el + np.maximum(p_raw, F.STEP_S))
        total = np.maximum(np.expm1(self.w_blend * log_raw + (1 - self.w_blend) * p_log), el + F.STEP_S)
        k = el / F.STEP_S + 1
        low, high = self._interval(total, el, k, level)
        out = pd.DataFrame({"observed_points": k.astype(int), "elapsed_s": el, "total_s": total, "remaining_s": total - el,
                            "low_s": low, "high_s": high})
        log.info("predicted %d trip(s) in %.0f ms", len(out), (time.time() - t0) * 1000)
        return out

    def predict_with_features(self, requests, level=0.8):
        df = self.feature_frame(requests)
        return self.predict(requests, level), df

    def _interval(self, total, elapsed, k, level):
        """Empirical interval from validation log errors by prefix length (not a model-based confidence interval)."""
        floor = elapsed + F.STEP_S
        if not self.err_q:
            return np.full(len(total), np.nan), np.full(len(total), np.nan)
        lo_q, hi_q = f"{(1 - level) / 2:.3f}", f"{1 - (1 - level) / 2:.3f}"
        buckets = k_bucket(k)
        lo, hi = np.full(len(total), np.nan), np.full(len(total), np.nan)
        for i, b in enumerate(buckets):
            q = self.err_q["quantiles"].get(b) or self.err_q["quantiles"].get("all")
            if not q or lo_q not in q or hi_q not in q:
                continue
            lt = np.log1p(total[i])
            lo[i] = max(np.expm1(lt - q[hi_q]), floor[i])           # error = log(pred) - log(true)
            hi[i] = max(np.expm1(lt - q[lo_q]), floor[i])
        return lo, hi

    def predict_curve(self, pts: np.ndarray, base: TripRequest, step: int = 1, level: float = 0.8) -> pd.DataFrame:
        """Prediction after each prefix of a trace (how the estimate evolves as more points arrive)."""
        n = len(pts)
        ks = list(range(1, n + 1, max(1, step)))
        if ks[-1] != n:
            ks.append(n)
        reqs = [TripRequest(pts[:k], base.ts, base.call, base.stand, base.taxi) for k in ks]
        out = self.predict(reqs, level)
        out.insert(0, "k", ks)
        return out
