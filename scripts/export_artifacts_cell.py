# =====================================================================================
# REPLACES cell 28 ("9. Final fit and submission") of the v4 notebook.
# It does exactly what the old cell did (same seeds, same models, same submission.csv) and ALSO keeps the
# fitted models and writes everything the dashboard needs into ./artifacts and ./taxi_eta_artifacts.zip.
# Run it after the validation cells (7 and 8) so that tot_va, best, tot_raw, tot_log, W_BLEND exist.
# Copy the content of the zip into taxi_eta_app/models/ (see README).
# =====================================================================================
import os, json, shutil, datetime


def oof_aux_keep(Dfit, apply_list, groups, seed=SEED):
    """Identical to oof_aux (same folds, same seeds) but also returns the fitted models so they can be saved."""
    oof = np.zeros((len(Dfit), len(AUX_TARGETS)))
    app = [np.zeros((len(d), len(AUX_TARGETS))) for d in apply_list]
    models = {}
    Xf = Dfit[AUX_IN]
    for fi, (a, b) in enumerate(GroupKFold(AUX_FOLDS).split(Xf, groups=groups)):
        for j, tname in enumerate(AUX_TARGETS):
            m = lgb.train(dict(AUX_PARAMS, seed=seed + fi), lgb.Dataset(Xf.iloc[a], Dfit[tname].values[a], categorical_feature=CATS), AUX_ROUNDS)
            oof[b, j] = m.predict(Xf.iloc[b])
            for A, d in zip(app, apply_list):
                A[:, j] += m.predict(d[AUX_IN]) / AUX_FOLDS
            models[(fi, j)] = m
    return oof, app, models


# ---------------------------------------------------------------- final fit (as before)
all_trips = np.ones(len(T), dtype=bool)
pri_full, lookups, g_full = build_priors(T, all_trips)
tab_full = build_loc_table(T, all_trips)

Df = build_frame(F, T, pri_full)
Df = add_loc_features(Df, tab_full, fold=TRIP_FOLD[Df["tidx"].values])
Xte = build_test_frame(test, lookups, g_full, tab_full)

oof_f, (app_te,), aux_models = oof_aux_keep(Df, [Xte], groups=Df["tidx"].values)
Df, Xte = attach_aux(Df, oof_f), attach_aux(Xte, app_te)

n_raw, n_log = int(m_raw.best_iteration * 1.1), int(m_log.best_iteration * 1.1)
raw_models, log_models, p_raw, p_log = [], [], [], []
for s in FINAL_SEEDS:
    prm = dict(PARAMS, seed=s)
    mr = lgb.train(prm, lgb.Dataset(Df[FEATURES], Df["remaining_s"], categorical_feature=CATS), n_raw)
    ml = lgb.train(prm, lgb.Dataset(Df[FEATURES], np.log1p(Df["total_s"]), categorical_feature=CATS), n_log)
    raw_models.append(mr); log_models.append(ml)
    p_raw.append(mr.predict(Xte[FEATURES])); p_log.append(ml.predict(Xte[FEATURES]))

el_te = Xte["elapsed_time_s"].values
log_raw = np.log1p(el_te + np.maximum(np.mean(p_raw, axis=0), STEP_S))
log_log = np.mean(p_log, axis=0)
total_hat = np.maximum(np.expm1(W_BLEND * log_raw + (1 - W_BLEND) * log_log), el_te + STEP_S)

id_col, y_col = sample_sub.columns[0], sample_sub.columns[1]
sub = sample_sub[[id_col]].merge(pd.DataFrame({id_col: test["TRIP_ID"].values, y_col: total_hat}), on=id_col, how="left")
sub[y_col] = sub[y_col].fillna(base_total)
sub.to_csv("submission.csv", index=False)
print(sub.shape, sub[y_col].describe().round(1).to_dict())


# ---------------------------------------------------------------- export for the dashboard
def export_serving_artifacts(out="artifacts", n_demo=300):
    shutil.rmtree(out, ignore_errors=True)
    for d in ("aux", "final", "priors", "loc"):
        os.makedirs(f"{out}/{d}")
    # 1) models (LightGBM text format)
    for (fi, j), m in aux_models.items():
        m.save_model(f"{out}/aux/aux_f{fi}_t{j}.txt")
    for s, mr, ml in zip(FINAL_SEEDS, raw_models, log_models):
        mr.save_model(f"{out}/final/raw_{s}.txt"); ml.save_model(f"{out}/final/log_{s}.txt")
    # 2) lookup tables as CSV (portable across pandas versions, no pickle)
    for name, s in lookups.items():
        s.rename("value").rename_axis("key").to_csv(f"{out}/priors/{name}.csv")
    for k in ("k", "kf", "c", "cf"):
        tab_full[k].to_csv(f"{out}/loc/{k}.csv")
    # 3) configuration: everything the serving code must agree on with training
    cfg = {"version": "v4-" + datetime.date.today().strftime("%Y%m%d"),
           "created_utc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
           "FEATURES": FEATURES, "AUX_IN": AUX_IN, "CATS": CATS, "AUX_FOLDS": AUX_FOLDS, "W_BLEND": float(W_BLEND),
           "PRIOR_KEYS": PRIOR_KEYS, "g_full": float(g_full), "loc_glob": [float(x) for x in tab_full["glob"]],
           "final_seeds": list(FINAL_SEEDS), "lightgbm_version": lgb.__version__, "pandas_version": pd.__version__, "n_raw_trees": n_raw, "n_log_trees": n_log, "n_training_cuts": int(len(Df)),
           "constants": {"STEP_S": STEP_S, "MAX_SPEED_KMH": MAX_SPEED_KMH, "R_EARTH": R_EARTH, "CENTER": list(CENTER), "AIRPORT": list(AIRPORT)}}
    json.dump(cfg, open(f"{out}/config.json", "w"), indent=1)
    # 4) validation metrics and slice tables (what the Model page shows)
    mj = {"validation_rows": int(len(tot_va)), "score": SCORE, "blend_weight_raw": float(W_BLEND),
          "baseline": {"rmsle": rmsle(tot_va, base_total), "rmse_s": rmse(base_total, tot_va)},
          "models": {"raw": {"rmsle": rmsle(tot_va, tot_raw), "rmse_s": rmse(tot_raw - el_va, rem_va)},
                     "log": {"rmsle": rmsle(tot_va, tot_log), "rmse_s": rmse(tot_log - el_va, rem_va)},
                     "blend": {"rmsle": rmsle(tot_va, best), "rmse_s": rmse(best - el_va, rem_va)}}}
    if "tables" in globals():
        mj["slices"] = {n: [{"slice": str(i), **{c: round(float(v), 4) for c, v in r.items()}} for i, r in t.iterrows()] for n, t in tables.items()}
    json.dump(mj, open(f"{out}/metrics.json", "w"), indent=1)
    # 5) empirical error quantiles by prefix length: the dashboard's prediction intervals
    e = np.log1p(best) - np.log1p(tot_va)
    kk = el_va / STEP_S + 1
    qs = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    edges, labels = [0, 2, 5, 10, 20, 40, 80, 1e9], ["1-2", "3-5", "6-10", "11-20", "21-40", "41-80", "81+"]
    cat = pd.cut(kk, edges, labels=labels)
    quant = {"all": {f"{q:.3f}": float(np.quantile(e, q)) for q in qs}}
    for lab in labels:
        m = np.asarray(cat == lab)
        if m.sum() >= 200:
            quant[lab] = {f"{q:.3f}": float(np.quantile(e[m], q)) for q in qs}
    json.dump({"definition": "log1p(prediction) - log1p(actual total), validation set", "quantiles": quant,
               "rows": {lab: int(np.asarray(cat == lab).sum()) for lab in labels}}, open(f"{out}/error_quantiles.json", "w"), indent=1)
    # 6) demo trips for the Replay tab: trips never used as training cuts (priors and the location table may still include them)
    rs = np.random.default_rng(SEED)
    cand = np.setdiff1d(T.index[T["n"].between(40, 150) & T["ok"]].values, np.unique(Df["tidx"].values))
    rows = []
    for ti in rs.choice(cand, size=min(n_demo, len(cand)), replace=False):
        r = train.iloc[int(ti)]
        rows.append({"trip_id": int(ti), "timestamp": int(r["TIMESTAMP"]), "call_type": r["CALL_TYPE"],
                     "origin_stand": r["ORIGIN_STAND"], "taxi_id": int(r["TAXI_ID"]), "polyline": r["POLYLINE"],
                     "true_total_s": float((int(T.loc[ti, "n"]) - 1) * STEP_S)})
    pd.DataFrame(rows).to_csv(f"{out}/demo_trips.csv", index=False)
    shutil.make_archive("taxi_eta_artifacts", "zip", out)
    print(f"exported {len(os.listdir(out))} items to ./{out} and taxi_eta_artifacts.zip "
          f"({os.path.getsize('taxi_eta_artifacts.zip') / 1e6:.1f} MB)")


export_serving_artifacts()
