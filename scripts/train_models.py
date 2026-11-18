"""Train + compare every model in the zoo.

Protocol
  * fitting / CV: 78 training trips (BAD_REF_TRIPS removed), unreliable fixes removed, every 3rd fix
    kept (1 Hz fixes are near-duplicates; striding cuts autocorrelation and cost);
  * 5-fold GroupKFold by trip -> no trip appears in both a fit and a validation fold;
  * then refit on all 78 trips and score the 10 student-test trips (all fixes);
  * student-test classifier-only route continuity from the argmax arc of every fix.
Outputs results/model_comparison.csv, results/cv_folds.csv, results/curves/*.csv,
        results/st_probs/<model>.npy, models/zoo/<model>.joblib (small models only).
"""
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss as sk_log_loss
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap.data import ROOT, load_trip  # noqa: E402
from pathsnap.evaluate import BAD_REF_TRIPS, candidate_metrics, obs_groups, top1_mask  # noqa: E402
from pathsnap.models import FEATURES, SEED, fit_model, predict_model, zoo  # noqa: E402
from pathsnap.route import continuity  # noqa: E402

RES = os.path.join(ROOT, "results")
STRIDE = 3


class _ArcsOnly:
    def __init__(self, arcs):
        self.arcs = arcs


def load_train():
    f = pd.read_parquet(os.path.join(RES, "dataset", "train.parquet"))
    o = pd.read_parquet(os.path.join(RES, "dataset", "train_obs.parquet"))
    bad = o.loc[o.unreliable, ["trip", "obs"]].assign(_bad=1)
    f = f[~f.trip.isin(BAD_REF_TRIPS) & (f.obs % STRIDE == 0)]
    f = f.merge(bad, on=["trip", "obs"], how="left")
    f = f[f._bad.isna()].drop(columns="_bad").reset_index(drop=True)
    return f


def load_st():
    return pd.read_parquet(os.path.join(RES, "dataset", "student_test.parquet")).reset_index(drop=True)


def cap_rows(df, cap, rng):
    if cap is None or len(df) <= cap:
        return df
    g = df[["trip", "obs"]].drop_duplicates()
    keep = g.sample(n=int(len(g) * cap / len(df)), random_state=rng)
    return df.merge(keep, on=["trip", "obs"])


def st_continuity(st, p):
    gid = obs_groups(st)
    pick = st[top1_mask(p, gid)]
    vals = []
    for tid, g in pick.groupby("trip"):
        arcs = np.loadtxt(os.path.join(ROOT, "data", "student_test", tid, f"{tid}.arcs"), dtype=np.int64)
        vals.append(continuity(g.sort_values("obs")["arc"].to_numpy(), _ArcsOnly(arcs)))
    return float(np.mean(vals))


def curves(train, fold_tr, fold_va):
    """Epoch-wise train/validation loss for the two models that actually have training histories."""
    from lightgbm import LGBMClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    os.makedirs(os.path.join(RES, "curves"), exist_ok=True)
    Xtr, ytr = train.loc[fold_tr, FEATURES].to_numpy(), train.loc[fold_tr, "y"].to_numpy()
    Xva, yva = train.loc[fold_va, FEATURES].to_numpy(), train.loc[fold_va, "y"].to_numpy()
    lgb = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=40,
                         subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED,
                         verbose=-1)
    lgb.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xva, yva)], eval_metric="binary_logloss")
    ev = lgb.evals_result_
    pd.DataFrame({"iteration": np.arange(1, 401), "train_logloss": ev["training"]["binary_logloss"],
                  "val_logloss": ev["valid_1"]["binary_logloss"]}).to_csv(
        os.path.join(RES, "curves", "lightgbm_fold0.csv"), index=False)
    sc = StandardScaler().fit(Xtr)
    a, b = sc.transform(Xtr), sc.transform(Xva)
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=512,
                        learning_rate_init=1e-3, random_state=SEED)
    rows = []
    rng = np.random.default_rng(SEED)
    for ep in range(1, 41):
        idx = rng.permutation(len(a))
        for s in range(0, len(a), 50_000):
            mlp.partial_fit(a[idx[s:s + 50_000]], ytr[idx[s:s + 50_000]], classes=[0, 1])
        rows.append((ep, sk_log_loss(ytr, mlp.predict_proba(a)[:, 1]),
                     sk_log_loss(yva, mlp.predict_proba(b)[:, 1])))
    pd.DataFrame(rows, columns=["epoch", "train_logloss", "val_logloss"]).to_csv(
        os.path.join(RES, "curves", "mlp_fold0.csv"), index=False)


def main(only=None):
    train, st = load_train(), load_st()
    print(f"train rows {len(train):,} ({train.trip.nunique()} trips), student-test rows {len(st):,}")
    groups = train["trip"].to_numpy()
    folds = list(GroupKFold(n_splits=5).split(train, groups=groups))
    Xst = st[FEATURES].to_numpy()
    gst = obs_groups(st)
    os.makedirs(os.path.join(RES, "st_probs"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "models", "zoo"), exist_ok=True)
    fold_rows, summary = [], []
    for name, (family, build, cfg, cap) in zoo().items():
        if only and name not in only:
            continue
        cv = []
        for k, (tr, va) in enumerate(folds):
            ftr = cap_rows(train.iloc[tr], cap, SEED + k)
            fva = train.iloc[va].reset_index(drop=True)
            t0 = time.time()
            m = fit_model(name, build, ftr[FEATURES].to_numpy(), ftr["y"].to_numpy(), obs_groups(ftr))
            p = predict_model(m, fva[FEATURES].to_numpy(), obs_groups(fva))
            met = candidate_metrics(fva, p)
            met.update(model=name, fold=k, fit_s=time.time() - t0)
            cv.append(met)
            fold_rows.append(met)
            print(f"  {name} fold {k}: top1 {met['top1']:.4f} mF1 {met['macro_f1']:.4f} "
                  f"LL {met['log_loss']:.4f}", flush=True)
        cvd = pd.DataFrame(cv)
        full = cap_rows(train, cap, SEED)
        t0 = time.time()
        m = fit_model(name, build, full[FEATURES].to_numpy(), full["y"].to_numpy(), obs_groups(full))
        fit_s = time.time() - t0
        t0 = time.time()
        p = predict_model(m, Xst, gst)
        inf_s = time.time() - t0
        np.save(os.path.join(RES, "st_probs", f"{name}.npy"), p.astype(np.float32))
        stm = candidate_metrics(st, p)
        row = dict(model=name, family=family, config=cfg, fit_rows=len(full),
                   cv_top1=cvd.top1.mean(), cv_top1_std=cvd.top1.std(),
                   cv_macro_f1=cvd.macro_f1.mean(), cv_macro_f1_std=cvd.macro_f1.std(),
                   cv_log_loss=cvd.log_loss.mean(),
                   st_top1=stm["top1"], st_macro_f1=stm["macro_f1"], st_precision=stm["precision"],
                   st_recall=stm["recall"], st_f1_thr05=stm["f1_thr05"], st_log_loss=stm["log_loss"],
                   st_route_continuity=st_continuity(st, p), fit_s=fit_s,
                   infer_ms_per_fix=1000 * inf_s / (gst.max() + 1))
        summary.append(row)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}),
              flush=True)
        try:
            path = os.path.join(ROOT, "models", "zoo", f"{name}.joblib")
            joblib.dump(m, path, compress=3)
            if os.path.getsize(path) > 40e6:   # keep the repo small: big forests are reproducible
                os.remove(path)
        except Exception as e:  # noqa: BLE001
            print("  (not saved)", e)
    tag = "" if not only else "_" + "_".join(only)
    pd.DataFrame(fold_rows).to_csv(os.path.join(RES, f"cv_folds{tag}.csv"), index=False)
    pd.DataFrame(summary).to_csv(os.path.join(RES, f"model_comparison{tag}.csv"), index=False)
    if not only:
        curves(train, folds[0][0], folds[0][1])


if __name__ == "__main__":
    main(sys.argv[1:] or None)
