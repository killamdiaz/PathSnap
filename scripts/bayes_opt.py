"""Bayesian hyperparameter optimisation (Optuna TPE) of the selected candidate scorer.

Objective : mean observation-level Top-1 accuracy over a 3-fold GroupKFold (by trip) on the 78 usable
            training trips, fixes strided by 6 for search speed. Median pruning after each fold.
Afterwards: untuned vs tuned under the *same* 5-fold CV (stride 3) and on the 10 student-test trips.
The student-test partition is never seen by the search.
Outputs results/bo_trials.csv, results/bo_study.db, results/bo_best.json, results/tuned_vs_untuned.csv
"""
import json
import os
import sys
import time

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap.data import ROOT  # noqa: E402
from pathsnap.evaluate import candidate_metrics, obs_groups  # noqa: E402
from pathsnap.models import FEATURES, SEED, RankerModel, zoo  # noqa: E402
from scripts.train_models import load_st, load_train  # noqa: E402

RES = os.path.join(ROOT, "results")
N_TRIALS = int(os.environ.get("BO_TRIALS", 40))
MODEL = os.environ.get("BO_MODEL", "LightGBM")


def make(params):
    from lightgbm import LGBMClassifier
    base = dict(random_state=SEED, verbose=-1, n_jobs=-1, subsample_freq=1)
    if MODEL == "LambdaRank":
        return RankerModel(**params)
    return LGBMClassifier(**base, **params)


def fit_predict(params, tr, va):
    m = make(params)
    Xtr, ytr = tr[FEATURES].to_numpy(), tr["y"].to_numpy()
    if isinstance(m, RankerModel):
        m.fit(Xtr, ytr, obs_groups(tr))
        return m, m.predict_proba(va[FEATURES].to_numpy(), obs_groups(va))
    m.fit(Xtr, ytr)
    return m, m.predict_proba(va[FEATURES].to_numpy())[:, 1]


def space(trial):
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 1200, log=True),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 300, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        min_split_gain=trial.suggest_float("min_split_gain", 0.0, 0.5),
    )


def main():
    train = load_train()
    search = train[train.obs % 6 == 0].reset_index(drop=True)
    folds3 = list(GroupKFold(n_splits=3).split(search, groups=search.trip))

    def objective(trial):
        params = space(trial)
        scores = []
        for k, (tr, va) in enumerate(folds3):
            fva = search.iloc[va].reset_index(drop=True)
            _, p = fit_predict(params, search.iloc[tr], fva)
            scores.append(candidate_metrics(fva, p)["top1"])
            trial.report(float(np.mean(scores)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    storage = f"sqlite:///{os.path.join(RES, 'bo_study.db')}"
    study = optuna.create_study(direction="maximize", study_name=f"pathsnap_{MODEL}", storage=storage,
                                load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=10),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1))
    t0 = time.time()
    done = len([t for t in study.trials if t.state.is_finished()])
    study.optimize(objective, n_trials=max(0, N_TRIALS - done), show_progress_bar=False)
    bo_s = time.time() - t0
    study.trials_dataframe().to_csv(os.path.join(RES, "bo_trials.csv"), index=False)
    best = study.best_params
    json.dump({"model": MODEL, "best_value_cv3_top1": study.best_value, "best_params": best,
               "n_trials": len(study.trials), "search_seconds": bo_s}, open(os.path.join(RES, "bo_best.json"), "w"), indent=2)
    print("best", study.best_value, best, f"{bo_s:.0f}s")

    # ---- untuned vs tuned, identical protocol ----
    untuned = {"LightGBM": dict(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=40,
                                subsample=0.8, colsample_bytree=0.8),
               "LambdaRank": {}}[MODEL]
    st = load_st()
    folds5 = list(GroupKFold(n_splits=5).split(train, groups=train.trip))
    rows = []
    for tag, params in (("untuned", untuned), ("tuned", best)):
        cv = []
        for tr, va in folds5:
            fva = train.iloc[va].reset_index(drop=True)
            _, p = fit_predict(params, train.iloc[tr], fva)
            cv.append(candidate_metrics(fva, p))
        t0 = time.time()
        m, _ = fit_predict(params, train, train.iloc[:10])
        fit_s = time.time() - t0
        t0 = time.time()
        if isinstance(m, RankerModel):
            p = m.predict_proba(st[FEATURES].to_numpy(), obs_groups(st))
        else:
            p = m.predict_proba(st[FEATURES].to_numpy())[:, 1]
        inf = time.time() - t0
        sm = candidate_metrics(st, p)
        cvd = pd.DataFrame(cv)
        rows.append(dict(variant=tag, cv_macro_f1=cvd.macro_f1.mean(), cv_top1=cvd.top1.mean(),
                         cv_log_loss=cvd.log_loss.mean(), st_top1=sm["top1"], st_macro_f1=sm["macro_f1"],
                         st_log_loss=sm["log_loss"], fit_s=fit_s,
                         infer_ms_per_fix=1000 * inf / (obs_groups(st).max() + 1)))
        np.save(os.path.join(RES, "st_probs", f"{MODEL}_{tag}.npy"), p.astype(np.float32))
        if tag == "tuned":
            import joblib
            os.makedirs(os.path.join(ROOT, "models", "final"), exist_ok=True)
            joblib.dump(m, os.path.join(ROOT, "models", "final", "model.joblib"), compress=3)
            json.dump({"model": f"{MODEL} (Bayesian-tuned)", "params": params, "features": FEATURES,
                       "candidates": {"k": 12, "radius_m": 80.0}, "seed": SEED,
                       "trained_on": "78 training trips (00000033, 00000054 excluded), every 3rd fix"},
                      open(os.path.join(ROOT, "models", "final", "config.json"), "w"), indent=2)
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "tuned_vs_untuned.csv"), index=False)


if __name__ == "__main__":
    main()
