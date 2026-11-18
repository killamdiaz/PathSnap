"""HMM/Viterbi experiments.

1. Out-of-fold (5-fold GroupKFold) probabilities of the final tuned model for every fix of the 78
   usable training trips (the decoder must be tuned on probabilities that are as uncertain as the ones
   it will see at inference time).
2. Bayesian optimisation (TPE) of the decoder parameters on those training trips:
   objective = fix-level accuracy of the Viterbi path on reliable fixes.
3. Student-test: classifier-only vs classifier+HMM for every model in the zoo (same decoder params),
   point accuracy, route continuity, length-weighted route precision/recall/F1, RMF, breaks, timings.
4. Folium maps for every student-test trip with the final model.
Outputs results/oof_train_probs.parquet, results/hmm_bo_trials.csv, models/final/decoder.json,
        results/hmm_comparison.csv, results/hmm_per_trip.csv, maps/st_<trip>.html
"""
import json
import os
import sys
import time

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap.data import ROOT, list_trips, load_route, load_trip  # noqa: E402
from pathsnap.decoder import DecoderParams, precompute_transitions, viterbi  # noqa: E402
from pathsnap.evaluate import BAD_REF_TRIPS  # noqa: E402
from pathsnap.features import FEATURES, build_candidates  # noqa: E402
from pathsnap.models import SEED, RankerModel, zoo  # noqa: E402
from pathsnap.route import assemble, continuity, route_scores  # noqa: E402
from pathsnap.viz import trip_map  # noqa: E402
from scripts.bayes_opt import MODEL, make  # noqa: E402
from scripts.train_models import STRIDE, load_train  # noqa: E402

RES = os.path.join(ROOT, "results")
DEC_TRIALS = int(os.environ.get("DEC_TRIALS", 30))


def probs_matrix(tc, p_rows, frame):
    P = np.zeros(tc.arc.shape)
    P[frame["obs"].to_numpy(), frame["slot"].to_numpy()] = p_rows
    return P


def oof_probs(best_params):
    full = pd.read_parquet(os.path.join(RES, "dataset", "train.parquet"))
    full = full[~full.trip.isin(BAD_REF_TRIPS)].reset_index(drop=True)
    fit_rows = load_train()
    trips = np.array(sorted(full.trip.unique()))
    out = np.zeros(len(full), np.float32)
    for k, (tr_i, va_i) in enumerate(GroupKFold(n_splits=5).split(trips, groups=trips)):
        tr_trips, va_trips = set(trips[tr_i]), set(trips[va_i])
        fr = fit_rows[fit_rows.trip.isin(tr_trips)]
        m = make(best_params)
        if isinstance(m, RankerModel):
            m.fit(fr[FEATURES].to_numpy(), fr["y"].to_numpy(),
                  pd.factorize(fr.trip + ":" + fr.obs.astype(str))[0])
        else:
            m.fit(fr[FEATURES].to_numpy(), fr["y"].to_numpy())
        mask = full.trip.isin(va_trips).to_numpy()
        va = full[mask]
        if isinstance(m, RankerModel):
            out[mask] = m.predict_proba(va[FEATURES].to_numpy(), va["obs"].to_numpy() + 10**7 * pd.factorize(va.trip)[0])
        else:
            out[mask] = m.predict_proba(va[FEATURES].to_numpy())[:, 1]
        print("oof fold", k, flush=True)
    full["p"] = out
    full[["trip", "obs", "slot", "p"]].to_parquet(os.path.join(RES, "oof_train_probs.parquet"))
    return full


def prepare(partition, trips):
    """Candidates + transition tensors + reference labels per trip (reference only for scoring)."""
    obs = pd.read_parquet(os.path.join(RES, "dataset", f"{partition}_obs.parquet"))
    prep = {}
    for tid in trips:
        t0 = time.time()
        trip = load_trip(tid, partition)
        tc = build_candidates(trip)
        cache = precompute_transitions(tc)
        o = obs[obs.trip == tid].sort_values("obs")
        prep[tid] = dict(trip=trip, tc=tc, cache=cache, ref=o.ref_arc.to_numpy(),
                         reliable=~o.unreliable.to_numpy(), route=load_route(tid, partition),
                         prep_s=time.time() - t0)
        print(f"  prepared {partition} {tid} in {time.time() - t0:.1f}s", flush=True)
    return prep


def decode(pp, P, params):
    tc = pp["tc"]
    valid = tc.arc >= 0
    t0 = time.perf_counter()
    path, breaks = viterbi(P, valid, pp["cache"], params)
    hmm = tc.arc[np.arange(len(P)), path]
    vit_s = time.perf_counter() - t0
    clf = tc.arc[np.arange(len(P)), np.where(valid, P, -1).argmax(1)]
    return clf, hmm, breaks, vit_s


def tune_decoder(prep_train, oof):
    groups = {tid: g for tid, g in oof.groupby("trip")}
    Ps = {tid: probs_matrix(pp["tc"], groups[tid]["p"].to_numpy(), groups[tid]) for tid, pp in prep_train.items()}

    def score(params):
        acc = []
        w = []
        for tid, pp in prep_train.items():
            _, hmm, _, _ = decode(pp, Ps[tid], params)
            r = pp["reliable"]
            acc.append((hmm[r] == pp["ref"][r]).mean())
            w.append(r.sum())
        return float(np.average(acc, weights=w))

    def objective(trial):
        return score(DecoderParams(
            alpha=trial.suggest_float("alpha", 0.3, 3.0, log=True),
            beta0=trial.suggest_float("beta0", 1.0, 40.0, log=True),
            beta_dt=trial.suggest_float("beta_dt", 0.1, 10.0, log=True),
            turn_w=trial.suggest_float("turn_w", 0.0, 4.0),
            uturn=trial.suggest_float("uturn", 0.0, 15.0),
            gap_s=trial.suggest_categorical("gap_s", [30.0, 60.0, 120.0, 300.0, 1e9]),
        ))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=8))
    study.enqueue_trial(DecoderParams().as_dict() | {"gap_s": 120.0})
    t0 = time.time()
    study.optimize(objective, n_trials=DEC_TRIALS)
    df = study.trials_dataframe()
    df.to_csv(os.path.join(RES, "hmm_bo_trials.csv"), index=False)
    best = DecoderParams(**{**DecoderParams().as_dict(), **study.best_params})
    default_score = df.loc[0, "value"]
    print(f"decoder BO: default {default_score:.4f} -> best {study.best_value:.4f} in {time.time() - t0:.0f}s", best)
    return best, dict(default_train_acc=default_score, tuned_train_acc=study.best_value,
                      seconds=time.time() - t0, trials=len(df))


def main():
    best_lgb = json.load(open(os.path.join(RES, "bo_best.json")))["best_params"]
    oof = oof_probs(best_lgb)
    train_trips = sorted(set(list_trips("train")) - BAD_REF_TRIPS)
    prep_train = prepare("train", train_trips)
    params, bo_info = tune_decoder(prep_train, oof)
    os.makedirs(os.path.join(ROOT, "models", "final"), exist_ok=True)
    json.dump({"decoder": params.as_dict(), "tuning": bo_info},
              open(os.path.join(ROOT, "models", "final", "decoder.json"), "w"), indent=2)

    st = pd.read_parquet(os.path.join(RES, "dataset", "student_test.parquet")).reset_index(drop=True)
    st_trips = list_trips("student_test")
    prep_st = prepare("student_test", st_trips)
    probs = {name: np.load(os.path.join(RES, "st_probs", f"{name}.npy")) for name in zoo()
             if os.path.exists(os.path.join(RES, "st_probs", f"{name}.npy"))}
    for tag in ("untuned", "tuned"):
        p = os.path.join(RES, "st_probs", f"{MODEL}_{tag}.npy")
        if os.path.exists(p):
            probs[f"{MODEL}_{tag}"] = np.load(p)
    rows, per_trip = [], []
    for name, p in probs.items():
        agg = {k: [] for k in ("clf_acc", "hmm_acc", "clf_cont", "hmm_cont", "clf_f1", "hmm_f1",
                               "clf_rmf", "hmm_rmf", "n", "vit_s", "breaks", "jumps")}
        for tid in st_trips:
            pp = prep_st[tid]
            fr = st[st.trip == tid]
            P = probs_matrix(pp["tc"], p[fr.index.to_numpy()], fr)
            clf, hmm, breaks, vit_s = decode(pp, P, params)
            net, ref = pp["tc"].net, pp["ref"]
            r_clf, _ = assemble(clf, net)
            r_hmm, jumps = assemble(hmm, net)
            s_clf, s_hmm = route_scores(r_clf, pp["route"], net), route_scores(r_hmm, pp["route"], net)
            rec = dict(model=name, trip=tid, n=len(ref), clf_acc=(clf == ref).mean(), hmm_acc=(hmm == ref).mean(),
                       clf_cont=continuity(clf, net, pp["tc"].xy), hmm_cont=continuity(hmm, net, pp["tc"].xy),
                       clf_f1=s_clf["route_f1"], hmm_f1=s_hmm["route_f1"], clf_rmf=s_clf["rmf"],
                       hmm_rmf=s_hmm["rmf"], vit_s=vit_s, breaks=len(breaks) - 1, jumps=jumps,
                       hmm_route_precision=s_hmm["route_precision"], hmm_route_recall=s_hmm["route_recall"])
            per_trip.append(rec)
            for k in agg:
                agg[k].append(rec[k])
            if name == f"{MODEL}_tuned":
                trip_map(pp["trip"], pp["tc"], hmm, r_hmm, clf_arcs=clf, ref_route=pp["route"],
                         out=os.path.join(ROOT, "maps", f"st_{tid}.html"),
                         title=f"Trip {tid}: HMM acc {rec['hmm_acc']:.3f}, route F1 {rec['hmm_f1']:.3f}")
        w = np.array(agg["n"])
        rows.append(dict(model=name, **{k: float(np.average(agg[k], weights=w)) for k in
                                        ("clf_acc", "hmm_acc")},
                         **{k: float(np.mean(agg[k])) for k in ("clf_cont", "hmm_cont", "clf_f1", "hmm_f1",
                                                                "clf_rmf", "hmm_rmf")},
                         viterbi_ms_per_fix=1000 * np.sum(agg["vit_s"]) / w.sum(),
                         breaks=int(np.sum(agg["breaks"])), jumps=int(np.sum(agg["jumps"]))))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "hmm_comparison.csv"), index=False)
    pd.DataFrame(per_trip).to_csv(os.path.join(RES, "hmm_per_trip.csv"), index=False)


if __name__ == "__main__":
    main()
