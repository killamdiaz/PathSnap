"""All result figures, generated only from the CSV/NPY files the experiment scripts wrote."""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap.data import ROOT  # noqa: E402
from pathsnap.evaluate import obs_groups, top1_mask  # noqa: E402

RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25})
C1, C2, C3 = "#2a6f97", "#e07a5f", "#81b29a"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)


def model_plots():
    mc = pd.read_csv(os.path.join(RES, "model_comparison.csv"))
    folds = pd.read_csv(os.path.join(RES, "cv_folds.csv"))
    order = mc.sort_values("cv_macro_f1").model.tolist()
    mc = mc.set_index("model").loc[order]
    y = np.arange(len(mc))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(y, mc.cv_macro_f1, xerr=mc.cv_macro_f1_std, color=C1, capsize=3)
    ax.set_yticks(y, order); ax.set_xlim(mc.cv_macro_f1.min() - .03, 1)
    ax.set_xlabel("5-fold GroupKFold macro F1 (mean ± sd over folds)")
    ax.set_title("Cross-validation macro F1 — 78 training trips")
    for i, v in enumerate(mc.cv_macro_f1):
        ax.text(v + .002, i, f"{v:.4f}", va="center", fontsize=8)
    save(fig, "models_cv_macro_f1.png")

    fig, ax = plt.subplots(figsize=(8, 4.2))
    w = .38
    ax.barh(y - w / 2, mc.st_top1, w, color=C1, label="Top-1 accuracy")
    ax.barh(y + w / 2, mc.st_macro_f1, w, color=C2, label="Macro F1")
    ax.set_yticks(y, order); ax.set_xlim(min(mc.st_top1.min(), mc.st_macro_f1.min()) - .03, 1)
    ax.legend(frameon=False, loc="lower right"); ax.set_title("Student-test (10 trips): classifier-only")
    save(fig, "models_student_test.png")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(y, mc.cv_log_loss, .38, color=C1, label="CV")
    ax.barh(y + .38, mc.st_log_loss, .38, color=C2, label="Student-test")
    ax.set_yticks(y + .19, order); ax.set_xlabel("log loss over each fix's candidates (lower is better)")
    ax.legend(frameon=False); ax.set_title("Probability quality")
    save(fig, "models_log_loss.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    data = [folds[folds.model == m].top1.to_numpy() for m in order]
    ax.boxplot(data, orientation="horizontal", tick_labels=order)
    ax.set_xlabel("Top-1 accuracy per CV fold"); ax.set_title("Stability across trip-grouped folds")
    save(fig, "models_cv_folds_box.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(y, mc.infer_ms_per_fix, color=C3); ax.set_xscale("log")
    ax.set_yticks(y, order); ax.set_xlabel("inference ms per GPS fix (12 candidates), log scale")
    ax.set_title("Inference cost (student-test)")
    save(fig, "models_inference_time.png")

    # confusion matrices (candidate level, top-1 decision rule)
    st = pd.read_parquet(os.path.join(RES, "dataset", "student_test.parquet")).reset_index(drop=True)
    gid = obs_groups(st)
    names = [m for m in order if os.path.exists(os.path.join(RES, "st_probs", f"{m}.npy"))]
    cols = 5
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 2.9 * rows))
    for ax, m in zip(axes.ravel(), names):
        p = np.load(os.path.join(RES, "st_probs", f"{m}.npy"))
        cm = confusion_matrix(st.y, top1_mask(p, gid), normalize="true")
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", color="white" if v > .5 else "black", fontsize=9)
        ax.set_title(m, fontsize=10); ax.set_xticks([0, 1], ["other", "picked"]); ax.set_yticks([0, 1], ["wrong arc", "true arc"])
        ax.grid(False)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    fig.suptitle("Candidate-level confusion matrices on student-test (row-normalised)")
    save(fig, "models_confusion.png")


def curve_plots():
    for name, x in (("lightgbm_fold0", "iteration"), ("mlp_fold0", "epoch")):
        p = os.path.join(RES, "curves", f"{name}.csv")
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.plot(d[x], d.train_logloss, color=C1, label="train (folds 1-4)")
        ax.plot(d[x], d.val_logloss, color=C2, label="validation (fold 0 trips)")
        ax.set_xlabel(x); ax.set_ylabel("binary log loss"); ax.legend(frameon=False)
        ax.set_title(("LightGBM" if "lightgbm" in name else "MLP") + " training history")
        save(fig, f"curve_{name}.png")


def bo_plots():
    p = os.path.join(RES, "bo_trials.csv")
    if not os.path.exists(p):
        return
    t = pd.read_csv(p)
    t = t[t.state == "COMPLETE"].sort_values("number")
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.scatter(t.number, t.value, color=C1, s=16, label="trial")
    ax.plot(t.number, t.value.cummax(), color=C2, label="best so far")
    ax.set_xlabel("trial"); ax.set_ylabel("3-fold CV top-1 accuracy"); ax.legend(frameon=False)
    ax.set_title("Bayesian optimisation (TPE) history")
    save(fig, "bo_history.png")
    tv = pd.read_csv(os.path.join(RES, "tuned_vs_untuned.csv")).set_index("variant")
    mets = ["cv_macro_f1", "cv_top1", "st_top1", "st_macro_f1"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    x = np.arange(len(mets))
    ax[0].bar(x - .2, tv.loc["untuned", mets], .4, color=C3, label="untuned")
    ax[0].bar(x + .2, tv.loc["tuned", mets], .4, color=C1, label="tuned")
    ax[0].set_xticks(x, mets, rotation=15); ax[0].set_ylim(tv[mets].min().min() - .02, 1); ax[0].legend(frameon=False)
    ax[1].bar(["untuned", "tuned"], tv.loc[["untuned", "tuned"], "st_log_loss"], color=[C3, C1])
    ax[1].set_ylabel("student-test log loss")
    fig.suptitle("Untuned vs Bayesian-tuned")
    save(fig, "bo_tuned_vs_untuned.png")
    params = [c for c in t.columns if c.startswith("params_")]
    fig, axes = plt.subplots(2, int(np.ceil(len(params) / 2)), figsize=(14, 5))
    for ax, c in zip(axes.ravel(), params):
        ax.scatter(t[c], t.value, s=12, color=C1)
        ax.set_title(c.replace("params_", ""), fontsize=9)
        if t[c].min() > 0 and t[c].max() / t[c].min() > 50:
            ax.set_xscale("log")
    fig.suptitle("Search space coverage vs objective")
    save(fig, "bo_params.png")


def hmm_plots():
    p = os.path.join(RES, "hmm_comparison.csv")
    if not os.path.exists(p):
        return
    h = pd.read_csv(p)
    h = h[h.model != "LambdaRank_untuned"].sort_values("hmm_acc")
    y = np.arange(len(h))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    ax[0].barh(y - .2, h.clf_acc, .4, color=C3, label="classifier only")
    ax[0].barh(y + .2, h.hmm_acc, .4, color=C1, label="classifier + HMM")
    ax[0].set_yticks(y, h.model); ax[0].set_xlim(min(h.clf_acc.min(), h.hmm_acc.min()) - .03, 1)
    ax[0].set_xlabel("student-test fix-level accuracy"); ax[0].legend(frameon=True, loc="lower left")
    ax[1].barh(y - .2, h.clf_cont, .4, color=C3, label="classifier only")
    ax[1].barh(y + .2, h.hmm_cont, .4, color=C1, label="classifier + HMM")
    ax[1].set_yticks(y, h.model); ax[1].set_xlabel("route continuity"); ax[1].set_xlim(.6, 1.005)
    ax[1].legend(frameon=True, loc="lower left")
    fig.suptitle("Effect of HMM/Viterbi decoding (same decoder parameters for every model)")
    save(fig, "hmm_effect.png")
    pt = pd.read_csv(os.path.join(RES, "hmm_per_trip.csv"))
    final = [json.load(open(os.path.join(ROOT, "models", "final", "config.json")))["results_key"]]
    if final:
        f = pt[pt.model == final[0]].sort_values("trip")
        x = np.arange(len(f))
        fig, ax = plt.subplots(figsize=(10, 3.8))
        ax.bar(x - .27, f.clf_acc, .27, color=C3, label="classifier-only acc")
        ax.bar(x, f.hmm_acc, .27, color=C1, label="HMM acc")
        ax.bar(x + .27, f.hmm_f1, .27, color=C2, label="HMM route F1")
        ax.set_xticks(x, f.trip, rotation=30); ax.set_ylim(.5, 1.01); ax.legend(frameon=False, ncol=3)
        ax.set_title(f"Per student-test trip — {final[0]}")
        save(fig, "hmm_per_trip.png")
    d = os.path.join(RES, "hmm_bo_trials.csv")
    if os.path.exists(d):
        t = pd.read_csv(d)
        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        ax.scatter(t.number, t.value, s=16, color=C1); ax.plot(t.number, t.value.cummax(), color=C2)
        ax.set_xlabel("trial"); ax.set_ylabel("train fix accuracy (OOF probabilities)")
        ax.set_title("Decoder parameters — Bayesian optimisation")
        save(fig, "hmm_bo_history.png")


if __name__ == "__main__":
    model_plots()
    curve_plots()
    bo_plots()
    hmm_plots()
    print("figures written to", FIG)
