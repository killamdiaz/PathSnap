"""Render the report tables (markdown) directly from the result files -> results/report_tables.md"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap.data import ROOT  # noqa: E402

RES = os.path.join(ROOT, "results")


def md(df, floatfmt=4):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda v: f"{v:.{floatfmt}f}")
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in df.to_numpy())


def main():
    out = ["# PathSnap — report tables (auto-generated from results/)", ""]
    inv = pd.read_csv(os.path.join(RES, "eda_inventory.csv"), dtype={"trip": str})
    g = inv.groupby("partition").agg(trips=("trip", "size"), fixes=("n_fixes", "sum"),
                                     median_fixes=("n_fixes", "median"), total_km=("track_km", "sum"),
                                     median_dt1=("dt_eq_1s", "median"), max_gap_s=("max_gap_s", "max"),
                                     median_nodes=("nodes", "median"), median_arcs=("arcs", "median"),
                                     median_snap_m=("median_snap_m", "median")).reset_index()
    out += ["## Dataset inventory (EDA, train + student_test)", md(g, 2), ""]
    mc = pd.read_csv(os.path.join(RES, "model_comparison.csv"))
    hm = os.path.join(RES, "hmm_comparison.csv")
    if os.path.exists(hm):
        h = pd.read_csv(hm).set_index("model")
        mc["route_continuity_clf"] = mc.model.map(h.clf_cont)
        mc["hmm_top1"] = mc.model.map(h.hmm_acc)
    t = mc[["model", "family", "config", "cv_macro_f1", "st_top1", "route_continuity_clf" if "route_continuity_clf" in mc else "st_route_continuity",
            "st_macro_f1", "st_log_loss", "infer_ms_per_fix"]].copy()
    t.columns = ["Model", "Algorithm family", "Key configuration", "CV Macro F1", "Top-1 Accuracy (student-test)",
                 "Route continuity (classifier-only)", "Student-Test Macro F1", "Log Loss", "Inference ms/fix"]
    out += ["## 3.1 Overall model comparison", md(t), "",
            "## Extra model metrics", md(mc.drop(columns=["family", "config"]).round(4)), ""]
    bo = os.path.join(RES, "bo_best.json")
    if os.path.exists(bo):
        b = json.load(open(bo))
        out += ["## 4. Bayesian optimisation", f"model: {b['model']}, trials: {b['n_trials']}, "
                f"search time: {b['search_seconds']:.0f} s, best 3-fold CV top-1: {b['best_value_cv3_top1']:.4f}", "",
                md(pd.DataFrame(b["best_params"].items(), columns=["hyperparameter", "best value"])), "",
                md(pd.read_csv(os.path.join(RES, "tuned_vs_untuned.csv"))), ""]
    if os.path.exists(hm):
        out += ["## 5. Classifier-only vs classifier + HMM (student-test)", md(pd.read_csv(hm)), ""]
        dj = os.path.join(ROOT, "models", "final", "decoder.json")
        if os.path.exists(dj):
            d = json.load(open(dj))
            out += ["Decoder parameters (Bayesian-tuned on OOF train probabilities):", "",
                    md(pd.DataFrame(d["decoder"].items(), columns=["parameter", "value"])), "",
                    f"Decoder tuning: {json.dumps(d['tuning'])}", ""]
        pt = pd.read_csv(os.path.join(RES, "hmm_per_trip.csv"), dtype={"trip": str})
        fin = [m for m in pt.model.unique() if m.endswith("_tuned")]
        if fin:
            out += [f"## Per-trip results ({fin[0]})", md(pt[pt.model == fin[0]].drop(columns="model")), ""]
    with open(os.path.join(RES, "report_tables.md"), "w") as fh:
        fh.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
