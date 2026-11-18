"""Build the candidate/feature table + reference labels for the train and student_test partitions.

Output: results/dataset/<partition>.parquet  (one row per (trip, observation, candidate arc))
        results/dataset/<partition>_obs.parquet (one row per observation: reference arc, miss flag ...)
        results/candidate_sweep.csv  (candidate miss rate vs K / radius on TRAIN only, justifies K,R)
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap.data import ROOT, list_trips, load_route, load_trip  # noqa: E402
from pathsnap.features import build_candidates  # noqa: E402
from pathsnap.labels import align_route  # noqa: E402

OUT = os.path.join(ROOT, "results", "dataset")
os.makedirs(OUT, exist_ok=True)


def run(partition):
    rows, obs_rows, sweep = [], [], []
    for tid in list_trips(partition):
        t0 = time.time()
        trip = load_trip(tid, partition)
        tc = build_candidates(trip, k=20, radius=150.0)  # wide set; narrowed below + used for the sweep
        route = load_route(tid, partition)
        ref, pos, unrel = align_route(tc.xy, tc.motion["head"], tc.motion["head_valid"], route, tc.net)
        if partition == "train":
            d = tc.feats.pivot(index="obs", columns="slot", values="d").reindex(range(len(ref))).to_numpy()
            for K in (4, 6, 8, 10, 12, 16, 20):
                for R in (30, 50, 80, 120, 150):
                    ok = (tc.arc[:, :K] == ref[:, None]) & (np.nan_to_num(d[:, :K], nan=1e9) <= R)
                    sweep.append((tid, K, R, 1 - ok.any(1).mean()))
        # keep the production setting K=12, R=80 m (rebuild ranks/gaps for that set)
        tc12 = build_candidates(trip)
        f = tc12.feats
        f["y"] = (f["arc"].to_numpy() == ref[f["obs"].to_numpy()]).astype(np.int8)
        rows.append(f)
        miss = ~(tc12.arc == ref[:, None]).any(1)
        obs_rows.append(pd.DataFrame({"trip": tid, "obs": np.arange(len(ref)), "ref_arc": ref,
                                      "route_pos": pos, "unreliable": unrel, "cand_miss": miss,
                                      "t": tc12.t}))
        print(f"{partition} {tid}: {len(ref)} obs, {len(f)} rows, miss {miss.mean():.3%}, "
              f"{time.time() - t0:.1f}s", flush=True)
    pd.concat(rows).to_parquet(os.path.join(OUT, f"{partition}.parquet"))
    pd.concat(obs_rows).to_parquet(os.path.join(OUT, f"{partition}_obs.parquet"))
    if sweep:
        s = pd.DataFrame(sweep, columns=["trip", "K", "radius_m", "miss_rate"])
        s.groupby(["K", "radius_m"]).miss_rate.mean().reset_index().to_csv(
            os.path.join(ROOT, "results", "candidate_sweep.csv"), index=False)


if __name__ == "__main__":
    for p in sys.argv[1:] or ["train", "student_test"]:
        run(p)
