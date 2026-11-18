"""Candidate-level and observation-level metrics shared by training, tuning and the API."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

# trips whose reference route does not describe the recorded track (median 130-200 m off for most of
# the trip; see results/data_quality.md). Excluded from fitting and CV.
BAD_REF_TRIPS = {"00000033", "00000054"}


def obs_groups(df: pd.DataFrame) -> np.ndarray:
    return pd.factorize(df["trip"].astype(str) + ":" + df["obs"].astype(str))[0]


def top1_mask(p: np.ndarray, gid: np.ndarray) -> np.ndarray:
    s = pd.DataFrame({"p": p, "g": gid})
    best = s.groupby("g")["p"].idxmax().to_numpy()
    m = np.zeros(len(p), bool)
    m[best] = True
    return m


def candidate_metrics(df: pd.DataFrame, p: np.ndarray) -> dict:
    """df must have a 0..n-1 RangeIndex and columns trip, obs, y."""
    gid = obs_groups(df)
    y = df["y"].to_numpy()
    pick = top1_mask(p, gid)
    n_obs = gid.max() + 1
    top1 = y[pick].sum() / n_obs
    # softmax-over-candidates log loss (what the HMM emission actually consumes)
    z = pd.Series(p).groupby(gid).transform("sum").to_numpy()
    pn = p / np.maximum(z, 1e-12)
    has_pos = pd.Series(y).groupby(gid).transform("max").to_numpy() == 1
    ll = -np.mean(np.log(np.clip(pn[(y == 1) & has_pos], 1e-6, 1)))
    return dict(top1=top1, macro_f1=f1_score(y, pick, average="macro"),
                precision=precision_score(y, pick, average="macro", zero_division=0),
                recall=recall_score(y, pick, average="macro", zero_division=0),
                f1_thr05=f1_score(y, p >= 0.5, average="macro"), log_loss=ll)
