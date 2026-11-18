"""Model zoo for the candidate-scoring task, plus the probability interface the decoder needs.

Every model maps a feature row (one GPS fix x one candidate arc) to P(candidate is the reference arc).
The ranker is the odd one out: it is trained listwise per observation and its scores are turned into
probabilities with a per-observation softmax.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRanker
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .features import FEATURES

SEED = 352093  # course code 352 + SRN 093


def per_obs_softmax(scores: np.ndarray, groups: np.ndarray) -> np.ndarray:
    s = pd.Series(scores)
    m = s.groupby(groups).transform("max").to_numpy()
    e = np.exp(scores - m)
    z = pd.Series(e).groupby(groups).transform("sum").to_numpy()
    return e / z


class RankerModel:
    """LightGBM LambdaRank over each observation's candidate list (listwise learning-to-rank)."""

    def __init__(self, **kw):
        self.kw = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=40,
                       subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED,
                       verbose=-1, n_jobs=-1)
        self.kw.update(kw)
        self.m = LGBMRanker(objective="lambdarank", lambdarank_truncation_level=3,
                            eval_at=[1], **self.kw)

    def fit(self, X, y, group_ids):
        order = np.argsort(group_ids, kind="stable")
        _, counts = np.unique(group_ids[order], return_counts=True)
        self.m.fit(X[order], y[order], group=counts)
        return self

    def predict_proba(self, X, group_ids):
        return per_obs_softmax(self.m.predict(X), group_ids)


def zoo():
    """name -> (family, builder, config string, row cap for fitting or None)."""
    return {
        "LogReg": ("Linear", lambda: make_pipeline(
            StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, class_weight=None)),
            "L2, C=1, standardised", None),
        "GaussNB": ("Probabilistic", lambda: make_pipeline(
            QuantileTransformer(output_distribution="normal", random_state=SEED), GaussianNB()),
            "quantile->normal features", None),
        "KNN": ("Instance-based", lambda: make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=35, weights="distance", n_jobs=-1)),
            "k=35, distance-weighted", 120_000),
        "DecisionTree": ("Tree", lambda: DecisionTreeClassifier(
            max_depth=12, min_samples_leaf=50, random_state=SEED), "depth 12, leaf 50", None),
        "RandomForest": ("Bagging ensemble", lambda: RandomForestClassifier(
            n_estimators=200, min_samples_leaf=20, max_features="sqrt", max_samples=0.5,
            n_jobs=-1, random_state=SEED), "200 trees, leaf 20, 50% bootstrap", None),
        "ExtraTrees": ("Randomised ensemble", lambda: ExtraTreesClassifier(
            n_estimators=200, min_samples_leaf=20, max_features=0.6, n_jobs=-1, random_state=SEED),
            "200 trees, leaf 20, 60% features", None),
        "HistGB": ("Boosting", lambda: HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0,
            random_state=SEED), "300 iters, lr .08, 31 leaves", None),
        "LightGBM": ("Boosting", lambda: LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=40,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1,
            n_jobs=-1), "400 trees, lr .05, 31 leaves", None),
        "MLP": ("Neural network", lambda: make_pipeline(
            StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=512,
                                            learning_rate_init=1e-3, max_iter=40, random_state=SEED)),
            "64-32 ReLU, Adam, 40 epochs", 400_000),
        "LambdaRank": ("Learning-to-rank", lambda: RankerModel(),
                       "LightGBM lambdarank, per-fix lists", None),
    }


def fit_model(name, builder, X, y, gid):
    m = builder()
    if isinstance(m, RankerModel):
        return m.fit(X, y, gid)
    return m.fit(X, y)


def predict_model(m, X, gid):
    if isinstance(m, RankerModel):
        return m.predict_proba(X, gid)
    return m.predict_proba(X)[:, 1]


__all__ = ["FEATURES", "SEED", "zoo", "fit_model", "predict_model", "per_obs_softmax", "RankerModel"]
