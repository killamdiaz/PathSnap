"""End-to-end inference for one trip: GPS + road network -> matched arc per fix -> route.

Inputs are a Trip (track/nodes/arcs) and a trained artifact. The reference route is never an input.
"""
from __future__ import annotations

import json
import os
import time

import joblib
import numpy as np

from .data import ROOT, Trip
from .decoder import DecoderParams, precompute_transitions, viterbi
from .features import FEATURES, build_candidates
from .models import predict_model
from .route import assemble, continuity

ARTIFACT_DIR = os.environ.get("PATHSNAP_ARTIFACT", os.path.join(ROOT, "models", "final"))


def load_artifact(path: str = ARTIFACT_DIR):
    model = joblib.load(os.path.join(path, "model.joblib"))
    with open(os.path.join(path, "config.json")) as fh:
        cfg = json.load(fh)
    assert cfg["features"] == FEATURES, "feature list changed since the model was trained"
    return model, cfg


def score_candidates(tc, model) -> np.ndarray:
    f = tc.feats
    gid = f["obs"].to_numpy()
    p = predict_model(model, f[FEATURES].to_numpy(), gid)
    n, k = tc.arc.shape
    P = np.zeros((n, k))
    P[f["obs"].to_numpy(), f["slot"].to_numpy()] = p
    return P


def predict_trip(trip: Trip, model, params: DecoderParams, tc=None, cache=None) -> dict:
    timings = {}
    t0 = time.perf_counter()
    if tc is None:
        tc = build_candidates(trip)
    timings["candidates_features_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    P = score_candidates(tc, model)
    timings["classifier_s"] = time.perf_counter() - t0
    valid = tc.arc >= 0
    clf_slot = np.where(valid, P, -1).argmax(1)
    clf_arcs = tc.arc[np.arange(len(P)), clf_slot]
    t0 = time.perf_counter()
    if cache is None:
        cache = precompute_transitions(tc)
    path, breaks = viterbi(P, valid, cache, params)
    hmm_arcs = tc.arc[np.arange(len(P)), path]
    route, jumps = assemble(hmm_arcs, tc.net)
    timings["hmm_viterbi_s"] = time.perf_counter() - t0
    return dict(tc=tc, cache=cache, probs=P, clf_arcs=clf_arcs, hmm_arcs=hmm_arcs, route=route,
                breaks=breaks, jumps=jumps, timings=timings,
                clf_continuity=continuity(clf_arcs, tc.net, tc.xy),
                hmm_continuity=continuity(hmm_arcs, tc.net, tc.xy))
