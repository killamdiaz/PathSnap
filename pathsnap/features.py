"""Candidate generation + per-(observation, candidate-arc) feature table.

Everything here is computed from the GPS track and the road network only (no reference route).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import Trip
from .geo import LocalProjection, point_segment
from .motion import motion_table
from .network import RoadNetwork

K_CAND = 12          # candidate arcs kept per observation (twins count separately)
RADIUS_M = 80.0      # search radius; chosen from the candidate-miss curve on training trips
WIN = 3              # +-3 observations for the corridor-consensus distance

FEATURES = [
    "d", "d_rank", "d_gap", "d_rel", "win_d", "win_gap",
    "head_cos", "head_ok", "wide_cos", "speed", "log_len", "t_frac", "overhang",
    "n_cand", "n_close", "twin_in", "deg", "dt_prev", "turn", "end_room",
]


@dataclass
class TripCandidates:
    trip_id: str
    xy: np.ndarray
    t: np.ndarray
    arc: np.ndarray      # (N,K) arc id or -1
    offset: np.ndarray   # (N,K) metres from arc start of the projection
    feats: pd.DataFrame  # one row per valid (obs, slot)
    net: RoadNetwork
    proj: LocalProjection
    motion: dict


def build_candidates(trip: Trip, k: int = K_CAND, radius: float = RADIUS_M) -> TripCandidates:
    proj = LocalProjection.for_track(trip.track)
    xy = proj.to_xy(trip.track[:, :2])
    t = trip.track[:, 2]
    net = RoadNetwork(trip.nodes, trip.arcs, proj, focus_xy=xy)
    mo = motion_table(xy, t)
    arc, d, tf, traw = net.candidates(xy, k=k, radius=radius)
    n = len(xy)
    valid = arc >= 0
    safe = np.where(valid, arc, 0)

    L = net.length[safe]
    ah = net.heading[safe]
    dmin = np.where(valid, d, np.inf).min(1, keepdims=True)
    rank = np.argsort(np.argsort(np.where(valid, d, np.inf), 1), 1)

    # corridor consensus: mean distance of the neighbouring fixes to the same arc
    acc = np.zeros((n, k))
    for o in range(-WIN, WIN + 1):
        j = np.clip(np.arange(n) + o, 0, n - 1)
        dd, _, _ = point_segment(xy[j][:, None, :], net.A[safe], net.B[safe])
        acc += dd
    win_d = acc / (2 * WIN + 1)
    win_min = np.where(valid, win_d, np.inf).min(1, keepdims=True)

    hv = mo["head_valid"][:, None]
    head_cos = np.where(hv, np.cos(mo["head"][:, None] - ah), 0.0)
    wv = mo["wide_valid"][:, None]
    wide_cos = np.where(wv, np.cos(mo["head_wide"][:, None] - ah), 0.0)

    cand_sets = [set(r[r >= 0].tolist()) for r in arc]
    twin = net.twin[safe]
    twin_in = np.array([[tw in s for tw in row] for row, s in zip(twin, cand_sets)], float)

    n_cand = valid.sum(1, keepdims=True).repeat(k, 1)
    n_close = (np.where(valid, d, np.inf) <= 25).sum(1, keepdims=True).repeat(k, 1)
    deg = net.out_deg[net.arcs[safe, 1]] + net.in_deg[net.arcs[safe, 0]]
    overhang = np.abs(traw - tf) * L
    # metres left before the arc ends in the direction of travel
    end_room = (1 - tf) * L

    obs, slot = np.nonzero(valid)
    f = {
        "obs": obs, "slot": slot, "arc": arc[obs, slot],
        "d": d[obs, slot], "d_rank": rank[obs, slot], "d_gap": (d - dmin)[obs, slot],
        "d_rel": (d / (dmin + 1.0))[obs, slot],
        "win_d": win_d[obs, slot], "win_gap": (win_d - win_min)[obs, slot],
        "head_cos": head_cos[obs, slot], "head_ok": np.broadcast_to(hv, (n, k))[obs, slot].astype(float),
        "wide_cos": wide_cos[obs, slot], "speed": mo["speed"][obs],
        "log_len": np.log1p(L[obs, slot]), "t_frac": tf[obs, slot], "overhang": overhang[obs, slot],
        "n_cand": n_cand[obs, slot], "n_close": n_close[obs, slot], "twin_in": twin_in[obs, slot],
        "deg": deg[obs, slot], "dt_prev": np.minimum(mo["dt_prev"][obs], 600.0),
        "turn": mo["turn"][obs], "end_room": np.minimum(end_room[obs, slot], 500.0),
    }
    feats = pd.DataFrame(f)
    feats.insert(0, "trip", trip.trip_id)
    return TripCandidates(trip.trip_id, xy, t, arc, tf * L, feats, net, proj, mo)
