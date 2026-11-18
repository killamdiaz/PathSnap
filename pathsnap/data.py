"""Raw-file loading. A trip folder <id>/ holds <id>.track, .nodes, .arcs and (optionally) .route.

The prediction pipeline only ever calls `load_trip(..., with_route=False)`; the .route file is read
separately by evaluation code (`load_route`) so reference labels can never leak into inference.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("PATHSNAP_DATA", os.path.join(ROOT, "data"))


@dataclass
class Trip:
    trip_id: str
    track: np.ndarray   # (N,3) lon, lat, t[s]
    nodes: np.ndarray   # (V,2) lon, lat
    arcs: np.ndarray    # (E,2) from_node, to_node (int)


def trip_dir(trip_id: str, partition: str, data_dir: str = DATA_DIR) -> str:
    return os.path.join(data_dir, partition, trip_id)


def list_trips(partition: str, data_dir: str = DATA_DIR) -> list[str]:
    p = os.path.join(data_dir, partition)
    if not os.path.isdir(p):
        return []
    return sorted(d for d in os.listdir(p) if os.path.isfile(os.path.join(p, d, f"{d}.track")))


def _load(path, **kw):
    return np.loadtxt(path, ndmin=2, **kw)


def load_trip(trip_id: str, partition: str, data_dir: str = DATA_DIR) -> Trip:
    d = trip_dir(trip_id, partition, data_dir)
    track = _load(os.path.join(d, f"{trip_id}.track"))
    nodes = _load(os.path.join(d, f"{trip_id}.nodes"))
    arcs = _load(os.path.join(d, f"{trip_id}.arcs"), dtype=np.int64)
    track = clean_track(track)
    return Trip(trip_id, track, nodes, arcs)


def load_route(trip_id: str, partition: str, data_dir: str = DATA_DIR) -> np.ndarray:
    """Reference arc sequence. Evaluation-only: never pass this into the prediction pipeline."""
    p = os.path.join(trip_dir(trip_id, partition, data_dir), f"{trip_id}.route")
    return np.loadtxt(p, dtype=np.int64, ndmin=1)


def clean_track(track: np.ndarray) -> np.ndarray:
    """Drop non-finite rows, implausible coordinates, and non-increasing timestamps (keeps first)."""
    ok = np.isfinite(track).all(1)
    ok &= (np.abs(track[:, 0]) <= 180) & (np.abs(track[:, 1]) <= 90)
    track = track[ok]
    if len(track) == 0:
        return track
    keep = np.ones(len(track), bool)
    last = track[0, 2]
    for i in range(1, len(track)):
        if track[i, 2] <= last:
            keep[i] = False
        else:
            last = track[i, 2]
    return track[keep]
