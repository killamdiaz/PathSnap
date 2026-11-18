"""Per-observation motion descriptors computed from the GPS track only.

Heading is estimated over a *distance* window rather than a fixed index window: from point i we walk
backwards/forwards until the vehicle has moved at least `span_m` metres. This keeps the heading stable
when the car is crawling or stopped (where index-based headings become pure GPS jitter), and lets us
flag observations where no reliable heading exists at all.
"""
from __future__ import annotations

import numpy as np

from .geo import angle_diff, bearing


def distance_window_heading(xy: np.ndarray, span_m: float = 12.0, max_steps: int = 30):
    n = len(xy)
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    lo = np.searchsorted(cum, cum - span_m / 2, side="right") - 1
    hi = np.searchsorted(cum, cum + span_m / 2, side="left")
    idx = np.arange(n)
    lo = np.clip(np.maximum(lo, idx - max_steps), 0, n - 1)
    hi = np.clip(np.minimum(hi, idx + max_steps), 0, n - 1)
    v = xy[hi] - xy[lo]
    moved = np.linalg.norm(v, axis=1)
    head = bearing(v[:, 0], v[:, 1])
    valid = moved >= 0.5 * span_m
    return head, valid, moved


def motion_table(xy: np.ndarray, t: np.ndarray) -> dict[str, np.ndarray]:
    n = len(xy)
    step = np.zeros(n)
    step[1:] = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    dt_prev = np.full(n, 1.0)
    dt_prev[1:] = np.diff(t)
    dt_next = np.full(n, 1.0)
    dt_next[:-1] = np.diff(t)
    speed = step / np.maximum(dt_prev, 1e-3)
    speed[0] = speed[1] if n > 1 else 0.0
    head, hvalid, _ = distance_window_heading(xy)
    head_wide, wvalid, _ = distance_window_heading(xy, span_m=40.0, max_steps=60)
    # turning: change of wide heading between the points 20 m behind and 20 m ahead
    turn = np.zeros(n)
    if n > 2:
        h_prev = np.roll(head_wide, 3)
        h_next = np.roll(head_wide, -3)
        turn = angle_diff(h_prev, h_next)
        turn[:3] = 0
        turn[-3:] = 0
    return dict(step=step, dt_prev=dt_prev, dt_next=dt_next, speed=speed,
                head=head, head_valid=hvalid, head_wide=head_wide, wide_valid=wvalid, turn=turn)
