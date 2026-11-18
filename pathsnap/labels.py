"""Reference labels per GPS observation (training / evaluation only).

The .route file gives the ordered arcs the vehicle drove, not which arc each GPS fix belongs to.
We recover a per-observation reference arc with a monotone alignment: observation i is assigned
route position j(i), with j non-decreasing in i (the vehicle never goes back in the route list), by
minimising

    sum_i  dist(p_i, arc_j(i))  +  lambda * [heading valid] * (1 - cos(heading_i - bearing_j(i))) / 2

The heading term separates a road's two directed twins and the in/out legs of a U-turn, which pure
distance cannot. Observations whose aligned arc is further than `far_m` are flagged `unreliable`
(GPS outliers / reference-route gaps); they are kept in the data but reported separately.
"""
from __future__ import annotations

import numpy as np

from .geo import point_segment


def align_route(xy, head, head_valid, route, net, lam: float = 25.0, far_m: float = 60.0):
    route = np.asarray(route)
    A, B = net.A[route], net.B[route]
    rb = net.heading[route]
    n, m = len(xy), len(route)
    idx = np.arange(m)
    back = np.zeros((n, m), np.int32)
    prev = None
    for i in range(n):
        d, _, _ = point_segment(xy[i][None, :], A, B)
        cost = d.copy()
        if head_valid[i]:
            cost += lam * (1 - np.cos(head[i] - rb)) / 2
        if prev is None:
            cur = cost
            back[i] = idx
        else:
            pm = np.minimum.accumulate(prev)
            arg = np.maximum.accumulate(np.where(prev == pm, idx, 0))
            cur = cost + pm
            back[i] = arg
        prev = cur
    j = int(np.argmin(prev))
    pos = np.empty(n, np.int64)
    for i in range(n - 1, -1, -1):
        pos[i] = j
        j = back[i, j]
    arc = route[pos]
    d, _, _ = point_segment(xy, net.A[arc], net.B[arc])
    return arc, pos, d > far_m
