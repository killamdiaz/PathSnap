"""Turn per-observation arcs into a route and score it."""
from __future__ import annotations

import numpy as np


def collapse(seq):
    out = []
    for a in seq:
        a = int(a)
        if not out or out[-1] != a:
            out.append(a)
    return out


def continuity(obs_arcs, net, xy=None) -> float:
    """Route continuity: share of consecutive *distinct* matched arcs (in fix order) where the second
    is drivable from the first. Strict version (xy=None): directly adjacent, to_node(a)==from_node(b).
    Default version (xy given): reachable along the directed network within 3x the straight-line
    distance between the two fixes + 100 m (a 1 Hz fix can legitimately skip a short arc)."""
    obs_arcs = np.asarray(obs_arcs)
    change = np.flatnonzero(obs_arcs[1:] != obs_arcs[:-1]) + 1
    if len(change) == 0:
        return 1.0
    a, b = obs_arcs[change - 1], obs_arcs[change]
    direct = net.arcs[a, 1] == net.arcs[b, 0]
    if xy is None:
        return float(direct.mean())
    ok = direct.copy()
    for k in np.flatnonzero(~direct):
        i = change[k]
        g = float(np.linalg.norm(xy[i] - xy[i - 1]))
        bound = 3 * g + 100.0
        ok[k] = net.node_dist(int(net.arcs[a[k], 1]), int(net.arcs[b[k], 0]), bound) <= bound
    return float(ok.mean())


def assemble(obs_arcs, net, max_bridge_m: float = 20000.0):
    """Collapse repeats and bridge non-adjacent consecutive arcs with the shortest directed path.
    Returns (route arc list, number of unbridgeable jumps)."""
    s = collapse(obs_arcs)
    route, jumps = [], 0
    for i, a in enumerate(s):
        if i > 0:
            prev = route[-1]
            u, v = int(net.arcs[prev, 1]), int(net.arcs[a, 0])
            if u != v:
                path = net.node_path(u, v, max_bridge_m)
                if path is None:
                    jumps += 1
                else:
                    route.extend(path)
        route.append(a)
    return collapse(route), jumps


def route_scores(pred_route, ref_route, net) -> dict:
    """Length-weighted overlap between predicted and reference arc sets, plus the route mismatch
    fraction RMF = (wrongly added + wrongly missed length) / reference length."""
    P, R = set(map(int, pred_route)), set(map(int, ref_route))
    L = net.length
    inter = sum(L[a] for a in P & R)
    lp = sum(L[a] for a in P)
    lr = sum(L[a] for a in R)
    prec = inter / lp if lp else 0.0
    rec = inter / lr if lr else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    rmf = ((lp - inter) + (lr - inter)) / lr if lr else np.nan
    return dict(route_precision=prec, route_recall=rec, route_f1=f1, rmf=rmf,
                pred_len_km=lp / 1000, ref_len_km=lr / 1000)
