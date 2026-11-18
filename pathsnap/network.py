"""Road network in the trip's local plane + candidate search + bounded shortest paths."""
from __future__ import annotations

import heapq

import numpy as np
from scipy.spatial import cKDTree

from .geo import LocalProjection, point_segment, bearing


class RoadNetwork:
    def __init__(self, nodes_lonlat: np.ndarray, arcs: np.ndarray, proj: LocalProjection,
                 focus_xy: np.ndarray | None = None, margin_m: float = 3000.0, sample_m: float = 8.0):
        self.proj = proj
        self.arcs = arcs
        self.nodes_xy = proj.to_xy(nodes_lonlat)
        a = self.nodes_xy[arcs[:, 0]]
        b = self.nodes_xy[arcs[:, 1]]
        self.A, self.B = a, b
        self.length = np.linalg.norm(b - a, axis=1)
        self.heading = bearing(b[:, 0] - a[:, 0], b[:, 1] - a[:, 1])
        V = len(nodes_lonlat)
        self.out_deg = np.bincount(arcs[:, 0], minlength=V)
        self.in_deg = np.bincount(arcs[:, 1], minlength=V)
        # adjacency: node -> list of (arc_id, to_node, length)
        order = np.argsort(arcs[:, 0], kind="stable")
        self._adj_order = order
        self._adj_start = np.searchsorted(arcs[order, 0], np.arange(V + 1))
        # reverse twin lookup (b->a for a->b)
        key = {(int(f), int(t)): i for i, (f, t) in enumerate(arcs)}
        self.twin = np.array([key.get((int(t), int(f)), -1) for f, t in arcs], dtype=np.int64)
        # spatial index over arc samples (only arcs near the track are indexed)
        if focus_xy is not None:
            lo = focus_xy.min(0) - margin_m
            hi = focus_xy.max(0) + margin_m
            near = ((np.maximum(a, b) >= lo) & (np.minimum(a, b) <= hi)).all(1)
            ids = np.flatnonzero(near)
        else:
            ids = np.arange(len(arcs))
        n_s = np.maximum(1, np.ceil(self.length[ids] / sample_m).astype(int)) + 1
        rep = np.repeat(ids, n_s)
        frac = np.concatenate([np.linspace(0, 1, k) for k in n_s])
        pts = self.A[rep] + frac[:, None] * (self.B[rep] - self.A[rep])
        self._sample_arc = rep
        self._tree = cKDTree(pts)
        self._sp_cache: dict[int, tuple[float, dict, dict]] = {}

    # ---------------- candidates ----------------
    def candidates(self, xy: np.ndarray, k: int = 12, radius: float = 80.0, probe: int = 96):
        """For every GPS point: up to k nearest arcs within `radius` m (exact segment distance).
        If nothing lies inside the radius we still return the k nearest arcs, so no observation is left
        without a state (they get flagged through the distance features)."""
        n = len(xy)
        _, si = self._tree.query(xy, k=min(probe, self._tree.n))
        out_arc = np.full((n, k), -1, np.int64)
        out_d = np.full((n, k), np.inf)
        out_t = np.zeros((n, k))
        out_traw = np.zeros((n, k))
        for i in range(n):
            arcs = np.unique(self._sample_arc[si[i]])
            d, t, traw = point_segment(xy[i], self.A[arcs], self.B[arcs])
            inside = d <= radius
            sel = np.argsort(d)
            if inside.any():
                sel = sel[inside[sel]]
            sel = sel[:k]
            m = len(sel)
            out_arc[i, :m] = arcs[sel]
            out_d[i, :m] = d[sel]
            out_t[i, :m] = t[sel]
            out_traw[i, :m] = traw[sel]
        return out_arc, out_d, out_t, out_traw

    # ---------------- shortest paths ----------------
    def _dijkstra(self, src: int, bound: float):
        dist = {src: 0.0}
        pred: dict[int, int] = {}  # node -> arc used to reach it
        heap = [(0.0, src)]
        order, start, arcs, L = self._adj_order, self._adj_start, self.arcs, self.length
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, np.inf) or d > bound:
                continue
            for e in order[start[u]:start[u + 1]]:
                v = int(arcs[e, 1])
                nd = d + L[e]
                if nd < dist.get(v, np.inf) and nd <= bound:
                    dist[v] = nd
                    pred[v] = int(e)
                    heapq.heappush(heap, (nd, v))
        return dist, pred

    def node_dist(self, src: int, dst: int, bound: float) -> float:
        hit = self._sp_cache.get(src)
        if hit is None or (hit[0] < bound and dst not in hit[1]):
            dist, pred = self._dijkstra(src, max(bound, 1500.0))
            hit = (max(bound, 1500.0), dist, pred)
            self._sp_cache[src] = hit
        return hit[1].get(dst, np.inf)

    def node_path(self, src: int, dst: int, bound: float = 20000.0) -> list[int] | None:
        """Arc ids of the shortest path src -> dst, or None."""
        if src == dst:
            return []
        hit = self._sp_cache.get(src)
        if hit is None or dst not in hit[1]:
            dist, pred = self._dijkstra(src, bound)
            hit = (bound, dist, pred)
            self._sp_cache[src] = hit
        dist, pred = hit[1], hit[2]
        if dst not in dist:
            return None
        path = []
        v = dst
        while v != src:
            e = pred[v]
            path.append(e)
            v = int(self.arcs[e, 0])
        return path[::-1]
