"""HMM + Viterbi route reconstruction.

Hidden state at observation i  : one of its candidate slots (slot s <-> arc tc.arc[i, s]); the
                                  slot->arc table is kept so the decoded state maps back to a road arc.
Emission                        : log of the classifier's probability, renormalised over the
                                  observation's candidates and sharpened/flattened by `alpha`.
Transition  a(i-1) -> b(i)      : route distance r along the directed network vs. straight GPS
                                  displacement g, Laplace in |r - g| with a scale that grows with the
                                  time gap (beta0 + beta_dt * dt); plus a turn penalty proportional
                                  to the heading change between the two arcs and an explicit U-turn
                                  penalty (b is a's reverse twin, or we move backwards on one arc).
Gaps / dead ends                : if dt > gap_s, or no candidate of i is reachable from any candidate
                                  of i-1, the chain is broken and restarted at i from emissions only;
                                  the route between the two pieces is bridged later by shortest path.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .geo import angle_diff

NEG = -1e18


@dataclass
class DecoderParams:
    alpha: float = 1.0       # emission exponent
    beta0: float = 6.0       # metres, base Laplace scale
    beta_dt: float = 2.0     # metres per second of gap added to the scale
    turn_w: float = 1.0      # log-penalty for a 180 deg heading change between arcs
    uturn: float = 6.0       # log-penalty for a U-turn
    back_tol: float = 12.0   # metres of apparent backwards motion tolerated on the same arc
    gap_s: float = 120.0     # break the chain after a gap longer than this

    def as_dict(self):
        return asdict(self)


@dataclass
class TransitionCache:
    r: list          # per step (K,K) route distance (inf = unreachable)
    turn: list       # per step (K,K) heading change in [0,1]
    ut: list         # per step (K,K) U-turn flag
    g: np.ndarray    # straight displacement per step
    dt: np.ndarray


def precompute_transitions(tc) -> TransitionCache:
    net, arc, off = tc.net, tc.arc, tc.offset
    n, k = arc.shape
    pair_cache: dict[tuple[int, int], float] = {}
    R, T, U = [None], [None], [None]
    g = np.zeros(n)
    g[1:] = np.linalg.norm(np.diff(tc.xy, axis=0), axis=1)
    dt = np.ones(n)
    dt[1:] = np.diff(tc.t)
    for i in range(1, n):
        a_arcs, b_arcs = arc[i - 1], arc[i]
        bound = min(2.5 * g[i] + 150.0 + 20.0 * dt[i], 8000.0)
        r = np.full((k, k), np.inf)
        ut = np.zeros((k, k), bool)
        for ia, a in enumerate(a_arcs):
            if a < 0:
                continue
            la, ta = net.length[a], int(net.arcs[a, 1])
            for ib, b in enumerate(b_arcs):
                if b < 0:
                    continue
                if a == b:
                    step = off[i, ib] - off[i - 1, ia]
                    if step >= -12.0:
                        r[ia, ib] = max(step, 0.0)
                    else:
                        ut[ia, ib] = True
                        r[ia, ib] = -step
                    continue
                key = (a, b)
                link = pair_cache.get(key)
                if link is None:
                    link = 0.0 if ta == int(net.arcs[b, 0]) else net.node_dist(ta, int(net.arcs[b, 0]), bound)
                    pair_cache[key] = link
                if np.isfinite(link):
                    r[ia, ib] = (la - off[i - 1, ia]) + link + off[i, ib]
                    ut[ia, ib] = net.twin[a] == b
        va = np.where(a_arcs >= 0, a_arcs, 0)
        vb = np.where(b_arcs >= 0, b_arcs, 0)
        turn = angle_diff(net.heading[va][:, None], net.heading[vb][None, :]) / np.pi
        turn[a_arcs[:, None] == b_arcs[None, :]] = 0.0
        R.append(r.astype(np.float32))
        T.append(turn.astype(np.float32))
        U.append(ut)
    return TransitionCache(R, T, U, g, dt)


def viterbi(prob: np.ndarray, valid: np.ndarray, cache: TransitionCache, p: DecoderParams):
    """prob (N,K) classifier probabilities; returns chosen slot per observation and chain-break indices."""
    n, k = prob.shape
    pn = prob * valid
    pn = pn / np.maximum(pn.sum(1, keepdims=True), 1e-12)
    logE = np.where(valid, p.alpha * np.log(np.maximum(pn, 1e-9)), NEG)
    back = np.zeros((n, k), np.int16)
    score = logE[0].copy()
    breaks = [0]
    seg_end = {}                      # last index of a chain -> its best final state
    for i in range(1, n):
        r = cache.r[i]
        scale = p.beta0 + p.beta_dt * max(cache.dt[i] - 1.0, 0.0)
        logT = -np.abs(r - cache.g[i]) / scale - p.turn_w * cache.turn[i] - p.uturn * cache.ut[i]
        logT = np.where(np.isfinite(r), logT, NEG)
        cand = score[:, None] + logT
        best = cand.argmax(0)
        best_val = cand[best, np.arange(k)]
        if cache.dt[i] > p.gap_s or best_val.max() <= NEG / 2:
            seg_end[i - 1] = int(np.argmax(score))
            breaks.append(i)
            score = logE[i].copy()
            continue
        back[i] = best
        score = best_val + logE[i]
        score -= score.max()
    path = np.zeros(n, np.int64)
    s = int(np.argmax(score))
    brk = set(breaks)
    for i in range(n - 1, -1, -1):
        path[i] = s
        if i == 0:
            break
        s = seg_end[i - 1] if i in brk else int(back[i, s])
    return path, breaks
