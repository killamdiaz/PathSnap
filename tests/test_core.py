import os
import shutil
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap.data import DATA_DIR, load_trip, load_route  # noqa: E402
from pathsnap.decoder import DecoderParams, TransitionCache, viterbi  # noqa: E402
from pathsnap.geo import angle_diff, point_segment  # noqa: E402


def test_point_segment():
    d, t, _ = point_segment(np.array([5.0, 3.0]), np.array([0.0, 0.0]), np.array([10.0, 0.0]))
    assert d == pytest.approx(3.0) and t == pytest.approx(0.5)
    d, t, traw = point_segment(np.array([-4.0, 3.0]), np.array([0.0, 0.0]), np.array([10.0, 0.0]))
    assert d == pytest.approx(5.0) and t == 0 and traw < 0


def test_angle_diff_wraps():
    assert angle_diff(np.array(3.1), np.array(-3.1)) == pytest.approx(2 * np.pi - 6.2)


def test_viterbi_prefers_reachable_path():
    # 3 fixes, 2 candidates; candidate 1 has higher emission at fix 1 but is unreachable from either
    prob = np.array([[0.9, 0.1], [0.4, 0.6], [0.9, 0.1]])
    inf = np.inf
    r = [None, np.array([[10, inf], [inf, inf]], np.float32), np.array([[10, inf], [inf, inf]], np.float32)]
    z = [None, np.zeros((2, 2), np.float32), np.zeros((2, 2), np.float32)]
    u = [None, np.zeros((2, 2), bool), np.zeros((2, 2), bool)]
    cache = TransitionCache(r, z, u, np.array([0, 10, 10.0]), np.ones(3))
    path, breaks = viterbi(prob, np.ones_like(prob, bool), cache, DecoderParams())
    assert path.tolist() == [0, 0, 0] and breaks == [0]


@pytest.mark.skipif(not os.path.isdir(os.path.join(DATA_DIR, "student_test")), reason="data not present")
def test_inference_never_needs_route(tmp_path):
    """Copy a trip without its .route file and run the full feature + decode path on it."""
    from pathsnap.decoder import precompute_transitions
    from pathsnap.features import build_candidates
    src = os.path.join(DATA_DIR, "student_test", "00000052")
    dst = tmp_path / "evalpart" / "00000052"
    dst.mkdir(parents=True)
    for ext in ("track", "nodes", "arcs"):
        shutil.copy(os.path.join(src, f"00000052.{ext}"), dst)
    trip = load_trip("00000052", "evalpart", data_dir=str(tmp_path))
    tc = build_candidates(trip)
    P = np.exp(-tc.feats.pivot(index="obs", columns="slot", values="d").fillna(1e3).to_numpy() / 10)
    path, _ = viterbi(P, tc.arc >= 0, precompute_transitions(tc), DecoderParams())
    assert len(path) == len(trip.track)
    with pytest.raises(OSError):
        load_route("00000052", "evalpart", data_dir=str(tmp_path))
