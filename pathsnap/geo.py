"""Geometry. Each trip is projected into its own azimuthal-equidistant (AEQD) plane centred on the
GPS track, so distances are in metres and accurate to well under 0.1 % over the ~50 km extent of a trip.
Euclidean distance in degrees would be wrong (1 deg lon = 111 km * cos(lat))."""
from __future__ import annotations

import numpy as np
from pyproj import Transformer


class LocalProjection:
    def __init__(self, lon0: float, lat0: float):
        self.lon0, self.lat0 = lon0, lat0
        self._fwd = Transformer.from_crs(
            "EPSG:4326", f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m", always_xy=True)
        self._inv = Transformer.from_crs(
            f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m", "EPSG:4326", always_xy=True)

    @classmethod
    def for_track(cls, track: np.ndarray) -> "LocalProjection":
        return cls(float(np.median(track[:, 0])), float(np.median(track[:, 1])))

    def to_xy(self, lonlat: np.ndarray) -> np.ndarray:
        x, y = self._fwd.transform(lonlat[:, 0], lonlat[:, 1])
        return np.column_stack([x, y])

    def to_lonlat(self, xy: np.ndarray) -> np.ndarray:
        lon, lat = self._inv.transform(xy[:, 0], xy[:, 1])
        return np.column_stack([lon, lat])


def point_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray):
    """Vectorised point-to-segment. p,a,b broadcastable (...,2).
    Returns (distance, t in [0,1] along a->b, unclipped t)."""
    ab = b - a
    L2 = np.maximum((ab * ab).sum(-1), 1e-9)
    t_raw = ((p - a) * ab).sum(-1) / L2
    t = np.clip(t_raw, 0.0, 1.0)
    proj = a + t[..., None] * ab
    d = np.sqrt(((p - proj) ** 2).sum(-1))
    return d, t, t_raw


def bearing(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Compass-style angle (radians) of a vector in the local plane."""
    return np.arctan2(dx, dy)


def angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Absolute smallest difference between two angles, radians in [0, pi]."""
    d = np.abs((a - b + np.pi) % (2 * np.pi) - np.pi)
    return d
