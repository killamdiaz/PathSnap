"""Folium map of one matched trip: GPS fixes, nearby network, reference route (offline reports only),
classifier-only picks and the HMM route, each in its own toggleable layer with a legend."""
from __future__ import annotations

import folium
import numpy as np
from branca.element import MacroElement, Template

LEGEND = """{% macro html(this, kwargs) %}
<div style="position:fixed; bottom:24px; left:24px; z-index:9999; background:white; padding:10px 12px;
 border-radius:8px; box-shadow:0 1px 6px rgba(0,0,0,.25); font:12px/1.5 system-ui, sans-serif">
 <b>{{this.title}}</b><br>
 <span style="color:#1d4ed8">&#9679;</span> GPS fixes<br>
 <span style="color:#9aa5b1">&#9473;&#9473;</span> road network (near track)<br>
 <span style="color:#16a34a">&#9473;&#9473;</span> reference route<br>
 <span style="color:#f59e0b">&#9679;</span> classifier-only pick changed by HMM (off by default)<br>
 <span style="color:#dc2626">&#9473;&#9473;</span> PathSnap HMM route
</div>{% endmacro %}"""


class _Legend(MacroElement):
    def __init__(self, title):
        super().__init__()
        self._template = Template(LEGEND)
        self.title = title


def _arc_line(trip, a):
    f, t = trip.arcs[a]
    return [[trip.nodes[f, 1], trip.nodes[f, 0]], [trip.nodes[t, 1], trip.nodes[t, 0]]]


def trip_map(trip, tc, hmm_arcs, route, clf_arcs=None, ref_route=None, out=None, title="PathSnap"):
    lat, lon = trip.track[:, 1], trip.track[:, 0]
    m = folium.Map(location=[lat.mean(), lon.mean()], zoom_start=14, tiles="OpenStreetMap")
    near = np.unique(tc.arc[tc.arc >= 0])
    g = folium.FeatureGroup(name="Road network (candidate arcs)", show=True)
    for a in near:
        folium.PolyLine(_arc_line(trip, a), color="#9aa5b1", weight=2, opacity=.55).add_to(g)
    g.add_to(m)
    step = max(1, len(lat) // 1500)
    g = folium.FeatureGroup(name="GPS fixes")
    for i in range(0, len(lat), step):
        folium.CircleMarker([lat[i], lon[i]], radius=1.6, color="#1d4ed8", weight=0, fill=True,
                            fill_opacity=.55, tooltip=f"fix {i}, t={trip.track[i, 2]:.0f}s, arc {hmm_arcs[i]}").add_to(g)
    g.add_to(m)
    if ref_route is not None:
        g = folium.FeatureGroup(name="Reference route")
        for a in ref_route:
            folium.PolyLine(_arc_line(trip, a), color="#16a34a", weight=8, opacity=.45).add_to(g)
        g.add_to(m)
    g = folium.FeatureGroup(name="PathSnap HMM route")
    for a in route:
        folium.PolyLine(_arc_line(trip, a), color="#dc2626", weight=3.5, opacity=.95).add_to(g)
    g.add_to(m)
    if clf_arcs is not None:
        diff = np.flatnonzero(clf_arcs != hmm_arcs)
        g = folium.FeatureGroup(name=f"Classifier-only picks that HMM changed ({len(diff)})", show=False)
        for i in diff[:: max(1, len(diff) // 800)]:
            a = clf_arcs[i]
            pt = np.mean(_arc_line(trip, a), axis=0)
            folium.CircleMarker(pt.tolist(), radius=3, color="#f59e0b", fill=True, fill_opacity=.9,
                                tooltip=f"fix {i}: classifier arc {a} -> HMM arc {hmm_arcs[i]}").add_to(g)
        g.add_to(m)
    folium.Marker([lat[0], lon[0]], tooltip="start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([lat[-1], lon[-1]], tooltip="end", icon=folium.Icon(color="red", icon="stop")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().add_child(_Legend(title))
    m.fit_bounds([[lat.min(), lon.min()], [lat.max(), lon.max()]])
    if out:
        m.save(out)
    return m
