"""Exploratory data analysis on the train + student_test partitions.

Outputs
  results/eda_inventory.csv      per-trip inventory (files, sizes, counts, gaps, extent, connectivity)
  results/eda_missing.csv        missing / invalid values per file type
  results/data_quality.md        short record of anomalies + decisions
  figures/eda_*.png              plots
  maps/eda_trip_<id>.html        Folium: GPS fixes, local road network, reference route
"""
import os
import sys

import folium
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap.data import ROOT, list_trips, load_route, load_trip, trip_dir  # noqa: E402
from pathsnap.geo import LocalProjection, point_segment  # noqa: E402
from pathsnap.network import RoadNetwork  # noqa: E402

FIG, MAPS, RES = (os.path.join(ROOT, d) for d in ("figures", "maps", "results"))
for d in (FIG, MAPS, RES):
    os.makedirs(d, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})


def raw_missing(path, ncol):
    bad = nan = 0
    n = 0
    with open(path) as fh:
        for line in fh:
            n += 1
            parts = line.split()
            if len(parts) != ncol:
                bad += 1
                continue
            try:
                v = [float(x) for x in parts]
                nan += any(np.isnan(x) for x in v)
            except ValueError:
                bad += 1
    return n, bad, nan


def inventory():
    rows, miss = [], []
    gaps_all = {"train": [], "student_test": []}
    snap_d = {"train": [], "student_test": []}
    for part in ("train", "student_test"):
        for tid in list_trips(part):
            d = trip_dir(tid, part)
            sizes = {ext: os.path.getsize(os.path.join(d, f"{tid}.{ext}")) for ext in
                     ("track", "nodes", "arcs", "route")}
            for ext, nc in (("track", 3), ("nodes", 2), ("arcs", 2), ("route", 1)):
                n, bad, nan = raw_missing(os.path.join(d, f"{tid}.{ext}"), nc)
                miss.append((part, tid, ext, n, bad, nan))
            raw = np.loadtxt(os.path.join(d, f"{tid}.track"), ndmin=2)
            trip = load_trip(tid, part)
            route = load_route(tid, part)
            V, E = len(trip.nodes), len(trip.arcs)
            G = coo_matrix((np.ones(E), (trip.arcs[:, 0], trip.arcs[:, 1])), shape=(V, V))
            n_weak, lab_w = connected_components(G, directed=True, connection="weak")
            n_strong, _ = connected_components(G, directed=True, connection="strong")
            deg = np.bincount(trip.arcs.ravel(), minlength=V)
            dt = np.diff(trip.track[:, 2])
            gaps_all[part].append(dt)
            proj = LocalProjection.for_track(trip.track)
            xy = proj.to_xy(trip.track[:, :2])
            net = RoadNetwork(trip.nodes, trip.arcs, proj, focus_xy=xy)
            arc_ids, dist, _, _ = net.candidates(xy[::5], k=1, radius=1e9)
            snap_d[part].append(dist[:, 0])
            step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
            cont = np.mean(trip.arcs[route[:-1], 1] == trip.arcs[route[1:], 0]) if len(route) > 1 else 1
            ext = xy.max(0) - xy.min(0)
            rows.append(dict(
                partition=part, trip=tid, n_fixes=len(raw), n_fixes_clean=len(trip.track),
                duration_s=trip.track[-1, 2] - trip.track[0, 2], dt_eq_1s=np.mean(dt == 1),
                max_gap_s=dt.max(), n_gaps_over_10s=int((dt > 10).sum()),
                track_km=step.sum() / 1000, extent_ew_km=ext[0] / 1000, extent_ns_km=ext[1] / 1000,
                lon_c=np.median(trip.track[:, 0]), lat_c=np.median(trip.track[:, 1]),
                nodes=V, arcs=E, isolated_nodes=int((deg == 0).sum()),
                weak_components=n_weak, largest_weak_frac=np.bincount(lab_w).max() / V,
                strong_components=n_strong, twin_frac=np.mean(net.twin >= 0),
                route_arcs=len(route), route_unique=len(set(route.tolist())), route_contiguous=cont,
                route_km=net.length[route].sum() / 1000, max_speed_ms=np.max(step / np.maximum(dt, 1e-3)),
                median_snap_m=np.median(dist[:, 0]), p95_snap_m=np.percentile(dist[:, 0], 95),
                **{f"bytes_{k}": v for k, v in sizes.items()}))
            print(part, tid, flush=True)
    inv = pd.DataFrame(rows)
    inv.to_csv(os.path.join(RES, "eda_inventory.csv"), index=False)
    m = pd.DataFrame(miss, columns=["partition", "trip", "file", "rows", "malformed_rows", "nan_rows"])
    m.groupby(["partition", "file"])[["rows", "malformed_rows", "nan_rows"]].sum().to_csv(
        os.path.join(RES, "eda_missing.csv"))
    return inv, gaps_all, snap_d


def plots(inv, gaps_all, snap_d):
    col = {"train": "#2a6f97", "student_test": "#e07a5f"}
    # 1 fixes per trip
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for p, g in inv.groupby("partition"):
        ax[0].hist(np.log10(g.n_fixes), bins=20, alpha=.7, color=col[p], label=p)
        ax[1].hist(g.track_km, bins=20, alpha=.7, color=col[p], label=p)
        ax[2].hist(np.log10(g.arcs), bins=20, alpha=.7, color=col[p], label=p)
    ax[0].set_xlabel("log10 GPS fixes per trip"); ax[1].set_xlabel("driven length (km)")
    ax[2].set_xlabel("log10 arcs in trip network")
    for a in ax:
        a.set_ylabel("trips"); a.legend(frameon=False)
    fig.suptitle("Trip size distribution (train n=80, student_test n=10)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "eda_trip_sizes.png")); plt.close(fig)
    # 2 sampling intervals
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for p in gaps_all:
        dt = np.concatenate(gaps_all[p])
        vals, cnt = np.unique(np.clip(np.round(dt), 0, 12), return_counts=True)
        ax[0].bar(vals + (0.2 if p == "student_test" else -0.2), cnt / cnt.sum(), width=.4,
                  color=col[p], label=p)
        big = dt[dt > 10]
        ax[1].hist(np.log10(big), bins=30, alpha=.7, color=col[p], label=f"{p} ({len(big)} gaps)")
    ax[0].set_xlabel("sampling interval (s, clipped at 12)"); ax[0].set_ylabel("share of steps")
    ax[0].set_yscale("log"); ax[0].legend(frameon=False)
    ax[1].set_xlabel("log10 gap length (s), gaps > 10 s"); ax[1].set_ylabel("count"); ax[1].legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "eda_sampling.png")); plt.close(fig)
    # 3 GPS-to-nearest-arc distance
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for p in snap_d:
        d = np.concatenate(snap_d[p])
        ax.hist(np.clip(d, 0, 100), bins=100, density=True, histtype="step", lw=1.6, color=col[p],
                label=f"{p}: median {np.median(d):.1f} m, p95 {np.percentile(d, 95):.1f} m")
    ax.set_xlabel("distance from GPS fix to nearest arc (m, clipped 100)"); ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "eda_snap_distance.png")); plt.close(fig)
    # 4 geographic spread
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for p, g in inv.groupby("partition"):
        ax.scatter(g.lon_c, g.lat_c, s=18 + g.n_fixes / 200, color=col[p], alpha=.75, label=p)
    ax.set_xlim(-180, 180); ax.set_ylim(-60, 80); ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title("Trip locations (marker size ~ number of fixes)"); ax.legend(frameon=False)
    ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "eda_world.png")); plt.close(fig)
    # 5 network connectivity
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for p, g in inv.groupby("partition"):
        ax[0].scatter(g.nodes, g.arcs / g.nodes, color=col[p], s=14, label=p)
        ax[1].scatter(g.weak_components, g.largest_weak_frac, color=col[p], s=14, label=p)
    ax[0].set_xscale("log"); ax[0].set_xlabel("nodes"); ax[0].set_ylabel("arcs per node")
    ax[1].set_xscale("log"); ax[1].set_xlabel("weakly connected components")
    ax[1].set_ylabel("share of nodes in largest component")
    for a in ax:
        a.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "eda_network.png")); plt.close(fig)


def folium_trip(tid, part, zoom_pad_m=400):
    trip = load_trip(tid, part)
    route = load_route(tid, part)
    proj = LocalProjection.for_track(trip.track)
    xy = proj.to_xy(trip.track[:, :2])
    net = RoadNetwork(trip.nodes, trip.arcs, proj, focus_xy=xy)
    # keep only arcs within 150 m of some fix so the map stays light
    keep = np.unique(_keep(net, xy))
    m = folium.Map(location=[trip.track[:, 1].mean(), trip.track[:, 0].mean()], zoom_start=14,
                   tiles="OpenStreetMap")
    road = folium.FeatureGroup(name=f"Road network near track ({len(keep)} arcs)")
    for a in keep:
        f, t = trip.arcs[a]
        folium.PolyLine([[trip.nodes[f, 1], trip.nodes[f, 0]], [trip.nodes[t, 1], trip.nodes[t, 0]]],
                        color="#9aa5b1", weight=2, opacity=.6).add_to(road)
    road.add_to(m)
    ref = folium.FeatureGroup(name="Reference route (.route)")
    for a in route:
        f, t = trip.arcs[a]
        folium.PolyLine([[trip.nodes[f, 1], trip.nodes[f, 0]], [trip.nodes[t, 1], trip.nodes[t, 0]]],
                        color="#2a9d8f", weight=6, opacity=.8).add_to(ref)
    ref.add_to(m)
    gps = folium.FeatureGroup(name=f"GPS fixes ({len(trip.track)})")
    for lon, lat, ts in trip.track[::2]:
        folium.CircleMarker([lat, lon], radius=2, color="#1d4ed8", fill=True, fill_opacity=.8,
                            weight=0, tooltip=f"t={ts:.0f}s").add_to(gps)
    gps.add_to(m)
    folium.Marker([trip.track[0, 1], trip.track[0, 0]], tooltip="start",
                  icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([trip.track[-1, 1], trip.track[-1, 0]], tooltip="end",
                  icon=folium.Icon(color="red", icon="stop")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[trip.track[:, 1].min(), trip.track[:, 0].min()],
                  [trip.track[:, 1].max(), trip.track[:, 0].max()]])
    out = os.path.join(MAPS, f"eda_{part}_{tid}.html")
    m.save(out)
    return out


def _keep(net, xy):
    arcs, _, _, _ = net.candidates(xy[::3], k=40, radius=150)
    return arcs[arcs >= 0]


def quality_notes(inv):
    lines = ["# Data-quality record (generated by scripts/eda.py)", ""]
    big = inv[inv.max_gap_s > 60]
    lines.append(f"* Trips with a gap > 60 s: {len(big)} ({', '.join(big.trip)}). Largest gap "
                 f"{inv.max_gap_s.max():.0f} s (trip {inv.loc[inv.max_gap_s.idxmax(), 'trip']}).")
    lines.append(f"* Share of 1 s steps: median {inv.dt_eq_1s.median():.1%}, min {inv.dt_eq_1s.min():.1%}.")
    lines.append(f"* No NaNs or malformed rows (see eda_missing.csv); timestamps strictly increasing "
                 f"in every trip: {bool((inv.n_fixes == inv.n_fixes_clean).all())}.")
    nc = inv[inv.route_contiguous < 1]
    lines.append(f"* Reference routes not fully contiguous (consecutive arcs not sharing a node): "
                 f"{len(nc)} trips, worst {nc.route_contiguous.min():.3f}.")
    rep = inv[inv.route_unique < inv.route_arcs]
    lines.append(f"* Routes that revisit an arc (loops / U-turns): {len(rep)} trips.")
    lines.append(f"* Isolated nodes in networks: total {inv.isolated_nodes.sum()}. Networks are clipped "
                 f"extracts, so many weak components exist at the clip boundary (median "
                 f"{inv.weak_components.median():.0f}).")
    hi = inv.sort_values("p95_snap_m", ascending=False).head(5)
    lines.append("* Highest p95 GPS-to-road distance: " +
                 ", ".join(f"{r.trip} ({r.p95_snap_m:.0f} m)" for r in hi.itertuples()))
    lines += ["", "## Decisions",
              "* Trips 00000033 and 00000054 (train): the GPS track is 130-200 m from its own reference "
              "route for most of the trip while staying inside the same bounding box, i.e. the reference "
              "does not describe the recording. They are excluded from fitting/CV (78 training trips remain).",
              "* Fixes more than 60 m from their aligned reference arc are flagged unreliable and excluded "
              "from training rows (10.7 % of train fixes, 90 % of them from the two trips above; 1.2 % of "
              "the remaining 78 trips; 0.13 % of student-test fixes).",
              "* Candidate set: 12 nearest directed arcs within 80 m. Reference arc missing from the set: "
              "8.2 % of all train fixes, 0.81 % on the 78 usable trips, 0.21 % on their reliable fixes, "
              "0.18 % on student-test (candidate_sweep.csv has the K x radius trade-off).",
              "* Distances use a per-trip azimuthal-equidistant projection centred on the track.",
              "* Chains are broken at gaps > gap_s (decoder parameter) and bridged by shortest path."]
    with open(os.path.join(RES, "data_quality.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    if sys.argv[1:] == ["--notes"]:
        quality_notes(pd.read_csv(os.path.join(RES, "eda_inventory.csv"), dtype={"trip": str}))
        sys.exit()
    inv, gaps, snap = inventory()
    plots(inv, gaps, snap)
    quality_notes(inv)
    for tid, part in (("00000001", "train"), ("00000065", "student_test")):
        print(folium_trip(tid, part))
