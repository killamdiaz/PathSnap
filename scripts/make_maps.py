"""Folium maps of every student-test trip produced by the *deployed* artifact (models/final), exactly as
the API runs it. Reference route is drawn for the offline report only (the API never returns it).
Outputs maps/st_<trip>.html and results/deployed_student_test.csv"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap.data import ROOT, list_trips, load_route, load_trip  # noqa: E402
from pathsnap.decoder import DecoderParams  # noqa: E402
from pathsnap.labels import align_route  # noqa: E402
from pathsnap.pipeline import ARTIFACT_DIR, load_artifact, predict_trip  # noqa: E402
from pathsnap.route import route_scores  # noqa: E402
from pathsnap.viz import trip_map  # noqa: E402

model, cfg = load_artifact()
dec = DecoderParams(**json.load(open(os.path.join(ARTIFACT_DIR, "decoder.json")))["decoder"])
rows = []
for tid in list_trips("student_test"):
    trip = load_trip(tid, "student_test")
    out = predict_trip(trip, model, dec)
    route = load_route(tid, "student_test")
    tc = out["tc"]
    ref, _, _ = align_route(tc.xy, tc.motion["head"], tc.motion["head_valid"], route, tc.net)
    rs = route_scores(out["route"], route, tc.net)
    r = dict(trip=tid, fixes=len(ref), clf_acc=(out["clf_arcs"] == ref).mean(), hmm_acc=(out["hmm_arcs"] == ref).mean(),
             clf_cont=out["clf_continuity"], hmm_cont=out["hmm_continuity"], **rs, breaks=len(out["breaks"]) - 1,
             jumps=out["jumps"], **{k: round(v, 3) for k, v in out["timings"].items()})
    rows.append(r)
    trip_map(trip, tc, out["hmm_arcs"], out["route"], clf_arcs=out["clf_arcs"], ref_route=route,
             out=os.path.join(ROOT, "maps", f"st_{tid}.html"),
             title=f"Trip {tid} | HMM {r['hmm_acc']:.1%} vs classifier {r['clf_acc']:.1%} | route F1 {rs['route_f1']:.3f}")
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(ROOT, "results", "deployed_student_test.csv"), index=False)
w = df.fixes.to_numpy()
print("weighted clf acc %.4f  hmm acc %.4f  mean route F1 %.4f" % (np.average(df.clf_acc, weights=w),
                                                                  np.average(df.hmm_acc, weights=w), df.route_f1.mean()))
