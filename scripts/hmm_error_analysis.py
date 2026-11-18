"""Where does HMM decoding help or hurt? Deployed artifact on the 10 student-test trips.
Output results/hmm_error_analysis.json"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap.data import ROOT, list_trips, load_route, load_trip  # noqa: E402
from pathsnap.decoder import DecoderParams  # noqa: E402
from pathsnap.labels import align_route  # noqa: E402
from pathsnap.pipeline import ARTIFACT_DIR, load_artifact, predict_trip  # noqa: E402

model, _ = load_artifact()
dec = DecoderParams(**json.load(open(os.path.join(ARTIFACT_DIR, "decoder.json")))["decoder"])
tot = dict(fixes=0, hmm_fixed=0, hmm_broke=0, both_wrong=0, candidate_miss=0,
           wrong_direction_twin_clf=0, wrong_direction_twin_hmm=0, hmm_err_adjacent_arc=0)
for tid in list_trips("student_test"):
    trip = load_trip(tid, "student_test")
    o = predict_trip(trip, model, dec)
    tc, net = o["tc"], o["tc"].net
    ref, _, _ = align_route(tc.xy, tc.motion["head"], tc.motion["head_valid"], load_route(tid, "student_test"), net)
    c, h = o["clf_arcs"] == ref, o["hmm_arcs"] == ref
    tot["fixes"] += len(ref)
    tot["hmm_fixed"] += int((~c & h).sum())
    tot["hmm_broke"] += int((c & ~h).sum())
    tot["both_wrong"] += int((~c & ~h).sum())
    tot["candidate_miss"] += int((~(tc.arc == ref[:, None]).any(1)).sum())
    tot["wrong_direction_twin_clf"] += int((~c & (net.twin[o["clf_arcs"]] == ref)).sum())
    tot["wrong_direction_twin_hmm"] += int((~h & (net.twin[o["hmm_arcs"]] == ref)).sum())
    ha = o["hmm_arcs"]
    adj = (net.arcs[ha, 1] == net.arcs[ref, 0]) | (net.arcs[ref, 1] == net.arcs[ha, 0])
    tot["hmm_err_adjacent_arc"] += int((~h & adj).sum())
json.dump(tot, open(os.path.join(ROOT, "results", "hmm_error_analysis.json"), "w"), indent=2)
print(tot)
