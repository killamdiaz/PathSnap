"""PathSnap inference API (Laptop 1).

Run:  PATHSNAP_EVAL_DIR=/path/to/evaluator_test  uvicorn api.main:app --host 0.0.0.0 --port 8000

Endpoints
  GET  /health            model + decoder loaded, which trip folder is being served
  GET  /models            the models compared in development (CV + student-test scores) and the deployed one
  GET  /trips             the evaluation trips available (ids + size only; no reference data)
  POST /predict {trip_id} full pipeline on that trip; returns GPS, predicted route geometry and the
                          accuracy computed server-side against the trip's .route file

The reference route is read only inside `server_side_evaluation`, after prediction has finished, and
only scalar metrics leave the server.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import re
import sys
import time

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap.data import ROOT, Trip, clean_track, list_trips  # noqa: E402
from pathsnap.decoder import DecoderParams  # noqa: E402
from pathsnap.labels import align_route  # noqa: E402
from pathsnap.pipeline import ARTIFACT_DIR, load_artifact, predict_trip  # noqa: E402
from pathsnap.route import route_scores  # noqa: E402

EVAL_DIR = os.environ.get("PATHSNAP_EVAL_DIR", os.path.join(ROOT, "data", "evaluator_test"))
if not os.path.isdir(EVAL_DIR):
    EVAL_DIR = os.path.join(ROOT, "data", "student_test")

log = logging.getLogger("pathsnap")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S")

MODEL, CFG = load_artifact()
DEC = DecoderParams(**json.load(open(os.path.join(ARTIFACT_DIR, "decoder.json")))["decoder"])
log.info("model %s loaded | decoder %s | serving trips from %s", CFG["model"], DEC.as_dict(), EVAL_DIR)

app = FastAPI(title="PathSnap API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictRequest(BaseModel):
    trip_id: str


def _trip_files(trip_id: str) -> str:
    if not re.fullmatch(r"[0-9A-Za-z_-]{1,40}", trip_id) or trip_id not in _available():
        raise HTTPException(404, f"unknown trip {trip_id}")
    return os.path.join(EVAL_DIR, trip_id)


def _available():
    return [t for t in os.listdir(EVAL_DIR) if os.path.isfile(os.path.join(EVAL_DIR, t, f"{t}.track"))]


def _load_inputs(trip_id: str) -> Trip:
    d = _trip_files(trip_id)
    track = clean_track(np.loadtxt(os.path.join(d, f"{trip_id}.track"), ndmin=2))
    nodes = np.loadtxt(os.path.join(d, f"{trip_id}.nodes"), ndmin=2)
    arcs = np.loadtxt(os.path.join(d, f"{trip_id}.arcs"), dtype=np.int64, ndmin=2)
    return Trip(trip_id, track, nodes, arcs)


def server_side_evaluation(trip_id: str, out: dict) -> dict:
    """Reads the reference route AFTER prediction; returns scalar metrics only."""
    p = os.path.join(_trip_files(trip_id), f"{trip_id}.route")
    if not os.path.isfile(p):
        return {"available": False}
    ref_route = np.loadtxt(p, dtype=np.int64, ndmin=1)
    tc = out["tc"]
    ref, _, unreliable = align_route(tc.xy, tc.motion["head"], tc.motion["head_valid"], ref_route, tc.net)
    rs = route_scores(out["route"], ref_route, tc.net)
    return {"available": True,
            "hmm_point_accuracy": float((out["hmm_arcs"] == ref).mean()),
            "classifier_point_accuracy": float((out["clf_arcs"] == ref).mean()),
            "route_precision": rs["route_precision"], "route_recall": rs["route_recall"],
            "route_f1": rs["route_f1"], "route_mismatch_fraction": rs["rmf"],
            "predicted_km": rs["pred_len_km"], "reference_km": rs["ref_len_km"],
            "unreliable_reference_fixes": int(unreliable.sum())}


@app.get("/health")
def health():
    return {"status": "ok", "model": CFG["model"], "decoder": DEC.as_dict(), "trips_dir": EVAL_DIR,
            "n_trips": len(_available())}


def _read_csv(name: str) -> list[dict]:
    p = os.path.join(ROOT, "results", name)
    if not os.path.isfile(p):
        return []
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


@app.get("/models")
def models():
    """The models compared during development (training/CV + student-test results from results/),
    and the deployed pipeline. Nothing here is computed from the evaluator trips."""
    hmm = {r["model"]: r for r in _read_csv("hmm_comparison.csv")}
    rows = []
    for r in _read_csv("model_comparison.csv"):
        f = lambda k, src=r: round(float(src[k]), 4) if src.get(k) not in (None, "") else None  # noqa: E731
        h = hmm.get(r["model"], {})
        rows.append({"model": r["model"], "family": r["family"], "config": r["config"],
                     "cv_top1": f("cv_top1"), "cv_macro_f1": f("cv_macro_f1"), "cv_log_loss": f("cv_log_loss"),
                     "st_top1": f("st_top1"), "st_hmm_top1": f("hmm_acc", h) if h else None,
                     "infer_ms_per_fix": f("infer_ms_per_fix")})
    rows.sort(key=lambda r: -(r["cv_top1"] or 0))
    deployed = CFG.get("results_key", "LambdaRank").replace("_untuned", "")
    return {"deployed": {"model": CFG["model"], "name": deployed, "decoder": DEC.as_dict(),
                         "selection": CFG.get("selection_note", "highest mean 5-fold GroupKFold top-1 accuracy")},
            "compared": rows}


@app.get("/trips")
def trips():
    out = []
    for t in sorted(_available()):
        tr = np.loadtxt(os.path.join(EVAL_DIR, t, f"{t}.track"), ndmin=2)
        out.append({"trip_id": t, "n_fixes": int(len(tr)), "duration_s": float(tr[-1, 2] - tr[0, 2])})
    log.info("GET /trips -> %d trips", len(out))
    return {"trips": out}


@app.post("/predict")
def predict(req: PredictRequest):
    t_all = time.perf_counter()
    log.info("POST /predict trip=%s received", req.trip_id)
    trip = _load_inputs(req.trip_id)
    out = predict_trip(trip, MODEL, DEC)
    metrics = server_side_evaluation(req.trip_id, out)
    tc = out["tc"]
    step = max(1, len(trip.track) // 4000)
    gps = trip.track[::step, [1, 0]].round(6).tolist()
    route_geo = [[[float(trip.nodes[f, 1]), float(trip.nodes[f, 0])], [float(trip.nodes[t, 1]), float(trip.nodes[t, 0])]]
                 for f, t in trip.arcs[out["route"]]]
    elapsed = time.perf_counter() - t_all
    resp = {
        "trip_id": req.trip_id, "n_fixes": int(len(trip.track)), "gps": gps, "gps_step": step,
        "route": route_geo, "route_arcs": [int(a) for a in out["route"]],
        "matched_arcs": [int(a) for a in out["hmm_arcs"][::step]],
        "classifier_route_continuity": out["clf_continuity"], "hmm_route_continuity": out["hmm_continuity"],
        "chain_breaks": len(out["breaks"]) - 1, "unbridged_jumps": out["jumps"],
        "candidates_per_fix": float((tc.arc >= 0).sum(1).mean()),
        "timings_s": {**{k: round(v, 3) for k, v in out["timings"].items()}, "total": round(elapsed, 3)},
        "evaluation": metrics, "model": CFG["model"],
    }
    if metrics.get("available"):
        log.info("POST /predict trip=%s fixes=%d | HMM acc %.4f (classifier-only %.4f) | route F1 %.4f "
                 "| continuity %.3f -> %.3f | %.2fs", req.trip_id, len(trip.track),
                 metrics["hmm_point_accuracy"], metrics["classifier_point_accuracy"], metrics["route_f1"],
                 out["clf_continuity"], out["hmm_continuity"], elapsed)
    else:
        log.info("POST /predict trip=%s fixes=%d | no reference available | %.2fs", req.trip_id,
                 len(trip.track), elapsed)
    return resp
