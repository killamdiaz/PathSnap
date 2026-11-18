# PathSnap — GPS map-matching with a learned candidate scorer + HMM/Viterbi

UE24CS352A Machine Learning hackathon. Team: _<names / SRNs>_

PathSnap snaps noisy GPS fixes onto a directed road network. For every fix we collect candidate road
arcs, score each (fix, arc) pair with a trained model, then decode the most plausible drivable
sequence with a Hidden Markov Model + Viterbi. The output is a road arc per GPS fix plus a
continuous route. A FastAPI service runs the pipeline (Laptop 1), and a browser UI drives it (Laptop 2).

```
 .track .nodes .arcs ──► per-trip AEQD projection ──► candidates (12 arcs ≤ 80 m) ──► 20 features
                                                                                      │
             route + metrics ◄── shortest-path bridging ◄── Viterbi (HMM) ◄── P(arc | fix) model
```

## Repository layout

| Folder | Contents |
|---|---|
| `pathsnap/` | library: `data` (loading, cleaning), `geo` (projection, segment geometry), `motion` (distance-window heading, speed, turning), `network` (spatial index, candidates, bounded Dijkstra), `labels` (heading-aware monotone alignment of fixes to the reference route), `features`, `models` (model zoo), `evaluate` (metrics), `decoder` (HMM + Viterbi), `route` (assembly, continuity, route scores), `pipeline` (end-to-end inference), `viz` (Folium) |
| `scripts/` | `eda.py`, `build_dataset.py`, `train_models.py`, `bayes_opt.py`, `hmm_eval.py`, `plots.py` |
| `api/` | FastAPI backend (`main.py`) |
| `ui/` | evaluator UI (`index.html`, `serve.py`) |
| `models/final/` | deployed artifact: `model.joblib`, `config.json`, `decoder.json` |
| `models/zoo/` | the other trained models that are small enough to commit |
| `results/` | every CSV/JSON the experiments wrote (model comparison, CV folds, Optuna trials and study DB, HMM comparison, EDA inventory, data-quality notes) |
| `figures/` | every plot, all generated from `results/` |
| `maps/` | interactive Folium maps (EDA + every student-test trip) |

## Setup

Python 3.10+ (developed on 3.12).

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Put the raw data (not committed) like this:

```
data/train/<trip_id>/<trip_id>.{track,nodes,arcs,route}          # 80 trips
data/student_test/<trip_id>/<trip_id>.{track,nodes,arcs,route}   # 10 trips
data/evaluator_test/<trip_id>/...                                 # 10 trips, given at evaluation
```

(or point `PATHSNAP_DATA` at another folder with the same structure).

## Reproduce every result (≈ 1.5 h on an 8-core laptop)

```bash
python scripts/eda.py              # inventory, missing values, data-quality notes, EDA plots + Folium maps
python scripts/build_dataset.py    # candidates, features, reference labels  -> results/dataset/
python scripts/train_models.py     # 10 models, 5-fold GroupKFold + student-test -> results/model_comparison.csv
python scripts/bayes_opt.py        # Optuna TPE on the selected model -> models/final/model.joblib
python scripts/hmm_eval.py         # decoder tuning on OOF train probs + classifier vs HMM -> models/final/decoder.json
python scripts/plots.py            # all figures in figures/
```

Seeds are fixed (`pathsnap.models.SEED`). The evaluator-test partition is never read by any of these scripts.

## Deployment (two laptops, same Wi-Fi)

**Laptop 1: model + API**

```bash
source .venv/bin/activate
export PATHSNAP_EVAL_DIR=/absolute/path/to/evaluator_test     # folder holding the 10 evaluation trips
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Find its IP with `ipconfig getifaddr en0` (macOS), `hostname -I` (Linux) or `ipconfig` (Windows).
Each request is logged in this terminal, for example:

```
10:42:07  INFO  POST /predict trip=00000065 received
10:42:10  INFO  POST /predict trip=00000065 fixes=768 | HMM acc 0.9a | route F1 0.9b | continuity ... | 2.91s
```

If `PATHSNAP_EVAL_DIR` is not set and `data/evaluator_test` does not exist, the API serves `data/student_test`.

**Laptop 2: UI** (needs only Python's standard library and a browser with internet access for map tiles)

```bash
python ui/serve.py            # http://localhost:5500
```

Open `http://localhost:5500/?api=http://<laptop1-ip>:8000`. The page lists the trips the API serves.
Clicking one sends `POST /predict {"trip_id": ...}` and draws the GPS fixes and the predicted route,
with the accuracy that the API computed server-side.

API quick test from any machine:

```bash
curl http://<laptop1-ip>:8000/health
curl http://<laptop1-ip>:8000/trips
curl -X POST http://<laptop1-ip>:8000/predict -H 'Content-Type: application/json' -d '{"trip_id":"00000065"}'
```

**Firewall:** allow inbound TCP 8000 on Laptop 1 (macOS asks on first run; on Windows, allow Python
through Windows Defender Firewall).

## Leakage controls

* `load_trip()` never reads `.route`. Reference routes are read only by training/evaluation code
  and, in the API, by `server_side_evaluation()` *after* prediction. Only scalar metrics are returned.
* Train/student-test/evaluator partitions are separate trip folders. CV uses `GroupKFold` by trip.
  Bayesian search and decoder tuning use training trips only.
* There are no hard-coded trip ids anywhere in the prediction path. The two training trips excluded
  as bad references (`pathsnap/evaluate.py`) only affect which data are used for fitting.

## AI-assistance disclosure

_<state here, as required by the course policy, how AI tools were used>_
