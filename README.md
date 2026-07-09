# Aurika Person Tracking Pipeline (v2)

A clean, modular person tracking pipeline delivering high ID consistency on surveillance video using **YOLO11l** and **ByteTrack**.

**Production detector:** YOLO11l (COCO pretrained) — selected after benchmarking YOLO11m, YOLO11l, and YOLO11x.  
**Production tracker:** ByteTrack (via Ultralytics)

---

## Project Structure

```
Aurika-Tracking-v2/
├── configs/
│   └── config.yaml             # Single-point tracker configuration (paths + ByteTrack params)
├── scripts/
│   ├── benchmark.py            # Two-stage detector benchmark (supports --use-cache)
│   ├── model_resolver.py       # Environment-agnostic model path resolver (local / Kaggle)
│   ├── optimize_tracker.py     # Hyperparameter optimization sweep tool
│   ├── cache_detections.py     # YOLO inference cache generator (Stage A)
│   └── run_tracker_cached.py   # Cached detections tracker runner (Stage B)
├── tracker/
│   ├── config_loader.py        # Dynamic local vs Kaggle path mapping & config parser
│   ├── model_loader.py         # YOLO model loader (local file or Ultralytics hub)
│   ├── video_loader.py         # OpenCV VideoCapture wrapper with termination diagnostics
│   ├── tracking_engine.py      # ByteTrack wrapper (Ultralytics BYTETracker)
│   ├── tracker_factory.py      # Factory abstraction to instantiate ByteTrack/BoT-SORT
│   ├── device.py               # Reusable priority-based device selection (CUDA -> MPS -> CPU)
│   ├── detection_cache.py      # Detection cache serializer & metadata validator
│   └── visualization.py        # Frame drawing engine with premium aesthetics
├── runs/                       # Output destination (generated dynamically)
├── requirements.txt            # Python package dependencies
├── README.md                   # This file
└── run.py                      # Pipeline orchestrator: detection + tracking, frame-by-frame
```

---

## Detection Caching Infrastructure (Experiment 003)

To accelerate tracker experiments, the pipeline supports a decoupled **Detection Cache Subsystem** that divides tracking into two stages:

### Stage A: One-time Detector Inference
Generate a serialized cache of YOLO detections (storing bounding boxes, confidence, class IDs) and metadata (video hash, model version):
```bash
python scripts/cache_detections.py --model yolo11l --video videos/Dark_lighting.mp4
```
This produces `runs/cache/detections.pkl` and `runs/cache/metadata.json`.

### Stage B: Cached Tracker Runner (Zero GPU/Detector Overhead)
Evaluate different trackers, parameters, or sweeps using the pre-calculated detections in **2 minutes** instead of 15 minutes (a **6-7x speedup**):
```bash
python scripts/run_tracker_cached.py --tracker bytetrack --cache runs/cache/detections.pkl --sample-every 3
```

### Cached Benchmarking
Execute the benchmark suite using cached detections:
```bash
python scripts/benchmark.py --mode tracker --model yolo11l --tracker bytetrack --use-cache
```

The cached runs produce **100% identical metrics, tracks, and outputs** (within floating-point precision) to their non-cached counterparts, verified by automatic metadata hash checking.

---

## Restaurant Intelligence Layer (Experiment 004)

The pipeline integrates a Restaurant Intelligence Layer to extract spatial-temporal analytics from tracked people:

- **Zone Monitoring:** Tracks live occupancy and dwell times across configurable polygon zones (e.g. Dining, Reception, Waiting, Entrance, Kitchen) loaded from `configs/zones.yaml`.
- **Entries & Exits:** Employs virtual counting lines to log client traffic flows (only counting each ID once).
- **Zone Transitions:** Tracks customer paths and transition statistics between zones.
- **Trajectory Heatmap:** Smooths and visualizes residence hotspots, outputting `heatmap.png`.
- **Structured CSV Logs:** Generates granular exports for events, occupancy, zone stats, and track lifetimes.

### Running Analytics

Run the analytics pipeline using pre-cached detections:
```bash
python scripts/run_analytics.py --use-cache --cache runs/cache/detections.pkl
```

This generates the following outputs in `runs/analytics/`:
- `events.csv`: Granular event log (e.g. entering/exiting zones, line crossings).
- `occupancy.csv`: Live occupancy counts per zone over time.
- `zone_statistics.csv`: Vistor transition counts and average zone dwell times.
- `dwell_times.csv`: Overall visitor dwell durations.
- `heatmap.png`: Trajectory hotspot visualization.
- `analytics_output.mp4`: Annotated video with zone boundaries, live occupants, entry/exit counters, and dwell timers.
- `analytics_report.md`: Executive summary with store recommendations.

---

## Interactive Scene Calibration (Experiment 005)

To replace manual coordinate entry, the pipeline includes a mouse-driven **Interactive Calibration GUI** to draw polygons and counters on a static background reference frame extracted from the camera feed.

### Calibrating layout profiles

Run the calibrator on a video to draw zones and counting lines:
```bash
python scripts/calibrate_scene.py --video videos/Dark_lighting.mp4 --output configs/restaurant_default.yaml
```

**GUI Instructions:**
- Press `n` on keyboad ➔ Type zone name in console ➔ Left-click vertices on the image.
- Press `l` on keyboard ➔ Type counting line name in console ➔ Click start point, then end point.
- Press `c` ➔ Save/close the active polygon zone or counting line.
- Press `d` ➔ Delete the last created zone/line.
- Press `s` ➔ Save coordinates and exit.
- Press `q` or `ESC` ➔ Exit without saving.

This automatically saves the configuration values under `configs/restaurant_default.yaml` and reference frame files inside `runs/calibration/`:
- `background_snapshot.png`: Clean background camera snapshot frame.
- `preview.png` & `scene_layout.png`: Layout blueprints.
- `calibration_report.md`: Coordinates specifications summary report.

### Loading Layout Profiles

You can calibrate and store multiple camera layouts (e.g. `configs/restaurant_A.yaml`, `configs/restaurant_B.yaml`) and pass them to the analytics runner using the `--layout` argument:
```bash
python scripts/run_analytics.py --use-cache --cache runs/cache/detections.pkl --layout restaurant_A
```

---



## Features

- **YOLO11l detector** — benchmarked winner across recall, track recovery, and inference speed on dark surveillance footage.
- **ByteTrack** — low-miss-rate multi-object tracker with configurable buffer and match thresholds.
- **Dynamic environment detection** — auto-detects local macOS/Linux vs Kaggle GPU kernel; loads assets and saves results to the correct locations with no code changes.
- **Configurable tracker parameters** — all ByteTrack thresholds, buffers, and match ratios are controlled from `configs/config.yaml`.
- **Termination diagnostics** — if the video ends early (e.g., H.264 corruption), the pipeline logs the last successfully processed frame and the explicit termination reason instead of silently exiting.
- **Premium visualisation** — per-ID color-coded bounding boxes with corner brackets, confidence scores, and track lifetime counters.

---

## Local Setup & Run

1. Clone or copy this repository to your workspace.
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place the input video at `videos/Dark_lighting.mp4`.  
   YOLO11l weights are downloaded automatically by Ultralytics on first run.
4. Run the tracking pipeline:
   ```bash
   python run.py
   ```
5. Output is written to `runs/output.mp4`.

---

## Kaggle Deployment

This pipeline runs on Kaggle without modification:

1. Create a private Kaggle Dataset containing `Dark_lighting.mp4` and attach it to your notebook.
2. Clone or upload the codebase into `/kaggle/working/`.
3. Enable GPU acceleration in the notebook settings.
4. Install dependencies and run:
   ```bash
   pip install -r requirements.txt
   python run.py
   ```
5. The processed video is saved to `/kaggle/working/runs/output.mp4`.

> YOLO11l weights are fetched automatically from the Ultralytics hub on Kaggle — no manual upload required.

---

## Benchmark

To re-run the detector benchmark:

```bash
python scripts/benchmark.py
```

This evaluates YOLO11m, YOLO11l, and YOLO11x on your video using identical conditions, then runs ByteTrack on the top-2 detectors and produces:

```
runs/benchmark/
  yolo11m/   detection_output.mp4   detection_metrics.json
  yolo11l/   detection_output.mp4   detection_metrics.json
  yolo11x/   detection_output.mp4   detection_metrics.json
  {top2}/    tracking_output.mp4    tracking_metrics.json
  detection_comparison.csv
  tracking_comparison.csv
  benchmark_report.md
```

**Selection criteria** — the winner is chosen on tracking quality (track lifetime, recovery rate, ID stability), not raw detection count.

---

## Detector Selection History

| Model | Avg Active Tracks | Recovered | Median ms | Result |
|---|---|---|---|---|
| YOLO11m | — | — | 88.6 ms | Stage 2 only |
| **YOLO11l** | **10.19** | **514** | 111.5 ms | ✅ **Production winner** |
| YOLO11x | — | — | 225.4 ms | Eliminated (too slow) |

---

## Track Memory Layer (Experiment 008)

The pipeline integrates a **Track Memory Layer** that acts as the single source of truth for behavioral state tracking:

- **Persistent TrackState:** Continuously updates 33 lifecycle and motion properties (role, lifecycle status, zone exit/entry dwell times, velocities, trajectory vectors, visibility index, quality score).
- **Lifecycle State Machine:** Strictly routes states through a deterministic state machine: `NEW` ➔ `CONFIRMED` ➔ `ACTIVE` ➔ `TEMP_OCCLUDED` ➔ `RECOVERED` ➔ `EXITED` ➔ `ARCHIVED`.
- **Event Engine:** Records lifecycle and milestone events (such as `TrackCreated`, `ZoneEntered`, `WaitingStarted`, `DiningFinished`).
- **Timelines & Transition Matrix:** Automatically generates chronological customer flow matrices and timelines.

### Running Experiment 008

To run the persistent tracking memory simulation:
```bash
python scripts/run_experiment_008.py
```

This validates all track state lifecycle transitions and generates output reports under `runs/experiment008/`.

