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
│   ├── benchmark.py            # Two-stage detector benchmark (detection → top-2 → tracking)
│   └── model_resolver.py       # Environment-agnostic model path resolver (local / Kaggle)
├── tracker/
│   ├── config_loader.py        # Dynamic local vs Kaggle path mapping & config parser
│   ├── model_loader.py         # YOLO model loader (local file or Ultralytics hub)
│   ├── video_loader.py         # OpenCV VideoCapture wrapper with termination diagnostics
│   ├── tracking_engine.py      # ByteTrack wrapper (Ultralytics BYTETracker)
│   └── visualization.py        # Frame drawing engine with premium aesthetics
├── runs/                       # Output destination (generated dynamically)
├── requirements.txt            # Python package dependencies
├── README.md                   # This file
└── run.py                      # Pipeline orchestrator: detection + tracking, frame-by-frame
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
