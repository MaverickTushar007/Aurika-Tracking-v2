# Aurika Person Tracking Pipeline (v2)

This is a clean, modular person tracking pipeline designed to achieve high ID consistency on surveillance video (e.g., `Dark_lighting.mp4`) using YOLOv8 (`yolo_staff_customer.pt`) and ByteTrack.

## Project Structure

```
Aurika-Tracking-v2/
├── configs/
│   └── config.yaml             # Single-point tracker configuration file
├── tracker/
│   ├── config_loader.py        # Dynamic local vs Kaggle paths mapping & config parser
│   ├── model_loader.py         # YOLO detection loader
│   ├── video_loader.py         # OpenCV Video Capture wrapper
│   ├── tracking_engine.py      # ByteTrack wrapper utilizing Ultralytics components
│   └── visualization.py        # Frame drawing engine with premium aesthetics
├── runs/                       # Output destination (generated dynamically)
├── requirements.txt            # Python package dependencies
├── README.md                   # Setup and usage guide
└── run.py                      # Orchestrator running prediction + tracking frame-by-frame
```

## Features

- **Dynamic Environment Detection:** The code auto-detects if it is running on macOS/Linux locally or as a Kaggle GPU workstation, loading assets and saving results to the correct locations automatically.
- **Tracker Parameter Controls:** All parameters (thresholds, buffer frames, match ratios, etc.) are modifiable from `configs/config.yaml` without editing tracking code.
- **Harmonious Visual Aesthetics:** Bounding boxes are styled using thin frames with heavy corner brackets and semi-transparent colored overlays per ID for high visual appeal.

## Local Setup & Run

1. Clone or copy this repository to `~/Desktop/Aurika-Tracking-v2` (or your preferred workspace).
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the tracking pipeline:
   ```bash
   python run.py
   ```
4. Verify the visual output in `runs/output.mp4`.

## Kaggle Deployment

This pipeline is optimized to run without modification on Kaggle:

1. Create a private Kaggle Dataset named `aurika-assets` containing `Dark_lighting.mp4` and `yolo_staff_customer.pt`.
2. Open a Kaggle Notebook, enable GPU acceleration, and attach the `aurika-assets` dataset.
3. Clone/upload the codebase into `/kaggle/working/`.
4. Install dependencies and run:
   ```bash
   pip install -r requirements.txt
   python run.py
   ```
5. The processed video will be saved directly to `/kaggle/working/runs/output.mp4`.
