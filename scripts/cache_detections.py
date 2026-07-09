#!/usr/bin/env python3
"""
Aurika Tracking v2 — Detection Cache Generator
===============================================
Runs YOLO11 detector once on the video, saving all frame-by-frame detections
to a serialized cache file to accelerate future tracker experiments.

Usage:
    python scripts/cache_detections.py --model yolo11l --video videos/Dark_lighting.mp4
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import torch
import ultralytics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("CacheGenerator")

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT))

from model_resolver import ModelResolver
from tracker.model_loader import load_yolo_model
from tracker.device import get_device
from tracker.detection_cache import calculate_video_hash, save_detection_cache

# Load environment configs
resolver = ModelResolver(project_root=PROJECT_ROOT)
DEVICE = get_device()

def main() -> None:
    parser = argparse.ArgumentParser(description="Aurika YOLO Detection Cache Generator")
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11l",
        choices=["yolo11m", "yolo11l", "yolo11x"],
        help="YOLO11 model name to run"
    )
    parser.add_argument(
        "--video",
        type=str,
        default="videos/Dark_lighting.mp4",
        help="Path to the input video file"
    )
    parser.add_argument(
        "--conf-thresh",
        type=float,
        default=0.25,
        help="YOLO detection confidence threshold floor"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/cache",
        help="Directory to save the cache files"
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.is_absolute():
        # Match config loader resolution
        if resolver.is_kaggle:
            video_path = Path("/kaggle/input/datasets/tusharmarscitizen/video-analysis") / video_path.name
        else:
            video_path = PROJECT_ROOT / video_path

    if not video_path.exists():
        log.error(f"Video not found: {video_path}")
        raise SystemExit(1)

    log.info("╔══════════════════════════════════════════╗")
    log.info("║ Aurika Tracking v2 — Cache Generator     ║")
    log.info("╚══════════════════════════════════════════╝")
    log.info(f"Model          : {args.model}")
    log.info(f"Video          : {video_path} ({video_path.stat().st_size / 1e6:.1f} MB)")
    log.info(f"Conf Threshold : {args.conf_thresh}")
    log.info(f"Device         : {DEVICE.upper()}\n")

    # 1. Load YOLO model
    cfg_path = resolver.resolve(args.model)
    model = load_yolo_model(str(cfg_path))
    model.to(DEVICE)

    # 2. Open Video
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    log.info(f"Video specs: {W}x{H} @ {fps:.2f} FPS | Total frames: {total_frames}")

    # 3. Calculate video content hash
    log.info("Calculating video content hash...")
    video_hash = calculate_video_hash(video_path)
    log.info(f"Video Hash: {video_hash}")

    # 4. Inference Loop
    detections = []
    frame_idx = 0
    t_start = time.time()

    log.info("Running YOLO inference across all frames. Please wait...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Predict frame detections
        results = model.predict(frame, conf=args.conf_thresh, device=DEVICE, verbose=False)[0]
        
        # Extract boxes values to CPU numpy arrays for pickling
        boxes = results.boxes
        if len(boxes) > 0:
            xyxy_np = boxes.xyxy.cpu().numpy()
            conf_np = boxes.conf.cpu().numpy()
            cls_np = boxes.cls.cpu().numpy()
        else:
            xyxy_np = np.empty((0, 4), dtype=np.float32)
            conf_np = np.empty((0,), dtype=np.float32)
            cls_np = np.empty((0,), dtype=np.float32)

        detections.append({
            "frame_index": frame_idx,
            "timestamp": frame_idx / fps if fps > 0 else 0.0,
            "boxes": xyxy_np,
            "confidence": conf_np,
            "class_id": cls_np,
        })

        if frame_idx % 500 == 0 or frame_idx == total_frames:
            pct = (frame_idx / total_frames) * 100 if total_frames > 0 else 0.0
            log.info(f"  Processed frame {frame_idx}/{total_frames} ({pct:.1f}%)")

    cap.release()
    duration = time.time() - t_start
    log.info(f"Inference complete in {duration:.2f} seconds ({frame_idx / duration:.1f} FPS average).")

    # 5. Compile metadata & Save
    metadata = {
        "video_hash": video_hash,
        "video_filename": video_path.name,
        "yolo_version": ultralytics.__version__,
        "model_name": args.model,
        "confidence_threshold": args.conf_thresh,
        "image_size": [W, H],
        "total_frames": frame_idx,
        "total_detections": sum(len(d["boxes"]) for d in detections),
        "inference_duration_seconds": duration,
        "average_fps": frame_idx / duration if duration > 0 else 0.0,
    }

    out_dir = PROJECT_ROOT / args.output_dir
    cache_path = out_dir / "detections.pkl"
    
    save_detection_cache(cache_path, metadata, detections)

    # Save summary.md
    summary_md = [
        "# YOLO Detection Cache Summary",
        "",
        f"- **Model:** {args.model}",
        f"- **Video:** `{video_path.name}`",
        f"- **File Hash:** `{video_hash}`",
        f"- **Resolution:** {W}x{H}",
        f"- **Confidence Threshold:** {args.conf_thresh}",
        f"- **Total Frames Processed:** {frame_idx}",
        f"- **Total Detections Logged:** {metadata['total_detections']}",
        f"- **Inference Speed:** {metadata['average_fps']:.1f} FPS",
        f"- **Inference Duration:** {duration:.2f} s",
        f"- **Timestamp:** {metadata['date']}",
    ]
    with open(out_dir / "summary.md", "w") as fh:
        fh.write("\n".join(summary_md))

    log.info(f"Cache generated successfully in {out_dir}/")

if __name__ == "__main__":
    main()
