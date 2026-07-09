#!/usr/bin/env python3
"""
Aurika Tracking v2 — Cached Tracker Runner
===========================================
Runs the tracker pipeline using pre-calculated YOLO detections from the cache.
Avoids loading YOLO and performing GPU/CPU detector inference entirely.

Usage:
    python scripts/run_tracker_cached.py --tracker bytetrack --cache runs/cache/detections.pkl
"""

import argparse
import csv
import gc
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import psutil
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("CachedRunner")

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT))

from model_resolver import ModelResolver
from tracker.config_loader import PipelineConfig
from tracker.device import get_device
from tracker.tracker_factory import create_tracker
from tracker.detection_cache import CachedBoxes, calculate_video_hash, load_detection_cache
from benchmark import annotate_tracking_frame, _filter_persons

# Load environment configs
resolver = ModelResolver(project_root=PROJECT_ROOT)
DEVICE = get_device()

def main() -> None:
    parser = argparse.ArgumentParser(description="Aurika Tracking Cache Runner")
    parser.add_argument(
        "--tracker",
        type=str,
        default="bytetrack",
        choices=["bytetrack", "botsort"],
        help="Tracker algorithm to run"
    )
    parser.add_argument(
        "--cache",
        type=str,
        default="runs/cache/detections.pkl",
        help="Path to the YOLO detection cache pickle file"
    )
    parser.add_argument(
        "--video",
        type=str,
        default="videos/Dark_lighting.mp4",
        help="Path to the input video file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/tracker_cached",
        help="Directory to save output files"
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=1,
        help="Frame sampling rate (1 = all frames, 3 = match benchmark sampling)"
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.is_absolute():
        if resolver.is_kaggle:
            video_path = Path("/kaggle/input/datasets/tusharmarscitizen/video-analysis") / video_path.name
        else:
            video_path = PROJECT_ROOT / video_path

    if not video_path.exists():
        log.error(f"Video not found: {video_path}")
        raise SystemExit(1)

    cache_path = Path(args.cache)
    if not cache_path.is_absolute():
        cache_path = PROJECT_ROOT / cache_path

    if not cache_path.exists():
        log.error(f"Cache file not found: {cache_path}")
        raise SystemExit(1)

    log.info("╔══════════════════════════════════════════╗")
    log.info("║ Aurika Tracking v2 — Cached Tracker      ║")
    log.info("╚══════════════════════════════════════════╝")
    log.info(f"Tracker      : {args.tracker.upper()}")
    log.info(f"Cache File   : {cache_path}")
    log.info(f"Video        : {video_path}")
    log.info(f"Sample Rate  : every {args.sample_every}rd frame")
    log.info(f"Device       : {DEVICE.upper()}\n")

    # 1. Load pipeline tracker config
    pipeline_cfg = PipelineConfig()
    tracker_params = pipeline_cfg.tracker

    # 2. Calculate hash and validate cache metadata
    log.info("Validating detection cache metadata...")
    video_hash = calculate_video_hash(video_path)
    expected_meta = {
        "video_hash": video_hash,
        "model_name": "yolo11l",  # Production model
    }
    
    detections_list = load_detection_cache(cache_path, expected_meta)
    if detections_list is None:
        log.error("Failed to load a valid cache. Rejecting run.")
        raise SystemExit(1)

    # 3. Initialize tracker via factory
    tracker = create_tracker(args.tracker, tracker_params, device=DEVICE)

    # 4. Open Video
    cap = cv2.VideoCapture(str(video_path))
    total_frm = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_fps = max(1.0, src_fps / args.sample_every)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_video = cv2.VideoWriter(
        str(out_dir / "tracking_output.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        out_fps, (W, H),
    )

    # State tracking
    track_first: Dict[int, int] = {}
    track_last: Dict[int, int] = {}
    track_frames: Dict[int, List[int]] = defaultdict(list)

    counts_per_frame: List[int] = []
    latencies: List[float] = []
    peak_ram_mb = 0.0
    # Warmup (skip first 5 frames to match benchmark.py behavior)
    warmed = 0
    while cap.isOpened() and warmed < 5:
        ret, _ = cap.read()
        if not ret:
            break
        warmed += 1
    total_read = warmed
    current_lifetimes: Dict[int, int] = {}

    t_start = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        total_read += 1
        
        # Apply sampling rate
        if total_read % args.sample_every != 0:
            continue
        sampled_n += 1

        t0 = time.perf_counter()
        
        # Get frame detections from cached dictionary
        if total_read - 1 < len(detections_list):
            det_dict = detections_list[total_read - 1]
        else:
            det_dict = {"boxes": np.empty((0, 4)), "confidence": np.empty((0,)), "class_id": np.empty((0,))}

        # Construct CachedBoxes and filter to COCO person (0)
        boxes = CachedBoxes(det_dict["boxes"], det_dict["confidence"], det_dict["class_id"])
        pb = _filter_persons(boxes, [0])

        if pb is not None and len(pb) > 0:
            det = pb.cpu().numpy()
            tracks = tracker.update(det, frame)
        else:
            tracks = np.empty((0, 8), dtype=np.float32)

        latencies.append((time.perf_counter() - t0) * 1_000)

        for track in tracks:
            tid = int(track[4])
            if tid not in track_first:
                track_first[tid] = sampled_n
            track_last[tid] = sampled_n
            track_frames[tid].append(sampled_n)
            current_lifetimes[tid] = sampled_n - track_first[tid]

        counts_per_frame.append(len(tracks))
        peak_ram_mb = max(peak_ram_mb, psutil.Process().memory_info().rss / 1e6)

        annotated = annotate_tracking_frame(frame, tracks, f"{args.tracker.upper()} (CACHED)", total_read, current_lifetimes)
        out_video.write(annotated)

        if sampled_n % 500 == 0 or total_read == total_frm:
            pct = (total_read / total_frm) * 100 if total_frm > 0 else 0.0
            log.info(f"  Processed frame {total_read}/{total_frm} ({pct:.1f}%) | active: {len(tracks)}")

    runtime = time.time() - t_start
    cap.release()
    out_video.release()

    # Compute metrics
    all_ids = list(track_first.keys())
    total_unique_ids = len(all_ids)
    lifetimes_f = [track_last[t] - track_first[t] for t in all_ids]
    avg_lifetime = float(np.mean(lifetimes_f)) if lifetimes_f else 0.0
    max_lifetime = float(max(lifetimes_f)) if lifetimes_f else 0.0

    recovered = 0
    for tid in all_ids:
        frames_sorted = track_frames[tid]
        for i in range(1, len(frames_sorted)):
            if frames_sorted[i] - frames_sorted[i - 1] >= 2:
                recovered += 1
                break

    LOST_GAP = 10
    tracks_lost = sum(1 for t in all_ids if track_last[t] < sampled_n - LOST_GAP)
    track_fragmentation = sum(1 for t in all_ids if (track_last[t] - track_first[t]) < 15)

    lat = np.array(latencies) if latencies else np.array([0.0])
    median_latency = float(np.median(lat))
    median_fps = float(sampled_n / runtime) if runtime > 0 else 0.0

    metrics = {
        "tracker_type": args.tracker,
        "frames_sampled": sampled_n,
        "avg_tracks": float(np.mean(counts_per_frame)) if counts_per_frame else 0.0,
        "max_tracks": int(max(counts_per_frame)) if counts_per_frame else 0,
        "recovered_tracks": recovered,
        "tracks_lost": tracks_lost,
        "track_fragmentation": track_fragmentation,
        "id_switches": "N/A (requires ground truth)",
        "median_fps": median_fps,
        "median_ms": median_latency,
        "peak_ram_mb": float(peak_ram_mb),
        "runtime": float(runtime),
        "mota": "N/A (requires ground truth)",
        "motp": "N/A (requires ground truth)",
        "idf1": "N/A (requires ground truth)",
        "hota": "N/A (requires ground truth)",
    }

    # Save metrics
    with open(out_dir / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)

    with open(out_dir / "metrics.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value"])
        for k, v in metrics.items():
            writer.writerow([k, v])

    # Save report.md
    report = [
        f"# Aurika Tracking v2 — Cached Run Report: {args.tracker.upper()}",
        "",
        f"- **Tracker:** {args.tracker.upper()}",
        f"- **Cache Source:** `{cache_path.name}`",
        f"- **Frames Sampled:** {sampled_n}",
        f"- **Total Runtime:** {runtime:.2f} s",
        f"- **Median Processing Speed:** {median_fps:.1f} FPS",
        "",
        "## Performance Metrics Table",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    for k, v in metrics.items():
        report.append(f"| {k} | {v} |")

    with open(out_dir / "report.md", "w") as fh:
        fh.write("\n".join(report))

    log.info(f"Metrics saved to {out_dir / 'metrics.json'}")
    log.info(f"Report saved to {out_dir / 'report.md'}")
    log.info("Cached tracking run complete!")

if __name__ == "__main__":
    main()
