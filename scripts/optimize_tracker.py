#!/usr/bin/env python3
"""
Aurika Tracking v2 — ByteTrack Hyperparameter Optimizer
======================================================
Sweeps a single tracker parameter across specified values.
Generates metrics.json, metrics.csv, plots, and a comparison report.

Usage:
    python scripts/optimize_tracker.py --parameter track_buffer --values 30 45 60 90
"""

import argparse
import csv
import gc
import json
import logging
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
import psutil
import torch
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("Optimizer")

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT))

from model_resolver import ModelResolver
from tracker.model_loader import load_yolo_model
from tracker.device import get_device
from tracker.tracker_factory import create_tracker

# Load environment configs
resolver = ModelResolver(project_root=PROJECT_ROOT)
DEVICE = get_device()

if resolver.is_kaggle:
    VIDEO_PATH = Path("/kaggle/input/datasets/tusharmarscitizen/video-analysis/Dark_lighting.mp4")
else:
    VIDEO_PATH = PROJECT_ROOT / "videos" / "Dark_lighting.mp4"

SAMPLE_EVERY = 3
CONF_THRESH = 0.25
WARMUP_FRAMES = 5

DEFAULT_CONFIG = {
    "track_high_thresh": 0.25,
    "track_low_thresh": 0.10,
    "new_track_thresh": 0.25,
    "track_buffer": 30,
    "match_thresh": 0.80,
    "fuse_score": True,
    "gmc_method": "none",
}

def parse_value(val_str: str) -> Any:
    if val_str.lower() in ("true", "yes", "1"):
        return True
    if val_str.lower() in ("false", "no", "0"):
        return False
    try:
        if "." in val_str:
            return float(val_str)
        return int(val_str)
    except ValueError:
        return val_str

def _filter_persons(boxes, person_classes: List[int]):
    if boxes is None or len(boxes) == 0:
        return boxes
    cls_int = boxes.cls.cpu().int()
    mask = torch.zeros(len(boxes), dtype=torch.bool)
    for c in person_classes:
        mask |= (cls_int == c)
    return boxes[mask]

def run_tracking_with_config(
    model_key: str,
    tracker_config: Dict[str, Any],
    video_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """Runs the tracking loop with a specific config dictionary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = resolver.resolve(model_key)
    person_classes = [0]  # YOLO11l is COCO person (0)

    model = load_yolo_model(str(cfg))
    model.to(DEVICE)

    tracker = create_tracker("bytetrack", tracker_config, device=DEVICE)

    cap = cv2.VideoCapture(str(video_path))
    total_frm = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_fps = max(1.0, src_fps / SAMPLE_EVERY)

    out_video = cv2.VideoWriter(
        str(output_dir / "tracking_output.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        out_fps, (W, H),
    )

    # Warmup
    warmed = 0
    while cap.isOpened() and warmed < WARMUP_FRAMES:
        ret, wf = cap.read()
        if not ret:
            break
        model.predict(wf, conf=CONF_THRESH, device=DEVICE, verbose=False)
        warmed += 1

    track_first: Dict[int, int] = {}
    track_last: Dict[int, int] = {}
    track_frames: Dict[int, List[int]] = defaultdict(list)

    counts_per_frame: List[int] = []
    latencies: List[float] = []
    peak_ram_mb = 0.0
    sampled_n = 0
    total_read = warmed

    t_start = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        total_read += 1
        if total_read % SAMPLE_EVERY != 0:
            continue
        sampled_n += 1

        t0 = time.perf_counter()
        results = model.predict(frame, conf=CONF_THRESH, device=DEVICE, verbose=False)[0]
        pb = _filter_persons(results.boxes, person_classes)

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

        counts_per_frame.append(len(tracks))
        peak_ram_mb = max(peak_ram_mb, psutil.Process().memory_info().rss / 1e6)

        # Draw default boxes for export video
        overlay = frame.copy()
        for track in tracks:
            x1, y1, x2, y2 = map(int, track[:4])
            tid = int(track[4])
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(overlay, f"ID {tid}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        out_video.write(overlay)

    runtime = time.time() - t_start
    cap.release()
    out_video.release()

    del model
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()
    elif DEVICE == "cuda":
        torch.cuda.empty_cache()

    # Compute metrics
    all_ids = list(track_first.keys())
    total_unique_ids = len(all_ids)
    lifetimes_f = [track_last[t] - track_first[t] for t in all_ids]
    avg_lifetime = float(np.mean(lifetimes_f)) if lifetimes_f else 0.0

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
        "avg_tracks": float(np.mean(counts_per_frame)) if counts_per_frame else 0.0,
        "max_tracks": int(max(counts_per_frame)) if counts_per_frame else 0,
        "recovered_tracks": recovered,
        "tracks_lost": tracks_lost,
        "track_fragmentation": track_fragmentation,
        "median_fps": median_fps,
        "median_ms": median_latency,
        "peak_ram_mb": float(peak_ram_mb),
        "runtime": float(runtime),
        "config": str(tracker_config),
    }

    # Save metrics in run folder
    with open(output_dir / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)

    with open(output_dir / "metrics.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value"])
        for k, v in metrics.items():
            writer.writerow([k, v])

    # Write small report.md
    with open(output_dir / "report.md", "w") as fh:
        fh.write(f"# Run Report\n\nConfig: `{tracker_config}`\n\n")
        fh.write("| Metric | Value |\n|---|---|\n")
        for k, v in metrics.items():
            fh.write(f"| {k} | {v} |\n")

    return metrics

def generate_plots(parameter: str, x_values: List[Any], results: List[Dict], output_dir: Path) -> None:
    try:
        numeric_x = [float(x) for x in x_values]
        is_numeric = True
    except (ValueError, TypeError):
        numeric_x = [str(x) for x in x_values]
        is_numeric = False

    metrics_to_plot = [
        ("recovered_tracks", "Track Recoveries", "g-o" if is_numeric else "green"),
        ("tracks_lost", "Lost Tracks", "r-o" if is_numeric else "red"),
        ("track_fragmentation", "Track Fragmentation", "b-o" if is_numeric else "blue"),
        ("runtime", "Runtime (s)", "y-o" if is_numeric else "yellow"),
    ]

    for key, title, style in metrics_to_plot:
        y_values = [r[key] for r in results]
        
        plt.figure(figsize=(8, 5))
        if is_numeric:
            plt.plot(numeric_x, y_values, style, linewidth=2, markersize=8)
            plt.xlabel(parameter)
        else:
            plt.bar(numeric_x, y_values, color=style, width=0.4)
            plt.xlabel(parameter)
            
        plt.ylabel(title)
        plt.title(f"{title} vs {parameter}")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        
        plot_path = output_dir / f"{key}_vs_{parameter}.png"
        plt.savefig(str(plot_path))
        plt.close()

def generate_comparison_report(
    parameter: str,
    values: List[Any],
    results: List[Dict],
    output_dir: Path,
) -> None:
    """Generates the comparison CSV and comparison.md report."""
    # Write summary CSV
    csv_path = output_dir / "comparison_summary.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        headers = ["value", "avg_tracks", "max_tracks", "recovered_tracks", "tracks_lost", "track_fragmentation", "median_fps", "median_ms", "peak_ram_mb", "runtime"]
        writer.writerow(headers)
        for val, r in zip(values, results):
            writer.writerow([
                val, r["avg_tracks"], r["max_tracks"], r["recovered_tracks"],
                r["tracks_lost"], r["track_fragmentation"], r["median_fps"],
                r["median_ms"], r["peak_ram_mb"], r["runtime"]
            ])

    # Rank configurations based on tracking quality
    # score = recovered - lost - fragmentation
    ranked = []
    for val, r in zip(values, results):
        score = r["recovered_tracks"] - r["tracks_lost"] - r["track_fragmentation"]
        ranked.append((score, val, r))

    # Sort descending (higher score is better)
    ranked.sort(key=lambda x: x[0], reverse=True)
    best_config = ranked[0]
    worst_config = ranked[-1]

    # Overall recommendation logic: check if best config is noticeably better than baseline
    # Find baseline (using default value if present)
    baseline_val = DEFAULT_CONFIG[parameter]
    baseline_run = None
    for val, r in zip(values, results):
        if val == baseline_val:
            baseline_run = r
            break

    if baseline_run:
        b_score = baseline_run["recovered_tracks"] - baseline_run["tracks_lost"] - baseline_run["track_fragmentation"]
        diff = best_config[0] - b_score
        
        if diff > 15: # threshold for meaningful improvement
            decision = "Promote best config"
            recommendation = (
                f"The configuration `{parameter}={best_config[1]}` demonstrates a meaningful "
                f"improvement in tracking stability (Score: {best_config[0]} vs Baseline: {b_score}). "
                f"We recommend promoting this value to production configs/config.yaml."
            )
        else:
            decision = "Keep baseline"
            recommendation = (
                f"The differences between configurations are within measurement noise. "
                f"We recommend keeping the baseline value `{parameter}={baseline_val}` to maintain simplicity."
            )
    else:
        decision = "Recommend best config"
        recommendation = f"The best performing configuration is `{parameter}={best_config[1]}` with a tracking score of {best_config[0]}."

    # Write Markdown comparison
    md = [
        f"# Experiment 002 — ByteTrack Optimization: {parameter}",
        "",
        "## Executive Summary",
        "",
        f"> **Decision Rule Outcome:** `{decision}`",
        f"> **Overall Recommendation:** {recommendation}",
        "",
        "## Detailed Parameter Sweep Summary Table",
        "",
        f"| Value of `{parameter}` | Avg Tracks | Max Tracks | Recoveries ↑ | Lost ↓ | Fragmentation ↓ | Median FPS | Inference Time | Peak RAM |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for val, r in zip(values, results):
        md.append(
            f"| `{val}` | {r['avg_tracks']:.2f} | {r['max_tracks']} | {r['recovered_tracks']} | "
            f"{r['tracks_lost']} | {r['track_fragmentation']} | {r['median_fps']:.1f} | "
            f"{r['median_ms']:.1f} ms | {r['peak_ram_mb']:.0f} MB |"
        )

    md += [
        "",
        "## Configuration Ranking",
        "",
        f"1. **Best Configuration:** `{parameter}={best_config[1]}` (Score: {best_config[0]})",
        f"2. **Worst Configuration:** `{parameter}={worst_config[1]}` (Score: {worst_config[0]})",
        "",
        "## Visualization Charts",
        "",
        f"### 1. Recoveries vs {parameter}",
        f"![Recoveries vs {parameter}](recovered_tracks_vs_{parameter}.png)",
        "",
        f"### 2. Lost Tracks vs {parameter}",
        f"![Lost Tracks vs {parameter}](tracks_lost_vs_{parameter}.png)",
        "",
        f"### 3. Track Fragmentation vs {parameter}",
        f"![Track Fragmentation vs {parameter}](track_fragmentation_vs_{parameter}.png)",
        ""
    ]

    with open(output_dir / "comparison.md", "w") as fh:
        fh.write("\n".join(md))

    log.info(f"Summary CSV written to {csv_path}")
    log.info(f"Comparison report written to {output_dir / 'comparison.md'}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Aurika ByteTrack Hyperparameter Optimizer")
    parser.add_argument(
        "--parameter",
        type=str,
        required=True,
        choices=["track_buffer", "match_thresh", "track_high_thresh", "track_low_thresh", "new_track_thresh", "fuse_score"],
        help="The ByteTrack parameter to optimize"
    )
    parser.add_argument(
        "--values",
        type=str,
        nargs="+",
        required=True,
        help="Space separated values to sweep over"
    )
    args = parser.parse_args()

    parsed_vals = [parse_value(v) for v in args.values]

    log.info("╔══════════════════════════════════════════╗")
    log.info("║ Aurika Tracking v2 — Parameter Optimizer ║")
    log.info("╚══════════════════════════════════════════╝")
    log.info(f"Parameter to Sweep : {args.parameter}")
    log.info(f"Values to Evaluate : {parsed_vals}")
    log.info(f"Device             : {DEVICE.upper()}")
    log.info(f"Video              : {VIDEO_PATH}\n")

    if not VIDEO_PATH.exists():
        log.error(f"Video not found: {VIDEO_PATH}")
        raise SystemExit(1)

    exp_dir = PROJECT_ROOT / "runs" / "experiment002" / args.parameter
    exp_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []

    for val in parsed_vals:
        run_dir = exp_dir / str(val)
        
        # Override baseline configs
        current_config = dict(DEFAULT_CONFIG)
        current_config[args.parameter] = val
        
        log.info(f"🚀 Running optimization sweep value: {args.parameter} = {val}")
        r = run_tracking_with_config(
            model_key="yolo11l",
            tracker_config=current_config,
            video_path=VIDEO_PATH,
            output_dir=run_dir
        )
        results.append(r)

    # Generate visual charts
    log.info("📊 Plotting parameter sweep comparison charts...")
    generate_plots(args.parameter, parsed_vals, results, exp_dir)

    # Generate summary CSV & Markdown comparison report
    log.info("📝 Compiling comparison summaries...")
    generate_comparison_report(args.parameter, parsed_vals, results, exp_dir)

    log.info(f"🎉 Parameter sweep complete! Outputs stored in {exp_dir}")

if __name__ == "__main__":
    main()
