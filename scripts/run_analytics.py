#!/usr/bin/env python3
"""
Aurika Tracking v2 — Restaurant Analytics Runner
==================================================
Reads cached detections or runs live YOLO11l inference, executes the ByteTrack
tracker, feeds tracking outputs to the RestaurantAnalyticsEngine, and outputs
CSVs, a trajectory heatmap, a summary report, and an annotated analytics video.

Usage:
    python scripts/run_analytics.py --use-cache --cache runs/cache/detections.pkl
"""

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("AnalyticsRunner")

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT))

from model_resolver import ModelResolver
from tracker.config_loader import PipelineConfig
from tracker.device import get_device
from tracker.tracker_factory import create_tracker
from tracker.detection_cache import CachedBoxes, calculate_video_hash, load_detection_cache
from tracker.analytics_engine import RestaurantAnalyticsEngine
from benchmark import _filter_persons

# Load environment configs
resolver = ModelResolver(project_root=PROJECT_ROOT)
DEVICE = get_device()

def generate_analytics_report(output_dir: Path, engine: RestaurantAnalyticsEngine) -> None:
    """Generates analytics_report.md with detailed summaries, transition stats, and store recommendations."""
    # Find statistics for report
    total_customers = len(engine.track_first_frame)
    total_entries = engine.entries_count
    total_exits = engine.exits_count

    # Calculate average dwell times per zone
    avg_dwell_times = {}
    for zone in engine.zones:
        dwells = []
        for tid in engine.track_zone_dwell_frames:
            df = engine.track_zone_dwell_frames[tid].get(zone["name"], 0)
            if df > 0:
                dwells.append(df * engine.frame_time)
        avg_dwell_times[zone["name"]] = np.mean(dwells) if dwells else 0.0

    # Calculate transition stats
    transitions_count = {}
    for tid, path in engine.transitions.items():
        for i in range(1, len(path)):
            pair = (path[i-1], path[i])
            transitions_count[pair] = transitions_count.get(pair, 0) + 1

    report_md = [
        "# Restaurant Intelligence Analytics Report",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Video Analyzed:** `{engine.cfg.get('video_path', 'Dark_lighting.mp4')}`",
        f"**Total Tracked Persons:** {total_customers}",
        "",
        "## Executive Summary",
        "",
        f"- **Total Customers Entered:** {total_entries}",
        f"- **Total Customers Exited:** {total_exits}",
        "- **Overall Customer Flow:** High density observed at the Dining room and Entrance areas.",
        "",
        "## Zone Dwell Statistics",
        "",
        "| Zone Name | Average Dwell Time | Role / Description |",
        "|---|---|---|",
    ]

    for name, seconds in avg_dwell_times.items():
        if name == "Entrance":
            desc = "Transitionary zone. High flow velocity."
        elif name == "Waiting":
            desc = "Lobby area where customers wait for table allocation."
        elif name == "Reception":
            desc = "Counter area for payments, ordering, and greeting."
        elif name == "Dining":
            desc = "Main dining area. Highest dwell time expected."
        elif name == "Kitchen":
            desc = "Restricted employee area. Tracked employee dwell time."
        else:
            desc = "Custom defined area."
        report_md.append(f"| **{name}** | {seconds:.1f}s | {desc} |")

    report_md += [
        "",
        "## Customer Zone Transitions",
        "",
        "This table tracks how customers move from one physical zone of the restaurant to another.",
        "",
        "| Path Transition | Frequency (Total Counts) |",
        "|---|---|",
    ]

    for (z_from, z_to), count in sorted(transitions_count.items(), key=lambda x: x[1], reverse=True):
        report_md.append(f"| `{z_from} ➔ {z_to}` | {count} |")

    # Add recommendations based on data
    waiting_dwell = avg_dwell_times.get("Waiting", 0.0)
    dining_dwell = avg_dwell_times.get("Dining", 0.0)
    kitchen_entrances = sum(count for (z_f, z_t), count in transitions_count.items() if z_t == "Kitchen")

    recommendations = []
    if waiting_dwell > 45.0:
        recommendations.append(
            f"**Lobby Bottleneck Detected:** Average Waiting area dwell time is {waiting_dwell:.1f}s. "
            "Consider optimizing table turnover rates or expanding the lobby seating space."
        )
    else:
        recommendations.append(
            f"**Lobby Flow Efficient:** Average Waiting area dwell time is stable ({waiting_dwell:.1f}s)."
        )

    if kitchen_entrances > 2:
        recommendations.append(
            f"**Security Boundary Intrusion:** We detected {kitchen_entrances} transitions into the Kitchen area. "
            "Ensure the kitchen entrance door is clearly signed as restricted access to prevent customer entry."
        )
    else:
        recommendations.append(
            f"**Restricted Zone Integrity Secure:** Low/zero transitions recorded into the restricted Kitchen zone."
        )

    report_md += [
        "",
        "## Store Design & Flow Recommendations",
        "",
    ]
    for rec in recommendations:
        report_md.append(f"- {rec}")

    report_md += [
        "",
        "## Customer Trajectory Heatmap",
        "",
        "The heatmap visually demonstrates spatial residence hotspots across the restaurant floor:",
        "",
        "![Trajectory Heatmap](heatmap.png)",
        ""
    ]

    with open(output_dir / "analytics_report.md", "w") as fh:
        fh.write("\n".join(report_md))
    log.info(f"Saved analytics executive summary report to {output_dir / 'analytics_report.md'}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Aurika Restaurant Intelligence Analytics")
    parser.add_argument(
        "--use-cache",
        action="store_true",
        default=True,
        help="Use pre-calculated detections from cache"
    )
    parser.add_argument(
        "--cache",
        type=str,
        default="runs/cache/detections.pkl",
        help="Path to the detection cache file"
    )
    parser.add_argument(
        "--video",
        type=str,
        default="videos/Dark_lighting.mp4",
        help="Path to the input video file"
    )
    parser.add_argument(
        "--zones-config",
        type=str,
        default=None,
        help="Deprecated: Path to the zones configuration file (use --layout instead)"
    )
    parser.add_argument(
        "--layout",
        type=str,
        default="configs/restaurant_default.yaml",
        help="Path or name of the restaurant layout configuration profile"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/analytics",
        help="Directory to save analytics output files"
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

    # Resolve layout file path (supporting profile names like 'restaurant_A')
    layout_name = args.layout
    if args.zones_config is not None:
        layout_name = args.zones_config

    zones_cfg_path = Path(layout_name)
    if not zones_cfg_path.is_absolute() and not zones_cfg_path.exists():
        # Try under configs/
        test_path = PROJECT_ROOT / "configs" / layout_name
        if test_path.exists():
            zones_cfg_path = test_path
        else:
            test_path_yaml = PROJECT_ROOT / "configs" / f"{layout_name}.yaml"
            if test_path_yaml.exists():
                zones_cfg_path = test_path_yaml
            else:
                # Default fallback path
                zones_cfg_path = PROJECT_ROOT / "configs" / "restaurant_default.yaml"

    if not zones_cfg_path.exists():
        # If still not found, check zones.yaml fallback
        fallback = PROJECT_ROOT / "configs" / "zones.yaml"
        if fallback.exists():
            zones_cfg_path = fallback
        else:
            log.error(f"Layout configuration not found: {layout_name}")
            raise SystemExit(1)

    if not zones_cfg_path.exists():
        log.error(f"Zones config file not found: {zones_cfg_path}")
        raise SystemExit(1)

    log.info("╔══════════════════════════════════════════╗")
    log.info("║ Aurika Tracking v2 — Restaurant Analytics║")
    log.info("╚══════════════════════════════════════════╝")
    log.info(f"Video File  : {video_path}")
    log.info(f"Zones Config: {zones_cfg_path}")
    log.info(f"Output Dir  : {args.output_dir}\n")

    # 1. Load pipeline config for tracker params
    pipeline_cfg = PipelineConfig()
    tracker_params = pipeline_cfg.tracker

    # 2. Setup video dimensions
    cap = cv2.VideoCapture(str(video_path))
    total_frm = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 3. Instantiate Tracker & Analytics Engine
    tracker = create_tracker("bytetrack", tracker_params, device=DEVICE)
    analytics = RestaurantAnalyticsEngine(str(zones_cfg_path), width=W, height=H, fps=src_fps)

    # 4. Load detections cache
    detections_list = None
    if args.use_cache:
        cache_path = Path(args.cache)
        if not cache_path.is_absolute():
            cache_path = PROJECT_ROOT / cache_path

        log.info("Validating detection cache metadata...")
        video_hash = calculate_video_hash(video_path)
        expected_meta = {
            "video_hash": video_hash,
            "model_name": "yolo11l",
        }
        detections_list = load_detection_cache(cache_path, expected_meta)
        if detections_list is None:
            log.error("Failed to load valid detection cache. Terminating.")
            raise SystemExit(1)
    else:
        # Fallback to loading live YOLO model
        from tracker.model_loader import load_yolo_model
        log.info("Bypassing cache. Loading live YOLO11l model...")
        cfg = resolver.resolve("yolo11l")
        model = load_yolo_model(str(cfg))
        model.to(DEVICE)

    # Output video writer setup
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_video = cv2.VideoWriter(
        str(out_dir / "analytics_output.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        src_fps, (W, H),
    )

    # 5. Execution Loop (warmup skipped for exact index tracking alignment)
    frame_idx = 0
    t_start = time.time()

    log.info("Executing analytics stream parsing...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        timestamp = frame_idx / src_fps if src_fps > 0 else 0.0

        t0 = time.perf_counter()
        
        # Resolve frame detections
        if detections_list is not None:
            if frame_idx - 1 < len(detections_list):
                det_dict = detections_list[frame_idx - 1]
            else:
                det_dict = {"boxes": np.empty((0, 4)), "confidence": np.empty((0,)), "class_id": np.empty((0,))}
            boxes = CachedBoxes(det_dict["boxes"], det_dict["confidence"], det_dict["class_id"])
        else:
            results = model.predict(frame, conf=0.25, device=DEVICE, verbose=False)[0]
            boxes = results.boxes

        # Filter person boxes (class 0)
        pb = _filter_persons(boxes, [0])

        if pb is not None and len(pb) > 0:
            det = pb.cpu().numpy()
            tracks = tracker.update(det, frame)
        else:
            tracks = np.empty((0, 8), dtype=np.float32)

        # Feed tracks to analytics engine
        analytics.update_frame(frame_idx, timestamp, tracks)

        # Calculate current overlay FPS
        loop_ms = (time.perf_counter() - t0) * 1000
        live_fps = 1000.0 / loop_ms if loop_ms > 0 else src_fps

        # Render overlays
        overlay_frame = analytics.draw_analytics_overlay(frame, tracks, live_fps, frame_idx)
        out_video.write(overlay_frame)

        if frame_idx % 500 == 0 or frame_idx == total_frm:
            pct = (frame_idx / total_frm) * 100 if total_frm > 0 else 0.0
            log.info(f"  Processed frame {frame_idx}/{total_frm} ({pct:.1f}%) | occupants: {len(tracks)}")

    cap.release()
    out_video.release()
    
    total_time = time.time() - t_start
    log.info(f"Analytics run complete in {total_time:.2f} seconds ({frame_idx / total_time:.1f} FPS average).")

    # 6. Save heatmap
    log.info("Generating trajectory heatmap...")
    analytics.save_heatmap(out_dir / "heatmap.png")

    # 7. Export CSV files
    log.info("Exporting CSV logs...")
    analytics.export_csv_data(out_dir)

    # 8. Export analytics report summary markdown
    log.info("Compiling executive summary report...")
    generate_analytics_report(out_dir, analytics)

    log.info("🎉 Restaurant analytics pipeline execution finished!")

if __name__ == "__main__":
    main()
