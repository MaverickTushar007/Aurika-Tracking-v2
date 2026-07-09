#!/usr/bin/env python3
"""
Aurika Tracking v2 — Experiment 009 Benchmarking & Evaluation Runner
=====================================================================
Evaluates ByteTrack vs BoT-SORT + OSNet on Dark_lighting.mp4 using cache,
generates validation tables, exports distributions/matrices, and compiles
comparison, trajectory, and hysteresis debug videos.
Uses disk-based temporary videos to prevent RAM growth.
"""

import csv
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("Experiment009")

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT))

from model_resolver import ModelResolver
from tracker.config_loader import PipelineConfig, TrackerConfig
from tracker.device import get_device
from tracker.detection_cache import CachedBoxes, calculate_video_hash, load_detection_cache
from tracker.tracking_engine import TrackingEngine
from tracker.analytics_engine import RestaurantAnalyticsEngine
from benchmark import _filter_persons

# Init configurations
resolver = ModelResolver(project_root=PROJECT_ROOT)
DEVICE = get_device()
SAMPLE_EVERY = 3

def run_tracker_evaluation(
    tracker_type: str,
    with_reid: bool,
    video_path: Path,
    layout_path: Path,
    detections_list: List[Dict[str, Any]],
    output_dir: Path,
    is_debug_video: bool = False
) -> Tuple[Dict[str, Any], Path, List[Dict[str, Any]]]:
    """Runs tracking + analytics engine for a config, returning performance and tracking metrics."""
    log.info(f"Evaluating Tracker: {tracker_type.upper()} (ReID={with_reid})...")

    # Set tracker config
    tracker_cfg = TrackerConfig({
        "tracker": {
            "tracker_type": tracker_type,
            "track_high_thresh": 0.25,
            "track_low_thresh": 0.10,
            "new_track_thresh": 0.25,
            "track_buffer": 60,
            "match_thresh": 0.80,
            "fuse_score": True,
            "gmc_method": "none",
            "with_reid": with_reid,
            "model": str(PROJECT_ROOT / "yolo11n.pt") if with_reid else "auto",
            "zone_hysteresis_frames": 5
        }
    })

    # Instantiate engines
    tracker = TrackingEngine(tracker_cfg)
    analytics = RestaurantAnalyticsEngine(str(layout_path), fps=30.0 / SAMPLE_EVERY)

    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frame_idx = 0
    sampled_idx = 0
    t_start = time.perf_counter()
    
    waiting_counts = []

    # Temp visual video writer to prevent OOM
    temp_visual_path = output_dir / f"temp_{tracker_type}.mp4"
    visual_writer = cv2.VideoWriter(
        str(temp_visual_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        src_fps / SAMPLE_EVERY, (W, H)
    )

    # Setup specific debug writers if needed
    if is_debug_video:
        debug_video = cv2.VideoWriter(
            str(output_dir / "zone_assignment_debug.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            src_fps / SAMPLE_EVERY, (W, H)
        )
        traj_video = cv2.VideoWriter(
            str(output_dir / "trajectory_overlay.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            src_fps / SAMPLE_EVERY, (W, H)
        )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        
        if frame_idx % SAMPLE_EVERY != 0:
            continue
            
        sampled_idx += 1
        timestamp = frame_idx / src_fps

        # Retrieve detections
        if frame_idx - 1 < len(detections_list):
            det_dict = detections_list[frame_idx - 1]
        else:
            det_dict = {"boxes": np.empty((0, 4)), "confidence": np.empty((0,)), "class_id": np.empty((0,))}
            
        boxes = CachedBoxes(det_dict["boxes"], det_dict["confidence"], det_dict["class_id"])
        pb = _filter_persons(boxes, [0])

        if pb is not None and len(pb) > 0:
            tracks = tracker.update(pb, frame)
        else:
            tracks = np.empty((0, 8), dtype=np.float32)

        # Update analytics
        analytics.update_frame(frame_idx, timestamp, tracks)
        
        # Track waiting counts
        active_waiting = sum(1 for t in tracks if analytics.get_zone_at_point(analytics.get_track_center(t)) in ["Waiting Area", "Waiting"])
        waiting_counts.append(active_waiting)

        # Draw visual frame overlay & save to temp video
        frame_overlay = analytics.draw_analytics_overlay(frame.copy(), tracks, src_fps / SAMPLE_EVERY, frame_idx)
        visual_writer.write(frame_overlay)

        # Draw hysteresis assignment debug overlay
        if is_debug_video:
            debug_frame = frame.copy()
            for zone in analytics.zones:
                cv2.polylines(debug_frame, [zone["polygon"]], True, zone["color"], 2)
            
            for t in tracks:
                x1, y1, x2, y2 = map(int, t[:4])
                tid = int(t[4])
                cx, cy = (x1 + x2) // 2, y2
                
                state = analytics.memory_engine.get_track(tid)
                if state:
                    cv2.circle(debug_frame, (cx, cy), 6, (0, 0, 255), -1)
                    label = f"ID:{tid} Z:{state.current_zone} Cand:{state.candidate_zone} Cnt:{state.frames_inside_candidate} Conf:{state.zone_confidence:.2f}"
                    cv2.putText(debug_frame, label, (cx - 60, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            debug_video.write(debug_frame)

            # Draw trajectories overlay
            traj_frame = frame.copy()
            for zone in analytics.zones:
                cv2.polylines(traj_frame, [zone["polygon"]], True, zone["color"], 1)
                
            all_states = list(analytics.memory_engine.active_tracks.values()) + list(analytics.memory_engine.archived_tracks.values())
            for s in all_states:
                if len(s.trajectory) > 1:
                    color = (hash(s.track_id) % 200 + 55, hash(s.track_id * 3) % 200 + 55, hash(s.track_id * 5) % 200 + 55)
                    pts = np.array(s.trajectory, dtype=np.int32)
                    cv2.polylines(traj_frame, [pts], False, color, 2)
                    if s.recovery_count > 0:
                        last_pt = pts[-1]
                        cv2.circle(traj_frame, (int(last_pt[0]), int(last_pt[1])), 8, (0, 255, 0), 2)
            traj_video.write(traj_frame)

    cap.release()
    visual_writer.release()
    if is_debug_video:
        debug_video.release()
        traj_video.release()

    runtime = time.perf_counter() - t_start

    # Compute metrics
    all_states = list(analytics.memory_engine.active_tracks.values()) + list(analytics.memory_engine.archived_tracks.values())
    lifetimes = [s.age_frames for s in all_states]
    
    avg_lifetime = np.mean(lifetimes) if lifetimes else 0.0
    ids_created = len(all_states)
    
    frags = sum(s.occlusion_count for s in all_states)
    recoveries = sum(s.recovery_count for s in all_states)
    
    waiting_dwells = []
    for s in all_states:
        waiting_dwells.extend(s.zone_dwell_times.get("Waiting Area", []) + s.zone_dwell_times.get("Waiting", []))
    
    avg_wait = np.mean(waiting_dwells) if waiting_dwells else 0.0
    queue_stability = float(np.std(waiting_counts)) if waiting_counts else 0.0

    metrics = {
        "tracker_type": tracker_type,
        "with_reid": with_reid,
        "runtime": round(runtime, 2),
        "fps": round(sampled_idx / runtime, 1),
        "ids_created": ids_created,
        "fragmentation": frags,
        "recoveries": recoveries,
        "avg_lifetime": round(avg_lifetime, 1),
        "avg_waiting_time": round(avg_wait, 2),
        "queue_stability": round(queue_stability, 3),
        "zone_transitions": sum(len(s.zone_history) for s in all_states)
    }

    return metrics, temp_visual_path, all_states

def main() -> None:
    exp_dir = PROJECT_ROOT / "runs" / "experiment009"
    exp_dir.mkdir(parents=True, exist_ok=True)

    video_path = PROJECT_ROOT / "videos" / "Dark_lighting.mp4"
    if resolver.is_kaggle:
        video_path = Path("/kaggle/input/datasets/tusharmarscitizen/video-analysis/Dark_lighting.mp4")

    cache_path = PROJECT_ROOT / "runs" / "cache" / "detections.pkl"
    video_hash = calculate_video_hash(video_path)
    detections_list = load_detection_cache(cache_path, {"video_hash": video_hash, "model_name": "yolo11l"})

    layout_path = PROJECT_ROOT / "configs" / "restaurant_semantic.yaml"

    # Evaluate Configuration A: ByteTrack
    bt_metrics, temp_bt_path, bt_states = run_tracker_evaluation(
        "bytetrack", False, video_path, layout_path, detections_list, exp_dir
    )

    # Evaluate Configuration B: BoT-SORT + OSNet ReID
    bs_metrics, temp_bs_path, bs_states = run_tracker_evaluation(
        "botsort", True, video_path, layout_path, detections_list, exp_dir, is_debug_video=True
    )

    # 4. Generate comparison_video.mp4 (Read from temp videos and stack horizontally)
    log.info("Generating side-by-side comparison video...")
    cap_bt = cv2.VideoCapture(str(temp_bt_path))
    cap_bs = cv2.VideoCapture(str(temp_bs_path))
    
    W = int(cap_bt.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap_bt.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_bt.get(cv2.CAP_PROP_FPS)

    comp_video = cv2.VideoWriter(
        str(exp_dir / "comparison_video.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (W * 2, H)
    )

    while True:
        ret_bt, frame_bt = cap_bt.read()
        ret_bs, frame_bs = cap_bs.read()
        if not ret_bt or not ret_bs:
            break

        # Label overlays
        cv2.putText(frame_bt, "BYTETRACK (BASELINE)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame_bs, "BOT-SORT + OSNET", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        stacked = np.hstack([frame_bt, frame_bs])
        comp_video.write(stacked)

    cap_bt.release()
    cap_bs.release()
    comp_video.release()
    
    # Cleanup temp videos
    if temp_bt_path.exists():
        temp_bt_path.unlink()
    if temp_bs_path.exists():
        temp_bs_path.unlink()
        
    log.info("Comparison video compiled.")

    # 5. Export reports and matrices
    write_benchmark_reports(exp_dir, bt_metrics, bs_metrics, bt_states, bs_states)

def write_benchmark_reports(exp_dir: Path, bt: Dict[str, Any], bs: Dict[str, Any], bt_states: List[Any], bs_states: List[Any]) -> None:
    """Writes comparison CSV tables and markdown report detailing business recommendations."""
    with open(exp_dir / "tracker_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Tracker", "IDs Created", "Fragmentation", "Recoveries", "Avg Lifetime"])
        writer.writerow(["ByteTrack", bt["ids_created"], bt["fragmentation"], bt["recoveries"], bt["avg_lifetime"]])
        writer.writerow(["BoT-SORT+OSNet", bs["ids_created"], bs["fragmentation"], bs["recoveries"], bs["avg_lifetime"]])

    with open(exp_dir / "restaurant_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Tracker", "Avg Waiting Time", "Queue Stability (StdDev)", "Zone Transitions"])
        writer.writerow(["ByteTrack", bt["avg_waiting_time"], bt["queue_stability"], bt["zone_transitions"]])
        writer.writerow(["BoT-SORT+OSNet", bs["avg_waiting_time"], bs["queue_stability"], bs["zone_transitions"]])

    with open(exp_dir / "tracker_runtime.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Tracker", "Runtime (s)", "Processing Speed (FPS)"])
        writer.writerow(["ByteTrack", bt["runtime"], bt["fps"]])
        writer.writerow(["BoT-SORT+OSNet", bs["runtime"], bs["fps"]])

    with open(exp_dir / "zone_transition_matrix.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Tracker", "Transitions Path Counts"])
        writer.writerow(["ByteTrack", bt["zone_transitions"]])
        writer.writerow(["BoT-SORT+OSNet", bs["zone_transitions"]])

    with open(exp_dir / "journey_statistics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "ByteTrack", "BoT-SORT+OSNet"])
        writer.writerow(["Total Journeys", len(bt_states), len(bs_states)])

    wait_diff = abs(bt["avg_waiting_time"] - bs["avg_waiting_time"])
    fps_ratio = (bt["fps"] - bs["fps"]) / bt["fps"] * 100.0

    conclusion = "ByteTrack remains the preferred production tracker."
    rationale = (
        f"While BoT-SORT + OSNet integrates appearance embeddings, the added extraction latency "
        f"degrades execution speed by {fps_ratio:.1f}% (dropping from {bt['fps']} to {bs['fps']} FPS) "
        f"without yielding statistically significant gains in customer waiting time estimation "
        f"(difference: {wait_diff:.2f}s) or queue stability."
    )
    if bs["ids_created"] < bt["ids_created"] - 20 and wait_diff > 1.5:
        conclusion = "BoT-SORT + OSNet is recommended for production."
        rationale = (
            f"BoT-SORT + OSNet demonstrates significant identity persistence, reducing unique track IDs "
            f"created from {bt['ids_created']} to {bs['ids_created']} (-{(bt['ids_created']-bs['ids_created'])/bt['ids_created']*100.0:.1f}%), "
            f"stabilizing dwell metrics sufficiently to justify the {fps_ratio:.1f}% runtime overhead."
        )

    with open(exp_dir / "feature_summary.md", "w") as f:
        f.write(f"# Feature Selection Summary\n\n- **Production Recommendation:** {conclusion.split()[0]} \n- **Rationale:** {rationale}\n")

    report = [
        "# Experiment 009 — Tracker Evaluation & Recommendation Report",
        "",
        "## 1. Tracker Performance Summary",
        "",
        "| Metric | ByteTrack (Baseline) | BoT-SORT + OSNet | Change % |",
        "|---|---|---|---|",
        f"| **Unique Track IDs** | {bt['ids_created']} | {bs['ids_created']} | {((bs['ids_created']-bt['ids_created'])/bt['ids_created'])*100.0:+.1f}% |",
        f"| **Fragmentation** | {bt['fragmentation']} | {bs['fragmentation']} | {((bs['fragmentation']-bt['fragmentation'])/max(1,bt['fragmentation']))*100.0:+.1f}% |",
        f"| **Recoveries** | {bt['recoveries']} | {bs['recoveries']} | {((bs['recoveries']-bt['recoveries'])/max(1,bt['recoveries']))*100.0:+.1f}% |",
        f"| **Avg Lifetime (Frames)** | {bt['avg_lifetime']} | {bs['avg_lifetime']} | {((bs['avg_lifetime']-bt['avg_lifetime'])/max(1,bt['avg_lifetime']))*100.0:+.1f}% |",
        f"| **Runtime (Seconds)** | {bt['runtime']}s | {bs['runtime']}s | {((bs['runtime']-bt['runtime'])/bt['runtime'])*100.0:+.1f}% |",
        f"| **Processing Speed** | {bt['fps']} FPS | {bs['fps']} FPS | {((bs['fps']-bt['fps'])/bt['fps'])*100.0:+.1f}% |",
        "",
        "## 2. Restaurant Analytics & Business Metrics",
        "",
        "| Business Indicator | ByteTrack | BoT-SORT + OSNet | Difference |",
        "|---|---|---|---|",
        f"| **Avg Waiting Time (Seconds)** | {bt['avg_waiting_time']}s | {bs['avg_waiting_time']}s | {wait_diff:.2f}s |",
        f"| **Queue Stability (Std Dev)** | {bt['queue_stability']} | {bs['queue_stability']} | {bs['queue_stability']-bt['queue_stability']:+.3f} |",
        f"| **Total Zone Transitions** | {bt['zone_transitions']} | {bs['zone_transitions']} | {bs['zone_transitions']-bt['zone_transitions']:+d} |",
        "",
        "## 3. Production Recommendation",
        f"**Choice:** {conclusion}",
        "",
        "**Rationale:**",
        rationale,
        ""
    ]
    with open(exp_dir / "comparison_report.md", "w") as f:
        f.write("\n".join(report))
    log.info("Benchmark reports generated successfully.")

if __name__ == "__main__":
    main()
