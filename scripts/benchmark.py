#!/usr/bin/env python3
"""
Aurika Tracking v2 — Detector Benchmark
========================================
Stage 2 : Detection-only benchmark on every available model (sampled frames).
           Annotated detection video saved for every model.
Stage 3 : ByteTrack benchmark on the top-2 detectors only.
           Annotated tracking video saved for those two models.
Final    : Qualitative recommendation with explicit reasoning.

Usage:
    python scripts/benchmark.py

Output layout:
    runs/benchmark/
        {model}/
            detection_output.mp4
            detection_metrics.json
            tracking_output.mp4          # top-2 only
            tracking_metrics.json        # top-2 only
        detection_comparison.csv
        tracking_comparison.csv
        benchmark_report.md
"""

import csv
import gc
import json
import logging
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import psutil
import torch

from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker

# Model resolution — environment-agnostic (local vs Kaggle)
sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/ on path
from model_resolver import ModelResolver  # noqa: E402

# Production model loader — handles local file + Ultralytics auto-download
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root
from tracker.model_loader import load_yolo_model  # noqa: E402

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("Benchmark")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Environment-aware model resolver ────────────────────────────────────────
# Instantiated once at module level; benchmark logic never touches paths.
_resolver = ModelResolver(project_root=PROJECT_ROOT)

# Model metadata — NO "path" keys here.
# The resolver fills those in via build_registry() at startup.
_MODELS_META: Dict[str, Dict] = {
    "yolo11m": {
        # COCO: class 0 = person  |  class 1 = bicycle (must NOT be included)
        "person_classes": [0],
        "label": "YOLO11m  (COCO pretrained)",
    },
    "yolo11l": {
        "person_classes": [0],
        "label": "YOLO11l  (COCO pretrained)",
    },
    "yolo11x": {
        "person_classes": [0],
        "label": "YOLO11x  (COCO pretrained)",
    },
}

# Resolved at startup — each entry now contains a "path" key that is
# correct for the current environment (local Path or Ultralytics str).
MODELS: Dict[str, Dict] = _resolver.build_registry(_MODELS_META)

# ── Environment-aware video path ─────────────────────────────────────────────
if _resolver.is_kaggle:
    _VIDEO_STR = "/kaggle/input/datasets/tusharmarscitizen/video-analysis/Dark_lighting.mp4"
else:
    _VIDEO_STR = str(PROJECT_ROOT / "videos" / "Dark_lighting.mp4")
VIDEO_PATH  = Path(_VIDEO_STR)
OUTPUT_BASE = PROJECT_ROOT / "runs" / "benchmark"

SAMPLE_EVERY  = 3        # run inference on every Nth frame
CONF_THRESH   = 0.25     # detection confidence floor (matches production config)
SMALL_AREA    = 40 * 40  # bounding-box area (px²) ≤ this counts as "small person"
WARMUP_FRAMES = 5        # silent warm-up frames before timing starts

# ByteTrack hyperparameters — identical to configs/config.yaml baseline
BT_HIGH_THRESH  = 0.25
BT_LOW_THRESH   = 0.10
BT_NEW_THRESH   = 0.25
BT_BUFFER       = 30
BT_MATCH_THRESH = 0.80
BT_FUSE_SCORE   = True

# Device — prefer MPS (Apple Silicon) then fall back to CPU
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Colour palette & annotation helpers
# ---------------------------------------------------------------------------

_PALETTE = [
    (230, 115, 180), (0,   229, 255), (255, 107,   0), (46,  204, 113),
    (241, 196,  15), (255, 105, 180), (254, 211,  48), (235,  77,  75),
    (26,  188, 156), (52,  152, 219), (155,  89, 182), (231,  76,  60),
]


def _color(idx: int) -> Tuple[int, int, int]:
    return _PALETTE[idx % len(_PALETTE)]


def _draw_box(
    img: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    color: Tuple[int, int, int],
    label: str,
) -> None:
    """Corner-bracket bounding box with semi-transparent fill and label pill."""
    # Glass fill
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.08, img, 0.92, 0, img)

    # Border
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

    # Corner brackets (L-shaped, 12 px arms)
    L, T = 12, 2
    # Top-left
    cv2.line(img, (x1,     y1),     (x1 + L, y1),     color, T, cv2.LINE_AA)
    cv2.line(img, (x1,     y1),     (x1,     y1 + L), color, T, cv2.LINE_AA)
    # Top-right
    cv2.line(img, (x2,     y1),     (x2 - L, y1),     color, T, cv2.LINE_AA)
    cv2.line(img, (x2,     y1),     (x2,     y1 + L), color, T, cv2.LINE_AA)
    # Bottom-left
    cv2.line(img, (x1,     y2),     (x1 + L, y2),     color, T, cv2.LINE_AA)
    cv2.line(img, (x1,     y2),     (x1,     y2 - L), color, T, cv2.LINE_AA)
    # Bottom-right
    cv2.line(img, (x2,     y2),     (x2 - L, y2),     color, T, cv2.LINE_AA)
    cv2.line(img, (x2,     y2),     (x2,     y2 - L), color, T, cv2.LINE_AA)

    # Label pill
    font, fs, ft = cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1
    (tw, th), _ = cv2.getTextSize(label, font, fs, ft)
    bg_y1 = y1 - th - 8 if (y1 - th - 8) > 0 else y2
    bg_y2 = bg_y1 + th + 8
    cv2.rectangle(img, (x1, bg_y1), (x1 + tw + 8, bg_y2), color, -1)
    cv2.putText(img, label, (x1 + 4, bg_y2 - 4),
                font, fs, (255, 255, 255), ft, cv2.LINE_AA)


def _hud(img: np.ndarray, line1: str, line2: str) -> None:
    """Overlay two HUD lines in the top-left corner."""
    for i, text in enumerate([line1, line2]):
        y = 24 + i * 22
        cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (20, 20, 20), 3, cv2.LINE_AA)   # dark shadow
        cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (255, 248, 180), 1, cv2.LINE_AA)  # bright text


def annotate_detection_frame(
    frame: np.ndarray,
    person_boxes,            # ultralytics Boxes | None
    model_key: str,
    src_frame_n: int,
) -> np.ndarray:
    out = frame.copy()
    n = 0
    if person_boxes is not None and len(person_boxes) > 0:
        xyxy  = person_boxes.xyxy.cpu().numpy()
        confs = person_boxes.conf.cpu().numpy()
        for i, (box, c) in enumerate(zip(xyxy, confs)):
            x1, y1, x2, y2 = map(int, box)
            _draw_box(out, x1, y1, x2, y2, _color(i), f"Person {c:.2f}")
        n = len(person_boxes)
    _hud(out, f"[DET] {model_key}", f"frame {src_frame_n}  |  persons: {n}")
    return out


def annotate_tracking_frame(
    frame: np.ndarray,
    tracks: np.ndarray,      # (N, 8) from BYTETracker
    model_key: str,
    src_frame_n: int,
    lifetimes: Dict[int, int],
) -> np.ndarray:
    out = frame.copy()
    for track in tracks:
        x1, y1, x2, y2 = map(int, track[:4])
        tid  = int(track[4])
        conf = float(track[5])
        life = lifetimes.get(tid, 0)
        _draw_box(out, x1, y1, x2, y2, _color(tid),
                  f"ID {tid} ({conf:.2f}) life={life}f")
    _hud(out, f"[TRK] {model_key}",
         f"frame {src_frame_n}  |  active: {len(tracks)}")
    return out

# ---------------------------------------------------------------------------
# ByteTrack argument namespace (mirrors configs/config.yaml baseline)
# ---------------------------------------------------------------------------

class _BenchTrackerArgs:
    track_high_thresh = BT_HIGH_THRESH
    track_low_thresh  = BT_LOW_THRESH
    new_track_thresh  = BT_NEW_THRESH
    track_buffer      = BT_BUFFER
    match_thresh      = BT_MATCH_THRESH
    fuse_score        = BT_FUSE_SCORE
    gmc_method        = "none"

# ---------------------------------------------------------------------------
# Shared utility: per-model person class filter
# ---------------------------------------------------------------------------

def _filter_persons(boxes, person_classes: List[int]):
    """Return a Boxes slice keeping only the specified class IDs."""
    if boxes is None or len(boxes) == 0:
        return boxes
    cls_int = boxes.cls.cpu().int()
    mask = torch.zeros(len(boxes), dtype=torch.bool)
    for c in person_classes:
        mask |= (cls_int == c)
    return boxes[mask]

# ---------------------------------------------------------------------------
# Stage 2 — Detection benchmark
# ---------------------------------------------------------------------------

def run_detection_benchmark(
    model_key: str,
    cfg: Dict,
    video_path: Path,
    output_base: Path,
) -> Dict:
    """
    Runs detection-only pass on every SAMPLE_EVERY-th frame.
    Saves an annotated detection video.
    Returns a metrics dict (or dict with 'error' key on failure).
    """
    sep = "=" * 62
    log.info(sep)
    log.info(f"  STAGE 2 · DETECTION · {model_key}")
    log.info(sep)

    out_dir = output_base / model_key
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path      = cfg["path"]   # Path (local) or str (Ultralytics hub)
    person_classes  = cfg["person_classes"]

    # Log what we are about to load
    path_display = Path(model_path).name if isinstance(model_path, Path) else str(model_path)
    log.info(f"  Model: {path_display}")

    # Load model via the shared resolver-aware loader
    try:
        model = load_yolo_model(str(model_path))
        model.to(DEVICE)
    except Exception as exc:
        log.error(f"  Failed to load {model_key}: {exc}")
        return {"model": model_key, "label": cfg["label"], "error": str(exc)}

    # Open video
    cap       = cv2.VideoCapture(str(video_path))
    total_frm = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps   = cap.get(cv2.CAP_PROP_FPS)
    W         = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H         = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_fps   = max(1.0, src_fps / SAMPLE_EVERY)

    out_video = cv2.VideoWriter(
        str(out_dir / "detection_output.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        out_fps, (W, H),
    )

    # ── Warmup (suppress from timing) ──────────────────────────────────────
    warmed = 0
    while cap.isOpened() and warmed < WARMUP_FRAMES:
        ret, wf = cap.read()
        if not ret:
            break
        model.predict(wf, conf=CONF_THRESH, device=DEVICE, verbose=False)
        warmed += 1

    # ── Benchmark loop ──────────────────────────────────────────────────────
    det_counts:  List[int]   = []
    confidences: List[float] = []
    small_dets:  List[int]   = []
    latencies:   List[float] = []
    peak_ram_mb  = 0.0
    sampled_n    = 0
    total_read   = warmed

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        total_read += 1

        # Skip non-sampled frames (no inference, no video output)
        if total_read % SAMPLE_EVERY != 0:
            continue
        sampled_n += 1

        t0      = time.perf_counter()
        results = model.predict(frame, conf=CONF_THRESH, device=DEVICE, verbose=False)[0]
        latencies.append((time.perf_counter() - t0) * 1_000)

        pb = _filter_persons(results.boxes, person_classes)
        n  = len(pb) if pb is not None else 0
        det_counts.append(n)

        if n > 0:
            confidences.extend(pb.conf.cpu().numpy().tolist())
            areas = ((pb.xyxy[:, 2] - pb.xyxy[:, 0]) *
                     (pb.xyxy[:, 3] - pb.xyxy[:, 1])).cpu().numpy()
            small_dets.append(int((areas < SMALL_AREA).sum()))
        else:
            small_dets.append(0)

        peak_ram_mb = max(peak_ram_mb, psutil.Process().memory_info().rss / 1e6)

        out_video.write(annotate_detection_frame(frame, pb, model_key, total_read))

        if sampled_n % 500 == 0:
            pct = total_read / total_frm * 100
            log.info(f"  [{model_key}] {sampled_n} samples  "
                     f"(src {total_read}/{total_frm}  {pct:.0f}%)")

    cap.release()
    out_video.release()
    log.info(f"  Detection video → {out_dir / 'detection_output.mp4'}")

    # ── Cleanup ─────────────────────────────────────────────────────────────
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # ── Compute metrics ──────────────────────────────────────────────────────
    lat = np.array(latencies) if latencies else np.array([0.0])
    metrics = {
        "model":             model_key,
        "label":             cfg["label"],
        "frames_sampled":    sampled_n,
        # informational — NOT used to rank winner
        "avg_detections":    float(np.mean(det_counts))   if det_counts else 0.0,
        # used for ranking (quality indicators)
        "avg_confidence":    float(np.mean(confidences))   if confidences else 0.0,
        "median_confidence": float(np.median(confidences)) if confidences else 0.0,
        "avg_small_dets":    float(np.mean(small_dets))   if small_dets  else 0.0,
        "median_ms":         float(np.median(lat)),
        "p95_ms":            float(np.percentile(lat, 95)),
        "peak_ram_mb":       float(peak_ram_mb),
    }

    with open(out_dir / "detection_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)

    log.info(f"  avg_det={metrics['avg_detections']:.2f}  "
             f"avg_conf={metrics['avg_confidence']:.3f}  "
             f"small={metrics['avg_small_dets']:.2f}  "
             f"median_ms={metrics['median_ms']:.1f}")
    return metrics

# ---------------------------------------------------------------------------
# Top-2 selection (detection phase only)
# ---------------------------------------------------------------------------

def select_top_two(det_results: List[Dict]) -> List[str]:
    """
    Ranks models on three criteria — NO raw detection count.

    Score = 0.40 × avg_confidence_norm       (calibration quality)
           + 0.35 × avg_small_dets_norm      (small-person coverage)
           + 0.25 × speed_score              (1 - ms_norm, lower latency wins)

    The two highest-scoring models advance to Stage 3.
    """
    valid = [r for r in det_results if "error" not in r]
    if len(valid) <= 2:
        return [r["model"] for r in valid]

    max_conf  = max(r["avg_confidence"] for r in valid) or 1.0
    max_small = max(r["avg_small_dets"] for r in valid) or 1.0
    max_ms    = max(r["median_ms"]      for r in valid) or 1.0

    log.info("\nDetection ranking scores:")
    scored = []
    for r in valid:
        conf_n  = r["avg_confidence"] / max_conf
        small_n = r["avg_small_dets"] / max_small
        speed_s = 1.0 - (r["median_ms"] / max_ms)
        score   = 0.40 * conf_n + 0.35 * small_n + 0.25 * speed_s
        scored.append((score, r["model"]))
        log.info(f"  {r['model']:<25}  score={score:.4f}  "
                 f"conf_n={conf_n:.3f}  small_n={small_n:.3f}  speed_s={speed_s:.3f}")

    scored.sort(reverse=True)
    top2 = [m for _, m in scored[:2]]
    log.info(f"\n  → Top-2 advancing to tracking: {top2}")
    return top2

# ---------------------------------------------------------------------------
# Stage 3 — Tracking benchmark (top-2 only)
# ---------------------------------------------------------------------------

def run_tracking_benchmark(
    model_key: str,
    cfg: Dict,
    video_path: Path,
    output_base: Path,
) -> Dict:
    """
    Runs detection + ByteTrack on every SAMPLE_EVERY-th frame.
    Saves an annotated tracking video.
    Returns a rich tracking metrics dict.

    Key tracking metrics collected:
      avg_track_lifetime  — mean sampled-frames a track survived (longer = better)
      recovered_tracks    — tracks that disappeared then reappeared under same ID
      tracks_lost         — tracks that never reappeared (permanently dropped)
      track_birth_rate    — unique IDs created per 100 sampled frames (lower = stable)
      max_tracks          — peak simultaneous active tracks (crowding indicator)
    """
    sep = "=" * 62
    log.info(sep)
    log.info(f"  STAGE 3 · TRACKING · {model_key}")
    log.info(sep)

    out_dir        = output_base / model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    person_classes = cfg["person_classes"]

    # Load model via the shared resolver-aware loader (local or hub)
    model = load_yolo_model(str(cfg["path"]))
    model.to(DEVICE)
    tracker = BYTETracker(args=_BenchTrackerArgs())

    cap       = cv2.VideoCapture(str(video_path))
    total_frm = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps   = cap.get(cv2.CAP_PROP_FPS)
    W, H      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_fps   = max(1.0, src_fps / SAMPLE_EVERY)

    out_video = cv2.VideoWriter(
        str(out_dir / "tracking_output.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        out_fps, (W, H),
    )

    # ── Warmup ──────────────────────────────────────────────────────────────
    warmed = 0
    while cap.isOpened() and warmed < WARMUP_FRAMES:
        ret, wf = cap.read()
        if not ret:
            break
        model.predict(wf, conf=CONF_THRESH, device=DEVICE, verbose=False)
        warmed += 1

    # ── Per-track state ──────────────────────────────────────────────────────
    # sampled frame index (1-based) at which each track first / last appeared
    track_first:  Dict[int, int] = {}
    track_last:   Dict[int, int] = {}
    # all sampled frame indices where the track was seen (for gap detection)
    track_frames: Dict[int, List[int]] = defaultdict(list)

    counts_per_frame: List[int]   = []
    latencies:        List[float] = []
    peak_ram_mb   = 0.0
    sampled_n     = 0
    total_read    = warmed
    # current_lifetimes[id] = sampled frames elapsed since first seen (for annotation)
    current_lifetimes: Dict[int, int] = {}

    # ── Tracking loop ────────────────────────────────────────────────────────
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        total_read += 1
        if total_read % SAMPLE_EVERY != 0:
            continue
        sampled_n += 1

        t0      = time.perf_counter()
        results = model.predict(frame, conf=CONF_THRESH, device=DEVICE, verbose=False)[0]
        pb      = _filter_persons(results.boxes, person_classes)

        if pb is not None and len(pb) > 0:
            det    = pb.cpu().numpy()
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

        out_video.write(
            annotate_tracking_frame(frame, tracks, model_key, total_read, current_lifetimes)
        )

        if sampled_n % 500 == 0:
            pct = total_read / total_frm * 100
            log.info(f"  [{model_key}] {sampled_n} samples  "
                     f"(src {total_read}/{total_frm}  {pct:.0f}%)  "
                     f"active tracks: {len(tracks)}")

    last_sampled = sampled_n
    cap.release()
    out_video.release()
    log.info(f"  Tracking video → {out_dir / 'tracking_output.mp4'}")

    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # ── Post-process tracking metrics ────────────────────────────────────────
    all_ids = list(track_first.keys())
    total_unique_ids = len(all_ids)

    # Average / max track lifetime (sampled frames from first to last seen)
    lifetimes_f = [track_last[t] - track_first[t] for t in all_ids]
    avg_lifetime = float(np.mean(lifetimes_f)) if lifetimes_f else 0.0
    max_lifetime = float(max(lifetimes_f))     if lifetimes_f else 0.0

    # Recovered tracks: track had ≥ 1 gap of ≥ 2 sampled frames, then resumed
    # (same track_id reappeared after disappearing — ByteTrack successfully reacquired)
    recovered = 0
    for tid in all_ids:
        frames_sorted = track_frames[tid]  # already appended in order
        for i in range(1, len(frames_sorted)):
            if frames_sorted[i] - frames_sorted[i - 1] >= 2:
                recovered += 1
                break  # count each track once

    # Permanently lost tracks: last seen more than 10 sampled frames before video end
    LOST_GAP = 10
    tracks_lost = sum(
        1 for t in all_ids
        if track_last[t] < last_sampled - LOST_GAP
    )

    # Track birth rate (IDs born per 100 sampled frames) — lower = more stable IDs
    birth_rate = (total_unique_ids / last_sampled * 100) if last_sampled > 0 else 0.0

    lat = np.array(latencies) if latencies else np.array([0.0])
    metrics = {
        "model":              model_key,
        "label":              cfg["label"],
        "frames_sampled":     last_sampled,
        "avg_tracks":         float(np.mean(counts_per_frame))  if counts_per_frame else 0.0,
        "max_tracks":         int(max(counts_per_frame))         if counts_per_frame else 0,
        "total_unique_ids":   total_unique_ids,
        "track_birth_rate":   float(birth_rate),
        "avg_track_lifetime": avg_lifetime,
        "max_track_lifetime": max_lifetime,
        "recovered_tracks":   recovered,
        "tracks_lost":        tracks_lost,
        "median_ms":          float(np.median(lat)),
        "p95_ms":             float(np.percentile(lat, 95)),
        "peak_ram_mb":        float(peak_ram_mb),
    }

    with open(out_dir / "tracking_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)

    log.info(f"  avg_tracks={metrics['avg_tracks']:.2f}  "
             f"max_tracks={metrics['max_tracks']}  "
             f"avg_lifetime={avg_lifetime:.1f}f  "
             f"recovered={recovered}  lost={tracks_lost}  "
             f"birth_rate={birth_rate:.2f}/100f")
    return metrics

# ---------------------------------------------------------------------------
# Console tables
# ---------------------------------------------------------------------------

def print_detection_table(results: List[Dict]) -> None:
    bar = "=" * 96
    print(f"\n{bar}")
    print("  STAGE 2 — DETECTION BENCHMARK RESULTS")
    print(f"  Note: avg_detections is informational only — NOT the ranking criterion.")
    print(bar)

    cw = [24, 7, 8, 9, 9, 10, 9, 11]
    cols = ["Model", "Frames", "Avg Det*", "Avg Conf", "Med Conf", "Avg Small", "Med ms", "Peak RAM MB"]
    print("  " + "  ".join(f"{c:<{w}}" for c, w in zip(cols, cw)))
    print("  " + "-" * 92)

    for r in results:
        if "error" in r:
            print(f"  {r['model']:<24}  [ERROR: {r['error'][:50]}]")
            continue
        vals = [
            r["model"],
            str(r["frames_sampled"]),
            f"{r['avg_detections']:.2f}",
            f"{r['avg_confidence']:.3f}",
            f"{r['median_confidence']:.3f}",
            f"{r['avg_small_dets']:.2f}",
            f"{r['median_ms']:.1f}",
            f"{r['peak_ram_mb']:.0f}",
        ]
        print("  " + "  ".join(f"{v:<{w}}" for v, w in zip(vals, cw)))

    print(f"{bar}")
    print("  * avg_detections is NOT used to select the winner (see benchmark docs).\n")


def print_tracking_table(results: List[Dict]) -> None:
    bar = "=" * 108
    print(f"\n{bar}")
    print("  STAGE 3 — TRACKING BENCHMARK RESULTS  (Top-2 only)")
    print(bar)

    cw = [24, 7, 10, 5, 10, 12, 12, 9, 6, 9]
    cols = ["Model", "Frames", "Avg Tracks", "Max", "Unique IDs",
            "Birth/100f ↓", "Avg Life ↑", "Recovered ↑", "Lost ↓", "Med ms"]
    print("  " + "  ".join(f"{c:<{w}}" for c, w in zip(cols, cw)))
    print("  " + "-" * 104)

    for r in results:
        vals = [
            r["model"],
            str(r["frames_sampled"]),
            f"{r['avg_tracks']:.2f}",
            str(r["max_tracks"]),
            str(r["total_unique_ids"]),
            f"{r['track_birth_rate']:.2f}",
            f"{r['avg_track_lifetime']:.1f}",
            str(r["recovered_tracks"]),
            str(r["tracks_lost"]),
            f"{r['median_ms']:.1f}",
        ]
        print("  " + "  ".join(f"{v:<{w}}" for v, w in zip(vals, cw)))
    print(f"{bar}\n")

# ---------------------------------------------------------------------------
# Qualitative recommendation
# ---------------------------------------------------------------------------

def generate_recommendation(
    det_results: List[Dict],
    track_results: List[Dict],
    top2: List[str],
) -> str:
    """
    Returns a multi-line qualitative recommendation string.
    Winner is selected on a weighted tracking score.
    The reasoning is written as explicit sentences, not just a label.
    """
    lines = ["\n" + "=" * 70, "  RECOMMENDATION", "=" * 70]

    valid = [r for r in track_results if "avg_track_lifetime" in r]
    if not valid:
        lines.append("  [ERROR] No tracking results — cannot produce recommendation.")
        return "\n".join(lines)

    # Weighted tracking score
    max_life  = max(r["avg_track_lifetime"] for r in valid) or 1.0
    max_rec   = max(r["recovered_tracks"]   for r in valid) or 1.0
    max_birth = max(r["track_birth_rate"]   for r in valid) or 1.0
    max_trk   = max(r["avg_tracks"]         for r in valid) or 1.0

    scored = []
    for r in valid:
        lifetime_s  =  r["avg_track_lifetime"] / max_life
        recovery_s  =  r["recovered_tracks"]   / max_rec
        # lower birth rate → more stable → higher score
        stability_s = (1.0 - r["track_birth_rate"] / max_birth)
        coverage_s  =  r["avg_tracks"] / max_trk
        score = (0.35 * lifetime_s
               + 0.25 * recovery_s
               + 0.25 * stability_s
               + 0.15 * coverage_s)
        scored.append((score, r))
    scored.sort(key=lambda x: x[0], reverse=True)

    winner_score, W = scored[0]
    runner_score, R = scored[1] if len(scored) > 1 else (None, None)

    lines.append(f"\n  WINNER  :  {W['model']}")
    lines.append(f"  Label   :  {W['label']}")
    lines.append(f"  Score   :  {winner_score:.4f}" +
                 (f"  (runner-up {R['model']}: {runner_score:.4f})" if R else ""))

    lines.append("\n  Reasons:\n")

    # ── Track lifetime ──
    if R and W["avg_track_lifetime"] >= R["avg_track_lifetime"]:
        delta = W["avg_track_lifetime"] - R["avg_track_lifetime"]
        lines.append(
            f"  ✓  Longest average track lifetime — {W['avg_track_lifetime']:.1f} sampled frames "
            f"vs {R['avg_track_lifetime']:.1f} for {R['model']} (Δ {delta:.1f} frames).\n"
            f"     Longer lifetimes mean each person stays under the same ID across more of\n"
            f"     the video, which is the single most important tracking quality metric."
        )
    else:
        lines.append(
            f"  ~  Track lifetime {W['avg_track_lifetime']:.1f} sampled frames — "
            f"not the longest but within acceptable range given other strengths."
        )

    # ── Recovery ──
    if R and W["recovered_tracks"] >= R["recovered_tracks"]:
        lines.append(
            f"\n  ✓  Best track recovery — {W['recovered_tracks']} tracks reacquired "
            f"vs {R['recovered_tracks']} for {R['model']}.\n"
            f"     ByteTrack re-linked persons after occlusion or missed detections more\n"
            f"     often, reducing the number of ID splits in crowded restaurant scenes."
        )
    else:
        lines.append(
            f"\n  ~  Recovery ({W['recovered_tracks']}) is similar to {R['model'] if R else 'N/A'}. "
            f"Not a differentiator here."
        )

    # ── Stability (birth rate) ──
    if R and W["track_birth_rate"] <= R["track_birth_rate"]:
        lines.append(
            f"\n  ✓  Most stable ID assignment — birth rate {W['track_birth_rate']:.2f} IDs/100 frames "
            f"vs {R['track_birth_rate']:.2f} for {R['model']}.\n"
            f"     A lower birth rate means fewer spurious track creations, indicating the\n"
            f"     detector produces cleaner, more consistent detections for ByteTrack."
        )
    else:
        lines.append(
            f"\n  ~  Birth rate {W['track_birth_rate']:.2f}/100f is slightly elevated vs "
            f"{R['track_birth_rate']:.2f}/100f for {R['model'] if R else 'N/A'}.\n"
            f"     Some ID fragmentation is occurring — monitor in the full-length run."
        )

    # ── Lost tracks ──
    if R and W["tracks_lost"] <= R["tracks_lost"]:
        lines.append(
            f"\n  ✓  Fewest permanently lost tracks — {W['tracks_lost']} vs "
            f"{R['tracks_lost']} for {R['model']}.\n"
            f"     Fewer lost tracks means persons remain trackable across the full clip."
        )
    elif W["tracks_lost"] > 0:
        lines.append(
            f"\n  ~  {W['tracks_lost']} tracks were permanently lost before video end\n"
            f"     (same as or slightly more than the runner-up). Acceptable for now."
        )

    # ── Speed ──
    rt = "real-time capable" if W["median_ms"] < 100 else "suitable for offline batch processing"
    lines.append(
        f"\n  ✓  Inference speed — {W['median_ms']:.1f} ms median / "
        f"{W['p95_ms']:.1f} ms p95 on {DEVICE.upper()}.\n"
        f"     {rt.capitalize()} at this frame rate."
    )

    # ── Caveats ──
    caveats = []
    if W["tracks_lost"] > 5:
        caveats.append(
            f"  ⚠  {W['tracks_lost']} permanently lost tracks — consider increasing "
            f"track_buffer or decreasing match_thresh to improve track persistence."
        )
    if W["track_birth_rate"] > 5.0:
        caveats.append(
            f"  ⚠  Birth rate {W['track_birth_rate']:.2f}/100f is elevated. "
            f"Tuning track_high_thresh may reduce ID fragmentation."
        )
    if W["median_ms"] > 120:
        alt = top2[1] if len(top2) > 1 and top2[0] == W["model"] else top2[0]
        caveats.append(
            f"  ⚠  Inference is slow ({W['median_ms']:.0f} ms/frame). "
            f"If real-time is later required, benchmark {alt} with optimised settings."
        )
    if caveats:
        lines.append("\n  Caveats:")
        lines.extend(caveats)

    # ── Next step ──
    lines.append(
        f"\n  Next step:\n"
        f"  Run a FULL validation across all {total_frm_label} frames of Dark_lighting.mp4\n"
        f"  with {W['model']} before promoting to production. This benchmark sampled\n"
        f"  every {SAMPLE_EVERY}rd frame; the full run is your production validation.\n"
        f"\n  Command: python run.py  (after updating configs/config.yaml model_path)"
    )

    lines.append("=" * 70)
    return "\n".join(lines)


# Placeholder filled at runtime
total_frm_label = "17,815"

# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

def save_report(
    det_results:   List[Dict],
    track_results: List[Dict],
    top2:          List[str],
    recommendation: str,
    output_base:   Path,
) -> None:
    """Writes detection_comparison.csv, tracking_comparison.csv, benchmark_report.md."""

    # Detection CSV
    det_fields = ["model", "frames_sampled", "avg_detections",
                  "avg_confidence", "median_confidence", "avg_small_dets",
                  "median_ms", "p95_ms", "peak_ram_mb"]
    with open(output_base / "detection_comparison.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=det_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(r for r in det_results if "error" not in r)

    # Tracking CSV
    if track_results:
        trk_fields = ["model", "frames_sampled", "avg_tracks", "max_tracks",
                      "total_unique_ids", "track_birth_rate", "avg_track_lifetime",
                      "max_track_lifetime", "recovered_tracks", "tracks_lost",
                      "median_ms", "p95_ms", "peak_ram_mb"]
        with open(output_base / "tracking_comparison.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=trk_fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(track_results)

    # Markdown report
    md = [
        "# Aurika Tracking v2 — Benchmark Report",
        "",
        f"> Video: `{VIDEO_PATH.name}` | Sample rate: every {SAMPLE_EVERY}rd frame "
        f"| Device: `{DEVICE.upper()}`",
        "",
        "## Stage 2 — Detection Results",
        "",
        "> `avg_detections` is **informational only** — it is NOT used to rank models.",
        "",
        "| Model | Frames | Avg Det* | Avg Conf | Med Conf | Avg Small | Med ms | Peak RAM |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in det_results:
        if "error" not in r:
            md.append(
                f"| {r['model']} | {r['frames_sampled']} | {r['avg_detections']:.2f} | "
                f"{r['avg_confidence']:.3f} | {r['median_confidence']:.3f} | "
                f"{r['avg_small_dets']:.2f} | {r['median_ms']:.1f} ms | {r['peak_ram_mb']:.0f} MB |"
            )
        else:
            md.append(f"| {r['model']} | — | — | — | — | — | — | ERROR |")

    md += [
        "",
        f"**Top-2 advancing to Stage 3 (tracking):** `{'`, `'.join(top2)}`",
        "",
        "## Stage 3 — Tracking Results",
        "",
        "| Model | Frames | Avg Tracks | Max | Unique IDs | Birth/100f ↓ | "
        "Avg Lifetime ↑ | Recovered ↑ | Lost ↓ | Med ms |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in track_results:
        md.append(
            f"| {r['model']} | {r['frames_sampled']} | {r['avg_tracks']:.2f} | "
            f"{r['max_tracks']} | {r['total_unique_ids']} | {r['track_birth_rate']:.2f} | "
            f"{r['avg_track_lifetime']:.1f} | {r['recovered_tracks']} | "
            f"{r['tracks_lost']} | {r['median_ms']:.1f} ms |"
        )

    md += ["", "## Recommendation", "", "```", recommendation, "```", ""]

    with open(output_base / "benchmark_report.md", "w") as fh:
        fh.write("\n".join(md))

    log.info(f"\n  Full report → {output_base / 'benchmark_report.md'}")
    log.info(f"  Detection CSV → {output_base / 'detection_comparison.csv'}")
    if track_results:
        log.info(f"  Tracking CSV  → {output_base / 'tracking_comparison.csv'}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global total_frm_label

    log.info("╔══════════════════════════════════════════╗")
    log.info("║  Aurika Tracking v2 — Detector Benchmark ║")
    log.info("╚══════════════════════════════════════════╝")
    log.info(f"Device      : {DEVICE.upper()}")
    log.info(f"Video       : {VIDEO_PATH}  ({VIDEO_PATH.stat().st_size / 1e6:.1f} MB)")
    log.info(f"Sample rate : every {SAMPLE_EVERY}rd frame")
    log.info(f"Models      : {list(MODELS.keys())}")
    log.info(f"Output      : {OUTPUT_BASE}\n")

    if not VIDEO_PATH.exists():
        log.error(f"Video not found: {VIDEO_PATH}")
        raise SystemExit(1)

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    total_frm_label = f"{int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):,}"
    cap.release()

    # ── Stage 2: Detection benchmark on all models ──────────────────────────
    det_results: List[Dict] = []
    for key, cfg in MODELS.items():
        det_results.append(
            run_detection_benchmark(key, cfg, VIDEO_PATH, OUTPUT_BASE)
        )

    print_detection_table(det_results)

    # ── Select top-2 ─────────────────────────────────────────────────────────
    top2 = select_top_two(det_results)

    if not top2:
        log.error("No valid models to advance to tracking. Exiting.")
        raise SystemExit(1)

    # ── Stage 3: Tracking benchmark on top-2 ─────────────────────────────────
    track_results: List[Dict] = []
    for key in top2:
        track_results.append(
            run_tracking_benchmark(key, MODELS[key], VIDEO_PATH, OUTPUT_BASE)
        )

    print_tracking_table(track_results)

    # ── Recommendation ────────────────────────────────────────────────────────
    rec = generate_recommendation(det_results, track_results, top2)
    print(rec)

    # ── Save all outputs ──────────────────────────────────────────────────────
    save_report(det_results, track_results, top2, rec, OUTPUT_BASE)

    print(f"\n  All outputs written to: {OUTPUT_BASE}")
    print("  Review the annotated videos in each model's subfolder,")
    print("  then run the full 17,815-frame pipeline with the winner.\n")


if __name__ == "__main__":
    main()
