# tracker/analytics_engine.py
"""
Aurika Tracking v2 — Restaurant Analytics Engine
Integrates the Track Memory Layer, parses geometric zone boundaries, calculates live occupancy/counts,
and draws dashboard analytics overlays.
"""

import csv
import logging
import time
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from tracker.track_memory import TrackMemoryEngine, TrackState
from tracker.events import BaseEvent, EnteredRestaurant, ExitedRestaurant

logger = logging.getLogger("AurikaTracking")

def check_line_intersection(p1: Tuple[float, float], p2: Tuple[float, float], q1: Tuple[float, float], q2: Tuple[float, float]) -> bool:
    """Checks if line segment p1p2 intersects with q1q2."""
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2)

class RestaurantAnalyticsEngine:
    """
    Orchestrates all restaurant analytics operations on tracking outputs.
    Refactored to utilize TrackMemoryEngine as the single source of truth.
    """
    def __init__(self, zones_config_path: str, width: int = 1280, height: int = 720, fps: float = 29.97):
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_time = 1.0 / fps if fps > 0 else 0.033

        # Load layout configuration via decoupled SemanticZoneEngine
        from tracker.semantic_zone_engine import SemanticZoneEngine
        self.zone_engine = SemanticZoneEngine(zones_config_path, width=self.width, height=self.height)
        self.zones = self.zone_engine.zones
        
        # Populate counting lines (with counted_ids sets for tracking crossings)
        self.counting_lines = []
        for line in self.zone_engine.counting_lines:
            self.counting_lines.append({
                "name": line["name"],
                "color": line["color"],
                "pt1": line["p1"],
                "pt2": line["p2"],
                "direction": line["direction"],
                "counted_ids": set()
            })

        # Load yaml for tracker settings
        with open(zones_config_path, "r") as f:
            self.cfg = yaml.safe_load(f) or {}

        # Track Memory Engine instantiation
        track_buffer = self.cfg.get("tracker", {}).get("track_buffer", 30)
        hysteresis = self.cfg.get("tracker", {}).get("zone_hysteresis_frames", 5)
        self.memory_engine = TrackMemoryEngine(track_buffer=track_buffer, frame_time=self.frame_time, zone_hysteresis_frames=hysteresis)

        # State Variables (Backward compatibility)
        self.track_history: Dict[int, List[Tuple[float, float]]] = {}  # track_id -> list of centers
        self.track_last_zone: Dict[int, str] = {}  # track_id -> last known zone name
        self.track_zone_entry_frame: Dict[int, int] = {}  # track_id -> frame index when entering current zone
        self.track_zone_dwell_frames: Dict[int, Dict[str, int]] = {}  # track_id -> zone -> count of frames
        self.track_first_frame: Dict[int, int] = {}  # track_id -> first frame seen
        self.track_last_frame: Dict[int, int] = {}  # track_id -> last frame seen
        self.transitions: Dict[int, List[str]] = {}  # track_id -> sequence of unique zones visited
        self.track_roles: Dict[int, str] = {}  # track_id -> role string ("Customer" or "Staff")

        # Global Counters
        self.entries_count = 0
        self.exits_count = 0
        self.current_occupants: Set[int] = set()

        # Events log
        self.events: List[Dict[str, Any]] = []
        self.label_overlap_count = 0

        # Heatmap accumulator
        self.heatmap_accum = np.zeros((height, width), dtype=np.float32)

        # History for occupancy logs
        self.occupancy_history: List[Dict[str, Any]] = []

    def get_track_center(self, track: np.ndarray) -> Tuple[float, float]:
        """Calculates bottom center (feet position) of bounding box for zone detection."""
        x1, y1, x2, y2 = track[:4]
        return ((x1 + x2) / 2.0, y2)

    def get_zone_at_point(self, pt: Tuple[float, float]) -> Optional[str]:
        """Returns name of the zone containing the point, or None."""
        return self.zone_engine.get_zone_at_point(pt)

    def update_frame(self, frame_idx: int, timestamp: float, tracks: np.ndarray) -> None:
        """Processes a frame's tracking data, updating TrackMemoryEngine and syncs analytics state."""
        # 1. Update memory layer
        active_states = self.memory_engine.update(frame_idx, timestamp, tracks, self.get_zone_at_point)
        
        current_frame_ids = set()
        zone_counts = {zone["name"]: 0 for zone in self.zones}

        # 2. Accumulate Heatmaps & Sync State Variables
        for track in tracks:
            tid = int(track[4])
            current_frame_ids.add(tid)
            self.current_occupants.add(tid)

            pt = self.get_track_center(track)
            px, py = int(clip(pt[0], 0, self.width - 1)), int(clip(pt[1], 0, self.height - 1))
            self.heatmap_accum[py, px] += 1.0

            # Line crossing checks
            state = self.memory_engine.get_track(tid)
            if state and len(state.trajectory) > 1:
                prev_pt = state.trajectory[-2]
                for line in self.counting_lines:
                    if tid not in line["counted_ids"]:
                        if check_line_intersection(prev_pt, pt, line["pt1"], line["pt2"]):
                            is_vertical = abs(line["pt2"][0] - line["pt1"][0]) <= abs(line["pt2"][1] - line["pt1"][1])
                            if is_vertical:
                                x_line = (line["pt1"][0] + line["pt2"][0]) / 2.0
                                if line["direction"] == "in":
                                    if not (prev_pt[0] <= x_line and pt[0] > x_line):
                                        continue
                                else:
                                    if not (prev_pt[0] >= x_line and pt[0] < x_line):
                                        continue
                            else:
                                y_line = (line["pt1"][1] + line["pt2"][1]) / 2.0
                                if line["direction"] == "in":
                                    if not (prev_pt[1] <= y_line and pt[1] > y_line):
                                        continue
                                else:
                                    if not (prev_pt[1] >= y_line and pt[1] < y_line):
                                        continue

                            line["counted_ids"].add(tid)
                            self.memory_engine.events_stream.append(
                                EnteredRestaurant(frame_idx, timestamp, tid, {}) if line["direction"] == "in" else ExitedRestaurant(frame_idx, timestamp, tid, {})
                            )

        # 3. Synchronize Active & Archived track structures for backward compatibility
        all_states = list(self.memory_engine.active_tracks.values()) + list(self.memory_engine.archived_tracks.values())
        for state in all_states:
            tid = state.track_id
            self.track_history[tid] = state.trajectory
            self.track_last_zone[tid] = state.current_zone
            self.track_first_frame[tid] = state.first_frame
            self.track_last_frame[tid] = state.last_frame
            self.transitions[tid] = state.zone_history
            self.track_roles[tid] = state.role

            # Calculate zone dwell times in frames
            self.track_zone_dwell_frames[tid] = {}
            for zone in self.zones:
                name = zone["name"]
                dwell_sec = sum(state.zone_dwell_times.get(name, []))
                # Add ongoing active zone time
                if state.current_zone == name:
                    entry_t = state.zone_entry_times[name][-1] if state.zone_entry_times.get(name) else state.last_timestamp
                    dwell_sec += (timestamp - entry_t)
                self.track_zone_dwell_frames[tid][name] = int(dwell_sec / self.frame_time)

            if state.current_zone in zone_counts:
                zone_counts[state.current_zone] += 1

        # 4. Sync Global counters & Clean occupants list
        self.entries_count = sum(1 for s in all_states if s.entered_restaurant)
        self.exits_count = sum(1 for s in all_states if s.exit_detected)
        self.current_occupants = {tid for tid, s in self.memory_engine.active_tracks.items()}

        # 5. Populate occupancy history
        log_entry = {"frame": frame_idx, "timestamp": timestamp}
        log_entry.update(zone_counts)
        self.occupancy_history.append(log_entry)

        # 6. Reconstruct backward compatible events dictionary
        self.events = []
        for e in self.memory_engine.events_stream:
            zone_name = e.metadata.get("zone", "")
            duration = e.metadata.get("dwell_seconds", 0.0)
            self.events.append({
                "timestamp": e.timestamp,
                "track_id": e.track_id,
                "event": e.event,
                "zone": zone_name,
                "duration": duration,
                "frame": e.frame
            })

    def draw_analytics_overlay(self, frame: np.ndarray, tracks: np.ndarray, current_fps: float, frame_idx: int) -> np.ndarray:
        """Paints polygon zones, counting lines, live occupancy overlays, and live dwell times."""
        overlay = frame.copy()
        
        # Draw Zones
        for zone in self.zones:
            cv2.fillPoly(overlay, [zone["polygon"]], zone["color"])
            cv2.polylines(frame, [zone["polygon"]], True, zone["color"], 2)
            
            M = cv2.moments(zone["polygon"])
            cx = int(M["m10"] / M["m00"]) if M["m00"] > 0 else zone["polygon"][0][0]
            cy = int(M["m01"] / M["m00"]) if M["m00"] > 0 else zone["polygon"][0][1]

            occ_count = sum(1 for track in tracks if self.get_zone_at_point(self.get_track_center(track)) == zone["name"])
            label = f"{zone['name']}: {occ_count}"
            cv2.putText(frame, label, (cx - 40, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Apply transparency to polygons
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

        # Draw Counting Lines
        for line in self.counting_lines:
            cv2.line(frame, line["pt1"], line["pt2"], line["color"], 3)
            mid_x = int((line["pt1"][0] + line["pt2"][0]) / 2)
            mid_y = int((line["pt1"][1] + line["pt2"][1]) / 2)
            cv2.putText(frame, f"{line['name']}: {len(line['counted_ids'])}", (mid_x - 50, mid_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, line["color"], 2, cv2.LINE_AA)

        # Draw individual tracks with live dwell times in their current zone (from memory layer)
        drawn_label_rects = []
        for track in tracks:
            x1, y1, x2, y2 = map(int, track[:4])
            tid = int(track[4])
            
            state = self.memory_engine.get_track(tid)
            if not state:
                continue

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
            
            # Style label as ID {tid} [{role}] {visit_sec}s Q:{qs:.2f}
            label = f"ID {tid} [{state.role}] {int(state.visit_duration)}s Q:{state.quality_score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            
            lx = x1
            ly = y1 - 5
            if ly < 15:
                ly = y2 + 15
                
            collides = True
            shifts = 0
            while collides and shifts < 3:
                collides = False
                for rx, ry, rw, rh in drawn_label_rects:
                    if not (lx + tw < rx or rx + rw < lx or ly + 3 < ry or ry + rh < ly - th):
                        collides = True
                        ly += 15
                        shifts += 1
                        break
            
            if collides:
                self.label_overlap_count += 1

            drawn_label_rects.append((lx, ly - th, tw, th + 3))
            cv2.putText(frame, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        # Premium dashboard sidebar (semi-transparent panel at top right)
        dash_w, dash_h = 320, 240
        dash_x, dash_y = self.width - dash_w - 20, 20
        sub_frame = frame[dash_y:dash_y+dash_h, dash_x:dash_x+dash_w]
        
        black_rect = np.zeros(sub_frame.shape, dtype=np.uint8)
        cv2.addWeighted(sub_frame, 0.4, black_rect, 0.6, 0, sub_frame)
        frame[dash_y:dash_y+dash_h, dash_x:dash_x+dash_w] = sub_frame

        # Compute dynamic counts from memory layer
        all_states = list(self.memory_engine.active_tracks.values()) + list(self.memory_engine.archived_tracks.values())
        customers_in_frame = sum(1 for s in self.memory_engine.active_tracks.values() if s.role == "Customer" and s.status != "TEMP_OCCLUDED")
        staff_in_frame = sum(1 for s in self.memory_engine.active_tracks.values() if s.role == "Staff" and s.status != "TEMP_OCCLUDED")
        
        # Calculations for premium metrics
        avg_visit = np.mean([s.visit_duration for s in all_states]) if all_states else 0.0
        
        waiting_dwells = []
        for s in all_states:
            w_dwell = sum(s.zone_dwell_times.get("Waiting Area", [0]) + s.zone_dwell_times.get("Waiting", [0]))
            if w_dwell > 0:
                waiting_dwells.append(w_dwell)
        avg_wait = np.mean(waiting_dwells) if waiting_dwells else 0.0

        active_recoveries = sum(s.recovery_count for s in all_states)
        archived_count = len(self.memory_engine.archived_tracks)

        # Draw Dashboard panel
        cv2.rectangle(frame, (dash_x, dash_y), (dash_x+dash_w, dash_y+dash_h), (255, 255, 255), 1)
        cv2.putText(frame, "AURIKA STATE DASHBOARD", (dash_x + 15, dash_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Active Customers   : {customers_in_frame}", (dash_x + 15, dash_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Active Staff       : {staff_in_frame}", (dash_x + 15, dash_y + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Avg Visit Duration : {int(avg_visit)}s", (dash_x + 15, dash_y + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Avg Waiting Time   : {int(avg_wait)}s", (dash_x + 15, dash_y + 135), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Active Recoveries  : {active_recoveries}", (dash_x + 15, dash_y + 160), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 128, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Archived Tracks    : {archived_count}", (dash_x + 15, dash_y + 185), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Total Occupants    : {len(tracks)}", (dash_x + 15, dash_y + 210), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        return frame

    def save_heatmap(self, output_path: Path) -> None:
        """Applies blur and jet color-mapping on accumulated coordinates and writes heatmap.png."""
        if np.max(self.heatmap_accum) == 0:
            cv2.imwrite(str(output_path), np.zeros((self.height, self.width, 3), dtype=np.uint8))
            return

        smoothed = cv2.GaussianBlur(self.heatmap_accum, (15, 15), 0)
        norm = np.clip((smoothed / np.max(smoothed)) * 255, 0, 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        cv2.imwrite(str(output_path), heatmap)
        logger.info(f"Saved trajectory heatmap to {output_path}")

    def export_csv_data(self, output_dir: Path) -> None:
        """Exports analytics metrics to Events, Occupancy, Dwell Times, and Zone Stats CSVs."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Export events.csv
        with open(output_dir / "events.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "track_id", "event", "zone", "duration", "frame"])
            for evt in self.events:
                writer.writerow([
                    f"{evt['timestamp']:.2f}", evt["track_id"], evt["event"],
                    evt["zone"], f"{evt['duration']:.2f}", evt["frame"]
                ])

        # 2. Export occupancy.csv
        with open(output_dir / "occupancy.csv", "w", newline="") as f:
            writer = csv.writer(f)
            zone_names = [z["name"] for z in self.zones]
            writer.writerow(["frame", "timestamp"] + zone_names)
            for entry in self.occupancy_history:
                row = [entry["frame"], f"{entry['timestamp']:.2f}"]
                for name in zone_names:
                    row.append(entry[name])
                writer.writerow(row)

        # 3. Export dwell_times.csv
        with open(output_dir / "dwell_times.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["track_id", "entry_frame", "exit_frame", "total_dwell_seconds"])
            for tid in self.track_first_frame:
                duration = (self.track_last_frame[tid] - self.track_first_frame[tid]) * self.frame_time
                writer.writerow([tid, self.track_first_frame[tid], self.track_last_frame[tid], f"{duration:.2f}"])

        # 4. Export zone_statistics.csv (transition counts & average zone dwell times)
        with open(output_dir / "zone_statistics.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "details", "value"])

            trans_pairs: Dict[Tuple[str, str], int] = defaultdict_int()
            for tid, path in self.transitions.items():
                for i in range(1, len(path)):
                    trans_pairs[(path[i-1], path[i])] += 1

            for (z_from, z_to), count in trans_pairs.items():
                writer.writerow(["transition", f"{z_from} -> {z_to}", count])

            for zone in self.zones:
                dwells = []
                for tid in self.track_zone_dwell_frames:
                    df = self.track_zone_dwell_frames[tid].get(zone["name"], 0)
                    if df > 0:
                        dwells.append(df * self.frame_time)
                avg_dwell = np.mean(dwells) if dwells else 0.0
                writer.writerow(["average_dwell_time", zone["name"], f"{avg_dwell:.2f}s"])


def clip(val, min_val, max_val):
    return min(max(val, min_val), max_val)

def defaultdict_int():
    class DefaultDict(dict):
        def __missing__(self, key):
            self[key] = 0
            return 0
    return DefaultDict()
