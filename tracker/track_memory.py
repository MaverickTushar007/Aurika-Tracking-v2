# tracker/track_memory.py
"""
Aurika Tracking v2 — Track Memory Layer & State Machine
Maintains persistent, long-lived TrackState objects and coordinates
lifecycle state transitions, motion analytics, and customer journeys.
Now integrates zone hysteresis logic.
"""

import math
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional, Set
from tracker.events import (
    BaseEvent, TrackCreated, TrackConfirmed, TrackLost, TrackRecovered, TrackArchived,
    EnteredRestaurant, ExitedRestaurant, ZoneEntered, ZoneExited, RoleAssigned, RoleChanged,
    WaitingStarted, WaitingFinished, DiningStarted, DiningFinished
)

logger = logging.getLogger("AurikaTracking")

VALID_TRANSITIONS = {
    "NEW": {"CONFIRMED", "TEMP_OCCLUDED", "EXITED"},
    "CONFIRMED": {"ACTIVE", "TEMP_OCCLUDED", "EXITED"},
    "ACTIVE": {"TEMP_OCCLUDED", "EXITED"},
    "TEMP_OCCLUDED": {"RECOVERED", "EXITED"},
    "RECOVERED": {"ACTIVE", "TEMP_OCCLUDED", "EXITED"},
    "EXITED": {"ARCHIVED"},
    "ARCHIVED": set()
}

class TrackState:
    """Persistent representation of a tracked individual's state and journey."""
    def __init__(self, track_id: int, frame: int, timestamp: float, bbox: Tuple[float, float, float, float], confidence: float, frame_time: float):
        self.track_id = track_id
        self.role = "Customer"
        self.status = "NEW"
        
        self.first_frame = frame
        self.last_frame = frame
        self.first_timestamp = timestamp
        self.last_timestamp = timestamp
        self.frame_time = frame_time
        
        self.age_frames = 1
        self.age_seconds = frame_time
        
        self.current_bbox = bbox
        self.bbox_history: List[Tuple[float, float, float, float]] = [bbox]
        
        self.confidence = confidence
        self.confidence_history: List[float] = [confidence]
        self.ema_confidence = confidence
        
        self.quality_score = 0.50
        self.visibility_ratio = 1.0
        self.detections_count = 1
        
        self.occlusion_count = 0
        self.recovery_count = 0
        self.lost_count = 0
        
        # Zone tracking fields
        self.current_zone: Optional[str] = None
        self.previous_zone: Optional[str] = None
        self.zone_history: List[str] = []
        
        self.zone_entry_times: Dict[str, List[float]] = {}
        self.zone_exit_times: Dict[str, List[float]] = {}
        self.zone_dwell_times: Dict[str, List[float]] = {}
        
        # Zone hysteresis fields
        self.zone_confidence = 1.0
        self.candidate_zone: Optional[str] = None
        self.frames_inside_candidate = 0
        self.zone_hysteresis_frames = 5
        
        self.entered_restaurant = False
        self.exit_detected = False
        
        self.entry_time = timestamp
        self.exit_time: Optional[float] = None
        self.visit_duration = 0.0
        
        # Motion fields
        cx, cy = (bbox[0] + bbox[2]) / 2.0, bbox[3] # feet position center
        self.trajectory: List[Tuple[float, float]] = [(cx, cy)]
        self.total_distance = 0.0
        self.average_speed = 0.0
        self.maximum_speed = 0.0
        self.average_direction = 0.0
        
        self.velocity_history: List[Tuple[float, float]] = []
        self.direction_angles: List[float] = []
        
        self.is_active = True
        self.metadata: Dict[str, Any] = {}

    def transition_to(self, new_state: str, frame: int, timestamp: float, reason: str, events_stream: List[BaseEvent]) -> None:
        """Executes a deterministic state transition, validating validity and logging events."""
        if new_state not in VALID_TRANSITIONS[self.status]:
            logger.warning(
                f"Illegal state transition requested: {self.status} ➔ {new_state} "
                f"for ID {self.track_id}. Skipping transition."
            )
            return

        old_state = self.status
        self.status = new_state
        logger.info(f"ID {self.track_id} transitioned: {old_state} ➔ {new_state} ({reason})")

        # Emit corresponding events
        if new_state == "CONFIRMED":
            events_stream.append(TrackConfirmed(frame, timestamp, self.track_id, {"reason": reason}))
        elif new_state == "TEMP_OCCLUDED":
            self.occlusion_count += 1
            events_stream.append(TrackLost(frame, timestamp, self.track_id, {"reason": reason}))
        elif new_state == "RECOVERED":
            self.recovery_count += 1
            events_stream.append(TrackRecovered(frame, timestamp, self.track_id, {"reason": reason}))
        elif new_state == "EXITED":
            self.is_active = False
            self.exit_detected = True
            self.exit_time = timestamp
            events_stream.append(ExitedRestaurant(frame, timestamp, self.track_id, {"reason": reason}))
        elif new_state == "ARCHIVED":
            self.is_active = False
            events_stream.append(TrackArchived(frame, timestamp, self.track_id, {"reason": reason}))

    def update(self, frame: int, timestamp: float, bbox: Tuple[float, float, float, float], confidence: float, zone_name: Optional[str], events_stream: List[BaseEvent], zone_hysteresis_frames: int = 5) -> None:
        """Updates internal behavioral and motion stats on a detection match frame."""
        self.last_frame = frame
        self.last_timestamp = timestamp
        self.age_frames += 1
        self.age_seconds = self.age_frames * self.frame_time
        self.visit_duration = timestamp - self.first_timestamp

        # Bounding box & confidence history
        self.current_bbox = bbox
        self.bbox_history.append(bbox)
        
        self.confidence = confidence
        self.confidence_history.append(confidence)
        self.detections_count += 1
        self.visibility_ratio = self.detections_count / self.age_frames

        # EMA Confidence
        alpha = 0.3
        self.ema_confidence = alpha * confidence + (1.0 - alpha) * self.ema_confidence

        # Motion & Velocity Analytics
        cx, cy = (bbox[0] + bbox[2]) / 2.0, bbox[3]
        prev_cx, prev_cy = self.trajectory[-1]
        self.trajectory.append((cx, cy))

        dt = max(0.01, timestamp - self.last_timestamp)
        vx = (cx - prev_cx) / dt
        vy = (cy - prev_cy) / dt
        self.velocity_history.append((vx, vy))

        speed = math.sqrt(vx**2 + vy**2)
        self.maximum_speed = max(self.maximum_speed, speed)
        
        dist = math.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
        self.total_distance += dist
        self.average_speed = self.total_distance / max(0.1, self.visit_duration)

        # Direction calculation (degrees)
        if dist > 1.0:
            angle = math.degrees(math.atan2(vy, vx))
            self.direction_angles.append(angle)
            self.average_direction = np.mean(self.direction_angles)

        # Zone engine integration with hysteresis
        self.update_zone(zone_name, frame, timestamp, events_stream, zone_hysteresis_frames)

        # Dynamic Quality Score calculation
        age_score = min(1.0, self.age_frames / 75.0)
        vis_score = self.visibility_ratio
        
        # Motion consistency score
        if len(self.trajectory) > 2:
            disps = [math.sqrt((self.trajectory[i][0]-self.trajectory[i-1][0])**2 + (self.trajectory[i][1]-self.trajectory[i-1][1])**2) for i in range(1, len(self.trajectory))]
            std = np.std(disps)
            motion_score = 1.0 / (1.0 + std / 8.0)
        else:
            motion_score = 0.5
            
        self.quality_score = 0.3 * age_score + 0.3 * self.ema_confidence + 0.2 * vis_score + 0.2 * motion_score

        # Auto state transitions
        if self.status == "NEW" and self.age_frames >= 3:
            self.transition_to("CONFIRMED", frame, timestamp, "Reached confirmation frame limit", events_stream)
        elif self.status in ["CONFIRMED", "RECOVERED"] and self.age_frames >= 10:
            self.transition_to("ACTIVE", frame, timestamp, "Reached active frame limit", events_stream)

    def update_zone(self, new_zone: Optional[str], frame: int, timestamp: float, events_stream: List[BaseEvent], zone_hysteresis_frames: int = 5) -> None:
        """Manages zone transitions with hysteresis integration and dwell duration logs."""
        self.zone_hysteresis_frames = zone_hysteresis_frames
        
        # Initial assignment bypasses hysteresis
        if not self.zone_history and self.current_zone is None:
            self.current_zone = new_zone
            self.zone_confidence = 1.0
            if new_zone is not None:
                self.zone_entry_times.setdefault(new_zone, []).append(timestamp)
                self.zone_history.append(new_zone)
                events_stream.append(ZoneEntered(frame, timestamp, self.track_id, {"zone": new_zone}))
                if "Entrance" in new_zone:
                    self.entered_restaurant = True
                    events_stream.append(EnteredRestaurant(frame, timestamp, self.track_id, {}))
            return

        if new_zone == self.current_zone:
            self.candidate_zone = None
            self.frames_inside_candidate = 0
            self.zone_confidence = 1.0
            return

        # Handle transition countdown
        if new_zone == self.candidate_zone:
            self.frames_inside_candidate += 1
        else:
            self.candidate_zone = new_zone
            self.frames_inside_candidate = 1

        self.zone_confidence = max(0.0, 1.0 - (self.frames_inside_candidate / self.zone_hysteresis_frames))

        if self.frames_inside_candidate < self.zone_hysteresis_frames:
            # Hysteresis not satisfied. Keep current zone.
            return

        # Hysteresis satisfied! Commit zone transition
        actual_new_zone = new_zone
        self.candidate_zone = None
        self.frames_inside_candidate = 0
        self.zone_confidence = 1.0

        # Exited old zone
        if self.current_zone is not None:
            old_z = self.current_zone
            self.zone_exit_times.setdefault(old_z, []).append(timestamp)
            
            entry_t = self.zone_entry_times[old_z][-1] if self.zone_entry_times.get(old_z) else self.first_timestamp
            dwell = timestamp - entry_t
            self.zone_dwell_times.setdefault(old_z, []).append(dwell)
            events_stream.append(ZoneExited(frame, timestamp, self.track_id, {"zone": old_z, "dwell_seconds": round(dwell, 2)}))

            if "Waiting" in old_z:
                events_stream.append(WaitingFinished(frame, timestamp, self.track_id, {"dwell": round(dwell, 2)}))
            elif "Dining" in old_z:
                events_stream.append(DiningFinished(frame, timestamp, self.track_id, {"dwell": round(dwell, 2)}))

        self.previous_zone = self.current_zone
        self.current_zone = actual_new_zone

        # Entered new zone
        if actual_new_zone is not None:
            self.zone_entry_times.setdefault(actual_new_zone, []).append(timestamp)
            if not self.zone_history or self.zone_history[-1] != actual_new_zone:
                self.zone_history.append(actual_new_zone)
            events_stream.append(ZoneEntered(frame, timestamp, self.track_id, {"zone": actual_new_zone}))

            if "Entrance" in actual_new_zone and not self.entered_restaurant:
                self.entered_restaurant = True
                events_stream.append(EnteredRestaurant(frame, timestamp, self.track_id, {}))

            if "Waiting" in actual_new_zone:
                events_stream.append(WaitingStarted(frame, timestamp, self.track_id, {}))
            elif "Dining" in actual_new_zone:
                events_stream.append(DiningStarted(frame, timestamp, self.track_id, {}))

        self.evaluate_role(frame, timestamp, events_stream)

    def evaluate_role(self, frame: int, timestamp: float, events_stream: List[BaseEvent]) -> None:
        """Classifies staff vs customer role dynamically based on cumulative zone dwells."""
        kitchen_dwell = sum(self.zone_dwell_times.get("Kitchen", [0]))
        reception_dwell = sum(self.zone_dwell_times.get("Reception", [0]))
        total_dwell = sum(sum(d) for d in self.zone_dwell_times.values()) if self.zone_dwell_times else 1.0

        new_role = "Customer"
        if kitchen_dwell > 5.0 or (reception_dwell > 20.0 and (reception_dwell / total_dwell) > 0.60):
            new_role = "Staff"

        if new_role != self.role:
            old_role = self.role
            self.role = new_role
            events_stream.append(RoleChanged(frame, timestamp, self.track_id, {"old_role": old_role, "new_role": new_role}))

class TrackMemoryEngine:
    """Manages all TrackState instances throughout their lifetimes."""
    def __init__(self, track_buffer: int = 30, frame_time: float = 0.033, zone_hysteresis_frames: int = 5):
        self.track_buffer = track_buffer
        self.frame_time = frame_time
        self.zone_hysteresis_frames = zone_hysteresis_frames
        
        self.active_tracks: Dict[int, TrackState] = {}
        self.archived_tracks: Dict[int, TrackState] = {}
        self.events_stream: List[BaseEvent] = []

    def update(self, frame: int, timestamp: float, tracks: np.ndarray, zone_resolver) -> List[TrackState]:
        """Updates TrackState for active and lost targets on every frame."""
        current_ids = set()

        for track in tracks:
            tid = int(track[4])
            x1, y1, x2, y2 = track[:4]
            conf = float(track[5])
            
            # Solve current zone center
            cx, cy = (x1 + x2) / 2.0, y2
            zone_name = zone_resolver((cx, cy))
            
            current_ids.add(tid)

            if tid not in self.active_tracks and tid not in self.archived_tracks:
                # Create NEW track state
                state = TrackState(tid, frame, timestamp, (x1, y1, x2, y2), conf, self.frame_time)
                state.update_zone(zone_name, frame, timestamp, self.events_stream, self.zone_hysteresis_frames)
                self.active_tracks[tid] = state
                self.events_stream.append(TrackCreated(frame, timestamp, tid, {"bbox": [x1, y1, x2, y2]}))
                self.events_stream.append(RoleAssigned(frame, timestamp, tid, {"role": "Customer"}))
            elif tid in self.active_tracks:
                # Update existing active track state
                state = self.active_tracks[tid]
                
                # Check recovery from occluded state
                if state.status == "TEMP_OCCLUDED":
                    state.transition_to("RECOVERED", frame, timestamp, "Track re-associated", self.events_stream)
                    
                state.update(frame, timestamp, (x1, y1, x2, y2), conf, zone_name, self.events_stream, self.zone_hysteresis_frames)

        # Identify lost tracks
        lost_ids = set(self.active_tracks.keys()) - current_ids
        for tid in lost_ids:
            state = self.active_tracks[tid]
            state.lost_count += 1
            
            # Transition to TEMP_OCCLUDED on first missing frame
            if state.status != "TEMP_OCCLUDED":
                state.transition_to("TEMP_OCCLUDED", frame, timestamp, "Lost tracking detection updates", self.events_stream)
            
            # Check if lost duration exceeds track buffer threshold for archiving
            missing_frames = frame - state.last_frame
            if missing_frames > self.track_buffer:
                # Exited & Archive
                state.update_zone(None, frame, timestamp, self.events_stream, self.zone_hysteresis_frames)
                state.transition_to("EXITED", frame, timestamp, f"Missing for {missing_frames} frames", self.events_stream)
                state.transition_to("ARCHIVED", frame, timestamp, "Archiving track memory", self.events_stream)
                
                # Move to archived registry
                self.archived_tracks[tid] = state
                del self.active_tracks[tid]

        return list(self.active_tracks.values())

    def get_track(self, track_id: int) -> Optional[TrackState]:
        """Retrieves read-only TrackState object if available."""
        if track_id in self.active_tracks:
            return self.active_tracks[track_id]
        return self.archived_tracks.get(track_id)
