import numpy as np
import logging
from typing import List, Tuple, Any, Dict, Set
from tracker.tracker_factory import create_tracker
from ultralytics.trackers.utils.gmc import GMC
from .config_loader import TrackerConfig

logger = logging.getLogger("AurikaTracking")

class TrackerArgs:
    """Namespace expected by Ultralytics' BYTETracker."""
    def __init__(self, config: TrackerConfig):
        self.track_high_thresh = config.track_high_thresh
        self.track_low_thresh = config.track_low_thresh
        self.new_track_thresh = config.new_track_thresh
        self.track_buffer = config.track_buffer
        self.match_thresh = config.match_thresh
        self.fuse_score = config.fuse_score
        self.gmc_method = config.gmc_method

class TrackingEngine:
    """Enhanced Wrapper managing the ByteTrack engine with Experiment 007 stability improvements."""
    def __init__(self, config: TrackerConfig):
        self.config = config
        self.args = TrackerArgs(config)
        self.tracker = create_tracker(self.config.tracker_type, self.config)
        
        if self.config.gmc_method and self.config.gmc_method.lower() != "none":
            logger.info(f"Initializing GMC with method: {self.config.gmc_method}")
            self.tracker.gmc = GMC(method=self.config.gmc_method)
        else:
            logger.info("GMC is disabled (set to 'none').")

        # Experiment 007 State
        self.smooth_conf: Dict[int, float] = {}
        self.track_prev_centers: Dict[int, Tuple[float, float]] = {}
        self.track_displacements: Dict[int, List[float]] = {}
        
        # Quality score tracking
        self.track_ages: Dict[int, int] = {}
        self.track_detections_count: Dict[int, int] = {}
        self.track_total_life: Dict[int, int] = {}
        
        # Instrumentation metrics
        self.id_lifetimes: Dict[int, int] = {}       # tid -> active frame count
        self.recovery_counts: Dict[int, int] = {}    # tid -> number of recoveries from lost
        self.lost_counts = 0
        self.fragmentations: Dict[int, int] = {}     # tid -> number of gaps
        self.occlusion_durations: Dict[int, int] = {} # tid -> total lost frames
        self.recovery_latencies: Dict[int, List[int]] = {} # tid -> list of recovery latencies
        self.track_prev_state: Dict[int, str] = {}   # tid -> last known state

    def reset(self) -> None:
        """Reset internal tracker state."""
        self.tracker.reset()
        self.smooth_conf.clear()
        self.track_prev_centers.clear()
        self.track_displacements.clear()
        self.track_ages.clear()
        self.track_detections_count.clear()
        self.track_total_life.clear()
        self.id_lifetimes.clear()
        self.recovery_counts.clear()
        self.lost_counts = 0
        self.fragmentations.clear()
        self.occlusion_durations.clear()
        self.recovery_latencies.clear()
        self.track_prev_state.clear()
        logger.info("Tracker state and instrumentation metrics reset.")

    def update(self, yolo_results: Any, frame: np.ndarray) -> np.ndarray:
        """
        Updates the tracker with YOLO predictions, applying Experiment 007 enhancements:
        1. Adaptive confidence thresholds based on scene brightness.
        2. Adaptive track buffers based on crowd density.
        3. Kalman velocity check motion consistency filtering.
        4. EMA confidence smoothing.
        5. Quality score check visualization filtering.
        6. Frame-by-frame tracker instrumentation.
        """
        if yolo_results is None or len(yolo_results) == 0:
            return np.empty((0, 8), dtype=np.float32)

        # Resolve bounding boxes (keeping as Boxes / CachedNumpyBoxes wrapper for BYTETracker compatibility)
        if hasattr(yolo_results, "boxes"):
            boxes = yolo_results.boxes
            mask = (boxes.cls == 0)
            det = boxes[mask].cpu().numpy()
        elif hasattr(yolo_results, "cls") and hasattr(yolo_results, "xyxy") and hasattr(yolo_results, "conf"):
            mask = (yolo_results.cls == 0)
            det = yolo_results[mask].cpu().numpy()
        else:
            det = yolo_results

        if det is None or len(det) == 0:
            return np.empty((0, 8), dtype=np.float32)

        # 1. Adaptive confidence threshold based on scene brightness
        if self.config.adaptive_confidence_enabled:
            brightness = float(np.mean(frame))
            # Brightness scale: lower values under dark environments to recover low-conf detections
            scale = max(0.6, min(1.2, brightness / 110.0))
            self.tracker.args.track_high_thresh = max(0.12, min(0.35, self.config.track_high_thresh * scale))
            self.tracker.args.track_low_thresh = max(0.04, min(0.18, self.config.track_low_thresh * scale))

        # 2. Adaptive track buffer based on crowd density
        if self.config.adaptive_track_buffer_enabled:
            crowd_density = len(self.tracker.tracked_stracks)
            if crowd_density > 10:
                # Tighten buffer under high density to avoid ID cross-associations
                self.tracker.max_frames_lost = max(12, int(self.args.track_buffer * 0.5))
            else:
                self.tracker.max_frames_lost = self.args.track_buffer

        # Run ByteTrack update
        tracks = self.tracker.update(det, frame)
        
        # 3. Post-process tracks: Smoothing, Motion Consistency, Quality Score, Instrumentation
        output_tracks = []
        alpha = self.config.confidence_smoothing_alpha

        for track in tracks:
            x1, y1, x2, y2 = track[:4]
            tid = int(track[4])
            score = float(track[5])
            cls_id = int(track[6])
            det_idx = int(track[7])

            # Increment active track life metrics
            self.track_total_life[tid] = self.track_total_life.get(tid, 0) + 1
            self.track_ages[tid] = self.track_ages.get(tid, 0) + 1

            # 4. Confidence Smoothing (EMA)
            prev_score = self.smooth_conf.get(tid, score)
            smoothed_score = alpha * score + (1.0 - alpha) * prev_score
            self.smooth_conf[tid] = smoothed_score

            # 5. Motion Consistency Check
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            motion_valid = True
            
            if tid in self.track_prev_centers:
                prev_cx, prev_cy = self.track_prev_centers[tid]
                disp = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
                
                # Check against bounding box diagonal limits (rejections of teleportations)
                diag = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                if self.config.motion_consistency_check and disp > 1.4 * diag:
                    motion_valid = False
                    logger.warning(f"Motion inconsistency: Target {tid} jumped {disp:.1f}px (Diagonal: {diag:.1f}px). Bypassing.")
                else:
                    if tid not in self.track_displacements:
                        self.track_displacements[tid] = []
                    self.track_displacements[tid].append(float(disp))
                    if len(self.track_displacements[tid]) > 20:
                        self.track_displacements[tid].pop(0)

            if not motion_valid:
                # Skip updating center coordinates for anomalous jump to prevent trajectory snapping
                continue

            self.track_prev_centers[tid] = (cx, cy)
            self.track_detections_count[tid] = self.track_detections_count.get(tid, 0) + 1

            # 6. Quality Score Calculation (penalize young/unconfirmed tracks)
            age_score = min(1.0, self.track_ages[tid] / 75.0)
            conf_score = smoothed_score
            visibility_score = self.track_detections_count[tid] / max(15, self.track_total_life[tid])
            
            disps = self.track_displacements.get(tid, [])
            if len(disps) > 1:
                std = np.std(disps)
                motion_score = 1.0 / (1.0 + std / 8.0)
            else:
                motion_score = 0.5

            quality_score = 0.3 * age_score + 0.3 * conf_score + 0.2 * visibility_score + 0.2 * motion_score

            # Only return tracks satisfying minimum quality threshold
            if quality_score >= self.config.quality_score_threshold:
                output_tracks.append([x1, y1, x2, y2, tid, smoothed_score, cls_id, det_idx])
                self.id_lifetimes[tid] = self.id_lifetimes.get(tid, 0) + 1

        # 7. Instrumentation (Lost and recovery states check for visible tracks only)
        # Track IDs currently active/lost states inside BYTETracker
        tracked_ids = {x.track_id for x in self.tracker.tracked_stracks if self.id_lifetimes.get(x.track_id, 0) > 0}
        lost_ids = {x.track_id for x in self.tracker.lost_stracks if self.id_lifetimes.get(x.track_id, 0) > 0}

        for tid in tracked_ids:
            prev = self.track_prev_state.get(tid, "Tracked")
            if prev == "Lost":
                # Recovered!
                self.recovery_counts[tid] = self.recovery_counts.get(tid, 0) + 1
                
                # Fetch latency
                lat = self.occlusion_durations.get(tid, 0)
                if tid not in self.recovery_latencies:
                    self.recovery_latencies[tid] = []
                self.recovery_latencies[tid].append(lat)
                self.occlusion_durations[tid] = 0 # reset current gap
            self.track_prev_state[tid] = "Tracked"

        for tid in lost_ids:
            prev = self.track_prev_state.get(tid, "Tracked")
            if prev == "Tracked":
                self.fragmentations[tid] = self.fragmentations.get(tid, 0) + 1
                self.lost_counts += 1
            
            # Increment current occlusion frame count
            self.occlusion_durations[tid] = self.occlusion_durations.get(tid, 0) + 1
            self.track_prev_state[tid] = "Lost"

        return np.array(output_tracks, dtype=np.float32) if output_tracks else np.empty((0, 8), dtype=np.float32)
