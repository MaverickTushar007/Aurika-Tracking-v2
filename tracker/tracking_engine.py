import numpy as np
import logging
from typing import List, Tuple, Any
from ultralytics.trackers.byte_tracker import BYTETracker
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
    """Wrapper class managing the ByteTrack engine."""
    def __init__(self, config: TrackerConfig):
        self.config = config
        self.args = TrackerArgs(config)
        self.tracker = BYTETracker(args=self.args)
        
        # Manually initialize GMC on the BYTETracker if a method is specified
        if self.config.gmc_method and self.config.gmc_method.lower() != "none":
            logger.info(f"Initializing GMC with method: {self.config.gmc_method}")
            self.tracker.gmc = GMC(method=self.config.gmc_method)
        else:
            logger.info("GMC is disabled (set to 'none').")

    def reset(self) -> None:
        """Reset internal tracker state."""
        self.tracker.reset()
        logger.info("Tracker state reset.")

    def update(self, yolo_results: Any, frame: np.ndarray) -> np.ndarray:
        """
        Updates the tracker with YOLO predictions.
        
        Args:
            yolo_results: The raw output box object from YOLO predictor.
            frame: The current video frame (BGR).
            
        Returns:
            np.ndarray: Array of shape (N, 8) with tracks formatted as:
                        [x1, y1, x2, y2, track_id, confidence, class_id, detection_idx]
        """
        if len(yolo_results) == 0:
            # Pass empty numpy boxes representation if no detections
            # If yolo_results is empty, tracker.update returns an empty array.
            return np.empty((0, 8), dtype=np.float32)

        # Filter detections: keep ONLY classes 0 ('customer') and 1 ('staff')
        # Since this tracker expects a Boxes object, we filter using YOLO's indexing
        boxes = yolo_results.boxes
        if len(boxes) == 0:
            return np.empty((0, 8), dtype=np.float32)

        mask = (boxes.cls == 0) | (boxes.cls == 1)
        filtered_boxes = boxes[mask]
        
        if len(filtered_boxes) == 0:
            return np.empty((0, 8), dtype=np.float32)

        # Convert to CPU/NumPy representation expected by the tracker
        det = filtered_boxes.cpu().numpy()
        
        # Run ByteTrack update
        tracks = self.tracker.update(det, frame)
        return tracks
