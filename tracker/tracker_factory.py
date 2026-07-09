import logging
from typing import Any
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.trackers.bot_sort import BOTSORT

logger = logging.getLogger("AurikaTracking")

class TrackerArgs:
    """Wrapper arguments expected by Ultralytics trackers."""
    def __init__(self, config: Any, tracker_type: str = "bytetrack", device: str | None = None):
        def get_val(name: str, default: Any) -> Any:
            if isinstance(config, dict):
                return config.get(name, default)
            return getattr(config, name, default)

        # Base ByteTrack configs
        self.track_high_thresh = get_val("track_high_thresh", 0.25)
        self.track_low_thresh = get_val("track_low_thresh", 0.10)
        self.new_track_thresh = get_val("new_track_thresh", 0.25)
        self.track_buffer = get_val("track_buffer", 30)
        self.match_thresh = get_val("match_thresh", 0.80)
        self.fuse_score = get_val("fuse_score", True)
        self.gmc_method = get_val("gmc_method", "none")
        
        # BoT-SORT specific defaults
        from tracker.device import get_device
        self.proximity_thresh = get_val("proximity_thresh", 0.5)
        self.appearance_thresh = get_val("appearance_thresh", 0.8)
        self.with_reid = get_val("with_reid", False)
        self.model = get_val("model", "auto")
        self.device = device if device is not None else get_device()

def create_tracker(tracker_type: str, config: Any, device: str | None = None) -> Any:
    """
    Returns an instantiated tracker instance (ByteTrack or BoT-SORT)
    conforming to the same interface (reset() and update(dets, frame)).
    """
    tracker_type_clean = tracker_type.lower().replace("-", "").replace("_", "")
    args = TrackerArgs(config, tracker_type_clean, device=device)
    
    if tracker_type_clean == "bytetrack":
        logger.info("Initializing ByteTrack tracker instance...")
        return BYTETracker(args=args)
    elif tracker_type_clean == "botsort":
        logger.info("Initializing BoT-SORT tracker instance...")
        return BOTSORT(args=args)
    else:
        raise ValueError(f"Unknown tracker type: {tracker_type}")
