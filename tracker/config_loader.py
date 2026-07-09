import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any

# Configure logging
logger = logging.getLogger("AurikaTracking")

class TrackerConfig:
    """Strongly typed class containing tracker parameters."""
    def __init__(self, raw_config: Dict[str, Any]):
        tracker_params = raw_config.get("tracker", {})
        self.tracker_type: str = str(tracker_params.get("tracker_type", "bytetrack"))
        self.track_high_thresh: float = float(tracker_params.get("track_high_thresh", 0.25))
        self.track_low_thresh: float = float(tracker_params.get("track_low_thresh", 0.1))
        self.new_track_thresh: float = float(tracker_params.get("new_track_thresh", 0.25))
        self.track_buffer: int = int(tracker_params.get("track_buffer", 30))
        self.match_thresh: float = float(tracker_params.get("match_thresh", 0.8))
        self.fuse_score: bool = bool(tracker_params.get("fuse_score", True))
        self.gmc_method: str = str(tracker_params.get("gmc_method", "none"))
        self.adaptive_confidence_enabled: bool = bool(tracker_params.get("adaptive_confidence_enabled", True))
        self.confidence_smoothing_alpha: float = float(tracker_params.get("confidence_smoothing_alpha", 0.3))
        self.motion_consistency_check: bool = bool(tracker_params.get("motion_consistency_check", True))
        self.adaptive_track_buffer_enabled: bool = bool(tracker_params.get("adaptive_track_buffer_enabled", True))
        self.quality_score_threshold: float = float(tracker_params.get("quality_score_threshold", 0.25))
        self.proximity_thresh: float = float(tracker_params.get("proximity_thresh", 0.5))
        self.appearance_thresh: float = float(tracker_params.get("appearance_thresh", 0.8))
        self.with_reid: bool = bool(tracker_params.get("with_reid", False))
        self.model: str = str(tracker_params.get("model", "auto"))
        self.zone_hysteresis_frames: int = int(tracker_params.get("zone_hysteresis_frames", 5))

class PipelineConfig:
    """Handles environment detection, loading configs, and resolving paths."""
    def __init__(self, config_path: str = "configs/config.yaml"):
        # Resolve config path
        resolved_path = Path(config_path)
        if not resolved_path.exists():
            # Look relative to the script file location
            resolved_path = Path(__file__).resolve().parent.parent / config_path
            
        if not resolved_path.exists():
            raise FileNotFoundError(f"Configuration file not found at {config_path} or resolved to {resolved_path}")
            
        with open(resolved_path, "r") as f:
            self._raw = yaml.safe_load(f)
            
        self.environment: str = self._detect_environment()
        logger.info(f"Active environment: {self.environment}")
        
        project_root = resolved_path.parent.parent.resolve()
        
        env_config = self._raw.get(self.environment, {})
        self.video_path: str = str(env_config.get("video_path", ""))
        self.model_path: str = str(env_config.get("model_path", ""))
        
        raw_output_dir = env_config.get("output_dir", "./runs")
        if not Path(raw_output_dir).is_absolute():
            self.output_dir: str = str((project_root / raw_output_dir).resolve())
        else:
            self.output_dir: str = str(raw_output_dir)
        
        self.tracker = TrackerConfig(self._raw)
        
        logger.info(f"Resolved Video Path: {self.video_path}")
        logger.info(f"Resolved Model Path: {self.model_path}")
        logger.info(f"Resolved Output Directory: {self.output_dir}")

    def _detect_environment(self) -> str:
        # Check for Kaggle-specific indicators
        if os.path.exists('/kaggle') or 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
            return "kaggle"
        return "local"
