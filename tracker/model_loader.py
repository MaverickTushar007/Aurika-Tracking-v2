import os
import logging
from pathlib import Path
from ultralytics import YOLO

logger = logging.getLogger("AurikaTracking")

def load_yolo_model(model_path: str) -> YOLO:
    """Loads the YOLO model and verifies classes."""
    resolved_path = Path(model_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"YOLO model file not found at {model_path}")
        
    logger.info(f"Loading YOLO model from {resolved_path}...")
    try:
        model = YOLO(str(resolved_path))
        # Basic validation of classes dictionary
        classes = model.names
        logger.info(f"YOLO model loaded successfully. Classes detected: {classes}")
        return model
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        raise
