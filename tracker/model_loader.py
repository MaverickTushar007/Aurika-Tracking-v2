import logging
from pathlib import Path
from ultralytics import YOLO

logger = logging.getLogger("AurikaTracking")


def load_yolo_model(model_path: str) -> YOLO:
    """
    Load a YOLO model.

    Supports:
    - Local weights (models/yolo11l.pt)
    - Absolute paths
    - Ultralytics model names (e.g. yolo11l.pt)
    """

    resolved_path = Path(model_path)

    if resolved_path.exists():
        logger.info(f"Loading local model: {resolved_path}")
        model = YOLO(str(resolved_path))
    else:
        logger.info(
            f"Local model not found. Attempting to download/load '{model_path}'..."
        )
        model = YOLO(model_path)

    logger.info(f"YOLO model loaded successfully.")
    logger.info(f"Classes: {model.names}")

    return model
