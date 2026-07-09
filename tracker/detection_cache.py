import hashlib
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

logger = logging.getLogger("AurikaTracking")

CACHE_VERSION = "1.0"

class CachedBoxes:
    """
    Lightweight Mock Boxes object containing PyTorch tensors.
    Simulates Ultralytics Boxes behavior including slicing and conversion to numpy.
    """
    def __init__(self, xyxy: Union[np.ndarray, torch.Tensor], conf: Union[np.ndarray, torch.Tensor], cls: Union[np.ndarray, torch.Tensor]):
        self._xyxy = torch.as_tensor(xyxy, dtype=torch.float32) if not isinstance(xyxy, torch.Tensor) else xyxy
        self._conf = torch.as_tensor(conf, dtype=torch.float32) if not isinstance(conf, torch.Tensor) else conf
        self._cls = torch.as_tensor(cls, dtype=torch.float32) if not isinstance(cls, torch.Tensor) else cls

    @property
    def xyxy(self) -> torch.Tensor:
        return self._xyxy

    @property
    def conf(self) -> torch.Tensor:
        return self._conf

    @property
    def cls(self) -> torch.Tensor:
        return self._cls

    @property
    def xywh(self) -> torch.Tensor:
        if len(self._xyxy) == 0:
            return torch.empty((0, 4), dtype=torch.float32)
        x1, y1, x2, y2 = self._xyxy.unbind(-1)
        w = x2 - x1
        h = y2 - y1
        x = x1 + w / 2
        y = y1 + h / 2
        return torch.stack((x, y, w, h), -1)

    def cpu(self) -> "CachedBoxes":
        return self

    def numpy(self) -> "CachedNumpyBoxes":
        return CachedNumpyBoxes(
            self._xyxy.cpu().numpy(),
            self._conf.cpu().numpy(),
            self._cls.cpu().numpy()
        )

    def __len__(self) -> int:
        return len(self._conf)

    def __getitem__(self, index: Any) -> "CachedBoxes":
        return CachedBoxes(self._xyxy[index], self._conf[index], self._cls[index])


class CachedNumpyBoxes:
    """
    Lightweight Mock Boxes object containing numpy arrays.
    Created by cpu().numpy() for compatibility with BYTETracker.
    """
    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls

    @property
    def xywh(self) -> np.ndarray:
        if len(self.xyxy) == 0:
            return np.empty((0, 4), dtype=np.float32)
        x1, y1, x2, y2 = self.xyxy[:, 0], self.xyxy[:, 1], self.xyxy[:, 2], self.xyxy[:, 3]
        w = x2 - x1
        h = y2 - y1
        x = x1 + w / 2
        y = y1 + h / 2
        return np.stack((x, y, w, h), axis=-1)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, index: Any) -> "CachedNumpyBoxes":
        return CachedNumpyBoxes(self.xyxy[index], self.conf[index], self.cls[index])


def calculate_video_hash(video_path: Path) -> str:
    """Calculate a fast head-and-tail MD5 hash of the video file to detect content changes."""
    if not video_path.exists():
        return ""
    file_size = video_path.stat().st_size
    md5 = hashlib.md5()
    
    # 8MB chunks for speed
    chunk_size = 8 * 1024 * 1024
    try:
        with open(video_path, "rb") as f:
            if file_size <= 2 * chunk_size:
                md5.update(f.read())
            else:
                # Read head
                md5.update(f.read(chunk_size))
                # Seek to tail
                f.seek(-chunk_size, 2)
                md5.update(f.read(chunk_size))
        # Include file size in hash calculation
        md5.update(str(file_size).encode())
        return md5.hexdigest()
    except Exception as e:
        logger.error(f"Error calculating hash for {video_path}: {e}")
        return ""


def save_detection_cache(
    cache_path: Path,
    metadata: Dict[str, Any],
    detections: List[Dict[str, Any]],
) -> None:
    """Serializes metadata and frame detections into the cache file."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Force set version
    metadata["cache_version"] = CACHE_VERSION
    metadata["date"] = time.strftime("%Y-%m-%d %H:%M:%S")

    # Save pickle package containing metadata and detections list
    payload = {
        "metadata": metadata,
        "detections": detections,
    }
    
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Save readable copy of metadata.json alongside for review
        meta_json_path = cache_path.parent / "metadata.json"
        with open(meta_json_path, "w") as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Saved detection cache to {cache_path}")
        logger.info(f"Saved cache metadata to {meta_json_path}")
    except Exception as e:
        logger.error(f"Failed to write cache: {e}")
        raise


def load_detection_cache(
    cache_path: Path,
    expected_meta: Dict[str, Any],
) -> Optional[List[Dict[str, Any]]]:
    """
    Loads and validates the detection cache.
    Returns the list of detections if valid, else returns None.
    """
    if not cache_path.exists():
        logger.warning(f"Detection cache not found at {cache_path}")
        return None

    try:
        with open(cache_path, "rb") as f:
            payload = pickle.load(f)
    except Exception as e:
        logger.error(f"Failed to read/deserialize detection cache at {cache_path}: {e}")
        return None

    cached_meta = payload.get("metadata", {})
    detections = payload.get("detections", [])

    # Validate version first
    if cached_meta.get("cache_version") != CACHE_VERSION:
        logger.warning(
            f"Cache version mismatch: cache={cached_meta.get('cache_version')} vs required={CACHE_VERSION}. Cache rejected."
        )
        return None

    # Validate crucial validation keys
    validation_keys = [
        ("video_hash", "Video content mismatch (hash mismatch)"),
        ("model_name", "Model name mismatch"),
        ("confidence_threshold", "Confidence threshold mismatch"),
        ("image_size", "Inference image size mismatch"),
    ]

    for key, err_msg in validation_keys:
        if key in expected_meta:
            cached_val = cached_meta.get(key)
            expected_val = expected_meta[key]
            if cached_val != expected_val:
                logger.warning(
                    f"Cache Rejected: {err_msg} (cached={cached_val!r} vs expected={expected_val!r})"
                )
                return None

    logger.info(f"Successfully loaded valid detection cache from {cache_path} (Model: {cached_meta.get('model_name')})")
    return detections
