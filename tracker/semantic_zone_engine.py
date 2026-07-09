# tracker/semantic_zone_engine.py
"""
Aurika Tracking v2 — Semantic Zone Engine
Decoupled API that reads polygon and counting line configurations
and resolves point-in-polygon queries and line crossing logic.
"""

import os
import yaml
import logging
import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger("AurikaTracking")

class SemanticZoneEngine:
    """Manages spatial zones and line coordinate parsing and point-in-polygon queries."""
    def __init__(self, layout_config_path: str, width: int = 1280, height: int = 720):
        self.config_path = layout_config_path
        self.width = width
        self.height = height
        self.zones: List[Dict[str, Any]] = []
        self.counting_lines: List[Dict[str, Any]] = []
        self.load_layout()

    def load_layout(self) -> None:
        """Parses zones and lines coordinates from the YAML config file, scaling them if normalized."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Layout config file not found: {self.config_path}")
            
        with open(self.config_path, "r") as f:
            data = yaml.safe_load(f) or {}

        # Parse Zones
        self.zones = []
        for z in data.get("zones", []):
            poly_list = z["polygon"]
            # Check if coordinates are normalized [0, 1]
            is_normalized = all(all(0.0 <= float(coord) <= 1.0 for coord in pt) for pt in poly_list)
            if is_normalized:
                scaled_pts = [[int(round(float(pt[0]) * self.width)), int(round(float(pt[1]) * self.height))] for pt in poly_list]
            else:
                scaled_pts = poly_list
            poly_pts = np.array(scaled_pts, dtype=np.int32)
            
            self.zones.append({
                "name": z["name"],
                "color": tuple(z.get("color", [255, 255, 255])),
                "polygon": poly_pts
            })

        # Parse Lines
        self.counting_lines = []
        for line in data.get("counting_lines", []):
            p1 = list(line["p1"])
            p2 = list(line["p2"])
            is_normalized = all(0.0 <= float(c) <= 1.0 for c in p1 + p2)
            if is_normalized:
                p1_scaled = (int(round(float(p1[0]) * self.width)), int(round(float(p1[1]) * self.height)))
                p2_scaled = (int(round(float(p2[0]) * self.width)), int(round(float(p2[1]) * self.height)))
            else:
                p1_scaled = tuple(p1)
                p2_scaled = tuple(p2)
                
            self.counting_lines.append({
                "name": line["name"],
                "color": tuple(line.get("color", [255, 255, 255])),
                "p1": p1_scaled,
                "p2": p2_scaled,
                "direction": line.get("direction", "in")
            })
            
        logger.info(f"Loaded layout: {len(self.zones)} zones, {len(self.counting_lines)} counting lines from {self.config_path}")

    def get_zone_at_point(self, point: Tuple[float, float]) -> Optional[str]:
        """Resolves which polygon zone contains the foot point via pointPolygonTest."""
        pt = (float(point[0]), float(point[1]))
        # Search zones
        for zone in self.zones:
            dist = cv2.pointPolygonTest(zone["polygon"], pt, False)
            if dist >= 0:
                return zone["name"]
        return None
