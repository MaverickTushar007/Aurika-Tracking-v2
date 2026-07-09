import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

logger = logging.getLogger("AurikaTracking")

def check_line_intersection(p1: Tuple[float, float], p2: Tuple[float, float], q1: Tuple[float, float], q2: Tuple[float, float]) -> bool:
    """Checks if segment p1p2 intersects with q1q2."""
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2)


def check_self_intersection(polygon: List[Tuple[int, int]]) -> bool:
    """Checks if any edge in the polygon intersects with another non-adjacent edge."""
    n = len(polygon)
    if n < 4:
        return False
    segments = [(polygon[i], polygon[(i+1)%n]) for i in range(n)]
    for i in range(n):
        for j in range(i + 2, n):
            # Skip adjacent lines sharing a vertex
            if (i == 0 and j == n - 1):
                continue
            if check_line_intersection(segments[i][0], segments[i][1], segments[j][0], segments[j][1]):
                return True
    return False


def check_duplicate_vertices(polygon: List[Tuple[int, int]]) -> bool:
    """Returns True if polygon contains duplicate points."""
    return len(polygon) != len(set(polygon))


def check_outside_boundary(polygon: List[Tuple[int, int]], width: int, height: int) -> bool:
    """Returns True if any point in the polygon is outside the width/height frame dimensions."""
    for x, y in polygon:
        if x < 0 or x >= width or y < 0 or y >= height:
            return True
    return False


def check_heavy_overlap(poly_a: List[Tuple[int, int]], poly_b: List[Tuple[int, int]]) -> float:
    """Computes a sampled point overlap ratio between two polygons."""
    poly_a_np = np.array(poly_a, dtype=np.int32)
    poly_b_np = np.array(poly_b, dtype=np.int32)
    
    bbox_a = cv2.boundingRect(poly_a_np)
    bbox_b = cv2.boundingRect(poly_b_np)
    
    x1 = max(bbox_a[0], bbox_b[0])
    y1 = max(bbox_a[1], bbox_b[1])
    x2 = min(bbox_a[0]+bbox_a[2], bbox_b[0]+bbox_b[2])
    y2 = min(bbox_a[1]+bbox_a[3], bbox_b[1]+bbox_b[3])
    
    if x1 >= x2 or y1 >= y2:
        return 0.0
        
    samples = 0
    overlap_count = 0
    step_x = max(1, (x2 - x1) // 10)
    step_y = max(1, (y2 - y1) // 10)
    
    for x in range(x1, x2, step_x):
        for y in range(y1, y2, step_y):
            samples += 1
            in_a = cv2.pointPolygonTest(poly_a_np, (float(x), float(y)), False) >= 0
            in_b = cv2.pointPolygonTest(poly_b_np, (float(x), float(y)), False) >= 0
            if in_a and in_b:
                overlap_count += 1
                
    if samples == 0:
        return 0.0
    return overlap_count / samples


class CalibrationManager:
    """Manages active zones, lines, vertex dragging, state validations, and file output mappings."""
    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height
        self.zones: List[Dict[str, Any]] = []
        self.counting_lines: List[Dict[str, Any]] = []

    def add_zone(self, name: str, points: List[Tuple[int, int]], color: Tuple[int, int, int]) -> Tuple[bool, str]:
        """Validates and adds a new polygon zone."""
        if len(points) < 3:
            return False, "Polygons must contain at least 3 vertices."
        if check_duplicate_vertices(points):
            return False, "Polygons cannot contain duplicate vertices."
        if check_self_intersection(points):
            return False, "Self-intersecting polygons are not allowed."
        if check_outside_boundary(points, self.width, self.height):
            return False, "Polygons contain coordinates outside the image frame size."

        # Check for heavy overlaps with existing zones
        for z in self.zones:
            overlap = check_heavy_overlap(points, z["polygon"])
            if overlap > 0.4:
                logger.warning(f"Warning: New zone '{name}' overlaps heavily ({overlap*100:.0f}%) with zone '{z['name']}'.")

        self.zones.append({
            "name": name,
            "polygon": points,
            "color": color
        })
        return True, "Zone added successfully."

    def add_counting_line(self, name: str, pt1: Tuple[int, int], pt2: Tuple[int, int], direction: str, color: Tuple[int, int, int]) -> Tuple[bool, str]:
        """Adds a counting line."""
        self.counting_lines.append({
            "name": name,
            "p1": list(pt1),
            "p2": list(pt2),
            "direction": direction,
            "color": list(color)
        })
        return True, "Counting line added successfully."

    def export_yaml_data(self) -> Dict[str, Any]:
        """Compiles clean coordinates mapping structure for export."""
        data = {
            "zones": [],
            "counting_lines": []
        }
        for z in self.zones:
            data["zones"].append({
                "name": z["name"],
                "color": list(z["color"]),
                "polygon": [list(pt) for pt in z["polygon"]]
            })
        for cl in self.counting_lines:
            data["counting_lines"].append({
                "name": cl["name"],
                "color": cl["color"],
                "p1": cl["p1"],
                "p2": cl["p2"],
                "direction": cl["direction"]
            })
        return data
