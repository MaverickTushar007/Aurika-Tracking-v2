import cv2
import numpy as np
from typing import Tuple, Dict

# Curated harmonious premium BGR color palette (OpenCV is BGR)
COLORS = [
    (180, 115, 230),  # Velvet Purple
    (255, 229, 0),    # Electric Cyan
    (0, 107, 255),    # Sunset Orange
    (113, 204, 46),   # Mint Emerald
    (15, 196, 241),   # Amber Gold
    (180, 105, 255),  # Rose Pink
    (48, 211, 254),   # Bright Yellow
    (75, 77, 235),    # Crimson Red
    (156, 188, 26)    # Soft Teal
]

CLASS_NAMES = {0: "Customer", 1: "Staff"}

def get_color(track_id: int) -> Tuple[int, int, int]:
    """Retrieves a unique color for a given track ID from the palette."""
    return COLORS[track_id % len(COLORS)]

def draw_corner_brackets(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: Tuple[int, int, int], thickness: int = 3, length: int = 15) -> None:
    """Draws premium corner brackets for a bounding box."""
    # Top-Left
    cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness, lineType=cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness, lineType=cv2.LINE_AA)
    
    # Top-Right
    cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness, lineType=cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness, lineType=cv2.LINE_AA)
    
    # Bottom-Left
    cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness, lineType=cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness, lineType=cv2.LINE_AA)
    
    # Bottom-Right
    cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness, lineType=cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness, lineType=cv2.LINE_AA)

def annotate_frame(frame: np.ndarray, tracks: np.ndarray) -> np.ndarray:
    """
    Draws tracking bounding boxes, IDs, confidences, and labels on a frame.
    
    Args:
        frame: Original image frame (BGR format)
        tracks: Array of tracks returned by tracking engine, each row is:
                [x1, y1, x2, y2, track_id, confidence, class_id, detection_idx]
                
    Returns:
        np.ndarray: Annotated frame
    """
    annotated = frame.copy()
    
    for track in tracks:
        x1, y1, x2, y2 = map(int, track[:4])
        track_id = int(track[4])
        conf = float(track[5])
        class_id = int(track[6])
        
        class_name = CLASS_NAMES.get(class_id, "Person")
        color = get_color(track_id)
        
        # 1. Draw semi-transparent glassmorphism box fill
        overlay = annotated.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.08, annotated, 0.92, 0, annotated)
        
        # 2. Draw thin boundary box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1, lineType=cv2.LINE_AA)
        
        # 3. Draw heavy corner brackets for premium sci-fi feel
        draw_corner_brackets(annotated, x1, y1, x2, y2, color, thickness=2, length=12)
        
        # 4. Prepare text label (e.g. ID 4 | Staff 0.89)
        label = f"ID {track_id} | {class_name} ({conf:.2f})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        text_thickness = 1
        
        (w, h), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
        
        # Determine clean placement for text pill (above box unless top-edge restricted)
        if y1 - h - 10 > 0:
            bg_y1, bg_y2 = y1 - h - 8, y1
            text_y = y1 - 4
        else:
            bg_y1, bg_y2 = y2, y2 + h + 8
            text_y = y2 + h + 4
            
        # Draw text pill background
        cv2.rectangle(annotated, (x1, bg_y1), (x1 + w + 8, bg_y2), color, -1)
        
        # Draw label text
        cv2.putText(annotated, label, (x1 + 4, text_y), font, font_scale, (255, 255, 255), text_thickness, lineType=cv2.LINE_AA)
        
    return annotated
