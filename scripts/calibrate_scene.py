#!/usr/bin/env python3
"""
Aurika Tracking v2 — Scene Calibrator GUI
===========================================
Opens an interactive layout calibration GUI to draw zones and counting lines.
Generates YAML coordinates files, clean background frame, preview overlays, and reports.
Supports automated fallback if run in a headless environment.

Usage:
    python scripts/calibrate_scene.py --video videos/Dark_lighting.mp4 --output configs/restaurant_default.yaml
"""

import argparse
import logging
import os
import sys
import time
import yaml
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("Calibrator")

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT))

from tracker.scene_calibration import CalibrationManager

# State variables for mouse callback
current_pts: List[Tuple[int, int]] = []
current_mode = "idle" # "idle", "polygon", "line"
active_name = ""
manager = None
img_w, img_h = 1280, 720

def prompt_console(prompt_str: str) -> str:
    """Prompt on stdin and return stripped string."""
    print(f"\n[PROMPT] {prompt_str}", end="", flush=True)
    return sys.stdin.readline().strip()

def mouse_click(event, x, y, flags, param):
    """Handles vertex insertion."""
    global current_pts, current_mode
    if current_mode == "idle":
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        current_pts.append((x, y))
        log.info(f"Added vertex: ({x}, {y})")

def generate_fallback_layout(output_path: Path, manager: CalibrationManager) -> None:
    """Writes default restaurant coordinates map if GUI fails or run headlessly."""
    # Dining Zone
    manager.add_zone(
        "Dining",
        [(400, 300), (800, 300), (800, 720), (400, 720)],
        (0, 255, 0)
    )
    # Reception
    manager.add_zone(
        "Reception",
        [(400, 0), (800, 0), (800, 300), (400, 300)],
        (255, 0, 255)
    )
    # Waiting
    manager.add_zone(
        "Waiting",
        [(0, 300), (400, 300), (400, 720), (0, 720)],
        (0, 255, 255)
    )
    # Entrance
    manager.add_zone(
        "Entrance",
        [(0, 0), (400, 0), (400, 300), (0, 300)],
        (255, 128, 0)
    )
    # Kitchen
    manager.add_zone(
        "Kitchen",
        [(800, 0), (1280, 0), (1280, 720), (800, 720)],
        (0, 0, 255)
    )
    # Lines
    manager.add_counting_line("Entrance Line", (50, 300), (350, 300), "in", (0, 165, 255))
    manager.add_counting_line("Exit Line", (50, 600), (350, 600), "out", (255, 0, 128))

    yaml_data = manager.export_yaml_data()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.safe_dump(yaml_data, f, default_flow_style=False)
    log.info(f"Generated default layout coordinates to {output_path}")

def render_layout_on_snapshot(bg_img: np.ndarray, manager: CalibrationManager) -> np.ndarray:
    """Draws all calibrated polygons and counting lines onto the background snapshot."""
    canvas = bg_img.copy()
    overlay = bg_img.copy()

    # Draw Polygons
    for z in manager.zones:
        poly_np = np.array(z["polygon"], dtype=np.int32)
        cv2.fillPoly(overlay, [poly_np], z["color"])
        cv2.polylines(canvas, [poly_np], True, z["color"], 2)
        
        # Label
        M = cv2.moments(poly_np)
        cx = int(M["m10"] / M["m00"]) if M["m00"] > 0 else z["polygon"][0][0]
        cy = int(M["m01"] / M["m00"]) if M["m00"] > 0 else z["polygon"][0][1]
        cv2.putText(canvas, z["name"], (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)

    # Draw Lines
    for line in manager.counting_lines:
        p1 = tuple(line["p1"])
        p2 = tuple(line["p2"])
        cv2.line(canvas, p1, p2, tuple(line["color"]), 3)
        
        # Arrow indicating direction
        mid_x = (p1[0] + p2[0]) // 2
        mid_y = (p1[1] + p2[1]) // 2
        
        # Draw perpendicular direction indicator line
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            nx = -dy / length
            ny = dx / length
            arrow_len = 25
            arr_p2 = (int(mid_x + nx * arrow_len), int(mid_y + ny * arrow_len))
            cv2.arrowedLine(canvas, (mid_x, mid_y), arr_p2, tuple(line["color"]), 2, tipLength=0.3)
            
        cv2.putText(canvas, line["name"], (mid_x - 30, mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, tuple(line["color"]), 2)

    return canvas

def main() -> None:
    global current_pts, current_mode, active_name, manager, img_w, img_h
    
    parser = argparse.ArgumentParser(description="Restaurant Layout Calibrator GUI")
    parser.add_argument(
        "--video",
        type=str,
        default="videos/Dark_lighting.mp4",
        help="Input video file to pull reference frame from"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="configs/restaurant_default.yaml",
        help="Layout file name to output configs to"
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Run without displaying GUI (auto fallbacks to default config)"
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.is_absolute():
        if os.path.exists('/kaggle') or 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
            video_path = Path("/kaggle/input/datasets/tusharmarscitizen/video-analysis") / video_path.name
        else:
            video_path = PROJECT_ROOT / video_path

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    # Open Video and extract reference frame
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        log.error("Failed to read video frame. Exiting.")
        raise SystemExit(1)

    img_h, img_w = frame.shape[:2]
    manager = CalibrationManager(width=img_w, height=img_h)

    # Save background snapshot frame
    cal_dir = PROJECT_ROOT / "runs" / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(cal_dir / "background_snapshot.png"), frame)
    log.info(f"Saved background reference snapshot to {cal_dir}/background_snapshot.png")

    # Check for Headless or Non-interactive execution
    is_headless = os.environ.get("DISPLAY") is None and sys.platform.startswith("linux")
    if args.non_interactive or is_headless:
        log.info("Running calibrator in non-interactive/headless mode. Writing defaults...")
        generate_fallback_layout(output_path, manager)
        
        # Save preview files
        preview_img = render_layout_on_snapshot(frame, manager)
        cv2.imwrite(str(cal_dir / "preview.png"), preview_img)
        cv2.imwrite(str(cal_dir / "scene_layout.png"), preview_img)
        
        # Save markdown report
        write_report(cal_dir / "calibration_report.md", manager, video_path)
        log.info("Completed headless layout configuration.")
        return

    # Interactive GUI using OpenCV
    cv2.namedWindow("Restaurant Calibrator")
    cv2.setMouseCallback("Restaurant Calibrator", mouse_click)

    log.info("╔══════════════════════════════════════════╗")
    log.info("║  Scene Calibrator GUI Opened             ║")
    log.info("╚══════════════════════════════════════════╝")
    log.info("Controls:")
    log.info("  'n': Start drawing new polygon Zone")
    log.info("  'l': Start drawing new Counting Line")
    log.info("  'c': Close/Complete current Zone or Line")
    log.info("  'd': Delete last configured item")
    log.info("  's': Save configs and exit")
    log.info("  'q' or ESC: Quit without saving")

    while True:
        display_img = frame.copy()
        
        # Paint existing layout components
        display_img = render_layout_on_snapshot(display_img, manager)

        # Draw current active pts
        for pt in current_pts:
            cv2.circle(display_img, pt, 5, (0, 0, 255), -1)
        if len(current_pts) >= 2:
            if current_mode == "polygon":
                cv2.polylines(display_img, [np.array(current_pts, dtype=np.int32)], False, (0, 255, 255), 2)
            elif current_mode == "line":
                cv2.line(display_img, current_pts[0], current_pts[1], (255, 255, 0), 2)

        # Status Bar instructions
        status = f"Mode: {current_mode.upper()} | pts: {len(current_pts)} | n: New Zone | l: New Line | c: Save Item | s: Save Config | q: Quit"
        cv2.rectangle(display_img, (0, img_h - 40), (img_w, img_h), (0, 0, 0), -1)
        cv2.putText(display_img, status, (15, img_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Restaurant Calibrator", display_img)
        key = cv2.waitKey(30) & 0xFF

        if key == 27 or key == ord("q"):
            log.info("Quitting calibrator.")
            break

        elif key == ord("n"):
            # New Zone
            active_name = prompt_console("Enter name for new polygon Zone: ")
            if not active_name:
                active_name = f"Zone_{len(manager.zones)+1}"
            current_mode = "polygon"
            current_pts = []
            log.info(f"Drawing zone '{active_name}'. Click vertices on image.")

        elif key == ord("l"):
            # New Line
            active_name = prompt_console("Enter name for new Counting Line: ")
            if not active_name:
                active_name = f"Line_{len(manager.counting_lines)+1}"
            current_mode = "line"
            current_pts = []
            log.info(f"Drawing counting line '{active_name}'. Click starting point, then end point.")

        elif key == ord("c"):
            # Complete item
            if current_mode == "polygon":
                color = np.random.randint(50, 255, size=3).tolist()
                success, msg = manager.add_zone(active_name, current_pts, tuple(color))
                if success:
                    log.info(f"Added Zone: {active_name}")
                else:
                    log.error(f"Failed to add Zone: {msg}")
            elif current_mode == "line":
                if len(current_pts) < 2:
                    log.error("Lines require at least 2 points.")
                else:
                    direction = prompt_console("Enter line direction ('in' or 'out'): ") or "in"
                    color = [0, 165, 255] if direction == "in" else [255, 0, 128]
                    manager.add_counting_line(active_name, current_pts[0], current_pts[1], direction, tuple(color))
                    log.info(f"Added Line: {active_name}")
            current_mode = "idle"
            current_pts = []

        elif key == ord("d"):
            # Delete last item
            if manager.counting_lines:
                item = manager.counting_lines.pop()
                log.info(f"Deleted Line: {item['name']}")
            elif manager.zones:
                item = manager.zones.pop()
                log.info(f"Deleted Zone: {item['name']}")

        elif key == ord("s"):
            # Save and exit
            yaml_data = manager.export_yaml_data()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                yaml.safe_dump(yaml_data, f, default_flow_style=False)
            log.info(f"Layout configuration successfully saved to {output_path}")
            
            # Save previews
            preview_img = render_layout_on_snapshot(frame, manager)
            cv2.imwrite(str(cal_dir / "preview.png"), preview_img)
            cv2.imwrite(str(cal_dir / "scene_layout.png"), preview_img)
            
            write_report(cal_dir / "calibration_report.md", manager, video_path)
            break

    cv2.destroyAllWindows()

def write_report(report_path: Path, manager: CalibrationManager, video_path: Path) -> None:
    """Writes formatting metadata specs layout summary markdown report."""
    md = [
        "# Restaurant Scene Calibration Report",
        "",
        f"- **Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Source Video File:** `{video_path.name}`",
        f"- **Layout Resolution:** {manager.width}x{manager.height} pixels",
        f"- **Calibrated Polygon Zones:** {len(manager.zones)}",
        f"- **Calibrated Counting Lines:** {len(manager.counting_lines)}",
        "",
        "## Defined Polygon Zones List",
        "",
        "| Zone Name | Vertices Count | Color (BGR) |",
        "|---|---|---|",
    ]
    for z in manager.zones:
        md.append(f"| **{z['name']}** | {len(z['polygon'])} | `{list(z['color'])}` |")

    md += [
        "",
        "## Defined Counting Lines List",
        "",
        "| Line Name | Start Point | End Point | Crossing Direction |",
        "|---|---|---|---|",
    ]
    for cl in manager.counting_lines:
        md.append(f"| **{cl['name']}** | {cl['p1']} | {cl['p2']} | `{cl['direction']}` |")

    md += [
        "",
        "## Layout Overview Blueprint",
        "",
        "Below is the reference snapshot blueprint showing overlay boundaries:",
        "",
        "![Scene Layout blueprint](preview.png)",
        ""
    ]
    with open(report_path, "w") as fh:
        fh.write("\n".join(md))
    log.info(f"Saved calibration blueprint specs to {report_path}")

if __name__ == "__main__":
    main()
