#!/usr/bin/env python3
import os
import cv2
import time
import argparse
import logging
from pathlib import Path

# Set up logging format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("AurikaTracking")

from tracker.config_loader import PipelineConfig
from tracker.model_loader import load_yolo_model
from tracker.video_loader import VideoLoader
from tracker.tracking_engine import TrackingEngine
from tracker.visualization import annotate_frame

def main():
    parser = argparse.ArgumentParser(description="Aurika Person Tracking Pipeline (v2)")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/config.yaml", 
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--max-frames", 
        type=int, 
        default=-1, 
        help="Maximum number of frames to process (-1 for all)"
    )
    args = parser.parse_args()

    logger.info("Initializing Aurika Tracking Pipeline...")
    
    # 1. Load pipeline configuration
    try:
        config = PipelineConfig(args.config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return

    # 2. Load YOLO model
    try:
        model = load_yolo_model(config.model_path)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return

    # 3. Load Video
    try:
        video_loader = VideoLoader(config.video_path)
    except Exception as e:
        logger.error(f"Error loading video: {e}")
        return

    # 4. Initialize Tracking Engine
    try:
        tracker = TrackingEngine(config.tracker)
    except Exception as e:
        logger.error(f"Error initializing tracking engine: {e}")
        return

    # 5. Setup output file
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video_path = output_dir / "output.mp4"
    logger.info(f"Target output video: {output_video_path}")

    # Set up OpenCV VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        str(output_video_path),
        fourcc,
        video_loader.fps,
        video_loader.get_resolution()
    )

    if not out.isOpened():
        logger.error(f"Failed to create VideoWriter for path: {output_video_path}")
        return

    # 6. Pipeline Run Loop
    frame_idx = 0
    start_time = time.time()
    
    logger.info("Starting processing loop. Please wait...")

    try:
        for frame in video_loader.frames():
            frame_idx += 1
            if args.max_frames > 0 and frame_idx > args.max_frames:
                logger.info(f"Reached max-frames limit of {args.max_frames}. Stopping.")
                break
            
            # Predict using YOLO model (set low confidence to capture candidates for ByteTrack)
            # Conf threshold is configured as min(0.01, track_low_thresh) to allow the tracker's high/low thresholds to filter
            conf_thresh = min(0.01, config.tracker.track_low_thresh)
            
            # Run YOLO prediction
            # verbose=False reduces terminal clutter during execution
            results = model.predict(frame, conf=conf_thresh, verbose=False)[0]
            
            # Run tracking update (automatically filters to only person classes)
            tracks = tracker.update(results, frame)
            
            # Annotate frame
            annotated_frame = annotate_frame(frame, tracks)
            
            # Write frame to output
            out.write(annotated_frame)
            
            # Periodic logging
            if frame_idx % 50 == 0 or frame_idx == video_loader.frame_count:
                progress = (frame_idx / video_loader.frame_count) * 100 if video_loader.frame_count > 0 else 0
                logger.info(f"Processed frame {frame_idx}/{video_loader.frame_count} ({progress:.1f}%)")
                
    except Exception as e:
        logger.error(f"An unexpected error occurred during frame processing: {e}")
        raise e
    finally:
        # Clean release of resources
        out.release()
        video_loader.release()
        
    duration = time.time() - start_time
    logger.info(f"Pipeline completed successfully in {duration:.2f} seconds.")
    logger.info(f"Final output video saved to: {output_video_path.resolve()}")

if __name__ == "__main__":
    main()
