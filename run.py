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
from tracker.video_loader import VideoLoader, TERM_EOF, TERM_READ_FAILURE
from tracker.tracking_engine import TrackingEngine
from tracker.visualization import annotate_frame
from tracker.device import get_device

DEVICE = get_device()

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
        model.to(DEVICE)
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
    stopped_by_max_frames = False  # True when the caller limit (--max-frames) triggered the stop
    
    logger.info("Starting processing loop. Please wait...")

    try:
        for frame in video_loader.frames():
            frame_idx += 1
            if args.max_frames > 0 and frame_idx > args.max_frames:
                stopped_by_max_frames = True
                # Override the termination reason so video_loader knows this
                # was a deliberate caller-side stop, not a codec failure.
                video_loader.termination_reason = TERM_EOF
                logger.info(f"Reached max-frames limit of {args.max_frames}. Stopping.")
                break
            
            # Predict using YOLO model (set low confidence to capture candidates for ByteTrack)
            # Conf threshold is configured as min(0.01, track_low_thresh) to allow the tracker's high/low thresholds to filter
            conf_thresh = min(0.01, config.tracker.track_low_thresh)
            
            # Run YOLO prediction
            # verbose=False reduces terminal clutter during execution
            results = model.predict(frame, conf=conf_thresh, device=DEVICE, verbose=False)[0]
            
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

    # ------------------------------------------------------------------
    # Termination diagnostics
    # ------------------------------------------------------------------
    term_reason = video_loader.termination_reason
    last_frame  = video_loader.last_frame_read  # last frame successfully decoded

    if term_reason == TERM_READ_FAILURE:
        # cap.read() returned False well before the expected frame count.
        # This is consistent with a corrupted input video (e.g. H.264 NAL
        # errors, missing picture headers, truncated bitstream).
        logger.error(
            f"PIPELINE TERMINATED DUE TO CORRUPTED INPUT VIDEO: "
            f"{config.video_path!r}\n"
            f"  Last successfully processed frame : {last_frame} "
            f"(of {video_loader.frame_count} expected)\n"
            f"  Reason                            : VideoCapture.read() returned False "
            f"at frame ~{last_frame + 1} — likely H.264 NAL / decoder error\n"
            f"  Elapsed time                      : {duration:.2f}s\n"
            f"  Partial output saved to           : {output_video_path.resolve()}\n"
            f"Action: Inspect the source video with 'ffprobe' or re-encode it with "
            f"'ffmpeg -i input.mp4 -c copy output.mp4' before re-running."
        )
    elif stopped_by_max_frames:
        logger.info(
            f"Pipeline stopped at --max-frames limit ({args.max_frames} frames) "
            f"in {duration:.2f}s. Last frame processed: {last_frame}. "
            f"Output saved to: {output_video_path.resolve()}"
        )
    else:
        logger.info(
            f"Pipeline completed successfully in {duration:.2f}s. "
            f"Frames processed: {last_frame}/{video_loader.frame_count}. "
            f"Output saved to: {output_video_path.resolve()}"
        )

if __name__ == "__main__":
    main()
