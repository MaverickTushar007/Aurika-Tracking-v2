import cv2
import logging
from pathlib import Path
from typing import Generator, Tuple

logger = logging.getLogger("AurikaTracking")

# Termination reason constants exposed for callers
TERM_EOF = "clean_eof"          # cap.read() returned False at expected end-of-stream
TERM_READ_FAILURE = "read_failure"  # cap.read() returned False unexpectedly (corrupt video / NAL error)
TERM_RELEASED = "not_started"   # generator never ran

class VideoLoader:
    """Helper wrapper for OpenCV VideoCapture."""
    def __init__(self, video_path: str):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found at {video_path}")
            
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise IOError(f"Could not open video file {video_path}")
            
        self.width: int = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height: int = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps: float = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_count: int = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # --- Termination diagnostics (populated by frames()) ---
        # The 1-based index of the last frame successfully read and yielded.
        self.last_frame_read: int = 0
        # One of the TERM_* constants above; set when the frames() generator exits.
        self.termination_reason: str = TERM_RELEASED
        
        logger.info(f"Video loaded: {self.video_path.name} | "
                    f"Resolution: {self.width}x{self.height} | "
                    f"FPS: {self.fps:.2f} | Total Frames: {self.frame_count}")

    def frames(self) -> Generator[cv2.Mat, None, None]:
        """Generator to yield video frames sequentially.

        After the generator is exhausted (or the caller breaks early), inspect
        ``self.last_frame_read`` for the index of the last successfully decoded
        frame and ``self.termination_reason`` for *why* iteration stopped:

        * ``TERM_EOF``          – all frames were read; clean end-of-stream.
        * ``TERM_READ_FAILURE`` – ``cap.read()`` returned False before reaching
          the expected frame count (corrupt video / H.264 NAL errors).
        """
        frames_yielded = 0
        read_failed = False
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    read_failed = True
                    break
                frames_yielded += 1
                self.last_frame_read = frames_yielded
                yield frame
        finally:
            # Determine why the loop ended.
            # A "clean EOF" is when we read at least as many frames as the
            # container reports (frame_count may be slightly off by 1 due to
            # container metadata; allow a small tolerance of 5 frames).
            if read_failed:
                tolerance = 5
                expected = self.frame_count
                if expected > 0 and frames_yielded >= (expected - tolerance):
                    # read() failed right at the expected boundary — treat as EOF
                    self.termination_reason = TERM_EOF
                else:
                    self.termination_reason = TERM_READ_FAILURE
            else:
                # Generator was exhausted without a failed read (e.g. max-frames
                # break from the caller side — caller sets reason separately).
                self.termination_reason = TERM_EOF
            self.release()

    def get_resolution(self) -> Tuple[int, int]:
        return self.width, self.height

    def release(self) -> None:
        """Release the video capture resource."""
        if self.cap.isOpened():
            self.cap.release()
            logger.info("Video resource released.")
