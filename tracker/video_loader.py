import cv2
import logging
from pathlib import Path
from typing import Generator, Tuple

logger = logging.getLogger("AurikaTracking")

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
        
        logger.info(f"Video loaded: {self.video_path.name} | "
                    f"Resolution: {self.width}x{self.height} | "
                    f"FPS: {self.fps:.2f} | Total Frames: {self.frame_count}")

    def frames(self) -> Generator[cv2.Mat, None, None]:
        """Generator to yield video frames sequentially."""
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                yield frame
        finally:
            self.release()

    def get_resolution(self) -> Tuple[int, int]:
        return self.width, self.height

    def release(self) -> None:
        """Release the video capture resource."""
        if self.cap.isOpened():
            self.cap.release()
            logger.info("Video resource released.")
