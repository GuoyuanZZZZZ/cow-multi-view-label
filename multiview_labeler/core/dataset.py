"""Dataset import utilities for multi-camera image and video sources."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from .calibration import CalibrationValidator, CalibrationIO
from .models import CameraCalibration


class MultiCameraDataset:
    def __init__(self) -> None:
        self.cameras: Dict[str, List[Path]] = {}
        self.calibrations: Dict[str, CameraCalibration] = {}
        self.frame_count = 0
        self.image_size = (960, 640)
        self.sync_map: List[Dict[str, int]] = []

    def import_images(self, camera_dirs: Dict[str, Path]) -> None:
        self.cameras = {}
        for camera_id, path in camera_dirs.items():
            files = sorted(
                [p for p in path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}]
            )
            if not files:
                raise ValueError(f"No images found for camera {camera_id} in {path}")
            self.cameras[camera_id] = files
        counts = {len(v) for v in self.cameras.values()}
        if len(counts) != 1:
            raise ValueError("Camera image sequences must have equal frame count for synchronization")
        self.frame_count = counts.pop()
        sample = cv2.imread(str(next(iter(self.cameras.values()))[0]))
        self.image_size = (sample.shape[1], sample.shape[0])
        self.sync_map = [{cid: idx for cid in self.cameras} for idx in range(self.frame_count)]

    def import_videos(self, video_paths: Dict[str, Path], extracted_root: Path, sample_stride: int = 1) -> None:
        extracted_root.mkdir(parents=True, exist_ok=True)
        extracted_dirs: Dict[str, Path] = {}
        for camera_id, video_path in video_paths.items():
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                raise ValueError(f"Could not open video for camera {camera_id}: {video_path}")
            out_dir = extracted_root / camera_id
            out_dir.mkdir(parents=True, exist_ok=True)
            frame_idx = 0
            saved_idx = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_idx % max(1, sample_stride) == 0:
                    cv2.imwrite(str(out_dir / f"frame_{saved_idx:06d}.png"), frame)
                    saved_idx += 1
                frame_idx += 1
            capture.release()
            extracted_dirs[camera_id] = out_dir
        self.import_images(extracted_dirs)

    def get_frame(self, camera_id: str, frame_idx: int) -> np.ndarray:
        return cv2.imread(str(self.cameras[camera_id][frame_idx]))

    def camera_ids(self) -> List[str]:
        return list(self.cameras.keys())

    def save_demo_calibration(self, path: Path) -> None:
        width, height = self.image_size
        calibrations = {}
        presets = [
            (-25, -2.5, 0.0, 0.0, 8.0),   # left side
            (25, 2.5, 0.0, 0.0, 8.0),    # right side
            (0, 0.0, -85.0, -5.0, 12.0), # top
            (-10, -0.9, 55.0, 1.2, 5.5), # ground front-left
            (10, 0.9, 55.0, 1.2, 5.5),   # ground front-right
            (-10, -0.9, 55.0, -1.2, 5.5),# ground rear-left
            (10, 0.9, 55.0, -1.2, 5.5),  # ground rear-right
        ]
        for idx, camera_id in enumerate(self.camera_ids()):
            yaw, tx, pitch, ty, tz = presets[min(idx, len(presets) - 1)]
            calibrations[camera_id] = CalibrationValidator.demo_calibrate(width, height, camera_id, yaw, tx, pitch_deg=pitch, ty=ty, tz=tz)
        CalibrationIO.save(path, calibrations)
        self.calibrations = calibrations
