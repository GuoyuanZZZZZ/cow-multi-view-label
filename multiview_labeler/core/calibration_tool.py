"""OpenCV-based chessboard calibration workflow."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np

from .models import CameraCalibration


@dataclass
class ChessboardSpec:
    pattern_size: Tuple[int, int] = (9, 6)
    square_size: float = 1.0

    def object_points(self) -> np.ndarray:
        cols, rows = self.pattern_size
        grid = np.zeros((rows * cols, 3), np.float32)
        grid[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        grid *= self.square_size
        return grid


class CalibrationTool:
    """Calibration helper inspired by common annotation/calibration tools.

    The workflow is:
    1. Detect chessboard corners for each camera.
    2. Calibrate intrinsics with ``cv2.calibrateCamera``.
    3. Estimate extrinsics with ``cv2.solvePnP`` using one shared board image.
    """

    @staticmethod
    def _iter_images(image_dir: Path) -> Iterable[Path]:
        for path in sorted(image_dir.iterdir()):
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                yield path

    @staticmethod
    def collect_corners(image_dir: Path, spec: ChessboardSpec) -> tuple[list[np.ndarray], list[np.ndarray], tuple[int, int]]:
        object_points: List[np.ndarray] = []
        image_points: List[np.ndarray] = []
        image_size = None
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        for image_path in CalibrationTool._iter_images(image_dir):
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_size = (gray.shape[1], gray.shape[0])
            ok, corners = cv2.findChessboardCorners(gray, spec.pattern_size)
            if not ok:
                continue
            refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            object_points.append(spec.object_points())
            image_points.append(refined)
        if not object_points or image_size is None:
            raise ValueError(f"No chessboard corners detected in {image_dir}")
        return object_points, image_points, image_size

    @staticmethod
    def calibrate_intrinsics(camera_id: str, image_dir: Path, spec: ChessboardSpec) -> CameraCalibration:
        object_points, image_points, image_size = CalibrationTool.collect_corners(image_dir, spec)
        ok, k, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, image_size, None, None)
        if not ok:
            raise ValueError(f"Calibration failed for camera {camera_id}")
        rmat, _ = cv2.Rodrigues(rvecs[0])
        return CameraCalibration(camera_id=camera_id, K=k, distCoeffs=dist.reshape(-1), R=rmat, t=tvecs[0].reshape(3))

    @staticmethod
    def calibrate_multi_camera(camera_image_dirs: Dict[str, Path], spec: ChessboardSpec) -> Dict[str, CameraCalibration]:
        calibrations = {
            camera_id: CalibrationTool.calibrate_intrinsics(camera_id, image_dir, spec)
            for camera_id, image_dir in camera_image_dirs.items()
        }
        reference_object = spec.object_points()
        for camera_id, image_dir in camera_image_dirs.items():
            _, image_points, _ = CalibrationTool.collect_corners(image_dir, spec)
            ok, rvec, tvec = cv2.solvePnP(reference_object, image_points[0], calibrations[camera_id].K, calibrations[camera_id].distCoeffs)
            if not ok:
                raise ValueError(f"Extrinsic solvePnP failed for camera {camera_id}")
            calibrations[camera_id].R = cv2.Rodrigues(rvec)[0]
            calibrations[camera_id].t = tvec.reshape(3)
        return calibrations
