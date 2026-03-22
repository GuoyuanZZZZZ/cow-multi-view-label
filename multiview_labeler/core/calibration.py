"""Calibration generation, validation, save/load."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict

import numpy as np

from .models import CameraCalibration


class CalibrationIO:
    @staticmethod
    def save(path: Path, calibrations: Dict[str, CameraCalibration]) -> None:
        payload = {cid: calib.to_json() for cid, calib in calibrations.items()}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def load(path: Path) -> Dict[str, CameraCalibration]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {cid: CameraCalibration.from_json(data) for cid, data in payload.items()}


class CalibrationValidator:
    @staticmethod
    def validate_projection(calibration: CameraCalibration, points3d: np.ndarray) -> float:
        reprojected = np.array([calibration.project(pt) for pt in points3d])
        return float(np.mean(np.linalg.norm(reprojected - reprojected.mean(axis=0), axis=1)))

    @staticmethod
    def demo_calibrate(width: int, height: int, camera_id: str, yaw_deg: float, tx: float) -> CameraCalibration:
        fx = fy = 900.0
        k = np.array([[fx, 0, width / 2], [0, fy, height / 2], [0, 0, 1]], dtype=np.float64)
        dist = np.zeros(5, dtype=np.float64)
        yaw = math.radians(yaw_deg)
        r = np.array(
            [
                [math.cos(yaw), 0, math.sin(yaw)],
                [0, 1, 0],
                [-math.sin(yaw), 0, math.cos(yaw)],
            ],
            dtype=np.float64,
        )
        t = np.array([tx, 0.0, 8.0], dtype=np.float64)
        return CameraCalibration(camera_id=camera_id, K=k, distCoeffs=dist, R=r, t=t)
