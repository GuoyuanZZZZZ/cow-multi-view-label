"""Data models used by the app."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np


@dataclass
class CameraCalibration:
    camera_id: str
    K: np.ndarray
    distCoeffs: np.ndarray
    R: np.ndarray
    t: np.ndarray

    def projection_matrix(self) -> np.ndarray:
        return self.K @ np.hstack([self.R, self.t.reshape(3, 1)])

    def project(self, xyz: np.ndarray) -> np.ndarray:
        pts2d, _ = cv2.projectPoints(
            xyz.reshape(1, 1, 3).astype(np.float32),
            cv2.Rodrigues(self.R)[0],
            self.t.astype(np.float32),
            self.K.astype(np.float32),
            self.distCoeffs.astype(np.float32),
        )
        return pts2d.reshape(2)

    def to_json(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "K": self.K.tolist(),
            "distCoeffs": self.distCoeffs.tolist(),
            "R": self.R.tolist(),
            "t": self.t.tolist(),
        }

    @staticmethod
    def from_json(data: dict) -> "CameraCalibration":
        return CameraCalibration(
            camera_id=data["camera_id"],
            K=np.array(data["K"], dtype=np.float64),
            distCoeffs=np.array(data["distCoeffs"], dtype=np.float64),
            R=np.array(data["R"], dtype=np.float64),
            t=np.array(data["t"], dtype=np.float64),
        )


@dataclass
class Keypoint2D:
    x: Optional[float] = None
    y: Optional[float] = None
    visibility: str = "absent"

    def is_valid(self) -> bool:
        return self.visibility != "absent" and self.x is not None and self.y is not None

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class FrameAnnotations:
    by_camera: Dict[str, List[Keypoint2D]] = field(default_factory=dict)
    points3d: List[Optional[List[float]]] = field(default_factory=list)
    qc: dict = field(default_factory=dict)


def ensure_keypoints(count: int) -> List[Keypoint2D]:
    return [Keypoint2D() for _ in range(count)]
