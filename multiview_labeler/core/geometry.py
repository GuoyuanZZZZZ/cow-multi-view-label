"""Triangulation, reprojection, and epipolar geometry."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .constants import KEYPOINTS
from .models import CameraCalibration, FrameAnnotations, Keypoint2D


class TriangulationEngine:
    @staticmethod
    def triangulate_point(observations: List[Tuple[CameraCalibration, Keypoint2D]]) -> Optional[np.ndarray]:
        if len(observations) < 2:
            return None
        rows = []
        for calib, kp in observations:
            if not kp.is_valid():
                continue
            p = calib.projection_matrix()
            rows.append(kp.x * p[2] - p[0])
            rows.append(kp.y * p[2] - p[1])
        if len(rows) < 4:
            return None
        _, _, vh = np.linalg.svd(np.array(rows))
        x = vh[-1]
        return (x[:3] / x[3]).astype(float)

    @staticmethod
    def triangulate_frame(frame: FrameAnnotations, calibrations: Dict[str, CameraCalibration]) -> List[Optional[List[float]]]:
        points3d: List[Optional[List[float]]] = []
        for kp_idx in range(len(KEYPOINTS)):
            observations = []
            for cid, calib in calibrations.items():
                kp = frame.by_camera[cid][kp_idx]
                if kp.is_valid():
                    observations.append((calib, kp))
            point = TriangulationEngine.triangulate_point(observations)
            points3d.append(point.tolist() if point is not None else None)
        return points3d

    @staticmethod
    def reprojection_errors(frame: FrameAnnotations, calibrations: Dict[str, CameraCalibration]) -> Dict[str, List[Optional[float]]]:
        errors: Dict[str, List[Optional[float]]] = {cid: [None] * len(KEYPOINTS) for cid in calibrations}
        for kp_idx, point in enumerate(frame.points3d):
            if point is None:
                continue
            xyz = np.array(point, dtype=float)
            for cid, calib in calibrations.items():
                kp = frame.by_camera[cid][kp_idx]
                if kp.is_valid():
                    uv = calib.project(xyz)
                    errors[cid][kp_idx] = float(np.linalg.norm(uv - np.array([kp.x, kp.y])))
        return errors

    @staticmethod
    def epipolar_line(calib_a: CameraCalibration, calib_b: CameraCalibration, point_a: Keypoint2D) -> Optional[np.ndarray]:
        if not point_a.is_valid():
            return None
        r_rel = calib_b.R @ calib_a.R.T
        t_rel = calib_b.t.reshape(3, 1) - r_rel @ calib_a.t.reshape(3, 1)
        tx = np.array(
            [[0, -t_rel[2, 0], t_rel[1, 0]], [t_rel[2, 0], 0, -t_rel[0, 0]], [-t_rel[1, 0], t_rel[0, 0], 0]],
            dtype=float,
        )
        f = np.linalg.inv(calib_b.K).T @ tx @ r_rel @ np.linalg.inv(calib_a.K)
        return f @ np.array([point_a.x, point_a.y, 1.0])
