"""Quality control checks for 3D reconstruction results."""
from __future__ import annotations

from typing import Dict

import numpy as np

from .constants import KEYPOINTS, LEFT_RIGHT_PAIRS, SKELETON
from .geometry import TriangulationEngine
from .models import CameraCalibration, FrameAnnotations


class QualityChecker:
    @staticmethod
    def evaluate(frame: FrameAnnotations, calibrations: Dict[str, CameraCalibration]) -> dict:
        report = {"reprojection": TriangulationEngine.reprojection_errors(frame, calibrations)}
        pts = [np.array(p, dtype=float) if p is not None else None for p in frame.points3d]
        bone_lengths = {}
        for a, b in SKELETON:
            if pts[a] is not None and pts[b] is not None:
                bone_lengths[f"{KEYPOINTS[a]}-{KEYPOINTS[b]}"] = float(np.linalg.norm(pts[a] - pts[b]))
        symmetry = {}
        for left, right in LEFT_RIGHT_PAIRS:
            root = 1 if left in (4, 5) else 2
            left_len = float(np.linalg.norm(pts[left] - pts[root])) if pts[left] is not None and pts[root] is not None else None
            right_len = float(np.linalg.norm(pts[right] - pts[root])) if pts[right] is not None and pts[root] is not None else None
            if left_len is not None and right_len is not None:
                symmetry[f"{KEYPOINTS[left]}-{KEYPOINTS[right]}"] = abs(left_len - right_len)
        anomalies = []
        for cid, values in report["reprojection"].items():
            for idx, err in enumerate(values):
                if err is not None and err > 18.0:
                    anomalies.append({"camera": cid, "keypoint": KEYPOINTS[idx], "type": "reprojection", "value": err})
        for name, diff in symmetry.items():
            if diff > 0.25:
                anomalies.append({"pair": name, "type": "symmetry", "value": diff})
        report["bone_lengths"] = bone_lengths
        report["symmetry"] = symmetry
        report["anomalies"] = anomalies
        return report
