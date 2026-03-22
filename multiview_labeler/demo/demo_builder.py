"""Synthetic demo data generation."""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from multiview_labeler.core.calibration import CalibrationIO, CalibrationValidator
from multiview_labeler.core.constants import KEYPOINTS, SKELETON


class DemoDataBuilder:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.image_size = (960, 640)

    def build(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        cams = ["cam01", "cam02", "cam03"]
        calibrations = {}
        width, height = self.image_size
        for idx, cid in enumerate(cams):
            (self.root / cid).mkdir(exist_ok=True)
            calibrations[cid] = CalibrationValidator.demo_calibrate(width, height, cid, -12 + idx * 12, -1.2 + idx * 1.2)
        points_template = np.array(
            [
                [0.0, 0.9, 0.0], [0.0, 0.45, 0.0], [0.0, 0.0, 0.0], [0.0, -0.6, 0.0],
                [-0.35, 0.35, 0.25], [0.35, 0.35, 0.25], [-0.35, -0.15, 0.25], [0.35, -0.15, 0.25],
            ],
            dtype=float,
        )
        for frame_idx in range(20):
            points = points_template.copy()
            points[:, 0] += math.sin(frame_idx / 3.0) * 0.25
            points[[0, 3], 1] += math.cos(frame_idx / 4.0) * 0.08
            points[:, 2] += math.sin(frame_idx / 5.0) * 0.2
            for cid, calib in calibrations.items():
                img = np.full((height, width, 3), (245, 248, 252), dtype=np.uint8)
                for i, pt in enumerate(points):
                    uv = calib.project(pt).astype(int)
                    if 0 <= uv[0] < width and 0 <= uv[1] < height:
                        cv2.circle(img, tuple(uv), 7, (30, 30, 220), -1)
                        cv2.putText(img, KEYPOINTS[i], (uv[0] + 8, uv[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (70, 70, 70), 1)
                for a, b in SKELETON:
                    cv2.line(img, tuple(calib.project(points[a]).astype(int)), tuple(calib.project(points[b]).astype(int)), (80, 140, 80), 2)
                cv2.putText(img, f"{cid} frame {frame_idx:03d}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 50, 50), 2)
                cv2.imwrite(str(self.root / cid / f"frame_{frame_idx:03d}.png"), img)
        CalibrationIO.save(self.root / "demo_calibration.json", calibrations)
        return self.root
