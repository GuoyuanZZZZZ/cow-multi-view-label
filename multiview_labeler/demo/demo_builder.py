"""Synthetic demo data generation."""
from __future__ import annotations

import json
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
        cams = ["cam_left", "cam_right", "cam_top", "cam_ground_fl", "cam_ground_fr", "cam_ground_rl", "cam_ground_rr"]
        calibrations = {}
        width, height = self.image_size
        presets = [
            (-25, -2.5, 0.0, 0.0, 8.0),
            (25, 2.5, 0.0, 0.0, 8.0),
            (0, 0.0, -85.0, -5.0, 12.0),
            (-10, -0.9, 55.0, 1.2, 5.5),
            (10, 0.9, 55.0, 1.2, 5.5),
            (-10, -0.9, 55.0, -1.2, 5.5),
            (10, 0.9, 55.0, -1.2, 5.5),
        ]
        for idx, cid in enumerate(cams):
            (self.root / cid).mkdir(exist_ok=True)
            yaw, tx, pitch, ty, tz = presets[idx]
            calibrations[cid] = CalibrationValidator.demo_calibrate(width, height, cid, yaw, tx, pitch_deg=pitch, ty=ty, tz=tz)
        points_template = np.array(
            [
                [0.0, 0.9, 0.0], [0.0, 0.45, 0.0], [0.0, 0.0, 0.0], [0.0, -0.6, 0.0],
                [-0.35, 0.35, 0.25], [0.35, 0.35, 0.25], [-0.35, -0.15, 0.25], [0.35, -0.15, 0.25],
            ],
            dtype=float,
        )

        def visible_indices(camera_id: str) -> set[int]:
            masks = {
                "cam_left": {0, 1, 2, 3, 4, 6},
                "cam_right": {0, 1, 2, 3, 5, 7},
                "cam_top": set(range(len(KEYPOINTS))),
                "cam_ground_fl": {0, 1, 2, 4, 5, 6},
                "cam_ground_fr": {0, 1, 2, 4, 5, 7},
                "cam_ground_rl": {1, 2, 3, 4, 6, 7},
                "cam_ground_rr": {1, 2, 3, 5, 6, 7},
            }
            return masks.get(camera_id, set(range(len(KEYPOINTS))))

        reference_payload = {}
        for frame_idx in range(20):
            points = points_template.copy()
            points[:, 0] += math.sin(frame_idx / 3.0) * 0.25
            points[[0, 3], 1] += math.cos(frame_idx / 4.0) * 0.08
            points[:, 2] += math.sin(frame_idx / 5.0) * 0.2
            frame_payload = {"2d": {}, "3d": [pt.tolist() for pt in points], "qc": {}}
            for cid, calib in calibrations.items():
                img = np.full((height, width, 3), (245, 248, 252), dtype=np.uint8)
                visible = visible_indices(cid)
                keypoints_payload = []
                for i, pt in enumerate(points):
                    if i in visible:
                        uv = calib.project(pt)
                        visible_in_frame = 0 <= uv[0] < width and 0 <= uv[1] < height
                        if visible_in_frame:
                            uv_int = uv.astype(int)
                            cv2.circle(img, tuple(uv_int), 7, (30, 30, 220), -1)
                            cv2.putText(img, KEYPOINTS[i], (uv_int[0] + 8, uv_int[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (70, 70, 70), 1)
                            keypoints_payload.append({"x": float(uv[0]), "y": float(uv[1]), "visibility": "visible"})
                        else:
                            keypoints_payload.append({"x": None, "y": None, "visibility": "absent"})
                    else:
                        keypoints_payload.append({"x": None, "y": None, "visibility": "absent"})
                frame_payload["2d"][cid] = keypoints_payload
                for a, b in SKELETON:
                    if a in visible and b in visible:
                        cv2.line(img, tuple(calib.project(points[a]).astype(int)), tuple(calib.project(points[b]).astype(int)), (80, 140, 80), 2)
                cv2.putText(img, f"{cid} frame {frame_idx:03d}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 50, 50), 2)
                cv2.imwrite(str(self.root / cid / f"frame_{frame_idx:03d}.png"), img)
            reference_payload[str(frame_idx)] = {"0": frame_payload}
        CalibrationIO.save(self.root / "demo_calibration.json", calibrations)
        (self.root / "reference_annotations.json").write_text(json.dumps(reference_payload, indent=2), encoding="utf-8")
        return self.root
