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

    def _project_keypoints(self, calib, points: np.ndarray) -> list[tuple[float, float] | None]:
        width, height = self.image_size
        projected: list[tuple[float, float] | None] = []
        for pt in points:
            uv = calib.project(pt)
            if 0 <= uv[0] < width and 0 <= uv[1] < height:
                projected.append((float(uv[0]), float(uv[1])))
            else:
                projected.append(None)
        return projected

    def _draw_cow(self, img: np.ndarray, projected: list[tuple[float, float] | None], visible: set[int]) -> None:
        body_points = [projected[idx] for idx in [0, 1, 2, 3] if projected[idx] is not None]
        hoof_points = [projected[idx] for idx in [4, 5, 6, 7] if projected[idx] is not None]
        if len(body_points) >= 2:
            xs = [pt[0] for pt in body_points]
            ys = [pt[1] for pt in body_points]
            center = (int(sum(xs) / len(xs)), int(sum(ys) / len(ys)))
            axis_major = max(36, int((max(xs) - min(xs)) * 0.55) + 20)
            axis_minor = max(28, int((max(ys) - min(ys)) * 0.45) + 18)
            cv2.ellipse(img, center, (axis_major, axis_minor), 0, 0, 360, (228, 223, 210), -1)
            cv2.ellipse(img, center, (axis_major, axis_minor), 0, 0, 360, (92, 80, 68), 2)
            patch_offsets = [(-axis_major // 3, -axis_minor // 5), (axis_major // 5, axis_minor // 8)]
            for off_x, off_y in patch_offsets:
                cv2.ellipse(img, (center[0] + off_x, center[1] + off_y), (max(10, axis_major // 5), max(8, axis_minor // 4)), -15, 0, 360, (70, 55, 45), -1)
        if projected[0] is not None and projected[1] is not None:
            nose = projected[0]
            neck = projected[1]
            head_center = (int((nose[0] * 0.65 + neck[0] * 0.35)), int((nose[1] * 0.65 + neck[1] * 0.35)))
            head_axes = (26, 18)
            cv2.ellipse(img, head_center, head_axes, -10, 0, 360, (228, 223, 210), -1)
            cv2.ellipse(img, head_center, head_axes, -10, 0, 360, (92, 80, 68), 2)
            ear_dx = 14 if nose[0] >= neck[0] else -14
            cv2.fillConvexPoly(img, np.array([[head_center[0], head_center[1] - 10], [head_center[0] + ear_dx, head_center[1] - 24], [head_center[0] + ear_dx // 2, head_center[1] - 4]], dtype=np.int32), (92, 80, 68))
        leg_pairs = [(1, 4), (1, 5), (2, 6), (2, 7)]
        for upper, hoof in leg_pairs:
            if projected[upper] is None or projected[hoof] is None:
                continue
            p0 = tuple(int(v) for v in projected[upper])
            p1 = tuple(int(v) for v in projected[hoof])
            cv2.line(img, p0, p1, (92, 80, 68), 5)
        if projected[3] is not None:
            tail_start = tuple(int(v) for v in projected[3])
            tail_end = (tail_start[0] - 24, tail_start[1] - 30)
            cv2.line(img, tail_start, tail_end, (92, 80, 68), 3)
            cv2.circle(img, tail_end, 4, (70, 55, 45), -1)
        ground_y = int(max([pt[1] for pt in hoof_points], default=img.shape[0] * 0.78) + 10)
        ground_y = max(0, min(img.shape[0] - 1, ground_y))
        cv2.line(img, (0, ground_y), (img.shape[1], ground_y), (190, 198, 182), 2)

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
                projected = self._project_keypoints(calib, points)
                self._draw_cow(img, projected, visible)
                keypoints_payload = []
                for i, uv in enumerate(projected):
                    if i in visible and uv is not None:
                        uv_int = (int(uv[0]), int(uv[1]))
                        cv2.circle(img, uv_int, 7, (30, 30, 220), -1)
                        cv2.putText(img, KEYPOINTS[i], (uv_int[0] + 8, uv_int[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (70, 70, 70), 1)
                        keypoints_payload.append({"x": float(uv[0]), "y": float(uv[1]), "visibility": "visible"})
                    else:
                        keypoints_payload.append({"x": None, "y": None, "visibility": "absent"})
                frame_payload["2d"][cid] = keypoints_payload
                for a, b in SKELETON:
                    if a in visible and b in visible and projected[a] is not None and projected[b] is not None:
                        cv2.line(img, tuple(int(v) for v in projected[a]), tuple(int(v) for v in projected[b]), (80, 140, 80), 2)
                cv2.putText(img, f"{cid} frame {frame_idx:03d}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 50, 50), 2)
                cv2.imwrite(str(self.root / cid / f"frame_{frame_idx:03d}.png"), img)
            reference_payload[str(frame_idx)] = {"0": frame_payload}
        CalibrationIO.save(self.root / "demo_calibration.json", calibrations)
        (self.root / "reference_annotations.json").write_text(json.dumps(reference_payload, indent=2), encoding="utf-8")
        return self.root
