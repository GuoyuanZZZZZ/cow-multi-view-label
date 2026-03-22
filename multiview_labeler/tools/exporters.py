"""Export helpers for annotations and QC reports."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from multiview_labeler.core.annotation import AnnotationManager
from multiview_labeler.core.constants import KEYPOINTS


class Exporter:
    @staticmethod
    def export_json(path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def export_2d_csv(path: Path, annotations: AnnotationManager) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "instance", "camera", "keypoint", "x", "y", "visibility"])
            for frame_idx, instances in sorted(annotations.frames.items()):
                for instance_idx, frame in sorted(instances.items()):
                    for cid, kps in frame.by_camera.items():
                        for idx, kp in enumerate(kps):
                            writer.writerow([frame_idx, instance_idx, cid, KEYPOINTS[idx], kp.x, kp.y, kp.visibility])

    @staticmethod
    def export_3d_csv(path: Path, annotations: AnnotationManager) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "instance", "keypoint", "x", "y", "z"])
            for frame_idx, instances in sorted(annotations.frames.items()):
                for instance_idx, frame in sorted(instances.items()):
                    for idx, point in enumerate(frame.points3d):
                        writer.writerow([frame_idx, instance_idx, KEYPOINTS[idx], *(point or [None, None, None])])
