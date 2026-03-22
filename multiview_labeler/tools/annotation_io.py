"""Import/export helpers for existing annotation files."""
from __future__ import annotations

import json
from pathlib import Path

from multiview_labeler.core.annotation import AnnotationManager
from multiview_labeler.core.models import FrameAnnotations, Keypoint2D


class AnnotationIO:
    @staticmethod
    def import_json(path: Path, annotations: AnnotationManager) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        annotations.frames = {}
        for frame_key, frame_value in payload.items():
            frame_idx = int(frame_key)
            annotations.frames[frame_idx] = {}
            if "2d" in frame_value:
                instances = {0: frame_value}
            else:
                instances = {int(instance_key): instance_value for instance_key, instance_value in frame_value.items()}
            for instance_idx, instance_payload in instances.items():
                annotations.frames[frame_idx][instance_idx] = FrameAnnotations(
                    by_camera={
                        cid: [Keypoint2D(**kp) for kp in keypoints]
                        for cid, keypoints in instance_payload.get("2d", {}).items()
                    },
                    points3d=instance_payload.get("3d", []),
                    qc=instance_payload.get("qc", {}),
                )
        annotations.annotations_changed.emit()
