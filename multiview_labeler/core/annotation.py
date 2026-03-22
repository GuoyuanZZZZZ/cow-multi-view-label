"""Annotation store and editing operations."""
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Dict, List, Tuple

from PySide6 import QtCore

from .constants import KEYPOINTS
from .dataset import MultiCameraDataset
from .models import FrameAnnotations, Keypoint2D, ensure_keypoints


class AnnotationManager(QtCore.QObject):
    annotations_changed = QtCore.Signal()

    def __init__(self, dataset: MultiCameraDataset) -> None:
        super().__init__()
        self.dataset = dataset
        self.instance_count = 1
        self.frames: Dict[int, Dict[int, FrameAnnotations]] = {}
        self.undo_stack: List[Tuple[int, int, str]] = []
        self.redo_stack: List[Tuple[int, int, str]] = []

    def frame(self, frame_idx: int, instance_idx: int = 0) -> FrameAnnotations:
        if frame_idx not in self.frames:
            self.frames[frame_idx] = {}
        if instance_idx not in self.frames[frame_idx]:
            self.frames[frame_idx][instance_idx] = FrameAnnotations(
                by_camera={cid: ensure_keypoints(len(KEYPOINTS)) for cid in self.dataset.camera_ids()},
                points3d=[None] * len(KEYPOINTS),
                qc={},
            )
        return self.frames[frame_idx][instance_idx]

    def snapshot(self, frame_idx: int, instance_idx: int = 0) -> str:
        frame = self.frame(frame_idx, instance_idx)
        return json.dumps(
            {
                "by_camera": {cid: [asdict(kp) for kp in kps] for cid, kps in frame.by_camera.items()},
                "points3d": frame.points3d,
                "qc": frame.qc,
            }
        )

    def restore(self, frame_idx: int, instance_idx: int, snapshot: str) -> None:
        payload = json.loads(snapshot)
        if frame_idx not in self.frames:
            self.frames[frame_idx] = {}
        self.frames[frame_idx][instance_idx] = FrameAnnotations(
            by_camera={cid: [Keypoint2D(**kp) for kp in kps] for cid, kps in payload["by_camera"].items()},
            points3d=payload["points3d"],
            qc=payload["qc"],
        )
        self.annotations_changed.emit()

    def push_undo(self, frame_idx: int, instance_idx: int = 0) -> None:
        self.undo_stack.append((frame_idx, instance_idx, self.snapshot(frame_idx, instance_idx)))
        self.redo_stack.clear()

    def undo(self, frame_idx: int, instance_idx: int = 0) -> None:
        if not self.undo_stack:
            return
        idx, inst, snap = self.undo_stack.pop()
        self.redo_stack.append((frame_idx, instance_idx, self.snapshot(frame_idx, instance_idx)))
        self.restore(idx, inst, snap)

    def redo(self, frame_idx: int, instance_idx: int = 0) -> None:
        if not self.redo_stack:
            return
        idx, inst, snap = self.redo_stack.pop()
        self.undo_stack.append((frame_idx, instance_idx, self.snapshot(frame_idx, instance_idx)))
        self.restore(idx, inst, snap)

    def set_point(self, frame_idx: int, camera_id: str, kp_idx: int, x: float, y: float, visibility: str, instance_idx: int = 0) -> None:
        self.push_undo(frame_idx, instance_idx)
        kp = self.frame(frame_idx, instance_idx).by_camera[camera_id][kp_idx]
        kp.x, kp.y, kp.visibility = float(x), float(y), visibility
        self.annotations_changed.emit()

    def set_visibility(self, frame_idx: int, camera_id: str, kp_idx: int, visibility: str, instance_idx: int = 0) -> None:
        self.push_undo(frame_idx, instance_idx)
        kp = self.frame(frame_idx, instance_idx).by_camera[camera_id][kp_idx]
        kp.visibility = visibility
        if visibility == "absent":
            kp.x = None
            kp.y = None
        self.annotations_changed.emit()

    def copy_previous_frame(self, frame_idx: int, instance_idx: int = 0) -> None:
        if frame_idx <= 0 or (frame_idx - 1) not in self.frames or instance_idx not in self.frames[frame_idx - 1]:
            return
        self.push_undo(frame_idx, instance_idx)
        self.restore(frame_idx, instance_idx, self.snapshot(frame_idx - 1, instance_idx))

    def interpolate_from_neighbors(self, frame_idx: int, instance_idx: int = 0) -> None:
        if frame_idx <= 0 or frame_idx + 1 >= self.dataset.frame_count:
            return
        if (
            frame_idx - 1 not in self.frames or instance_idx not in self.frames[frame_idx - 1] or
            frame_idx + 1 not in self.frames or instance_idx not in self.frames[frame_idx + 1]
        ):
            return
        self.push_undo(frame_idx, instance_idx)
        prev_frame = self.frame(frame_idx - 1, instance_idx)
        next_frame = self.frame(frame_idx + 1, instance_idx)
        dst = self.frame(frame_idx, instance_idx)
        for cid in self.dataset.camera_ids():
            for i, kp in enumerate(dst.by_camera[cid]):
                p = prev_frame.by_camera[cid][i]
                n = next_frame.by_camera[cid][i]
                if p.is_valid() and n.is_valid():
                    kp.x = (p.x + n.x) / 2.0
                    kp.y = (p.y + n.y) / 2.0
                    kp.visibility = "visible"
        self.annotations_changed.emit()
