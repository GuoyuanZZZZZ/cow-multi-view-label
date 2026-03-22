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
        self.frames: Dict[int, FrameAnnotations] = {}
        self.undo_stack: List[Tuple[int, str]] = []
        self.redo_stack: List[Tuple[int, str]] = []

    def frame(self, frame_idx: int) -> FrameAnnotations:
        if frame_idx not in self.frames:
            self.frames[frame_idx] = FrameAnnotations(
                by_camera={cid: ensure_keypoints(len(KEYPOINTS)) for cid in self.dataset.camera_ids()},
                points3d=[None] * len(KEYPOINTS),
                qc={},
            )
        return self.frames[frame_idx]

    def snapshot(self, frame_idx: int) -> str:
        frame = self.frame(frame_idx)
        return json.dumps(
            {
                "by_camera": {cid: [asdict(kp) for kp in kps] for cid, kps in frame.by_camera.items()},
                "points3d": frame.points3d,
                "qc": frame.qc,
            }
        )

    def restore(self, frame_idx: int, snapshot: str) -> None:
        payload = json.loads(snapshot)
        self.frames[frame_idx] = FrameAnnotations(
            by_camera={cid: [Keypoint2D(**kp) for kp in kps] for cid, kps in payload["by_camera"].items()},
            points3d=payload["points3d"],
            qc=payload["qc"],
        )
        self.annotations_changed.emit()

    def push_undo(self, frame_idx: int) -> None:
        self.undo_stack.append((frame_idx, self.snapshot(frame_idx)))
        self.redo_stack.clear()

    def undo(self, frame_idx: int) -> None:
        if not self.undo_stack:
            return
        idx, snap = self.undo_stack.pop()
        self.redo_stack.append((frame_idx, self.snapshot(frame_idx)))
        self.restore(idx, snap)

    def redo(self, frame_idx: int) -> None:
        if not self.redo_stack:
            return
        idx, snap = self.redo_stack.pop()
        self.undo_stack.append((frame_idx, self.snapshot(frame_idx)))
        self.restore(idx, snap)

    def set_point(self, frame_idx: int, camera_id: str, kp_idx: int, x: float, y: float, visibility: str) -> None:
        self.push_undo(frame_idx)
        kp = self.frame(frame_idx).by_camera[camera_id][kp_idx]
        kp.x, kp.y, kp.visibility = float(x), float(y), visibility
        self.annotations_changed.emit()

    def set_visibility(self, frame_idx: int, camera_id: str, kp_idx: int, visibility: str) -> None:
        self.push_undo(frame_idx)
        kp = self.frame(frame_idx).by_camera[camera_id][kp_idx]
        kp.visibility = visibility
        if visibility == "absent":
            kp.x = None
            kp.y = None
        self.annotations_changed.emit()

    def copy_previous_frame(self, frame_idx: int) -> None:
        if frame_idx <= 0 or (frame_idx - 1) not in self.frames:
            return
        self.push_undo(frame_idx)
        self.restore(frame_idx, self.snapshot(frame_idx - 1))

    def interpolate_from_neighbors(self, frame_idx: int) -> None:
        if frame_idx <= 0 or frame_idx + 1 >= self.dataset.frame_count:
            return
        if frame_idx - 1 not in self.frames or frame_idx + 1 not in self.frames:
            return
        self.push_undo(frame_idx)
        prev_frame = self.frame(frame_idx - 1)
        next_frame = self.frame(frame_idx + 1)
        dst = self.frame(frame_idx)
        for cid in self.dataset.camera_ids():
            for i, kp in enumerate(dst.by_camera[cid]):
                p = prev_frame.by_camera[cid][i]
                n = next_frame.by_camera[cid][i]
                if p.is_valid() and n.is_valid():
                    kp.x = (p.x + n.x) / 2.0
                    kp.y = (p.y + n.y) / 2.0
                    kp.visibility = "visible"
        self.annotations_changed.emit()
