"""Representative frame suggestion utilities inspired by JARVIS-style workflows."""
from __future__ import annotations

from typing import List

import cv2
import numpy as np

from multiview_labeler.core.dataset import MultiCameraDataset


class RepresentativeFrameSampler:
    @staticmethod
    def suggest(
        dataset: MultiCameraDataset,
        top_k: int = 8,
        camera_id: str | None = None,
        min_gap: int = 2,
        include_uniform: bool = True,
    ) -> List[int]:
        if dataset.frame_count <= 1:
            return [0] if dataset.frame_count == 1 else []
        cam = camera_id or dataset.camera_ids()[0]
        scores = []
        previous = None
        for frame_idx in range(dataset.frame_count):
            image = dataset.get_frame(cam, frame_idx)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            if previous is None:
                scores.append((frame_idx, 0.0))
            else:
                diff = cv2.absdiff(previous, gray)
                score = float(np.mean(diff) + np.std(diff))
                scores.append((frame_idx, score))
            previous = gray
        scores.sort(key=lambda item: item[1], reverse=True)
        chosen: List[int] = []
        for idx, _score in scores:
            if all(abs(idx - existing) >= min_gap for existing in chosen):
                chosen.append(idx)
            if len(chosen) >= max(1, min(top_k, len(scores))):
                break
        if include_uniform and dataset.frame_count > 1:
            uniform_count = min(max(2, top_k // 2), dataset.frame_count)
            uniform = np.linspace(0, dataset.frame_count - 1, num=uniform_count, dtype=int).tolist()
            chosen.extend(uniform)
        chosen = sorted(set(chosen))
        while len(chosen) > top_k:
            chosen.pop(len(chosen) // 2)
        return chosen
