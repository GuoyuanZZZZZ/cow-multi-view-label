"""Bootstrap helpers to create a ready-to-run demo workspace."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

from multiview_labeler.core.annotation import AnnotationManager
from multiview_labeler.core.calibration import CalibrationIO
from multiview_labeler.core.dataset import MultiCameraDataset
from multiview_labeler.demo.demo_builder import DemoDataBuilder


def bootstrap_demo() -> Tuple[MultiCameraDataset, AnnotationManager, Path]:
    root = Path(__file__).resolve().parents[2] / "demo_data"
    DemoDataBuilder(root).build()
    dataset = MultiCameraDataset()
    dataset.import_images({cid.name: cid for cid in root.iterdir() if cid.is_dir() and cid.name.startswith("cam")})
    dataset.calibrations = CalibrationIO.load(root / "demo_calibration.json")
    annotations = AnnotationManager(dataset)
    return dataset, annotations, root
