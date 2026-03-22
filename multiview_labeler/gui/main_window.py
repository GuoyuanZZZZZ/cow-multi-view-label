"""Main application window that wires UI + algorithms together."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from multiview_labeler.core.annotation import AnnotationManager
from multiview_labeler.core.calibration import CalibrationIO
from multiview_labeler.core.constants import KEYPOINTS, VISIBILITY_STATES
from multiview_labeler.core.dataset import MultiCameraDataset
from multiview_labeler.core.geometry import TriangulationEngine
from multiview_labeler.core.models import FrameAnnotations
from multiview_labeler.core.qc import QualityChecker
from multiview_labeler.gui.views import ImageView, Skeleton3DView
from multiview_labeler.tools.exporters import Exporter


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, dataset: MultiCameraDataset, annotations: AnnotationManager, demo_root: Path) -> None:
        super().__init__()
        self.dataset = dataset
        self.annotations = annotations
        self.demo_root = demo_root
        self.current_frame = 0
        self.current_keypoint = 0
        self.views: Dict[str, ImageView] = {}
        self.setWindowTitle("Multi-camera 2D/3D Labeler Demo")
        self.resize(1700, 980)
        self._build_ui()
        self.annotations.annotations_changed.connect(self.refresh_views)
        self.refresh_views()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        toolbar = QtWidgets.QHBoxLayout()
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frame_slider.setRange(0, max(0, self.dataset.frame_count - 1))
        self.frame_slider.valueChanged.connect(self.on_frame_changed)
        toolbar.addWidget(QtWidgets.QLabel("Frame"))
        toolbar.addWidget(self.frame_slider, 1)
        self.frame_label = QtWidgets.QLabel()
        toolbar.addWidget(self.frame_label)
        self.keypoint_combo = QtWidgets.QComboBox()
        self.keypoint_combo.addItems(KEYPOINTS)
        self.keypoint_combo.currentIndexChanged.connect(self.on_keypoint_changed)
        toolbar.addWidget(QtWidgets.QLabel("Keypoint"))
        toolbar.addWidget(self.keypoint_combo)
        self.visibility_combo = QtWidgets.QComboBox()
        self.visibility_combo.addItems(VISIBILITY_STATES)
        self.visibility_combo.currentTextChanged.connect(self.on_visibility_changed)
        toolbar.addWidget(QtWidgets.QLabel("Visibility"))
        toolbar.addWidget(self.visibility_combo)
        for text, callback in [
            ("Undo", self.on_undo), ("Redo", self.on_redo), ("Copy Prev", self.on_copy_prev),
            ("Interpolate", self.on_interpolate), ("Export", self.on_export),
            ("Save Calib", self.on_save_calib), ("Load Calib", self.on_load_calib),
        ]:
            btn = QtWidgets.QPushButton(text)
            btn.clicked.connect(callback)
            toolbar.addWidget(btn)
        main_layout.addLayout(toolbar)
        splitter = QtWidgets.QSplitter()
        views_widget = QtWidgets.QWidget()
        views_layout = QtWidgets.QGridLayout(views_widget)
        for idx, cid in enumerate(self.dataset.camera_ids()):
            view = ImageView(cid)
            view.point_changed.connect(self.on_point_changed)
            view.point_selected.connect(self.on_view_point_selected)
            self.views[cid] = view
            group = QtWidgets.QGroupBox(cid)
            group_layout = QtWidgets.QVBoxLayout(group)
            group_layout.addWidget(view)
            views_layout.addWidget(group, idx // 2, idx % 2)
        splitter.addWidget(views_widget)
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        self.gl_view = Skeleton3DView()
        right_layout.addWidget(self.gl_view, 2)
        self.info_box = QtWidgets.QPlainTextEdit()
        self.info_box.setReadOnly(True)
        right_layout.addWidget(self.info_box, 1)
        splitter.addWidget(right_panel)
        splitter.setSizes([1100, 600])
        main_layout.addWidget(splitter, 1)
        for key, cb in {"A": lambda: self.change_frame(-1), "D": lambda: self.change_frame(1), "Z": self.on_undo, "Y": self.on_redo, "C": self.on_copy_prev, "I": self.on_interpolate}.items():
            QtGui.QShortcut(QtGui.QKeySequence(key), self, activated=cb)

    def on_frame_changed(self, value: int) -> None:
        self.current_frame = value
        self.refresh_views()

    def change_frame(self, delta: int) -> None:
        self.frame_slider.setValue(max(0, min(self.dataset.frame_count - 1, self.current_frame + delta)))

    def on_keypoint_changed(self, index: int) -> None:
        self.current_keypoint = index
        for view in self.views.values():
            view.selected_idx = index
        self.refresh_views()

    def on_view_point_selected(self, index: int) -> None:
        self.current_keypoint = index
        self.keypoint_combo.blockSignals(True)
        self.keypoint_combo.setCurrentIndex(index)
        self.keypoint_combo.blockSignals(False)
        for view in self.views.values():
            view.selected_idx = index
        self.refresh_views()

    def on_point_changed(self, camera_id: str, kp_idx: int, x: float, y: float) -> None:
        self.annotations.set_point(self.current_frame, camera_id, kp_idx, x, y, self.visibility_combo.currentText())

    def on_visibility_changed(self, visibility: str) -> None:
        if visibility != "absent":
            self.refresh_views()
            return
        frame = self.annotations.frame(self.current_frame)
        changed = False
        for cid in self.dataset.camera_ids():
            if frame.by_camera[cid][self.current_keypoint].visibility != "absent":
                self.annotations.set_visibility(self.current_frame, cid, self.current_keypoint, visibility)
                changed = True
        if not changed:
            self.refresh_views()

    def on_undo(self) -> None:
        self.annotations.undo(self.current_frame)

    def on_redo(self) -> None:
        self.annotations.redo(self.current_frame)

    def on_copy_prev(self) -> None:
        self.annotations.copy_previous_frame(self.current_frame)

    def on_interpolate(self) -> None:
        self.annotations.interpolate_from_neighbors(self.current_frame)

    def on_save_calib(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save calibration", str(self.demo_root / "calibration_saved.json"), "JSON (*.json)")
        if path:
            CalibrationIO.save(Path(path), self.dataset.calibrations)

    def on_load_calib(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load calibration", str(self.demo_root), "JSON (*.json)")
        if path:
            self.dataset.calibrations = CalibrationIO.load(Path(path))
            self.refresh_views()

    def on_export(self) -> None:
        out_dir = self.demo_root / "exports"
        out_dir.mkdir(exist_ok=True)
        payload = {
            str(frame_idx): {"2d": {cid: [asdict(kp) for kp in kps] for cid, kps in frame.by_camera.items()}, "3d": frame.points3d, "qc": frame.qc}
            for frame_idx, frame in self.annotations.frames.items()
        }
        Exporter.export_json(out_dir / "annotations.json", payload)
        Exporter.export_2d_csv(out_dir / "keypoints_2d.csv", self.annotations)
        Exporter.export_3d_csv(out_dir / "keypoints_3d.csv", self.annotations)
        Exporter.export_json(out_dir / "qc_report.json", {str(k): v.qc for k, v in self.annotations.frames.items()})
        QtWidgets.QMessageBox.information(self, "Export", f"Exported files to {out_dir}")

    def _update_geometry_guides(self, frame: FrameAnnotations) -> None:
        width, height = self.dataset.image_size
        source_id = next((cid for cid in self.dataset.camera_ids() if frame.by_camera[cid][self.current_keypoint].is_valid()), None)
        for cid, view in self.views.items():
            if source_id and cid != source_id:
                line = TriangulationEngine.epipolar_line(self.dataset.calibrations[source_id], self.dataset.calibrations[cid], frame.by_camera[source_id][self.current_keypoint])
                view.set_epipolar_line(line, width, height)
            else:
                view.set_epipolar_line(None, width, height)

    def refresh_views(self) -> None:
        self.frame_label.setText(f"{self.current_frame + 1}/{self.dataset.frame_count}")
        frame = self.annotations.frame(self.current_frame)
        frame.points3d = TriangulationEngine.triangulate_frame(frame, self.dataset.calibrations)
        frame.qc = QualityChecker.evaluate(frame, self.dataset.calibrations)
        anomalies_by_cam = {cid: set() for cid in self.dataset.camera_ids()}
        for item in frame.qc.get("anomalies", []):
            if item.get("type") == "reprojection":
                anomalies_by_cam[item["camera"]].add(KEYPOINTS.index(item["keypoint"]))
        anomalies_3d = {KEYPOINTS.index(item["keypoint"]) for item in frame.qc.get("anomalies", []) if item.get("keypoint") in KEYPOINTS}
        self._update_geometry_guides(frame)
        for cid, view in self.views.items():
            img = self.dataset.get_frame(cid, self.current_frame)
            reprojected = [None if point is None else self.dataset.calibrations[cid].project(np.array(point, dtype=float)) for point in frame.points3d]
            view.set_image(img)
            view.set_points(frame.by_camera[cid], reprojected=reprojected, anomalies=anomalies_by_cam[cid])
            view.selected_idx = self.current_keypoint
        self.gl_view.set_points(frame.points3d, self.current_keypoint, anomalies_3d)
        point3d = frame.points3d[self.current_keypoint]
        recommendations = []
        if point3d is not None:
            for cid in self.dataset.camera_ids():
                if not frame.by_camera[cid][self.current_keypoint].is_valid():
                    reco = self.dataset.calibrations[cid].project(np.array(point3d, dtype=float))
                    recommendations.append({"camera": cid, "keypoint": KEYPOINTS[self.current_keypoint], "recommended_uv": [round(float(reco[0]), 2), round(float(reco[1]), 2)]})
        details = {
            "selected_keypoint": KEYPOINTS[self.current_keypoint],
            "sync_mode": "frame_index",
            "frame_sync": self.dataset.sync_map[self.current_frame] if self.dataset.sync_map else {},
            "recommendations": recommendations,
            "qc": frame.qc,
        }
        self.info_box.setPlainText(json.dumps(details, indent=2, ensure_ascii=False))
