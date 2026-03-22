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
from multiview_labeler.core.constants import KEYPOINTS, SKELETON, VISIBILITY_STATES
from multiview_labeler.core.dataset import MultiCameraDataset
from multiview_labeler.core.geometry import TriangulationEngine
from multiview_labeler.core.models import FrameAnnotations
from multiview_labeler.core.qc import QualityChecker
from multiview_labeler.gui.calibration_dialog import CalibrationDialog
from multiview_labeler.gui.pages import CalibrationPage, ConstraintsPage, ExportPage, FramesPage, ImportWizardPage, KeypointTable, ProjectPage
from multiview_labeler.gui.views import ImageView, Skeleton3DView
from multiview_labeler.tools.exporters import Exporter
from multiview_labeler.tools.frame_sampler import RepresentativeFrameSampler


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, dataset: MultiCameraDataset, annotations: AnnotationManager, demo_root: Path) -> None:
        super().__init__()
        self.dataset = dataset
        self.annotations = annotations
        self.demo_root = demo_root
        self.current_frame = 0
        self.current_keypoint = 0
        self.current_instance = 0
        self.views: Dict[str, ImageView] = {}
        self.current_qc_payload: dict = {}
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
        self.instance_spin = QtWidgets.QSpinBox()
        self.instance_spin.setRange(1, self.project_page.instance_count.value() if hasattr(self, "project_page") else 8)
        self.instance_spin.setValue(1)
        self.instance_spin.valueChanged.connect(self.on_instance_changed)
        toolbar.addWidget(QtWidgets.QLabel("Instance"))
        toolbar.addWidget(self.instance_spin)
        for text, callback in [
            ("Undo", self.on_undo), ("Redo", self.on_redo), ("Copy Prev", self.on_copy_prev),
            ("Interpolate", self.on_interpolate), ("Export", self.on_export),
            ("Run Calib", self.on_run_calibration), ("Save Calib", self.on_save_calib), ("Load Calib", self.on_load_calib),
            ("Zoom In", self.on_zoom_in), ("Zoom Out", self.on_zoom_out), ("Fit", self.on_zoom_fit),
        ]:
            btn = QtWidgets.QPushButton(text)
            btn.clicked.connect(callback)
            toolbar.addWidget(btn)
        main_layout.addLayout(toolbar)
        splitter = QtWidgets.QSplitter()
        self.pages = QtWidgets.QTabWidget()
        self.project_page = ProjectPage()
        self.project_page.instance_count.valueChanged.connect(self.on_project_instance_count_changed)
        self.pages.addTab(self.project_page, "Project")
        self.import_page = ImportWizardPage()
        self.import_page.import_videos_requested.connect(self.on_import_videos)
        self.pages.addTab(self.import_page, "Import")
        self.frames_page = FramesPage()
        self.frames_page.jump_to_frame.connect(self.on_frame_changed)
        self.frames_page.refresh_requested.connect(self.refresh_frame_suggestions)
        self.pages.addTab(self.frames_page, "Frames")
        annotation_page = QtWidgets.QWidget()
        annotation_layout = QtWidgets.QHBoxLayout(annotation_page)
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
        annotation_layout.addWidget(views_widget, 4)
        self.keypoint_table = KeypointTable()
        self.keypoint_table.point_selected.connect(self.on_keypoint_changed)
        annotation_side = QtWidgets.QWidget()
        annotation_side_layout = QtWidgets.QVBoxLayout(annotation_side)
        annotation_side_layout.addWidget(QtWidgets.QLabel("Keypoint Inspector"))
        annotation_side_layout.addWidget(self.keypoint_table, 2)
        annotation_side_layout.addWidget(QtWidgets.QLabel("Realtime Details"))
        self.info_box = QtWidgets.QPlainTextEdit()
        self.info_box.setReadOnly(True)
        annotation_side_layout.addWidget(self.info_box, 3)
        annotation_layout.addWidget(annotation_side, 1)
        self.pages.addTab(annotation_page, "Annotation")
        self.constraints_page = ConstraintsPage([f"{KEYPOINTS[a]}-{KEYPOINTS[b]}" for a, b in SKELETON])
        self.pages.addTab(self.constraints_page, "Constraints")
        splitter.addWidget(self.pages)
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        self.gl_view = Skeleton3DView()
        right_layout.addWidget(self.gl_view, 2)
        self.calibration_page = CalibrationPage()
        self.calibration_page.run_calibration.connect(self.on_run_calibration)
        self.calibration_page.save_calibration.connect(self.on_save_calib)
        self.calibration_page.load_calibration.connect(self.on_load_calib)
        right_layout.addWidget(self.calibration_page, 1)
        self.export_page = ExportPage()
        self.export_page.export_requested.connect(self.on_export)
        right_layout.addWidget(self.export_page, 1)
        splitter.addWidget(right_panel)
        splitter.setSizes([1100, 600])
        main_layout.addWidget(splitter, 1)
        for key, cb in {"A": lambda: self.change_frame(-1), "D": lambda: self.change_frame(1), "Z": self.on_undo, "Y": self.on_redo, "C": self.on_copy_prev, "I": self.on_interpolate}.items():
            QtGui.QShortcut(QtGui.QKeySequence(key), self, activated=cb)
        self.refresh_frame_suggestions()

    def on_frame_changed(self, value: int) -> None:
        self.current_frame = int(value)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(self.current_frame)
        self.frame_slider.blockSignals(False)
        self.refresh_views()

    def change_frame(self, delta: int) -> None:
        self.frame_slider.setValue(max(0, min(self.dataset.frame_count - 1, self.current_frame + delta)))

    def on_keypoint_changed(self, index: int) -> None:
        self.current_keypoint = index
        if isinstance(index, int):
            self.keypoint_combo.blockSignals(True)
            self.keypoint_combo.setCurrentIndex(index)
            self.keypoint_combo.blockSignals(False)
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
        self.annotations.set_point(self.current_frame, camera_id, kp_idx, x, y, self.visibility_combo.currentText(), instance_idx=self.current_instance)

    def on_visibility_changed(self, visibility: str) -> None:
        if visibility != "absent":
            self.refresh_views()
            return
        frame = self.annotations.frame(self.current_frame, self.current_instance)
        changed = False
        for cid in self.dataset.camera_ids():
            if frame.by_camera[cid][self.current_keypoint].visibility != "absent":
                self.annotations.set_visibility(self.current_frame, cid, self.current_keypoint, visibility, instance_idx=self.current_instance)
                changed = True
        if not changed:
            self.refresh_views()

    def on_undo(self) -> None:
        self.annotations.undo(self.current_frame, self.current_instance)

    def on_redo(self) -> None:
        self.annotations.redo(self.current_frame, self.current_instance)

    def on_copy_prev(self) -> None:
        self.annotations.copy_previous_frame(self.current_frame, self.current_instance)

    def on_interpolate(self) -> None:
        self.annotations.interpolate_from_neighbors(self.current_frame, self.current_instance)

    def on_instance_changed(self, value: int) -> None:
        self.current_instance = max(0, value - 1)
        self.refresh_views()

    def on_project_instance_count_changed(self, value: int) -> None:
        self.annotations.instance_count = value
        self.instance_spin.setRange(1, value)

    def on_import_videos(self) -> None:
        selected = {}
        for cid in self.dataset.camera_ids():
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, f"Select video for {cid}", str(self.demo_root), "Videos (*.mp4 *.mov *.avi *.mkv)")
            if not path:
                return
            selected[cid] = Path(path)
        extracted_root = self.demo_root / "imported_videos"
        self.dataset.import_videos(selected, extracted_root)
        self.import_page.set_video_rows([(cid, str(path)) for cid, path in selected.items()])
        self.refresh_frame_suggestions()
        self.refresh_views()

    def on_save_calib(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save calibration", str(self.demo_root / "calibration_saved.json"), "JSON (*.json)")
        if path:
            CalibrationIO.save(Path(path), self.dataset.calibrations)

    def on_run_calibration(self) -> None:
        camera_dirs = {cid: self.dataset.cameras[cid][0].parent for cid in self.dataset.camera_ids()}
        dialog = CalibrationDialog(camera_dirs, self)
        if dialog.exec() and dialog.calibrations:
            self.dataset.calibrations = dialog.calibrations
            self.calibration_page.set_status("Calibration updated from GUI dialog.")
            self.refresh_views()

    def on_load_calib(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load calibration", str(self.demo_root), "JSON (*.json)")
        if path:
            self.dataset.calibrations = CalibrationIO.load(Path(path))
            self.calibration_page.set_status(f"Loaded calibration from {path}")
            self.refresh_views()

    def on_zoom_in(self) -> None:
        for view in self.views.values():
            view.zoom_in()

    def on_zoom_out(self) -> None:
        for view in self.views.values():
            view.zoom_out()

    def on_zoom_fit(self) -> None:
        for view in self.views.values():
            view.fit_to_image()

    def on_export(self) -> None:
        out_dir = self.demo_root / "exports"
        out_dir.mkdir(exist_ok=True)
        payload = {
            str(frame_idx): {
                str(instance_idx): {"2d": {cid: [asdict(kp) for kp in kps] for cid, kps in frame.by_camera.items()}, "3d": frame.points3d, "qc": frame.qc}
                for instance_idx, frame in instances.items()
            }
            for frame_idx, instances in self.annotations.frames.items()
        }
        Exporter.export_json(out_dir / "annotations.json", payload)
        Exporter.export_2d_csv(out_dir / "keypoints_2d.csv", self.annotations)
        Exporter.export_3d_csv(out_dir / "keypoints_3d.csv", self.annotations)
        Exporter.export_json(out_dir / "qc_report.json", {str(frame_idx): {str(instance_idx): frame.qc for instance_idx, frame in instances.items()} for frame_idx, instances in self.annotations.frames.items()})
        QtWidgets.QMessageBox.information(self, "Export", f"Exported files to {out_dir}")

    def refresh_frame_suggestions(self) -> None:
        suggested = RepresentativeFrameSampler.suggest(self.dataset, top_k=8)
        self.frames_page.set_frames(suggested)

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
        self.project_page.update_summary(self.dataset.camera_ids(), self.dataset.frame_count, self.dataset.image_size)
        self.frame_label.setText(f"{self.current_frame + 1}/{self.dataset.frame_count}")
        frame = self.annotations.frame(self.current_frame, self.current_instance)
        frame.points3d = TriangulationEngine.triangulate_frame(frame, self.dataset.calibrations)
        frame.qc = QualityChecker.evaluate(frame, self.dataset.calibrations)
        constraint_status = {}
        for bone_name, length in frame.qc.get("bone_lengths", {}).items():
            constraint = self.constraints_page.constraints().get(bone_name)
            if constraint:
                target, tolerance = constraint
                delta = abs(length - target)
                constraint_status[bone_name] = {"target": target, "observed": length, "delta": delta, "ok": delta <= tolerance}
        self.constraints_page.set_status(constraint_status)
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
        self.keypoint_table.update_points(frame.by_camera, frame.points3d)
        point3d = frame.points3d[self.current_keypoint]
        recommendations = []
        if point3d is not None:
            for cid in self.dataset.camera_ids():
                if not frame.by_camera[cid][self.current_keypoint].is_valid():
                    reco = self.dataset.calibrations[cid].project(np.array(point3d, dtype=float))
                    recommendations.append({"camera": cid, "keypoint": KEYPOINTS[self.current_keypoint], "recommended_uv": [round(float(reco[0]), 2), round(float(reco[1]), 2)]})
        details = {
            "instance": self.current_instance,
            "selected_keypoint": KEYPOINTS[self.current_keypoint],
            "sync_mode": "frame_index",
            "frame_sync": self.dataset.sync_map[self.current_frame] if self.dataset.sync_map else {},
            "recommendations": recommendations,
            "qc": frame.qc,
            "constraints": constraint_status,
        }
        self.current_qc_payload = details
        self.info_box.setPlainText(json.dumps(details, indent=2, ensure_ascii=False))
        self.export_page.set_report(details)
