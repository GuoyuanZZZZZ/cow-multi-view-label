"""Calibration dialog for running OpenCV chessboard calibration from the GUI."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

from PySide6 import QtWidgets

from multiview_labeler.core.calibration import CalibrationIO
from multiview_labeler.core.calibration_tool import CalibrationTool, ChessboardSpec


class CalibrationDialog(QtWidgets.QDialog):
    def __init__(self, camera_dirs: Dict[str, Path], parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.camera_dirs = camera_dirs
        self.camera_videos: Dict[str, Path] = {}
        self.calibrations = None
        self.setWindowTitle("Chessboard Calibration")
        self.resize(620, 460)
        layout = QtWidgets.QFormLayout(self)
        self.cols_spin = QtWidgets.QSpinBox()
        self.cols_spin.setRange(3, 20)
        self.cols_spin.setValue(9)
        self.rows_spin = QtWidgets.QSpinBox()
        self.rows_spin.setRange(3, 20)
        self.rows_spin.setValue(6)
        self.square_spin = QtWidgets.QDoubleSpinBox()
        self.square_spin.setRange(0.001, 1000.0)
        self.square_spin.setValue(1.0)
        self.square_spin.setDecimals(3)
        self.status = QtWidgets.QPlainTextEdit()
        self.status.setReadOnly(True)
        self.source_combo = QtWidgets.QComboBox()
        self.source_combo.addItems(["Images", "Videos"])
        self.source_combo.currentTextChanged.connect(self._on_source_changed)
        self.video_stride_spin = QtWidgets.QSpinBox()
        self.video_stride_spin.setRange(1, 120)
        self.video_stride_spin.setValue(10)
        self.video_picker = QtWidgets.QTableWidget(0, 3)
        self.video_picker.setHorizontalHeaderLabels(["Camera", "Video", "Pick"])
        self.video_picker.horizontalHeader().setStretchLastSection(False)
        self.video_picker.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self._build_video_picker()
        self.run_btn = QtWidgets.QPushButton("Run Calibration")
        self.save_btn = QtWidgets.QPushButton("Save Result")
        self.save_btn.setEnabled(False)
        self.run_btn.clicked.connect(self.run_calibration)
        self.save_btn.clicked.connect(self.save_result)
        layout.addRow("Chessboard cols", self.cols_spin)
        layout.addRow("Chessboard rows", self.rows_spin)
        layout.addRow("Square size", self.square_spin)
        layout.addRow("Calibration source", self.source_combo)
        layout.addRow("Video sample stride", self.video_stride_spin)
        layout.addRow("Per-camera videos", self.video_picker)
        layout.addRow(self.run_btn)
        layout.addRow(self.save_btn)
        layout.addRow("Status", self.status)
        self._on_source_changed(self.source_combo.currentText())

    def _build_video_picker(self) -> None:
        rows = sorted(self.camera_dirs.keys())
        self.video_picker.setRowCount(len(rows))
        for row, camera_id in enumerate(rows):
            self.video_picker.setItem(row, 0, QtWidgets.QTableWidgetItem(camera_id))
            self.video_picker.setItem(row, 1, QtWidgets.QTableWidgetItem(""))
            btn = QtWidgets.QPushButton("Select")
            btn.clicked.connect(lambda _checked=False, cid=camera_id, r=row: self._pick_video(cid, r))
            self.video_picker.setCellWidget(row, 2, btn)

    def _pick_video(self, camera_id: str, row: int) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, f"Select calibration video for {camera_id}", str(self.camera_dirs[camera_id]), "Videos (*.mp4 *.mov *.avi *.mkv)")
        if path:
            self.camera_videos[camera_id] = Path(path)
            self.video_picker.setItem(row, 1, QtWidgets.QTableWidgetItem(path))

    def _on_source_changed(self, source: str) -> None:
        is_video = source == "Videos"
        self.video_stride_spin.setEnabled(is_video)
        self.video_picker.setEnabled(is_video)

    def run_calibration(self) -> None:
        spec = ChessboardSpec(pattern_size=(self.cols_spin.value(), self.rows_spin.value()), square_size=self.square_spin.value())
        try:
            if self.source_combo.currentText() == "Videos":
                missing = [cid for cid in self.camera_dirs if cid not in self.camera_videos]
                if missing:
                    raise ValueError(f"Please select calibration videos for: {', '.join(missing)}")
                self.calibrations = CalibrationTool.calibrate_multi_camera_from_videos(
                    {cid: self.camera_videos[cid] for cid in self.camera_dirs},
                    spec,
                    sample_stride=self.video_stride_spin.value(),
                )
            else:
                self.calibrations = CalibrationTool.calibrate_multi_camera(self.camera_dirs, spec)
            self.save_btn.setEnabled(True)
            lines = [
                "Calibration success",
                "Note: extrinsics were solved from one synchronized chessboard frame shared across all cameras.",
            ]
            for camera_id, calib in self.calibrations.items():
                lines.append(f"[{camera_id}] fx={calib.K[0,0]:.2f}, fy={calib.K[1,1]:.2f}, cx={calib.K[0,2]:.2f}, cy={calib.K[1,2]:.2f}")
            self.status.setPlainText("\n".join(lines))
        except Exception as exc:  # GUI feedback path
            self.status.setPlainText(f"Calibration failed: {exc}")

    def save_result(self) -> None:
        if not self.calibrations:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save calibration", "calibration_result.json", "JSON (*.json)")
        if path:
            CalibrationIO.save(Path(path), self.calibrations)
            self.accept()
