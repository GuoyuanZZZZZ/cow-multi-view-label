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
        self.calibrations = None
        self.setWindowTitle("Chessboard Calibration")
        self.resize(480, 220)
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
        self.run_btn = QtWidgets.QPushButton("Run Calibration")
        self.save_btn = QtWidgets.QPushButton("Save Result")
        self.save_btn.setEnabled(False)
        self.run_btn.clicked.connect(self.run_calibration)
        self.save_btn.clicked.connect(self.save_result)
        layout.addRow("Chessboard cols", self.cols_spin)
        layout.addRow("Chessboard rows", self.rows_spin)
        layout.addRow("Square size", self.square_spin)
        layout.addRow(self.run_btn)
        layout.addRow(self.save_btn)
        layout.addRow("Status", self.status)

    def run_calibration(self) -> None:
        spec = ChessboardSpec(pattern_size=(self.cols_spin.value(), self.rows_spin.value()), square_size=self.square_spin.value())
        try:
            self.calibrations = CalibrationTool.calibrate_multi_camera(self.camera_dirs, spec)
            self.save_btn.setEnabled(True)
            lines = []
            for camera_id, calib in self.calibrations.items():
                lines.append(f"[{camera_id}] fx={calib.K[0,0]:.2f}, fy={calib.K[1,1]:.2f}, cx={calib.K[0,2]:.2f}, cy={calib.K[1,2]:.2f}")
            self.status.setPlainText("Calibration success\n" + "\n".join(lines))
        except Exception as exc:  # GUI feedback path
            self.status.setPlainText(f"Calibration failed: {exc}")

    def save_result(self) -> None:
        if not self.calibrations:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save calibration", "calibration_result.json", "JSON (*.json)")
        if path:
            CalibrationIO.save(Path(path), self.calibrations)
            self.accept()
