"""Workbench-style pages/panels for the annotation tool."""
from __future__ import annotations

import json
from typing import Dict, List

from PySide6 import QtCore, QtWidgets

from multiview_labeler.core.constants import KEYPOINTS
from multiview_labeler.core.models import Keypoint2D


class ProjectPage(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        self.summary = QtWidgets.QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setPlainText(
            "Project workspace\n\n"
            "- Annotation page: multi-view labeling, zoom, triangulation, 3D view\n"
            "- Calibration page: chessboard calibration workflow\n"
            "- Export/QC page: report preview and export actions\n"
        )
        layout.addWidget(QtWidgets.QLabel("Workspace Overview"))
        layout.addWidget(self.summary)

    def update_summary(self, camera_ids: List[str], frame_count: int, image_size: tuple[int, int]) -> None:
        self.summary.setPlainText(
            json.dumps(
                {
                    "camera_ids": camera_ids,
                    "frame_count": frame_count,
                    "image_size": image_size,
                    "pages": ["project", "annotation", "calibration", "export_qc"],
                },
                indent=2,
                ensure_ascii=False,
            )
        )


class KeypointTable(QtWidgets.QTableWidget):
    point_selected = QtCore.Signal(int)

    def __init__(self) -> None:
        super().__init__(len(KEYPOINTS), 3)
        self.setHorizontalHeaderLabels(["Keypoint", "Views Labeled", "3D"]) 
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setStretchLastSection(True)
        for row, name in enumerate(KEYPOINTS):
            self.setItem(row, 0, QtWidgets.QTableWidgetItem(name))
            self.setItem(row, 1, QtWidgets.QTableWidgetItem("0"))
            self.setItem(row, 2, QtWidgets.QTableWidgetItem("No"))
        self.cellClicked.connect(lambda row, _col: self.point_selected.emit(row))

    def update_points(self, by_camera: Dict[str, List[Keypoint2D]], points3d: list) -> None:
        for row, name in enumerate(KEYPOINTS):
            labeled = sum(1 for cid in by_camera if by_camera[cid][row].is_valid())
            self.item(row, 1).setText(str(labeled))
            self.item(row, 2).setText("Yes" if points3d[row] is not None else "No")
            self.item(row, 0).setToolTip(name)


class CalibrationPage(QtWidgets.QWidget):
    run_calibration = QtCore.Signal()
    save_calibration = QtCore.Signal()
    load_calibration = QtCore.Signal()

    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        info = QtWidgets.QLabel(
            "Calibration Workspace\n"
            "1. Prepare chessboard images for each camera.\n"
            "2. Click Run Calibration to estimate K/dist/R/t.\n"
            "3. Save result to JSON for future sessions."
        )
        info.setWordWrap(True)
        button_row = QtWidgets.QHBoxLayout()
        for text, signal in [
            ("Run Calibration", self.run_calibration),
            ("Save Calibration", self.save_calibration),
            ("Load Calibration", self.load_calibration),
        ]:
            btn = QtWidgets.QPushButton(text)
            btn.clicked.connect(signal.emit)
            button_row.addWidget(btn)
        self.status = QtWidgets.QPlainTextEdit()
        self.status.setReadOnly(True)
        layout.addWidget(info)
        layout.addLayout(button_row)
        layout.addWidget(QtWidgets.QLabel("Calibration Status"))
        layout.addWidget(self.status)

    def set_status(self, text: str) -> None:
        self.status.setPlainText(text)


class ExportPage(QtWidgets.QWidget):
    export_requested = QtCore.Signal()

    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        self.report = QtWidgets.QPlainTextEdit()
        self.report.setReadOnly(True)
        export_btn = QtWidgets.QPushButton("Export 2D/3D/QC")
        export_btn.clicked.connect(self.export_requested.emit)
        layout.addWidget(QtWidgets.QLabel("QC / Export Workspace"))
        layout.addWidget(self.report)
        layout.addWidget(export_btn)

    def set_report(self, payload: dict) -> None:
        self.report.setPlainText(json.dumps(payload, indent=2, ensure_ascii=False))


class FramesPage(QtWidgets.QWidget):
    jump_to_frame = QtCore.Signal(int)
    refresh_requested = QtCore.Signal()

    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        info = QtWidgets.QLabel(
            "Representative Frames Workspace\n"
            "Inspired by JARVIS-style workflows: review automatically suggested frames\n"
            "with strong appearance changes to speed up annotation coverage."
        )
        info.setWordWrap(True)
        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.itemClicked.connect(self._on_item_clicked)
        refresh_btn = QtWidgets.QPushButton("Refresh Suggestions")
        refresh_btn.clicked.connect(self.refresh_requested.emit)
        layout.addWidget(info)
        layout.addWidget(self.list_widget)
        layout.addWidget(refresh_btn)

    def set_frames(self, frames: List[int]) -> None:
        self.list_widget.clear()
        for frame_idx in frames:
            self.list_widget.addItem(f"Frame {frame_idx + 1}")

    def _on_item_clicked(self, item: QtWidgets.QListWidgetItem) -> None:
        text = item.text().replace("Frame ", "")
        self.jump_to_frame.emit(int(text) - 1)
