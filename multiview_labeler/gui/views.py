"""2D and 3D visual widgets."""
from __future__ import annotations

import math
from typing import List, Optional

import cv2
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PySide6 import QtCore, QtGui

from multiview_labeler.core.constants import SKELETON, VISIBILITY_COLORS
from multiview_labeler.core.models import Keypoint2D


class ImageView(pg.GraphicsLayoutWidget):
    point_changed = QtCore.Signal(str, int, float, float)
    point_selected = QtCore.Signal(int)

    def __init__(self, camera_id: str) -> None:
        super().__init__()
        self.camera_id = camera_id
        self.view = self.addViewBox(lockAspect=True)
        self.view.invertY(True)
        self.image_item = pg.ImageItem()
        self.view.addItem(self.image_item)
        self.scatter = pg.ScatterPlotItem(pxMode=True, size=12)
        self.view.addItem(self.scatter)
        self.line_items: List[pg.GraphicsObject] = []
        self.selected_idx = 0
        self.dragging_idx: Optional[int] = None
        self.drag_distance = 16.0
        self.current_points: List[Keypoint2D] = []
        self.epipolar_item = pg.PlotDataItem(pen=pg.mkPen((180, 0, 180), width=2, style=QtCore.Qt.DashLine))
        self.view.addItem(self.epipolar_item)
        self.scatter.sigClicked.connect(self.on_scatter_clicked)

    def set_image(self, image_bgr: np.ndarray) -> None:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.image_item.setImage(np.flipud(np.swapaxes(rgb, 0, 1)))
        self.view.setRange(QtCore.QRectF(0, 0, image_bgr.shape[1], image_bgr.shape[0]))

    def set_points(self, points: List[Keypoint2D], reprojected=None, anomalies: Optional[set] = None) -> None:
        self.current_points = points
        spots = []
        for i, kp in enumerate(points):
            if kp.x is None or kp.y is None:
                continue
            color = VISIBILITY_COLORS.get(kp.visibility, (200, 200, 200))
            spots.append({
                "pos": (kp.x, kp.y),
                "data": i,
                "brush": pg.mkBrush(color),
                "pen": pg.mkPen("y" if anomalies and i in anomalies else color, width=3),
                "size": 14 if i == self.selected_idx else 11,
            })
        self.scatter.setData(spots)
        for item in self.line_items:
            self.view.removeItem(item)
        self.line_items = []
        for a, b in SKELETON:
            kpa, kpb = points[a], points[b]
            if kpa.x is not None and kpb.x is not None:
                line = pg.PlotDataItem([kpa.x, kpb.x], [kpa.y, kpb.y], pen=pg.mkPen((0, 180, 255), width=2))
                self.view.addItem(line)
                self.line_items.append(line)
        if reprojected:
            for pt in reprojected:
                if pt is None:
                    continue
                cross = pg.ScatterPlotItem([pt[0]], [pt[1]], symbol="x", pen=pg.mkPen("r", width=2), size=10)
                self.view.addItem(cross)
                self.line_items.append(cross)

    def set_epipolar_line(self, line: Optional[np.ndarray], width: int, height: int) -> None:
        if line is None:
            self.epipolar_item.setData([], [])
            return
        a, b, c = line
        pts = []
        for x in [0, width - 1]:
            if abs(b) > 1e-6:
                pts.append((x, (-c - a * x) / b))
        for y in [0, height - 1]:
            if abs(a) > 1e-6:
                pts.append(((-c - b * y) / a, y))
        pts = [(x, y) for x, y in pts if 0 <= x < width and 0 <= y < height]
        self.epipolar_item.setData([pts[0][0], pts[1][0]], [pts[0][1], pts[1][1]]) if len(pts) >= 2 else self.epipolar_item.setData([], [])

    def on_scatter_clicked(self, _plot, points) -> None:
        if points:
            self.selected_idx = int(points[0].data())
            self.point_selected.emit(self.selected_idx)

    def _find_nearby_point(self, x: float, y: float) -> Optional[int]:
        for idx, kp in enumerate(self.current_points):
            if kp.x is not None and kp.y is not None and math.hypot(kp.x - x, kp.y - y) <= self.drag_distance:
                return idx
        return None

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        scene_pos = self.view.mapSceneToView(event.position())
        if event.button() == QtCore.Qt.LeftButton:
            nearby = self._find_nearby_point(scene_pos.x(), scene_pos.y())
            if nearby is not None:
                self.selected_idx = nearby
                self.dragging_idx = nearby
                self.point_selected.emit(nearby)
            else:
                self.point_changed.emit(self.camera_id, self.selected_idx, scene_pos.x(), scene_pos.y())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self.dragging_idx is not None:
            scene_pos = self.view.mapSceneToView(event.position())
            self.point_changed.emit(self.camera_id, self.dragging_idx, scene_pos.x(), scene_pos.y())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        self.dragging_idx = None
        super().mouseReleaseEvent(event)


class Skeleton3DView(gl.GLViewWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setCameraPosition(distance=10)
        grid = gl.GLGridItem()
        grid.scale(1, 1, 1)
        self.addItem(grid)
        self.scatter = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), size=12, color=np.array([[1, 0, 0, 1]]))
        self.addItem(self.scatter)
        self.lines: List[gl.GLLinePlotItem] = []

    def set_points(self, points3d, selected_idx: int, anomalies: set) -> None:
        valid_points, colors = [], []
        for idx, point in enumerate(points3d):
            if point is None:
                continue
            valid_points.append(point)
            colors.append([1, 1, 0, 1] if idx == selected_idx else ([1, 0, 0, 1] if idx in anomalies else [0, 1, 1, 1]))
        if not valid_points:
            valid_points, colors = [[0, 0, 0]], [[0, 0, 0, 0]]
        self.scatter.setData(pos=np.array(valid_points, dtype=float), color=np.array(colors, dtype=float), size=14)
        for item in self.lines:
            self.removeItem(item)
        self.lines = []
        dense = [np.array(p, dtype=float) if p is not None else None for p in points3d]
        for a, b in SKELETON:
            if dense[a] is not None and dense[b] is not None:
                line = gl.GLLinePlotItem(pos=np.vstack([dense[a], dense[b]]), color=(0.3, 0.8, 0.2, 1), width=2, antialias=True)
                self.addItem(line)
                self.lines.append(line)
