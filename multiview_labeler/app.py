"""Application entry inside the package."""
from __future__ import annotations

import sys

import pyqtgraph as pg
from PySide6 import QtWidgets

from multiview_labeler.demo.bootstrap import bootstrap_demo
from multiview_labeler.gui.main_window import MainWindow
from multiview_labeler.gui.theme import build_stylesheet


def main() -> int:
    pg.setConfigOptions(antialias=True)
    dataset, annotations, root = bootstrap_demo()
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(build_stylesheet())
    window = MainWindow(dataset, annotations, root)
    window.show()
    return app.exec()
