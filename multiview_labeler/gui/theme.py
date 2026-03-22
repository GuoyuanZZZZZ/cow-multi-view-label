"""Application theme helpers."""

def build_stylesheet() -> str:
    return """
    QWidget {
        background: #1f2430;
        color: #e6e9ef;
        font-size: 13px;
    }
    QMainWindow, QPlainTextEdit, QTableWidget, QListWidget, QTabWidget::pane {
        background: #252b39;
        border: 1px solid #3b4252;
        border-radius: 8px;
    }
    QPushButton {
        background: #4f6ef7;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 12px;
        font-weight: 600;
    }
    QPushButton:hover { background: #6782ff; }
    QGroupBox {
        border: 1px solid #465066;
        border-radius: 10px;
        margin-top: 12px;
        font-weight: 700;
    }
    QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
    QTabBar::tab {
        background: #2d3548;
        color: #cfd6e6;
        padding: 10px 16px;
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
        margin-right: 4px;
    }
    QTabBar::tab:selected { background: #4f6ef7; color: white; }
    QHeaderView::section {
        background: #2d3548;
        color: #e6e9ef;
        padding: 6px;
        border: 0;
    }
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
        background: #11151d;
        border: 1px solid #465066;
        border-radius: 6px;
        padding: 6px;
    }
    QLabel { color: #dce3f2; }
    """
