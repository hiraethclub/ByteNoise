"""Application entry point for ByteNoise.

Run with:

    python -m bytenoise.main

or, after installation:

    python main.py
"""

from __future__ import annotations

import sys

from PyQt5.QtWidgets import QApplication

from .gui.main_window import MainWindow


# Dark theme stylesheet kept lightweight - we just want a comfortable default.
DARK_STYLE = """
QMainWindow, QWidget { background-color: #1a1c20; color: #d8dde2; }
QGroupBox {
    border: 1px solid #2c2f36;
    border-radius: 4px;
    margin-top: 10px;
    padding-top: 8px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 6px;
    color: #9ed6c1;
}
QPushButton {
    background-color: #2a2e36;
    border: 1px solid #3a3f48;
    border-radius: 3px;
    padding: 6px 12px;
}
QPushButton:hover { background-color: #353a44; }
QPushButton:pressed { background-color: #4a525e; }
QSlider::groove:horizontal {
    height: 4px;
    background: #2a2e36;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #7ad3b1;
    width: 14px;
    margin: -6px 0;
    border-radius: 7px;
}
QTabWidget::pane { border: 1px solid #2c2f36; }
QTabBar::tab {
    background: #1f2229;
    padding: 6px 12px;
    border: 1px solid #2c2f36;
    border-bottom: none;
}
QTabBar::tab:selected { background: #2a2e36; }
QLabel { color: #d8dde2; }
QStatusBar { background: #15171a; color: #9aa3ad; }
QComboBox {
    background-color: #2a2e36;
    border: 1px solid #3a3f48;
    border-radius: 3px;
    padding: 4px;
}
"""


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLE)
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
