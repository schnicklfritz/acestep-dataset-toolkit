"""ACE-Step Dataset Toolkit - application entry point.

Run with:  python dataset_manager.py
"""
import sys
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from ui.main_window import DatasetManager


if __name__ == "__main__":
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    window = DatasetManager()
    window.show()
    sys.exit(app.exec())

